//! Routing engine: integrates session affinity and the hybrid strategy to produce the final routing decision.
//!
//! [`RoutingEngine`] holds a [`HybridStrategy`] and a number of runtime parameters,
//! and exposes [`RoutingEngine::route`]: it first performs a session affinity check
//! (reuse if hit and the backend is still online), otherwise uses the hybrid strategy
//! to evaluate the candidate set, then selects the final backend from the scores via
//! softmax/greedy, and writes the session affinity back to the metadata store.
//!
//! Degraded routing: when [`DegradationStrategy::is_available`] determines that the
//! gateway is in a degraded state (KV metadata missing/stale or backend metadata
//! empty), and the hybrid strategy evaluation fails or returns empty, it
//! automatically falls back to [`DegradationStrategy`] for prefix-reuse replay
//! routing based on the local [`PrefixReuseHistory`]. Each successful routing
//! records the decision in `prefix_history` for later degraded-state replay.

use std::sync::Arc;
use std::time::Duration;

use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::{BackendId, RegionId};
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::degradation::DegradationStrategy;
use crate::hybrid::HybridStrategy;
use crate::prefix_history::{PrefixReuseHistory, DEFAULT_MAX_ENTRIES};
use crate::strategy::RoutingStrategy;

/// Routing decision result.
#[derive(Clone, Debug)]
pub struct RouteDecision {
    /// The identifier of the finally selected backend.
    pub backend: BackendId,
    /// The name of the strategy that triggered the decision (e.g. `hybrid` or `session_affinity`).
    pub strategy: String,
    /// KV overlap length between the selected backend and this request.
    pub kv_overlap: u32,
    /// Sub-scores of each sub-strategy for the selected backend, for tracing and logging.
    pub scores: Vec<(String, f64)>,
}

/// Routing engine.
pub struct RoutingEngine {
    /// Embedded hybrid strategy.
    pub hybrid: HybridStrategy,
    /// Session affinity TTL.
    pub session_affinity_ttl: Duration,
    /// Maximum retry count.
    pub max_retries: u32,
    /// The Region where this gateway resides.
    pub self_region: RegionId,
    /// Local prefix reuse history for degradation routing.
    pub prefix_history: Arc<PrefixReuseHistory>,
    /// Degradation strategy (fallback).
    pub degradation: DegradationStrategy,
    /// Whether to re-evaluate sub-strategies after selection to populate
    /// `RouteDecision::scores` for tracing.
    ///
    /// Defaults to `false`: the decision carries only the final hybrid score,
    /// avoiding a redundant O(N) re-evaluation of kv/load/topology sub-strategies.
    /// Set to `true` when detailed per-strategy tracing is required (e.g.
    /// debug builds, integration tests).
    pub trace_sub_scores: bool,
}

impl RoutingEngine {
    /// Create a new routing engine.
    ///
    /// Internally creates a [`PrefixReuseHistory`] with the default capacity
    /// (10000 entries) and constructs the corresponding [`DegradationStrategy`],
    /// so this constructor signature remains compatible with historical versions.
    pub fn new(
        hybrid: HybridStrategy,
        session_affinity_ttl: Duration,
        max_retries: u32,
        self_region: RegionId,
    ) -> Self {
        let prefix_history = Arc::new(PrefixReuseHistory::new(DEFAULT_MAX_ENTRIES));
        let degradation = DegradationStrategy::new(prefix_history.clone());
        Self {
            hybrid,
            session_affinity_ttl,
            max_retries,
            self_region,
            prefix_history,
            degradation,
            trace_sub_scores: false,
        }
    }

    /// Enable or disable per-strategy sub-score tracing in [`route`](Self::route).
    ///
    /// When enabled, the engine re-evaluates the kv/load/topology sub-strategies
    /// after selection to populate `RouteDecision::scores`. This roughly doubles
    /// the per-request cost (see `routing_hot_path::engine_route_full` bench:
    /// ~240 µs at n=20 with tracing vs ~120 µs without). Defaults to `false`.
    pub fn with_trace_sub_scores(mut self, enabled: bool) -> Self {
        self.trace_sub_scores = enabled;
        self
    }

    /// Borrow the shared prefix reuse history.
    pub fn prefix_history(&self) -> &Arc<PrefixReuseHistory> {
        &self.prefix_history
    }

    /// Execute the routing decision.
    ///
    /// Flow:
    /// 1. If the request carries a `session_id`, first look up the affinity record; reuse if hit and the backend is still online.
    /// 2. Otherwise collect all candidate backends and call the hybrid strategy to score.
    /// 3. When the hybrid strategy fails or returns empty and the degradation strategy is available, fall back to the degradation strategy to score.
    /// 4. Select the final backend from the scores via softmax/greedy.
    /// 5. Record the decision in prefix_history for later degraded-state replay.
    /// 6. Write the session affinity back to the metadata store.
    pub async fn route(
        &self,
        ctx: &RoutingContext,
        meta: &MetadataStore,
    ) -> Result<RouteDecision> {
        // 1. Session affinity check
        if let Some(session_id) = ctx.session_id.as_ref() {
            if let Some(affinity) = meta.session_get(session_id) {
                // Verify the backend is still online
                if meta.backend_get(&affinity.backend).is_some() {
                    // Compute the current KV overlap
                    let kv_overlap = meta
                        .kv_find_local_overlap(
                            ctx.block_hashes.as_slice(),
                            affinity.backend.clone(),
                        )
                        .await;
                    // Update the session affinity timestamp
                    meta.session_set(
                        session_id.clone(),
                        affinity.backend.clone(),
                        kv_overlap,
                    );
                    // A session affinity hit also counts as a valid dispatch; record it in the prefix history.
                    self.prefix_history
                        .record_dispatch(ctx.block_hashes.as_slice(), &affinity.backend);
                    return Ok(RouteDecision {
                        backend: affinity.backend,
                        strategy: "session_affinity".to_string(),
                        kv_overlap,
                        scores: Vec::new(),
                    });
                }
            }
        }

        // 2. Candidate set: pre-filter by model name when possible, otherwise take all
        let candidates: Vec<BackendId> = match ctx.model_name.as_deref() {
            Some(name) if !name.is_empty() => {
                let by_model = meta.model_find_backends(name);
                if by_model.is_empty() {
                    meta.backends_all()
                        .into_iter()
                        .map(|b| b.id)
                        .collect()
                } else {
                    by_model
                }
            }
            _ => meta
                .backends_all()
                .into_iter()
                .map(|b| b.id)
                .collect(),
        };

        if candidates.is_empty() {
            return Err(HierKvGatewayError::BackendUnavailable);
        }

        // 3. Hybrid strategy scoring; fall back to the degradation strategy on failure or empty result.
        let hybrid_result = self.hybrid.evaluate(ctx, &candidates, meta).await;
        let degradation_available = self.degradation.is_available(meta);
        let (scored, used_strategy): (Vec<ScoredBackend>, &'static str) = match hybrid_result {
            Ok(s) if !s.is_empty() => (s, self.hybrid.name()),
            Ok(_) => {
                // Hybrid strategy returned empty: try degradation.
                if degradation_available {
                    let deg = self.degradation.evaluate(ctx, &candidates, meta).await?;
                    if deg.is_empty() {
                        return Err(HierKvGatewayError::RoutingFailed(
                            "Both the hybrid strategy and the degradation strategy produced no candidate scores".to_string(),
                        ));
                    }
                    (deg, self.degradation.name())
                } else {
                    return Err(HierKvGatewayError::RoutingFailed(
                        "The hybrid strategy produced no candidate scores".to_string(),
                    ));
                }
            }
            Err(e) => {
                // Hybrid strategy errored: try degradation.
                if degradation_available {
                    let deg = self.degradation.evaluate(ctx, &candidates, meta).await?;
                    if deg.is_empty() {
                        return Err(e);
                    }
                    (deg, self.degradation.name())
                } else {
                    return Err(e);
                }
            }
        };

        // 4. Select the backend
        let selected = select_with_temperature(&scored, self.hybrid.temperature)
            .ok_or_else(|| HierKvGatewayError::RoutingFailed("Unable to select a backend from the scores".to_string()))?;

        // 5. Record the dispatch in the prefix history for later degraded-state replay.
        self.prefix_history
            .record_dispatch(ctx.block_hashes.as_slice(), &selected.backend_id);

        // 6. Query the KV overlap between the selected backend and this request
        let kv_overlap = meta
            .kv_find_local_overlap(
                ctx.block_hashes.as_slice(),
                selected.backend_id.clone(),
            )
            .await;

        // 7. Write back the session affinity
        if let Some(session_id) = ctx.session_id.as_ref() {
            meta.session_set(
                session_id.clone(),
                selected.backend_id.clone(),
                kv_overlap,
            );
        }

        // 8. Collect the sub-scores of each sub-strategy for the selected backend (for tracing only).
        //    This is opt-in via `trace_sub_scores`: the re-evaluation roughly
        //    doubles the per-request cost (see `routing_hot_path::engine_route_full`
        //    bench), so it is disabled by default and only turned on when an
        //    upper layer explicitly requests per-strategy tracing.
        let mut scores: Vec<(String, f64)> = Vec::new();
        if self.trace_sub_scores && used_strategy == self.hybrid.name() {
            let kv_scores = self.hybrid.kv.evaluate(ctx, &candidates, meta).await?;
            let load_scores = self.hybrid.load.evaluate(ctx, &candidates, meta).await?;
            let topo_scores = self
                .hybrid
                .topology
                .evaluate(ctx, &candidates, meta)
                .await?;
            for s in &kv_scores {
                if s.backend_id == selected.backend_id {
                    scores.push((self.hybrid.kv.name().to_string(), s.score));
                    break;
                }
            }
            for s in &load_scores {
                if s.backend_id == selected.backend_id {
                    scores.push((self.hybrid.load.name().to_string(), s.score));
                    break;
                }
            }
            for s in &topo_scores {
                if s.backend_id == selected.backend_id {
                    scores.push((self.hybrid.topology.name().to_string(), s.score));
                    break;
                }
            }
        }
        scores.push((used_strategy.to_string(), selected.score));

        Ok(RouteDecision {
            backend: selected.backend_id.clone(),
            strategy: used_strategy.to_string(),
            kv_overlap,
            scores,
        })
    }
}

/// Select a backend from the score list according to `temperature`.
///
/// - `temperature <= 0`: greedily pick the highest score.
/// - `temperature > 0`: sample from the `softmax(score / temperature)` distribution;
///   returns `None` when the score list is empty.
pub fn select_with_temperature(scores: &[ScoredBackend], temperature: f64) -> Option<ScoredBackend> {
    if scores.is_empty() {
        return None;
    }
    if temperature <= 0.0 {
        // Greedy: return the highest score
        return scores
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
    }
    // softmax sampling
    let max_score = scores
        .iter()
        .map(|s| s.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores
        .iter()
        .map(|s| ((s.score - max_score) / temperature).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        // Numerical anomaly: fall back to greedy
        return scores
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
    }
    // Sample using a thread-local RNG
    let mut rng = rand::rng();
    let weights: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    let dist = WeightedIndex::new(&weights).ok()?;
    let idx = dist.sample(&mut rng);
    Some(scores[idx].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::ids::BackendId;

    fn scored(backend: &str, score: f64) -> ScoredBackend {
        ScoredBackend {
            backend_id: BackendId::new("r1", backend),
            score,
            raw_cost: -score,
            meta_version: 0,
        }
    }

    #[test]
    fn greedy_picks_highest() {
        let s = vec![scored("a", 0.1), scored("b", 0.9), scored("c", 0.5)];
        let chosen = select_with_temperature(&s, 0.0).unwrap();
        assert_eq!(chosen.backend_id.instance.as_str(), "b");
    }

    #[test]
    fn softmax_returns_some_on_non_empty() {
        let s = vec![scored("a", 0.1), scored("b", 0.9)];
        let chosen = select_with_temperature(&s, 0.5).unwrap();
        // We do not require a specific one to be selected, but it must be in the set
        assert!(s.iter().any(|x| x.backend_id == chosen.backend_id));
    }

    #[test]
    fn empty_returns_none() {
        let s: Vec<ScoredBackend> = Vec::new();
        assert!(select_with_temperature(&s, 0.0).is_none());
        assert!(select_with_temperature(&s, 1.0).is_none());
    }
}
