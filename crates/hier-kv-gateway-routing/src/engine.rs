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
    /// Optional primary strategy that replaces the hybrid strategy in scoring.
    ///
    /// Set via [`with_primary_strategy`](Self::with_primary_strategy) — e.g. a
    /// [`RoundRobinStrategy`](crate::round_robin::RoundRobinStrategy) baseline
    /// selected through `routing.strategy = "round_robin"`. When `None`, the
    /// hybrid strategy scores candidates (the historical default).
    primary: Option<Box<dyn RoutingStrategy>>,
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
    /// Selection temperature (`<= 0` greedy, `> 0` softmax sampling), mirrored
    /// from the hybrid strategy at construction so it also governs selection
    /// when a non-hybrid primary strategy is installed.
    pub temperature: f64,
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
        let temperature = hybrid.temperature;
        Self {
            hybrid,
            primary: None,
            session_affinity_ttl,
            max_retries,
            self_region,
            prefix_history,
            degradation,
            temperature,
            trace_sub_scores: false,
        }
    }

    /// Install a primary strategy that replaces the embedded hybrid strategy
    /// for candidate scoring (builder style).
    ///
    /// The hybrid strategy is retained: its sub-strategies still back
    /// `trace_sub_scores`, and its temperature still governs final selection.
    pub fn with_primary_strategy(mut self, strategy: Box<dyn RoutingStrategy>) -> Self {
        self.primary = Some(strategy);
        self
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

    /// Borrow the adaptive weight controller attached to the embedded hybrid
    /// strategy, if any.
    ///
    /// The gateway's forwarding loop uses this handle to feed execution
    /// metrics (per-backend forward success/failure/latency, KV hit ratio)
    /// back into the controller after each request.
    pub fn adaptive_controller(&self) -> Option<&Arc<crate::adaptive::AdaptiveWeightController>> {
        self.hybrid.adaptive()
    }

    /// Snapshot of the hybrid weights most recently used for scoring.
    ///
    /// Returns the adaptive controller's cached effective weights when one is
    /// attached, otherwise the static configured weights. Intended for
    /// decision telemetry; callers should only attach the snapshot to events
    /// whose winning strategy is the hybrid one.
    pub fn weight_snapshot(&self) -> hier_kv_gateway_core::config::StrategyWeights {
        self.hybrid.weight_snapshot()
    }

    /// Execute the routing decision and return the single best backend.
    ///
    /// Equivalent to [`route_candidates`](Self::route_candidates) with
    /// `limit == 1`; see that method for the full flow.
    pub async fn route(
        &self,
        ctx: &RoutingContext,
        meta: &MetadataStore,
    ) -> Result<RouteDecision> {
        let mut ranked = self.route_candidates(ctx, meta, 1).await?;
        ranked.pop().ok_or_else(|| {
            HierKvGatewayError::RoutingFailed("Unable to select a backend from the scores".to_string())
        })
    }

    /// Execute the routing decision and return up to `limit` ranked candidates.
    ///
    /// Flow:
    /// 1. If the request carries a `session_id`, first look up the affinity record; on a hit the
    ///    affinity backend becomes the head of the returned list (remaining slots are filled with
    ///    failover candidates so the forwarding loop can still retry elsewhere).
    /// 2. Otherwise collect all candidate backends and call the primary (or hybrid) strategy to score.
    /// 3. When the strategy fails or returns empty and the degradation strategy is available, fall back to the degradation strategy to score.
    /// 4. Select up to `limit` backends from the scores: greedy order when `temperature <= 0`,
    ///    softmax sampling without replacement when `temperature > 0`.
    /// 5. Record the head decision in prefix_history for later degraded-state replay.
    /// 6. Write the session affinity of the head decision back to the metadata store.
    ///
    /// The returned list is ordered by preference: element 0 is what
    /// [`route`](Self::route) would have returned; subsequent elements are the
    /// failover order for the forwarding loop's retry logic.
    pub async fn route_candidates(
        &self,
        ctx: &RoutingContext,
        meta: &MetadataStore,
        limit: usize,
    ) -> Result<Vec<RouteDecision>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        // 1. Session affinity check: on a hit, pin the affinity backend as head.
        let mut head: Option<RouteDecision> = None;
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
                    head = Some(RouteDecision {
                        backend: affinity.backend,
                        strategy: "session_affinity".to_string(),
                        kv_overlap,
                        scores: Vec::new(),
                    });
                }
            }
        }

        if head.is_some() && limit == 1 {
            return Ok(vec![head.expect("checked is_some")]);
        }

        // 2. Candidate set: pre-filter by model name when possible, otherwise take all.
        //    The affinity head (when present) is excluded so it is not scored twice.
        let head_backend = head.as_ref().map(|h| &h.backend);
        let collect = |ids: Vec<BackendId>| -> Vec<BackendId> {
            let mut ids: Vec<BackendId> = ids
                .into_iter()
                .filter(|id| Some(id) != head_backend)
                .collect();
            // Deterministic candidate order: the metadata store is backed by
            // DashMaps whose iteration order is unspecified, which would make
            // order-sensitive strategies (round-robin rotation) and the
            // emitted decision events nondeterministic across runs.
            ids.sort();
            ids
        };
        let candidates: Vec<BackendId> = match ctx.model_name.as_deref() {
            Some(name) if !name.is_empty() => {
                let by_model = meta.model_find_backends(name);
                if by_model.is_empty() {
                    collect(meta.backends_all().into_iter().map(|b| b.id).collect())
                } else {
                    collect(by_model)
                }
            }
            _ => collect(meta.backends_all().into_iter().map(|b| b.id).collect()),
        };

        if candidates.is_empty() {
            // No failover candidates: the affinity head (if any) is all we have.
            return match head {
                Some(h) => Ok(vec![h]),
                None => Err(HierKvGatewayError::BackendUnavailable),
            };
        }

        // 3. Strategy scoring; fall back to the degradation strategy on failure or empty result.
        let primary = self.primary.as_deref().unwrap_or(&self.hybrid);
        let strategy_result = primary.evaluate(ctx, &candidates, meta).await;
        let degradation_available = self.degradation.is_available(meta);
        let (scored, used_strategy): (Vec<ScoredBackend>, &'static str) = match strategy_result {
            Ok(s) if !s.is_empty() => (s, primary.name()),
            Ok(_) => {
                // Primary strategy returned empty: try degradation.
                if degradation_available {
                    let deg = self.degradation.evaluate(ctx, &candidates, meta).await?;
                    if deg.is_empty() {
                        return Err(HierKvGatewayError::RoutingFailed(
                            "Both the primary strategy and the degradation strategy produced no candidate scores".to_string(),
                        ));
                    }
                    (deg, self.degradation.name())
                } else {
                    return Err(HierKvGatewayError::RoutingFailed(
                        "The primary strategy produced no candidate scores".to_string(),
                    ));
                }
            }
            Err(e) => {
                // Primary strategy errored: try degradation.
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

        // 4. Select up to `limit - head` backends (sampling without replacement).
        let slots = limit - head.as_ref().map_or(0, |_| 1);
        let ranked = select_ranked(&scored, self.temperature, slots);
        if ranked.is_empty() {
            return match head {
                Some(h) => Ok(vec![h]),
                None => Err(HierKvGatewayError::RoutingFailed(
                    "Unable to select a backend from the scores".to_string(),
                )),
            };
        }

        // 5. Record the dispatch in the prefix history for later degraded-state
        //    replay (only when the head did not already come from affinity).
        if head.is_none() {
            self.prefix_history
                .record_dispatch(ctx.block_hashes.as_slice(), &ranked[0].backend_id);
        }

        // 6. Write back the session affinity of the head decision.
        if head.is_none() {
            if let Some(session_id) = ctx.session_id.as_ref() {
                let kv_overlap = meta
                    .kv_find_local_overlap(
                        ctx.block_hashes.as_slice(),
                        ranked[0].backend_id.clone(),
                    )
                    .await;
                meta.session_set(
                    session_id.clone(),
                    ranked[0].backend_id.clone(),
                    kv_overlap,
                );
            }
        }

        // 7. Collect the sub-scores of each sub-strategy for the head pick (for tracing only).
        //    This is opt-in via `trace_sub_scores`: the re-evaluation roughly
        //    doubles the per-request cost (see `routing_hot_path::engine_route_full`
        //    bench), so it is disabled by default and only turned on when an
        //    upper layer explicitly requests per-strategy tracing.
        let mut head_scores: Vec<(String, f64)> = Vec::new();
        if self.trace_sub_scores && used_strategy == self.hybrid.name() {
            let kv_scores = self.hybrid.kv.evaluate(ctx, &candidates, meta).await?;
            let load_scores = self.hybrid.load.evaluate(ctx, &candidates, meta).await?;
            let topo_scores = self
                .hybrid
                .topology
                .evaluate(ctx, &candidates, meta)
                .await?;
            for s in &kv_scores {
                if s.backend_id == ranked[0].backend_id {
                    head_scores.push((self.hybrid.kv.name().to_string(), s.score));
                    break;
                }
            }
            for s in &load_scores {
                if s.backend_id == ranked[0].backend_id {
                    head_scores.push((self.hybrid.load.name().to_string(), s.score));
                    break;
                }
            }
            for s in &topo_scores {
                if s.backend_id == ranked[0].backend_id {
                    head_scores.push((self.hybrid.topology.name().to_string(), s.score));
                    break;
                }
            }
        }

        // 8. Materialize the ranked decisions, computing KV overlap per candidate.
        let mut decisions: Vec<RouteDecision> = Vec::with_capacity(slots);
        if let Some(h) = head {
            decisions.push(h);
        }
        for (i, pick) in ranked.iter().enumerate() {
            let kv_overlap = meta
                .kv_find_local_overlap(
                    ctx.block_hashes.as_slice(),
                    pick.backend_id.clone(),
                )
                .await;
            let mut scores: Vec<(String, f64)> = Vec::new();
            if i == 0 && head_scores.is_empty() == false {
                scores = head_scores.clone();
            }
            scores.push((used_strategy.to_string(), pick.score));
            decisions.push(RouteDecision {
                backend: pick.backend_id.clone(),
                strategy: used_strategy.to_string(),
                kv_overlap,
                scores,
            });
        }

        Ok(decisions)
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

/// Select up to `limit` backends from the score list, ordered by preference.
///
/// - `temperature <= 0`: greedy — equivalent to sorting by score descending
///   and truncating.
/// - `temperature > 0`: repeated softmax sampling *without replacement*, so
///   high-score candidates are likely to lead but the failover order still
///   covers distinct backends.
///
/// Returns fewer than `limit` entries when `scores` is smaller; returns an
/// empty list only when `scores` is empty or `limit == 0`.
pub fn select_ranked(
    scores: &[ScoredBackend],
    temperature: f64,
    limit: usize,
) -> Vec<ScoredBackend> {
    let mut pool: Vec<ScoredBackend> = scores.to_vec();
    let mut out: Vec<ScoredBackend> = Vec::with_capacity(limit.min(pool.len()));
    while out.len() < limit && !pool.is_empty() {
        let Some(pick) = select_with_temperature(&pool, temperature) else {
            break;
        };
        let Some(pos) = pool
            .iter()
            .position(|s| s.backend_id == pick.backend_id)
        else {
            break;
        };
        out.push(pool.remove(pos));
    }
    out
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

    #[test]
    fn select_ranked_greedy_orders_descending_and_truncates() {
        let s = vec![scored("a", 0.1), scored("b", 0.9), scored("c", 0.5)];
        let ranked = select_ranked(&s, 0.0, 2);
        let order: Vec<&str> = ranked
            .iter()
            .map(|x| x.backend_id.instance.as_str())
            .collect();
        assert_eq!(order, ["b", "c"]);
    }

    #[test]
    fn select_ranked_returns_all_when_limit_exceeds_pool() {
        let s = vec![scored("a", 0.1), scored("b", 0.9)];
        let ranked = select_ranked(&s, 0.0, 5);
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn select_ranked_never_repeats_a_backend() {
        // Sampling without replacement: even at high temperature every pick
        // must be distinct and cover the whole pool.
        let s = vec![scored("a", 1.0), scored("b", 0.9), scored("c", 0.8)];
        let ranked = select_ranked(&s, 5.0, 3);
        assert_eq!(ranked.len(), 3);
        let mut ids: Vec<&str> = ranked
            .iter()
            .map(|x| x.backend_id.instance.as_str())
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    #[test]
    fn select_ranked_limit_zero_is_empty() {
        let s = vec![scored("a", 0.1)];
        assert!(select_ranked(&s, 0.0, 0).is_empty());
    }

    // ---------------------------------------------------------------------
    // Engine-level: round-robin primary strategy
    // ---------------------------------------------------------------------

    mod round_robin_primary {
        use crate::engine::RoutingEngine;
        use crate::hybrid::HybridStrategy;
        use crate::kv_aware::KvAwareStrategy;
        use crate::load_aware::LoadAwareStrategy;
        use crate::model_aware::ModelAwareStrategy;
        use crate::round_robin::RoundRobinStrategy;
        use crate::topology_aware::TopologyAwareStrategy;
        use hier_kv_gateway_core::backend::{
            BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
            ModelInstance, Protocol, Quantization,
        };
        use hier_kv_gateway_core::config::StrategyWeights;
        use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId};
        use hier_kv_gateway_core::request::RoutingContext;
        use hier_kv_gateway_metadata::store::MetadataStore;
        use std::time::Duration;

        fn backend_info(region: &str, instance: &str, model: &str) -> BackendInfo {
            BackendInfo {
                id: BackendId::new(region, instance),
                backend_type: BackendType::VllmEngine,
                endpoint: Endpoint {
                    url: format!("http://{instance}.example:8000"),
                    protocol: Protocol::Http,
                },
                models: vec![ModelInstance {
                    model_name: model.to_string(),
                    model_architecture: "llama".to_string(),
                    quantization: Quantization::Fp16,
                    max_context_len: 4096,
                    supports_tool_calling: false,
                    supports_streaming: true,
                }],
                region: RegionId::new(region),
                indexer_domain: IndexerDomainId::new(0),
                capabilities: BackendCapabilities {
                    supports_kv_events: false,
                    supports_batching: true,
                    max_batch_size: 32,
                    gpu_count: 1,
                    gpu_memory_gb: 24,
                },
                kv_config: KvConfig {
                    block_size: 16,
                    cache_namespace: String::new(),
                    max_kv_blocks: 1024,
                },
                status: BackendStatus::Healthy,
            }
        }

        fn build_engine(self_region: &str) -> RoutingEngine {
            let hybrid = HybridStrategy::new(
                Box::new(KvAwareStrategy::default()),
                Box::new(ModelAwareStrategy::default()),
                Box::new(LoadAwareStrategy::default()),
                Box::new(TopologyAwareStrategy {
                    w_rtt: 1.0,
                    w_bw: 0.0,
                    self_region: RegionId::new(self_region),
                }),
                StrategyWeights {
                    kv: 0.35,
                    load: 0.30,
                    topology: 0.20,
                },
                0.0,
            );
            RoutingEngine::new(hybrid, Duration::from_secs(300), 3, RegionId::new(self_region))
                .with_primary_strategy(Box::new(RoundRobinStrategy::new()))
        }

        fn ctx(model: &str) -> RoutingContext {
            RoutingContext {
                model_name: Some(model.to_string()),
                estimated_output_tokens: 32,
                ..RoutingContext::default()
            }
        }

        #[tokio::test]
        async fn round_robin_primary_rotates_head_and_ranks_all() {
            let store = MetadataStore::new();
            for inst in ["a", "b", "c"] {
                store.register_backend(backend_info("r1", inst, "m"));
            }
            let engine = build_engine("r1");

            // First call: head = a, failover order covers all three.
            let d1 = engine
                .route_candidates(&ctx("m"), &store, 3)
                .await
                .expect("routing should succeed");
            assert_eq!(d1.len(), 3);
            assert_eq!(d1[0].strategy, "round_robin");
            let order1: Vec<&str> = d1
                .iter()
                .map(|d| d.backend.instance.as_str())
                .collect();
            assert_eq!(order1, ["a", "b", "c"]);

            // Second call: the cursor advanced — head = b, wrap-around order.
            let d2 = engine
                .route_candidates(&ctx("m"), &store, 3)
                .await
                .expect("routing should succeed");
            let order2: Vec<&str> = d2
                .iter()
                .map(|d| d.backend.instance.as_str())
                .collect();
            assert_eq!(order2, ["b", "c", "a"]);
        }

        #[tokio::test]
        async fn round_robin_primary_route_picks_rotate() {
            let store = MetadataStore::new();
            for inst in ["x", "y"] {
                store.register_backend(backend_info("r1", inst, "m"));
            }
            let engine = build_engine("r1");

            let first = engine.route(&ctx("m"), &store).await.unwrap();
            let second = engine.route(&ctx("m"), &store).await.unwrap();
            assert_ne!(first.backend, second.backend);
            let third = engine.route(&ctx("m"), &store).await.unwrap();
            assert_eq!(first.backend, third.backend);
        }
    }
}
