//! Hybrid intelligent routing strategy (default strategy).
//!
//! Fuses the scores of three sub-strategies — KV-aware, load-aware, and
//! topology-aware:
//!
//! 1. First apply the model-aware strategy as a hard filter on the candidate set (remove `score == 0`).
//! 2. Call `evaluate` on each available sub-strategy to obtain its own `ScoredBackend` list.
//! 3. Dynamically adjust weights: zero out the KV weight when KV is unavailable; reduce the load weight when any candidate's metrics are stale by more than 10s.
//! 4. For each candidate compute `hybrid_score = Σ(weight_s * normalize(score_s))`,
//!    where `normalize` normalizes the cost to the `[0, 1]` range; 0 means the lowest cost under that strategy.
//! 5. When `temperature > 0`, the caller samples via softmax; otherwise greedily pick the highest score.
//!
//! Outputs a `ScoredBackend` list sorted in descending order by `hybrid_score`; the `score` field is the hybrid score.

use std::collections::HashMap;

use async_trait::async_trait;

use hier_kv_gateway_core::config::StrategyWeights;
use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// Load metrics staleness threshold: when exceeded, the load weight is discounted.
const STALE_LOAD_THRESHOLD_SECS: u64 = 10;

/// Hybrid intelligent routing strategy.
pub struct HybridStrategy {
    /// KV-aware sub-strategy.
    pub kv: Box<dyn RoutingStrategy>,
    /// Model-aware sub-strategy (hard filter).
    pub model: Box<dyn RoutingStrategy>,
    /// Load-aware sub-strategy.
    pub load: Box<dyn RoutingStrategy>,
    /// Topology-aware sub-strategy.
    pub topology: Box<dyn RoutingStrategy>,
    /// Static weights for the three sub-strategies.
    pub weights: StrategyWeights,
    /// Routing temperature parameter: when > 0 the caller samples via softmax; when == 0 greedily pick the highest score.
    pub temperature: f64,
}

impl HybridStrategy {
    /// Construct a hybrid strategy with the given sub-strategies.
    pub fn new(
        kv: Box<dyn RoutingStrategy>,
        model: Box<dyn RoutingStrategy>,
        load: Box<dyn RoutingStrategy>,
        topology: Box<dyn RoutingStrategy>,
        weights: StrategyWeights,
        temperature: f64,
    ) -> Self {
        Self {
            kv,
            model,
            load,
            topology,
            weights,
            temperature,
        }
    }

    /// Compute the cost-normalized score (`1 - normalize(cost)`) for the candidate set under a given sub-strategy.
    ///
    /// That is: the candidate with the lowest cost gets 1.0, the one with the highest cost gets 0.0;
    /// when all candidates have the same cost, returns 1.0.
    fn normalize_costs(scores: &[ScoredBackend]) -> HashMap<BackendId, f64> {
        let mut min_cost = f64::INFINITY;
        let mut max_cost = f64::NEG_INFINITY;
        for s in scores {
            if !s.raw_cost.is_finite() {
                // Skip candidates marked with f64::MAX for exclusion: they are not counted here but still appear in the output (set to 0)
                continue;
            }
            if s.raw_cost < min_cost {
                min_cost = s.raw_cost;
            }
            if s.raw_cost > max_cost {
                max_cost = s.raw_cost;
            }
        }
        let mut out = HashMap::new();
        let span = max_cost - min_cost;
        for s in scores {
            if !s.raw_cost.is_finite() {
                // Candidate that does not satisfy constraints: normalized score is 0
                out.insert(s.backend_id.clone(), 0.0);
                continue;
            }
            let normalized = if span > 0.0 {
                (s.raw_cost - min_cost) / span
            } else {
                0.0
            };
            // Convert to "higher is better": the lower the cost, the higher the score
            out.insert(s.backend_id.clone(), 1.0 - normalized);
        }
        out
    }
}

#[async_trait]
impl RoutingStrategy for HybridStrategy {
    fn name(&self) -> &'static str {
        "hybrid"
    }

    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Hard filter via model-aware
        let model_scores = self.model.evaluate(ctx, candidates, meta).await?;
        let filtered: Vec<BackendId> = model_scores
            .iter()
            .filter(|s| s.score > 0.0)
            .map(|s| s.backend_id.clone())
            .collect();
        if filtered.is_empty() {
            return Err(HierKvGatewayError::RoutingFailed(
                "No candidate backend passed the model-aware filter".to_string(),
            ));
        }

        // 2. Dynamic weight adjustment
        let kv_available = self.kv.is_available(meta);
        let mut weight_kv = if kv_available { self.weights.kv } else { 0.0 };

        // Discount the load weight when any candidate's load metrics exceed the staleness threshold
        let mut load_stale = false;
        for cand in &filtered {
            if let Some(freshness) = meta.load_freshness(cand) {
                if freshness.as_secs() > STALE_LOAD_THRESHOLD_SECS {
                    load_stale = true;
                    break;
                }
            }
        }
        let mut weight_load = if load_stale {
            self.weights.load * 0.3
        } else {
            self.weights.load
        };
        let weight_topo = self.weights.topology;

        // Normalize weights so the sum is 1.0
        let total = weight_kv + weight_load + weight_topo;
        if total > 0.0 {
            weight_kv /= total;
            weight_load /= total;
        } else {
            // All zero: fall back to uniform weights
            weight_kv = 1.0 / 3.0;
            weight_load = 1.0 / 3.0;
        }
        let weight_topo = if total > 0.0 {
            weight_topo / total
        } else {
            1.0 / 3.0
        };

        // 3. Each sub-strategy scores the filtered candidate set
        let kv_scores = if weight_kv > 0.0 {
            self.kv.evaluate(ctx, &filtered, meta).await?
        } else {
            Vec::new()
        };
        let load_scores = if weight_load > 0.0 {
            self.load.evaluate(ctx, &filtered, meta).await?
        } else {
            Vec::new()
        };
        let topo_scores = if weight_topo > 0.0 {
            self.topology.evaluate(ctx, &filtered, meta).await?
        } else {
            Vec::new()
        };

        // 4. Normalize each sub-strategy's cost to [0, 1]
        let kv_norm = Self::normalize_costs(&kv_scores);
        let load_norm = Self::normalize_costs(&load_scores);
        let topo_norm = Self::normalize_costs(&topo_scores);

        // 5. Weighted sum to get hybrid_score
        let mut hybrid: Vec<ScoredBackend> = Vec::with_capacity(filtered.len());
        for cand in &filtered {
            let kv_s = kv_norm.get(cand).copied().unwrap_or(0.0);
            let load_s = load_norm.get(cand).copied().unwrap_or(0.0);
            let topo_s = topo_norm.get(cand).copied().unwrap_or(0.0);

            let hybrid_score = weight_kv * kv_s + weight_load * load_s + weight_topo * topo_s;

            // raw_cost takes the negative hybrid_score to keep semantic
            // consistency with "lower raw_cost is better" upstream of SortBackend
            // (higher hybrid_score is better).
            hybrid.push(ScoredBackend {
                backend_id: cand.clone(),
                score: hybrid_score,
                raw_cost: -hybrid_score,
                meta_version: 0,
            });
        }

        // 6. Sort by hybrid_score in descending order
        hybrid.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(hybrid)
    }

    fn is_available(&self, meta: &MetadataStore) -> bool {
        // Model-aware is a hard filter; its availability determines overall availability
        self.model.is_available(meta)
    }

    fn weight(&self) -> f64 {
        // As the top-level composite strategy, the weight is 1.0
        1.0
    }
}
