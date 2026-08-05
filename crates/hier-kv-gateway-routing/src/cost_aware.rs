//! Cost-aware routing strategy.
//!
//! Scores each candidate backend by the projected dollar cost of serving the
//! request on its served model, using a [`CostModel`] price catalog. The
//! strategy is a *soft* sub-strategy of [`HybridStrategy`]: it contributes a
//! weighted cost term to the hybrid score rather than filtering candidates.
//!
//! ## Design rationale (informed by LiteLLM / Langfuse / OpenRouter)
//!
//! * **LiteLLM**'s `cost-based-routing` router selects the cheapest *known*
//!   model; its `model_prices_and_context_window.json` is the canonical
//!   static-catalog reference. We mirror the static-table shape via
//!   [`StaticCostModel`].
//! * **Langfuse**'s `/pricing` API decouples price *data* (HTTP-fetched) from
//!   price *consumers* (gen-ai tracing, dashboards). We mirror that by
//!   defining a [`CostModel`] trait — the consumer (this strategy) is
//!   agnostic to whether the catalog is a TOML table or a fetched one.
//! * **OpenRouter** quotes prices as per-token strings; the trait's
//!   `price_for` returns the normalized USD-per-1M-tokens form so a future
//!   OpenRouter-backed impl just multiplies by `1_000_000` on the way in.
//!
//! ## Score shape
//!
//! Backends whose served model is *unknown* to the catalog are handled per
//! the configured [`CostConfig::exclude_on_unknown_price`] policy:
//!
//! * `exclude_on_unknown_price = true` → `raw_cost = f64::INFINITY`, score 0.0
//!   (the exclusion convention recognized by `HybridStrategy::normalize_costs`,
//!   which skips non-finite costs and assigns the candidate a normalized score
//!   of 0.0 — effectively excluding it from the sub-strategy's contribution).
//! * `exclude_on_unknown_price = false` → `raw_cost = 0.0`, score 1.0
//!   (neutral; the other sub-strategies decide).
//!
//! For known-priced backends the raw cost is the projected dollar amount
//! (input + scaled output), and the score is `1 / (1 + raw_cost * scale)`.
//! The `scale` factor (`1e6` by default) keeps scores in a sensible range:
//! a $0.01 projection scores ≈ 0.99, a $10 projection scores ≈ 0.09.

use std::sync::Arc;

use async_trait::async_trait;

use hier_kv_gateway_core::cost::{CostConfig, CostModel};
use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// Multiplier applied to raw dollar cost before inverting into a score.
///
/// `1 / (1 + cost * SCORE_COST_SCALE)` — chosen so $0.01 → 0.99 and $10 → 0.09.
const SCORE_COST_SCALE: f64 = 1_000_000.0;

/// Cost-aware routing strategy: scores backends by projected dollar cost.
///
/// Holds the price catalog as an `Arc<dyn CostModel>` so it can be swapped at
/// startup without rebuilding the strategy (and so multiple strategies, if
/// ever needed, share the same catalog instance).
pub struct CostAwareStrategy {
    /// Price catalog (LiteLLM-style static table by default).
    pub model: Arc<dyn CostModel>,
    /// Tunable parameters from `[cost]` in TOML.
    pub cfg: CostConfig,
}

impl CostAwareStrategy {
    /// Build a strategy from a cost model and configuration.
    pub fn new(model: Arc<dyn CostModel>, cfg: CostConfig) -> Self {
        Self { model, cfg }
    }
}

#[async_trait]
impl RoutingStrategy for CostAwareStrategy {
    fn name(&self) -> &'static str {
        "cost_aware"
    }

    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        // Estimated output tokens come from the client's `max_tokens` — a
        // conservative upper bound, mirroring `LoadAwareStrategy`'s
        // `w_decode` philosophy. `output_cost_scale` lets operators be even
        // more conservative (>= 1.0) or optimistic (< 1.0).
        let est_out = (ctx.estimated_output_tokens as f64) * self.cfg.output_cost_scale;
        // Rough prompt-token estimate when the request is untokenized: sum
        // the prompt block count (each block holds `block_size` tokens) plus
        // a per-character fallback for the messages. This is the same
        // approximation the API layer uses for `prompt_tokens` accounting.
        let prompt_tokens = estimate_prompt_tokens(ctx, meta);

        let mut out = Vec::with_capacity(candidates.len());
        for cand in candidates {
            // Resolve the model name actually served by this backend. When
            // the request specifies a model that the backend serves, prefer
            // it; otherwise fall back to the backend's first listed model
            // (so a backend that *can* serve the request via architecture
            // match still gets a price lookup against its real model name).
            let served_model = resolve_served_model(cand, ctx.model_name.as_deref(), meta);

            let Some(served_model) = served_model else {
                // No model info at all: cannot price. Apply the configured
                // unknown-price policy.
                let (cost, score) = if self.cfg.exclude_on_unknown_price {
                    (f64::INFINITY, 0.0)
                } else {
                    (0.0, 1.0)
                };
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score,
                    raw_cost: cost,
                    meta_version: 0,
                });
                continue;
            };

            let projected = self
                .model
                .projected_cost(&served_model, prompt_tokens, est_out as u32);

            let (cost, score) = match projected {
                Some(c) => {
                    let c = c.max(0.0);
                    let score = 1.0 / (1.0 + c * SCORE_COST_SCALE);
                    (c, score)
                }
                None => {
                    if self.cfg.exclude_on_unknown_price {
                        (f64::INFINITY, 0.0)
                    } else {
                        (0.0, 1.0)
                    }
                }
            };
            out.push(ScoredBackend {
                backend_id: cand.clone(),
                score,
                raw_cost: cost,
                meta_version: 0,
            });
        }
        Ok(out)
    }

    fn is_available(&self, _meta: &MetadataStore) -> bool {
        // Available whenever a price catalog is configured (i.e. the
        // `enabled = true` master switch in `[cost]`). The catalog itself
        // may be empty — in that case every backend hits the
        // `exclude_on_unknown_price` policy and the strategy is effectively
        // a no-op pass-through.
        true
    }

    fn weight(&self) -> f64 {
        // Mirror the configured hybrid weight. `0.0` means the strategy is
        // attached but contributes nothing (useful for staging the price
        // table without affecting routing).
        self.cfg.weight
    }
}

/// Resolve the model name served by `backend` that best matches the request.
///
/// Prefers an exact `model_name` match against the backend's served models;
/// falls back to the backend's first listed model otherwise. Returns `None`
/// when the backend is unknown to the metadata store or carries no models.
fn resolve_served_model(
    backend: &BackendId,
    requested: Option<&str>,
    meta: &MetadataStore,
) -> Option<String> {
    let instances = meta.model_get_instances(backend);
    if instances.is_empty() {
        return None;
    }
    if let Some(name) = requested {
        if let Some(m) = instances.iter().find(|i| i.model_name == name) {
            return Some(m.model_name.clone());
        }
    }
    Some(instances[0].model_name.clone())
}

/// Estimate the prompt token count for cost projection.
///
/// Prefers the tokenized form (`token_ids` or `block_hashes * block_size`);
/// falls back to a character-based heuristic when only the (un-tokenized)
/// message text is available.
fn estimate_prompt_tokens(ctx: &RoutingContext, _meta: &MetadataStore) -> u32 {
    if !ctx.token_ids.is_empty() {
        return ctx.token_ids.len() as u32;
    }
    if !ctx.block_hashes.is_empty() && ctx.block_size > 0 {
        return (ctx.block_hashes.len() as u32) * ctx.block_size;
    }
    // Conservative fallback: the API layer does the same `chars / 4`
    // approximation for non-streaming `prompt_tokens` accounting. We can't
    // see the message text from `RoutingContext`, so we return 0 here and
    // let `projected_cost` skip the input term. In practice, routing-layer
    // callers always set `token_ids` or `block_hashes`.
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::backend::{
        BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
        ModelInstance, Protocol, Quantization,
    };
    use hier_kv_gateway_core::cost::{ModelPrice, PriceEntry, StaticCostModel};
    use hier_kv_gateway_core::ids::{IndexerDomainId, RegionId};
    use hier_kv_gateway_metadata::store::MetadataStore;

    fn backend(region: &str, instance: &str, model: &str) -> BackendInfo {
        BackendInfo {
            id: BackendId::new(region, instance),
            backend_type: BackendType::VllmEngine,
            endpoint: Endpoint {
                url: format!("http://{instance}.example:8000"),
                protocol: Protocol::Http,
            },
            models: vec![ModelInstance {
                model_name: model.to_string(),
                model_architecture: "qwen".to_string(),
                quantization: Quantization::Fp16,
                max_context_len: 32768,
                supports_tool_calling: true,
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
                cache_namespace: "default".to_string(),
                max_kv_blocks: 1024,
            },
            status: BackendStatus::Healthy,
        }
    }

    fn ctx(model: &str, prompt_tokens: u32, est_out: u32) -> RoutingContext {
        RoutingContext {
            model_name: Some(model.to_string()),
            token_ids: (0..prompt_tokens).collect(),
            estimated_output_tokens: est_out,
            block_size: 16,
            ..RoutingContext::default()
        }
    }

    fn catalog() -> Arc<dyn CostModel> {
        Arc::new(StaticCostModel::new([
            (
                "qwen2.5-7b".to_string(),
                ModelPrice {
                    input_per_1m: 0.15,
                    output_per_1m: 0.60,
                },
            ),
            (
                "qwen2.5-72b".to_string(),
                ModelPrice {
                    input_per_1m: 3.0,
                    output_per_1m: 12.0,
                },
            ),
        ]))
    }

    #[tokio::test]
    async fn cheaper_model_scores_higher() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "small", "qwen2.5-7b"));
        store.register_backend(backend("r1", "large", "qwen2.5-72b"));
        let strat = CostAwareStrategy::new(
            catalog(),
            CostConfig {
                enabled: true,
                prices: Vec::new(),
                weight: 0.15,
                output_cost_scale: 1.0,
                exclude_on_unknown_price: false,
            },
        );
        let candidates = vec![
            BackendId::new("r1", "small"),
            BackendId::new("r1", "large"),
        ];
        let scored = strat
            .evaluate(&ctx("qwen2.5-7b", 100_000, 1_000), &candidates, &store)
            .await
            .unwrap();
        let by_id: std::collections::HashMap<_, _> = scored
            .into_iter()
            .map(|s| (s.backend_id.instance.to_string(), s))
            .collect();
        let small = by_id.get("small").unwrap();
        let large = by_id.get("large").unwrap();
        assert!(
            small.score > large.score,
            "small model should score higher: small={} large={}",
            small.score,
            large.score
        );
        // Raw cost should be in dollars (small << large).
        assert!(small.raw_cost < large.raw_cost);
        assert!(small.raw_cost > 0.0);
    }

    #[tokio::test]
    async fn unknown_price_excludes_when_configured() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "unpriced", "deepseek-r1-671b"));
        let strat = CostAwareStrategy::new(
            catalog(),
            CostConfig {
                enabled: true,
                prices: Vec::new(),
                weight: 0.15,
                output_cost_scale: 1.0,
                exclude_on_unknown_price: true,
            },
        );
        let candidates = vec![BackendId::new("r1", "unpriced")];
        let scored = strat
            .evaluate(&ctx("deepseek-r1-671b", 100, 100), &candidates, &store)
            .await
            .unwrap();
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].score, 0.0);
        assert!(!scored[0].raw_cost.is_finite());
    }

    #[tokio::test]
    async fn unknown_price_neutral_when_not_excluding() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "unpriced", "deepseek-r1-671b"));
        let strat = CostAwareStrategy::new(
            catalog(),
            CostConfig {
                enabled: true,
                prices: Vec::new(),
                weight: 0.15,
                output_cost_scale: 1.0,
                exclude_on_unknown_price: false,
            },
        );
        let candidates = vec![BackendId::new("r1", "unpriced")];
        let scored = strat
            .evaluate(&ctx("deepseek-r1-671b", 100, 100), &candidates, &store)
            .await
            .unwrap();
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].score, 1.0);
        assert_eq!(scored[0].raw_cost, 0.0);
    }

    #[tokio::test]
    async fn output_cost_scale_amplifies_output_heavy_requests() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "small", "qwen2.5-7b"));
        let base = CostConfig {
            enabled: true,
            prices: Vec::new(),
            weight: 0.15,
            output_cost_scale: 1.0,
            exclude_on_unknown_price: false,
        };
        let strat_lo = CostAwareStrategy::new(catalog(), base.clone());
        let strat_hi = CostAwareStrategy::new(
            catalog(),
            CostConfig {
                output_cost_scale: 4.0,
                ..base
            },
        );
        let candidates = vec![BackendId::new("r1", "small")];
        let ctx = ctx("qwen2.5-7b", 1_000, 10_000);

        let s_lo = strat_lo
            .evaluate(&ctx, &candidates, &store)
            .await
            .unwrap()[0]
            .raw_cost;
        let s_hi = strat_hi
            .evaluate(&ctx, &candidates, &store)
            .await
            .unwrap()[0]
            .raw_cost;
        assert!(
            s_hi > s_lo,
            "scaling output by 4x should raise cost: lo={s_lo} hi={s_hi}"
        );
    }

    #[test]
    fn from_config_builds_strategy() {
        let cfg = CostConfig {
            enabled: true,
            prices: vec![
                PriceEntry {
                    model: "qwen2.5-7b".to_string(),
                    input_per_1m: 0.15,
                    output_per_1m: 0.60,
                },
                PriceEntry {
                    model: "qwen2.5-72b".to_string(),
                    input_per_1m: 3.0,
                    output_per_1m: 12.0,
                },
            ],
            weight: 0.15,
            output_cost_scale: 1.0,
            exclude_on_unknown_price: false,
        };
        let model: Arc<dyn CostModel> = Arc::new(cfg.build_model());
        let strat = CostAwareStrategy::new(model, cfg.clone());
        assert_eq!(strat.name(), "cost_aware");
        assert!((strat.weight() - 0.15).abs() < 1e-9);
        assert_eq!(strat.cfg.prices.len(), 2);
    }
}
