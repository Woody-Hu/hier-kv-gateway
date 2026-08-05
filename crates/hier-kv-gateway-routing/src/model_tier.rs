//! Large/small model tiering routing strategy.
//!
//! This is the *strategy* half of large↔small model coordination; the *data*
//! half ([`ModelTierConfig`]) lives in the core crate.
//!
//! ## Two modes
//!
//! * [`TierRoutingPolicy::Pick`] — a **soft sub-strategy** (plugin in the
//!   hybrid ensemble). It scores each candidate by how well its tier matches
//!   the request's *complexity*: short prompt + no tools + low `max_tokens`
//!   ⇒ prefer small; long prompt or tool-calling ⇒ prefer large. The score
//!   is expressed as a `raw_cost` (lower = better) so the hybrid strategy's
//!   [`HybridStrategy::normalize_costs`] turns it into a `[0,1]` term.
//!
//! * [`TierRoutingPolicy::Fallback`] — a **primary strategy**. It ranks every
//!   small-model backend ahead of every large-model backend unconditionally.
//!   When installed via
//!   [`RoutingEngine::with_primary_strategy`](crate::engine::RoutingEngine::with_primary_strategy),
//!   the engine's ranked candidate list becomes "try small first, then
//!   large", and the forwarding loop's existing retry logic realizes the
//!   fallback chain for free — no new retry code needed. This mirrors
//!   LiteLLM's `fallbacks` feature, but realized at the routing layer instead
//!   of the forwarding layer.
//!
//! ## Tier resolution
//!
//! A backend may serve multiple models. The strategy resolves a backend's
//! tier by scanning its served model instances and looking each up in the
//! configured `tiers` table. When the request names a specific model, that
//! model's tier is preferred; otherwise the first listed model with a known
//! tier wins. Backends serving no tier-listed model are "unknown" — under
//! `Pick` they get a neutral mid-range cost, under `Fallback` they rank
//! between small and large so they're tried after small fails but before
//! large is exhausted.
//!
//! ## Honesty note
//!
//! "Seamless" large→small fallback on *quality* signals (e.g. the small
//! model produced a low-confidence answer) requires evaluating the response,
//! which is out of scope for a routing-only strategy. See the design doc for
//! the future-work sketch.

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::model_tier::{ModelTier, ModelTierConfig, TierRoutingPolicy};
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// Cost assigned to a backend whose tier matches the request's preference
/// (lower raw_cost ⇒ higher normalized score in the hybrid ensemble).
const COST_TIER_MATCH: f64 = 0.0;
/// Cost assigned to a backend whose tier does NOT match the request's
/// preference.
const COST_TIER_MISMATCH: f64 = 1.0;
/// Cost assigned to a backend whose tier is unknown. Placed mid-range so it
/// is neither preferred nor excluded — the other sub-strategies decide.
const COST_TIER_UNKNOWN: f64 = 0.5;

/// Score (higher = better) for a small-model backend under the `Fallback`
/// primary strategy. Small is tried first.
const SCORE_FALLBACK_SMALL: f64 = 1.0;
/// Score for an unknown-tier backend under `Fallback`. Tried after small
/// fails but before large.
const SCORE_FALLBACK_UNKNOWN: f64 = 0.5;
/// Score for a large-model backend under `Fallback`. Tried last.
const SCORE_FALLBACK_LARGE: f64 = 0.0;

/// Large/small model tiering routing strategy.
///
/// Holds an `Arc<ModelTierConfig>` so it can be shared with the cost-aware
/// strategy (which may want to read tier info for price scaling) and so
/// cloning the strategy for plugin registration is cheap.
pub struct ModelTierStrategy {
    /// The configured tier table + policy + weight.
    pub cfg: std::sync::Arc<ModelTierConfig>,
}

impl ModelTierStrategy {
    /// Build a strategy from a tier configuration.
    pub fn new(cfg: std::sync::Arc<ModelTierConfig>) -> Self {
        Self { cfg }
    }

    /// Resolve the [`ModelTier`] of a backend by scanning its served models.
    ///
    /// Prefers the model named in the request *when the backend actually
    /// serves it*; otherwise takes the first served model that has a tier
    /// entry. Returns `None` when the backend serves no tier-listed model.
    ///
    /// The "backend actually serves it" guard is essential: without it every
    /// candidate would inherit the requested model's tier, collapsing the
    /// small/large distinction the strategy exists to exploit.
    fn resolve_tier(
        &self,
        backend: &BackendId,
        requested: Option<&str>,
        meta: &MetadataStore,
    ) -> Option<ModelTier> {
        let instances = meta.model_get_instances(backend);
        if instances.is_empty() {
            return None;
        }
        // Prefer the requested model's tier when the backend serves it.
        if let Some(name) = requested {
            if instances.iter().any(|i| i.model_name == name) {
                if let Some(tier) = self.cfg.tier_for(name) {
                    return Some(tier);
                }
            }
        }
        // Otherwise: first served model with a known tier.
        for inst in &instances {
            if let Some(tier) = self.cfg.tier_for(&inst.model_name) {
                return Some(tier);
            }
        }
        None
    }

    /// Whether the request is "complex" under the `Pick` policy's thresholds.
    ///
    /// A request is complex when ANY of:
    ///   * prompt token count > `prompt_token_threshold`,
    ///   * `max_tokens` > `max_token_threshold`,
    ///   * `prefer_large_for_tools` is on and the request carries tools.
    ///
    /// Simple requests prefer small; complex requests prefer large.
    fn is_complex(&self, ctx: &RoutingContext) -> bool {
        match self.cfg.policy {
            TierRoutingPolicy::Pick {
                prompt_token_threshold,
                max_token_threshold,
                prefer_large_for_tools,
            } => {
                let prompt_tokens = estimate_prompt_tokens(ctx);
                if prompt_tokens > prompt_token_threshold {
                    return true;
                }
                if ctx.estimated_output_tokens > max_token_threshold {
                    return true;
                }
                if prefer_large_for_tools && ctx.requires_tool_calling {
                    return true;
                }
                false
            }
            // Fallback doesn't classify complexity; it's unconditional.
            TierRoutingPolicy::Fallback => false,
        }
    }
}

/// Estimate the prompt token count from the routing context.
///
/// Prefers the tokenized form; falls back to `block_hashes.len() * block_size`
/// when only block hashes are available. Mirrors the heuristic used by the
/// cost-aware strategy.
fn estimate_prompt_tokens(ctx: &RoutingContext) -> u32 {
    if !ctx.token_ids.is_empty() {
        return ctx.token_ids.len() as u32;
    }
    if !ctx.block_hashes.is_empty() && ctx.block_size > 0 {
        return (ctx.block_hashes.len() as u32) * ctx.block_size;
    }
    0
}

#[async_trait]
impl RoutingStrategy for ModelTierStrategy {
    fn name(&self) -> &'static str {
        "model_tier"
    }

    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        let requested = ctx.model_name.as_deref();
        match self.cfg.policy {
            TierRoutingPolicy::Pick { .. } => {
                // Soft sub-strategy: score via raw_cost so the hybrid
                // strategy's normalize_costs turns it into a [0,1] term.
                let complex = self.is_complex(ctx);
                let desired = if complex {
                    ModelTier::Large
                } else {
                    ModelTier::Small
                };
                let mut out = Vec::with_capacity(candidates.len());
                for cand in candidates {
                    let tier = self.resolve_tier(cand, requested, meta);
                    let raw_cost = match tier {
                        Some(t) if t == desired => COST_TIER_MATCH,
                        Some(_) => COST_TIER_MISMATCH,
                        None => COST_TIER_UNKNOWN,
                    };
                    out.push(ScoredBackend {
                        backend_id: cand.clone(),
                        // `score` is ignored by normalize_costs (which uses
                        // raw_cost); set it to mirror raw_cost for
                        // traceability when used as a primary.
                        score: 1.0 - raw_cost,
                        raw_cost,
                        meta_version: 0,
                    });
                }
                Ok(out)
            }
            TierRoutingPolicy::Fallback => {
                // Primary strategy: rank small ahead of large unconditionally.
                // The `score` field directly governs the ranked order
                // produced by `select_ranked`.
                let mut out = Vec::with_capacity(candidates.len());
                for cand in candidates {
                    let tier = self.resolve_tier(cand, requested, meta);
                    let (score, raw_cost) = match tier {
                        Some(ModelTier::Small) => (SCORE_FALLBACK_SMALL, -SCORE_FALLBACK_SMALL),
                        Some(ModelTier::Large) => (SCORE_FALLBACK_LARGE, -SCORE_FALLBACK_LARGE),
                        None => (SCORE_FALLBACK_UNKNOWN, -SCORE_FALLBACK_UNKNOWN),
                    };
                    out.push(ScoredBackend {
                        backend_id: cand.clone(),
                        score,
                        raw_cost,
                        meta_version: 0,
                    });
                }
                Ok(out)
            }
        }
    }

    fn is_available(&self, _meta: &MetadataStore) -> bool {
        // Available whenever a tier table is configured. An empty table
        // makes the strategy a no-op (every backend is "unknown"), so we
        // still report available — the weight normalization in the hybrid
        // strategy will simply give every candidate the same neutral score.
        !self.cfg.tiers.is_empty()
    }

    fn weight(&self) -> f64 {
        self.cfg.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::backend::{
        BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
        ModelInstance, Protocol, Quantization,
    };
    use hier_kv_gateway_core::ids::{IndexerDomainId, RegionId};
    use hier_kv_gateway_core::model_tier::{ModelTier, TierEntry, TierRoutingPolicy};
    use hier_kv_gateway_metadata::store::MetadataStore;
    use std::sync::Arc;

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

    fn store_with(small: &str, large: &str) -> MetadataStore {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "small", "qwen2.5-7b"));
        store.register_backend(backend("r1", "large", "qwen2.5-72b"));
        let _ = (small, large);
        store
    }

    fn tier_cfg(policy: TierRoutingPolicy) -> Arc<ModelTierConfig> {
        Arc::new(ModelTierConfig {
            enabled: true,
            weight: 0.20,
            policy,
            tiers: vec![
                TierEntry {
                    model: "qwen2.5-7b".to_string(),
                    tier: ModelTier::Small,
                },
                TierEntry {
                    model: "qwen2.5-72b".to_string(),
                    tier: ModelTier::Large,
                },
            ],
        })
    }

    fn ctx_simple(model: &str) -> RoutingContext {
        RoutingContext {
            model_name: Some(model.to_string()),
            token_ids: vec![1; 100], // well below 2048 threshold
            estimated_output_tokens: 64, // well below 1024 threshold
            block_size: 16,
            requires_tool_calling: false,
            ..RoutingContext::default()
        }
    }

    fn ctx_complex_prompt(model: &str) -> RoutingContext {
        RoutingContext {
            model_name: Some(model.to_string()),
            token_ids: vec![1; 4096], // above 2048 threshold
            estimated_output_tokens: 64,
            block_size: 16,
            requires_tool_calling: false,
            ..RoutingContext::default()
        }
    }

    fn ctx_complex_tools(model: &str) -> RoutingContext {
        RoutingContext {
            model_name: Some(model.to_string()),
            token_ids: vec![1; 100],
            estimated_output_tokens: 64,
            block_size: 16,
            requires_tool_calling: true, // triggers large preference
            ..RoutingContext::default()
        }
    }

    fn pick_cfg() -> Arc<ModelTierConfig> {
        tier_cfg(TierRoutingPolicy::Pick {
            prompt_token_threshold: 2048,
            max_token_threshold: 1024,
            prefer_large_for_tools: true,
        })
    }

    fn fallback_cfg() -> Arc<ModelTierConfig> {
        tier_cfg(TierRoutingPolicy::Fallback)
    }

    // -----------------------------------------------------------------
    // Pick policy (soft sub-strategy)
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn pick_simple_prefers_small() {
        let store = store_with("small", "large");
        let strat = ModelTierStrategy::new(pick_cfg());
        let candidates = vec![
            BackendId::new("r1", "small"),
            BackendId::new("r1", "large"),
        ];
        let scored = strat
            .evaluate(&ctx_simple("qwen2.5-7b"), &candidates, &store)
            .await
            .unwrap();
        let by_inst: std::collections::HashMap<_, _> = scored
            .into_iter()
            .map(|s| (s.backend_id.instance.to_string(), s))
            .collect();
        let small = by_inst.get("small").unwrap();
        let large = by_inst.get("large").unwrap();
        assert!(
            small.raw_cost < large.raw_cost,
            "simple request: small should have lower cost (higher score)"
        );
        assert_eq!(small.raw_cost, COST_TIER_MATCH);
        assert_eq!(large.raw_cost, COST_TIER_MISMATCH);
    }

    #[tokio::test]
    async fn pick_complex_prompt_prefers_large() {
        let store = store_with("small", "large");
        let strat = ModelTierStrategy::new(pick_cfg());
        let candidates = vec![
            BackendId::new("r1", "small"),
            BackendId::new("r1", "large"),
        ];
        let scored = strat
            .evaluate(&ctx_complex_prompt("qwen2.5-7b"), &candidates, &store)
            .await
            .unwrap();
        let by_inst: std::collections::HashMap<_, _> = scored
            .into_iter()
            .map(|s| (s.backend_id.instance.to_string(), s))
            .collect();
        let small = by_inst.get("small").unwrap();
        let large = by_inst.get("large").unwrap();
        assert!(
            large.raw_cost < small.raw_cost,
            "complex request: large should have lower cost (higher score)"
        );
        assert_eq!(large.raw_cost, COST_TIER_MATCH);
        assert_eq!(small.raw_cost, COST_TIER_MISMATCH);
    }

    #[tokio::test]
    async fn pick_tools_prefers_large_when_enabled() {
        let store = store_with("small", "large");
        let strat = ModelTierStrategy::new(pick_cfg());
        let candidates = vec![
            BackendId::new("r1", "small"),
            BackendId::new("r1", "large"),
        ];
        let scored = strat
            .evaluate(&ctx_complex_tools("qwen2.5-7b"), &candidates, &store)
            .await
            .unwrap();
        let by_inst: std::collections::HashMap<_, _> = scored
            .into_iter()
            .map(|s| (s.backend_id.instance.to_string(), s))
            .collect();
        let large = by_inst.get("large").unwrap();
        assert_eq!(large.raw_cost, COST_TIER_MATCH);
    }

    #[tokio::test]
    async fn pick_unknown_tier_is_neutral() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "unknown", "deepseek-r1-671b"));
        let strat = ModelTierStrategy::new(pick_cfg());
        let candidates = vec![BackendId::new("r1", "unknown")];
        let scored = strat
            .evaluate(&ctx_simple("deepseek-r1-671b"), &candidates, &store)
            .await
            .unwrap();
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].raw_cost, COST_TIER_UNKNOWN);
    }

    #[tokio::test]
    async fn pick_uses_requested_model_tier_when_available() {
        // Backend serves both small and large models; the request asks for
        // the large one — its tier should win over the first-listed small.
        let store = MetadataStore::new();
        let mut info = backend("r1", "multi", "qwen2.5-7b");
        info.models.push(ModelInstance {
            model_name: "qwen2.5-72b".to_string(),
            model_architecture: "qwen".to_string(),
            quantization: Quantization::Fp16,
            max_context_len: 32768,
            supports_tool_calling: true,
            supports_streaming: true,
        });
        store.register_backend(info);

        let strat = ModelTierStrategy::new(pick_cfg());
        let candidates = vec![BackendId::new("r1", "multi")];
        let scored = strat
            .evaluate(&ctx_simple("qwen2.5-72b"), &candidates, &store)
            .await
            .unwrap();
        // Requested model is large → matches large tier even though the
        // request is simple (Pick honours the requested model's tier for
        // backend resolution, then applies complexity scoring).
        assert_eq!(scored[0].raw_cost, COST_TIER_MISMATCH);
    }

    // -----------------------------------------------------------------
    // Fallback policy (primary strategy)
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn fallback_ranks_small_above_large() {
        let store = store_with("small", "large");
        let strat = ModelTierStrategy::new(fallback_cfg());
        let candidates = vec![
            BackendId::new("r1", "small"),
            BackendId::new("r1", "large"),
        ];
        let scored = strat
            .evaluate(&ctx_simple("qwen2.5-7b"), &candidates, &store)
            .await
            .unwrap();
        // Verify score-based ordering (select_ranked sorts descending).
        let mut sorted = scored.clone();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert_eq!(sorted[0].backend_id.instance.as_str(), "small");
        assert_eq!(sorted[1].backend_id.instance.as_str(), "large");
        assert!(sorted[0].score > sorted[1].score);
    }

    #[tokio::test]
    async fn fallback_unknown_ranks_between_small_and_large() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "small", "qwen2.5-7b"));
        store.register_backend(backend("r1", "unknown", "deepseek-r1-671b"));
        store.register_backend(backend("r1", "large", "qwen2.5-72b"));
        let strat = ModelTierStrategy::new(fallback_cfg());
        let candidates = vec![
            BackendId::new("r1", "small"),
            BackendId::new("r1", "unknown"),
            BackendId::new("r1", "large"),
        ];
        let scored = strat
            .evaluate(&ctx_simple("qwen2.5-7b"), &candidates, &store)
            .await
            .unwrap();
        let mut sorted = scored.clone();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let order: Vec<&str> = sorted
            .iter()
            .map(|s| s.backend_id.instance.as_str())
            .collect();
        assert_eq!(order, ["small", "unknown", "large"]);
    }

    #[test]
    fn is_available_requires_non_empty_tiers() {
        let strat = ModelTierStrategy::new(Arc::new(ModelTierConfig::default()));
        assert!(!strat.is_available(&MetadataStore::new()));
        let strat = ModelTierStrategy::new(pick_cfg());
        assert!(strat.is_available(&MetadataStore::new()));
    }

    #[test]
    fn weight_mirrors_config() {
        let strat = ModelTierStrategy::new(pick_cfg());
        assert!((strat.weight() - 0.20).abs() < 1e-9);
    }

    #[test]
    fn name_is_model_tier() {
        let strat = ModelTierStrategy::new(pick_cfg());
        assert_eq!(strat.name(), "model_tier");
    }
}
