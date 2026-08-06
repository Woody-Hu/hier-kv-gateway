//! KV-capacity-aware routing strategy.
//!
//! Estimates the KV-cache memory footprint of the incoming request (using the
//! analytical, allocation-free formulas from the `hier-kv-gateway-kv-estimate`
//! leaf crate — the same ones vLLM, SGLang, Mooncake and llm-d use, not a
//! simulation) and scores each candidate backend by how much headroom it has
//! for that footprint. This is the *behaviour* half of `[kv_estimate]` config;
//! the *data* half ([`KvEstimateConfig`], model spec catalog) lives in the
//! estimate crate and is surfaced via [`GatewayConfig::kv_estimate`].
//!
//! ## What it scores
//!
//! For every candidate backend the strategy:
//!
//! 1. Resolves the model the backend would actually serve (preferring an exact
//!    match against `RoutingContext::model_name`, else the backend's first
//!    listed model — same rule as [`CostAwareStrategy`]).
//! 2. Resolves the model's [`ModelSpec`] via the shared
//!    [`KvEstimationRegistry`] (builtin catalog + operator `[[kv_estimate.models]]`
//!    overrides + any registered custom estimator plugin).
//! 3. Builds an [`EstimateInput`] from the request's prompt-token count and the
//!    client's `max_tokens` (a conservative upper bound on output, mirroring
//!    `LoadAwareStrategy`'s `w_decode` philosophy) and computes the request's
//!    KV footprint in bytes/blocks.
//! 4. Reads the backend's reported resource headroom and compares it against
//!    the estimate:
//!    - **KV-block path** (preferred, exact): when the backend reports
//!      `kv_total_blocks`/`kv_used_blocks` and the request carries a
//!      `block_size`, available bytes = free blocks × per-block bytes, and the
//!      request's block demand is compared against the free-block count.
//!    - **GPU-memory path** (fallback, conservative): when no KV block totals
//!      are reported but GPU memory is, available bytes = free GPU memory ×
//!      `gpu_mem_safety_fraction` (KV is not the only GPU memory consumer, so
//!      only a safety fraction of the free memory is claimable).
//! 5. Scores: a backend with plenty of headroom gets a low `raw_cost`
//!    (≈ footprint / available, in `[0, 1]`) and therefore a high normalized
//!    score; a backend whose available headroom is *smaller than the estimate*
//!    is **excluded** (`raw_cost = ∞`, score 0) — this is the load-shedding
//!    decision the strategy exists to make. Backends with no capacity signal
//!    at all are treated as neutral (`raw_cost = 0`, score 1) so the other
//!    sub-strategies decide, matching [`CostAwareStrategy`]'s unknown-price
//!    convention.
//!
//! ## Relation to the existing `KvAwareStrategy`
//!
//! [`KvAwareStrategy`] scores by *prefix hit overlap* — how many of the
//! request's KV blocks a backend already has cached (so it can skip prefilling
//! them). [`KvCapacityStrategy`] scores by *remaining capacity* — whether the
//! backend has room to hold the request's KV at all. The two are
//! complementary: hit-overlap reduces the work, capacity-headroom decides
//! admission. They are independent sub-strategies in the hybrid ensemble and
//! are normalized separately, exactly like `LoadAwareStrategy` vs
//! `CostAwareStrategy`.
//!
//! ## Hot-path cost
//!
//! The per-backend work is: one `MetadataStore` lookup (metrics), one model
//! lookup (cheap), and the [`KvEstimationRegistry::estimate`] call — itself an
//! allocation-free `Copy`-spec lookup + a handful of integer multiplies (see
//! `tests/alloc_free.rs` in the estimate crate). The strategy adds no
//! allocation of its own on the scored path.
//!
//! [`CostAwareStrategy`]: crate::cost_aware::CostAwareStrategy
//! [`KvAwareStrategy`]: crate::kv_aware::KvAwareStrategy
//! [`GatewayConfig::kv_estimate`]: hier_kv_gateway_core::config::GatewayConfig::kv_estimate

use std::sync::Arc;

use async_trait::async_trait;

use hier_kv_gateway_core::config::KvEstimateConfig;
use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_kv_estimate::{EstimateInput, KvEstimationRegistry};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// One megabyte in bytes — `BackendMetrics::gpu_memory_*_mb` is reported in
/// decimal MB (10⁶ bytes), so the GPU-memory fallback path converts through
/// this constant.
const BYTES_PER_MB: f64 = 1_000_000.0;

/// KV-capacity-aware routing strategy.
///
/// Constructed once at startup from the `[kv_estimate]` config section (the
/// spec catalog is built once; the [`KvEstimationRegistry`] is wrapped in an
/// `Arc` and read concurrently on the routing hot path). Attached to the
/// hybrid ensemble as a [`RoutingPlugin`](crate::plugin::RoutingPlugin) when
/// `kv_estimate.enabled = true`.
pub struct KvCapacityStrategy {
    /// Shared estimator registry: builtin catalog + custom specs + plugins.
    pub registry: Arc<KvEstimationRegistry>,
    /// Tunable parameters from `[kv_estimate]` in TOML.
    pub cfg: KvEstimateConfig,
}

impl KvCapacityStrategy {
    /// Build a strategy from an estimator registry and configuration.
    pub fn new(registry: Arc<KvEstimationRegistry>, cfg: KvEstimateConfig) -> Self {
        Self { registry, cfg }
    }
}

#[async_trait]
impl RoutingStrategy for KvCapacityStrategy {
    fn name(&self) -> &'static str {
        "kv_capacity"
    }

    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        // Estimated output tokens come from the client's `max_tokens` — a
        // conservative upper bound, mirroring `LoadAwareStrategy`'s `w_decode`
        // and `CostAwareStrategy`'s output projection. The estimate therefore
        // never underestimates the KV growth the request can cause.
        let prompt_tokens = estimate_prompt_tokens(ctx);
        let block_size = ctx.block_size;
        let input = EstimateInput::new(prompt_tokens, ctx.estimated_output_tokens)
            .with_block_size(block_size);

        let mut out = Vec::with_capacity(candidates.len());
        for cand in candidates {
            // Resolve the model name actually served by this backend (prefers
            // an exact `model_name` match, else the backend's first model).
            let served_model = resolve_served_model(cand, ctx.model_name.as_deref(), meta);

            // No model info at all: cannot estimate. Treat as neutral so the
            // other sub-strategies decide (a backend with no model metadata
            // will already be filtered by the model-aware hard filter).
            let Some(served_model) = served_model else {
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 1.0,
                    raw_cost: 0.0,
                    meta_version: 0,
                });
                continue;
            };

            // Estimate the request's KV footprint for this model. `None` means
            // no estimator recognizes the model name — apply the configured
            // unknown-spec policy.
            let Some(est) = self.registry.estimate(&served_model, &input) else {
                let (cost, score) = if self.cfg.exclude_on_unknown_spec {
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

            // Read the backend's resource headroom. No metrics → no capacity
            // signal → neutral (let other sub-strategies decide).
            let Some(m) = meta.load_get_metrics(cand) else {
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 1.0,
                    raw_cost: 0.0,
                    meta_version: 0,
                });
                continue;
            };

            let (cost, score) = score_capacity(&est, &m, block_size, self.cfg.gpu_mem_safety_fraction);
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
        // Available whenever the strategy is attached (i.e. `enabled = true`
        // in `[kv_estimate]`). The catalog may be empty — in that case every
        // backend hits the unknown-spec policy and the strategy is effectively
        // a no-op pass-through, mirroring `CostAwareStrategy`.
        true
    }

    fn weight(&self) -> f64 {
        // Mirror the configured hybrid weight. `0.0` keeps the strategy
        // attached (so its availability probe runs) but contributes nothing.
        self.cfg.weight
    }
}

/// Score a backend's capacity headroom against an estimated KV footprint.
///
/// Returns `(raw_cost, score)` where lower `raw_cost` is better (the convention
/// `HybridStrategy::normalize_costs` expects) and `score = 1 / (1 + raw_cost)`
/// for admitted backends.
///
/// Capacity signal selection:
/// - **KV-block path** (exact): `kv_total_blocks > 0` and `block_size > 0`.
///   Available bytes = free blocks × per-block bytes.
/// - **GPU-memory path** (conservative fallback): `gpu_memory_total_mb > 0`.
///   Available bytes = free GPU MB × `safety_fraction`.
/// - **No signal**: neutral `(0.0, 1.0)`.
///
/// **Exclusion** (`raw_cost = ∞`, score 0) happens whenever the estimated
/// footprint exceeds the available headroom — the load-shedding decision the
/// strategy exists to make. For the KV-block path this is equivalent to
/// `est.blocks > free_blocks` (since `est.bytes = est.blocks × per_block_bytes`
/// and `available_bytes = free_blocks × per_block_bytes`); the byte-granular
/// check is used uniformly for both paths.
fn score_capacity(
    est: &hier_kv_gateway_kv_estimate::KvEstimate,
    m: &hier_kv_gateway_core::metrics::BackendMetrics,
    block_size: u32,
    gpu_mem_safety_fraction: f64,
) -> (f64, f64) {
    // Per-block bytes for this model = per-token bytes × block_size. The
    // estimate already carries `per_token_bytes`, so we can convert free
    // blocks → free bytes without a second spec lookup.
    let per_block_bytes = if block_size > 0 {
        est.per_token_bytes.saturating_mul(block_size as u64)
    } else {
        0
    };

    // ---- KV-block path: exact, paged-attention semantics ----------------
    let available_bytes = if m.kv_total_blocks > 0 && block_size > 0 && per_block_bytes > 0 {
        let free_blocks = m.kv_total_blocks.saturating_sub(m.kv_used_blocks);
        free_blocks as f64 * per_block_bytes as f64
    } else if m.gpu_memory_total_mb > 0 {
        // ---- GPU-memory path: conservative fallback ---------------------
        // KV is not the only GPU memory consumer (weights, activations), so
        // only a safety fraction of the currently free memory is claimable.
        let free_mb = m
            .gpu_memory_total_mb
            .saturating_sub(m.gpu_memory_used_mb);
        free_mb as f64 * BYTES_PER_MB * gpu_mem_safety_fraction
    } else {
        // ---- No capacity signal: neutral --------------------------------
        return (0.0, 1.0);
    };

    // No claimable memory at all → cannot admit → exclude.
    if available_bytes <= 0.0 {
        return (f64::INFINITY, 0.0);
    }

    // Load-shedding: the estimate exceeds the available headroom → exclude.
    let footprint = est.bytes as f64;
    if footprint > available_bytes {
        return (f64::INFINITY, 0.0);
    }

    // Admitted: raw_cost = utilization ratio in `[0, 1]`; lower = more
    // headroom = better. score = 1 / (1 + cost).
    let ratio = (footprint / available_bytes).clamp(0.0, 1.0);
    (ratio, 1.0 / (1.0 + ratio))
}

/// Resolve the model name served by `backend` that best matches the request.
///
/// Prefers an exact `model_name` match against the backend's served models;
/// falls back to the backend's first listed model otherwise. Returns `None`
/// when the backend is unknown to the metadata store or carries no models.
///
/// This is the same resolution rule [`CostAwareStrategy`] uses, kept local so
/// each strategy module stays self-contained.
///
/// [`CostAwareStrategy`]: crate::cost_aware::CostAwareStrategy
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
        if let Some(found) = instances.iter().find(|i| i.model_name == name) {
            return Some(found.model_name.clone());
        }
    }
    Some(instances[0].model_name.clone())
}

/// Estimate the prompt token count for KV projection.
///
/// Prefers the tokenized form (`token_ids` or `block_hashes * block_size`);
/// returns 0 when neither is available (the API layer always sets one before
/// routing). Mirrors [`CostAwareStrategy`]'s prompt estimate.
///
/// [`CostAwareStrategy`]: crate::cost_aware::CostAwareStrategy
fn estimate_prompt_tokens(ctx: &RoutingContext) -> u32 {
    if !ctx.token_ids.is_empty() {
        return ctx.token_ids.len() as u32;
    }
    if !ctx.block_hashes.is_empty() && ctx.block_size > 0 {
        return (ctx.block_hashes.len() as u32) * ctx.block_size;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::backend::{
        BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
        ModelInstance, Protocol, Quantization,
    };
    use hier_kv_gateway_core::ids::{IndexerDomainId, RegionId};
    use hier_kv_gateway_core::metrics::BackendMetrics;
    use hier_kv_gateway_kv_estimate::{KvDtype, ModelSpec, SpecCatalog};

    /// Llama-3-8B builtin: 32 layers, 8 KV heads, head_dim 128, BF16.
    /// per_token = 2*32*8*128*2 = 131_072 B; per_block (16) = 2_097_152 B.
    const LLAMA: &str = "Llama-3-8B";

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
                model_architecture: "llama".to_string(),
                quantization: Quantization::Bf16,
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
            token_ids: vec![1; prompt_tokens as usize],
            estimated_output_tokens: est_out,
            block_size: 16,
            ..RoutingContext::default()
        }
    }

    fn strategy() -> KvCapacityStrategy {
        KvCapacityStrategy::new(
            Arc::new(KvEstimationRegistry::with_builtins()),
            KvEstimateConfig {
                enabled: true,
                weight: 0.20,
                gpu_mem_safety_fraction: 0.5,
                exclude_on_unknown_spec: false,
                models: Vec::new(),
            },
        )
    }

    fn metrics(kv_used: u64, kv_total: u64) -> BackendMetrics {
        BackendMetrics {
            kv_used_blocks: kv_used,
            kv_total_blocks: kv_total,
            ..BackendMetrics::default()
        }
    }

    #[test]
    fn name_and_weight_mirror_config() {
        let s = strategy();
        assert_eq!(s.name(), "kv_capacity");
        assert!((s.weight() - 0.20).abs() < 1e-9);
        assert!(s.is_available(&MetadataStore::new()));
    }

    #[tokio::test]
    async fn backend_with_more_headroom_scores_higher() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "roomy", LLAMA));
        store.register_backend(backend("r1", "tight", LLAMA));
        // 4096 prompt tokens / block_size 16 = 256 blocks needed.
        // roomy: 1000 free blocks; tight: 300 free blocks. Both can fit.
        store.load_update(
            BackendId::new("r1", "roomy"),
            metrics(0, 1000),
        );
        store.load_update(
            BackendId::new("r1", "tight"),
            metrics(700, 1000),
        );
        let s = strategy();
        let candidates = vec![
            BackendId::new("r1", "roomy"),
            BackendId::new("r1", "tight"),
        ];
        let scored = s
            .evaluate(&ctx(LLAMA, 4096, 0), &candidates, &store)
            .await
            .unwrap();
        let by_id: std::collections::HashMap<_, _> = scored
            .into_iter()
            .map(|s| (s.backend_id.instance.to_string(), s))
            .collect();
        let roomy = by_id.get("roomy").unwrap();
        let tight = by_id.get("tight").unwrap();
        // Lower raw_cost = more headroom; roomy should beat tight.
        assert!(
            roomy.raw_cost < tight.raw_cost,
            "roomy should have lower cost: roomy={} tight={}",
            roomy.raw_cost,
            tight.raw_cost
        );
        assert!(roomy.score > tight.score);
        // Both admitted (finite cost, score > 0).
        assert!(roomy.raw_cost.is_finite());
        assert!(tight.raw_cost.is_finite());
        // roomy: 256/1000 = 0.256; tight: 256/300 ≈ 0.853.
        assert!((roomy.raw_cost - 0.256).abs() < 1e-3);
        assert!((tight.raw_cost - 0.853).abs() < 1e-3);
    }

    #[tokio::test]
    async fn over_capacity_backend_is_excluded() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "full", LLAMA));
        // 5 free blocks. A 4096-token request needs 256 blocks → excluded.
        store.load_update(BackendId::new("r1", "full"), metrics(995, 1000));
        let s = strategy();
        let candidates = vec![BackendId::new("r1", "full")];
        let scored = s
            .evaluate(&ctx(LLAMA, 4096, 0), &candidates, &store)
            .await
            .unwrap();
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].score, 0.0);
        assert!(!scored[0].raw_cost.is_finite());
    }

    #[tokio::test]
    async fn exact_fit_is_admitted() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "exact", LLAMA));
        // 256 free blocks, request needs exactly 256 → admitted (not > free).
        store.load_update(BackendId::new("r1", "exact"), metrics(744, 1000));
        let s = strategy();
        let candidates = vec![BackendId::new("r1", "exact")];
        let scored = s
            .evaluate(&ctx(LLAMA, 4096, 0), &candidates, &store)
            .await
            .unwrap();
        assert!(scored[0].raw_cost.is_finite());
        assert!(scored[0].score > 0.0);
    }

    #[tokio::test]
    async fn unknown_spec_neutral_by_default() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "mystery", "totally-unknown-model"));
        let s = strategy(); // exclude_on_unknown_spec = false
        let candidates = vec![BackendId::new("r1", "mystery")];
        let scored = s
            .evaluate(&ctx("totally-unknown-model", 100, 10), &candidates, &store)
            .await
            .unwrap();
        assert_eq!(scored[0].score, 1.0);
        assert_eq!(scored[0].raw_cost, 0.0);
    }

    #[tokio::test]
    async fn unknown_spec_excluded_when_configured() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "mystery", "totally-unknown-model"));
        let cfg = KvEstimateConfig {
            enabled: true,
            exclude_on_unknown_spec: true,
            ..KvEstimateConfig::default()
        };
        let s = KvCapacityStrategy::new(
            Arc::new(KvEstimationRegistry::with_builtins()),
            cfg,
        );
        let candidates = vec![BackendId::new("r1", "mystery")];
        let scored = s
            .evaluate(&ctx("totally-unknown-model", 100, 10), &candidates, &store)
            .await
            .unwrap();
        assert_eq!(scored[0].score, 0.0);
        assert!(!scored[0].raw_cost.is_finite());
    }

    #[tokio::test]
    async fn no_metrics_is_neutral() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "silent", LLAMA));
        // No load_update → load_get_metrics returns None.
        let s = strategy();
        let candidates = vec![BackendId::new("r1", "silent")];
        let scored = s
            .evaluate(&ctx(LLAMA, 4096, 0), &candidates, &store)
            .await
            .unwrap();
        assert_eq!(scored[0].score, 1.0);
        assert_eq!(scored[0].raw_cost, 0.0);
    }

    #[tokio::test]
    async fn gpu_memory_fallback_admits_and_excludes() {
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "gpu", LLAMA));
        // No KV block totals, but GPU memory reported. Free = 10_000 MB.
        // safety_fraction 0.5 → 5_000 MB = 5e9 B claimable.
        // per_token for Llama-3-8B = 131_072 B; 4096 tokens → 536_870_912 B.
        // ratio = 5.37e8 / 5e9 ≈ 0.107 → admitted, finite cost.
        store.load_update(
            BackendId::new("r1", "gpu"),
            BackendMetrics {
                kv_total_blocks: 0,
                kv_used_blocks: 0,
                gpu_memory_used_mb: 30_000,
                gpu_memory_total_mb: 40_000,
                ..BackendMetrics::default()
            },
        );
        let s = strategy();
        let candidates = vec![BackendId::new("r1", "gpu")];
        let scored = s
            .evaluate(&ctx(LLAMA, 4096, 0), &candidates, &store)
            .await
            .unwrap();
        assert!(scored[0].raw_cost.is_finite(), "should be admitted");
        assert!((scored[0].raw_cost - 0.107).abs() < 1e-2);

        // Now squeeze GPU memory so the estimate cannot fit → excluded.
        store.load_update(
            BackendId::new("r1", "gpu"),
            BackendMetrics {
                kv_total_blocks: 0,
                kv_used_blocks: 0,
                gpu_memory_used_mb: 39_999,
                gpu_memory_total_mb: 40_000,
                ..BackendMetrics::default()
            },
        );
        let scored = s
            .evaluate(&ctx(LLAMA, 4096, 0), &candidates, &store)
            .await
            .unwrap();
        assert!(!scored[0].raw_cost.is_finite(), "should be excluded");
        assert_eq!(scored[0].score, 0.0);
    }

    #[tokio::test]
    async fn custom_catalog_spec_is_used() {
        // A custom spec with a tiny footprint (1 layer, 1 head, dim 1, fp8):
        // per_token = 2*1*1*1*1 = 2 B. Even a backend with 1 free block of
        // block_size 16 (32 B) admits a 4096-token request (8192 B) only if
        // enough blocks; here we give 1000 free blocks → admitted.
        let store = MetadataStore::new();
        store.register_backend(backend("r1", "tiny", "my-tiny-model"));
        store.load_update(BackendId::new("r1", "tiny"), metrics(0, 1000));
        let catalog = SpecCatalog::new().insert(
            "my-tiny-model",
            ModelSpec::standard(1, 1, 1, KvDtype::Fp8),
        );
        let registry = Arc::new(KvEstimationRegistry::with_catalog(catalog));
        let s = KvCapacityStrategy::new(
            registry,
            KvEstimateConfig {
                enabled: true,
                weight: 0.2,
                gpu_mem_safety_fraction: 0.5,
                exclude_on_unknown_spec: false,
                models: Vec::new(),
            },
        );
        let candidates = vec![BackendId::new("r1", "tiny")];
        let scored = s
            .evaluate(&ctx("my-tiny-model", 4096, 0), &candidates, &store)
            .await
            .unwrap();
        // per_token=2, 4096 tokens, block_size 16 → 256 blocks, 8192 B.
        // 1000 free blocks × 32 B/block = 32000 B available.
        // ratio = 8192/32000 = 0.256.
        assert!(scored[0].raw_cost.is_finite());
        assert!((scored[0].raw_cost - 0.256).abs() < 1e-3);
    }

    #[test]
    fn score_capacity_monotonic_in_headroom() {
        // Same footprint, more headroom → lower cost, higher score.
        let est = hier_kv_gateway_kv_estimate::KvEstimate {
            bytes: 131_072 * 256 * 16, // 256 blocks worth
            blocks: 256,
            per_token_bytes: 131_072,
            effective_seq_len: 4096,
            batch_size: 1,
        };
        let roomy = BackendMetrics {
            kv_used_blocks: 0,
            kv_total_blocks: 1000, // 1000 free blocks
            ..BackendMetrics::default()
        };
        let tight = BackendMetrics {
            kv_used_blocks: 700,
            kv_total_blocks: 1000, // 300 free blocks
            ..BackendMetrics::default()
        };
        let (c_roomy, s_roomy) = score_capacity(&est, &roomy, 16, 0.5);
        let (c_tight, s_tight) = score_capacity(&est, &tight, 16, 0.5);
        assert!(c_roomy < c_tight);
        assert!(s_roomy > s_tight);
        // ratio never exceeds 1 in the admitted path.
        assert!(c_tight <= 1.0);
    }

    #[test]
    fn score_capacity_neutral_without_signal() {
        let est = hier_kv_gateway_kv_estimate::KvEstimate {
            bytes: 1_000_000,
            blocks: 0,
            per_token_bytes: 131_072,
            effective_seq_len: 4096,
            batch_size: 1,
        };
        // No KV block totals, no GPU memory → neutral.
        let m = BackendMetrics::default();
        let (cost, score) = score_capacity(&est, &m, 16, 0.5);
        assert_eq!(cost, 0.0);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn score_capacity_kv_block_path_excludes_on_overflow() {
        let est = hier_kv_gateway_kv_estimate::KvEstimate {
            bytes: 131_072 * 257 * 16, // 257 blocks worth
            blocks: 257,
            per_token_bytes: 131_072,
            effective_seq_len: 4096,
            batch_size: 1,
        };
        let m = BackendMetrics {
            kv_used_blocks: 999,
            kv_total_blocks: 1000, // 1 free block
            ..BackendMetrics::default()
        };
        let (cost, score) = score_capacity(&est, &m, 16, 0.5);
        assert!(!cost.is_finite());
        assert_eq!(score, 0.0);
    }
}
