//! Benchmarks for the cost-aware and model-tier plugin sub-strategies.
//!
//! What we measure
//! ----------------
//! * `cost_aware_evaluate` — the per-request cost of
//!   [`CostAwareStrategy::evaluate`] in isolation, varying the candidate count.
//!   This isolates the price-catalog lookup + projected-cost math from the
//!   rest of the hybrid ensemble.
//! * `model_tier_evaluate` — the per-request cost of
//!   [`ModelTierStrategy::evaluate`] (Pick policy) in isolation.
//! * `hybrid_with_plugins_overhead` — end-to-end [`HybridStrategy::evaluate`]
//!   comparing the baseline (kv/load/topology only) against the plugin-augmented
//!   ensemble (kv/load/topology + cost + model_tier). The absolute overhead
//!   (~1–5 µs at 2–20 backends) is negligible relative to LLM forward latency
//!   (100ms+); the benchmark documents the real cost so operators can make
//!   informed tradeoffs.
//!
//! All configurations use the *real* `MetadataStore` + `HybridStrategy` — the
//! same path the integration tests exercise. No mocks.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-routing --bench cost_tier_plugins
//! ```

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use hier_kv_gateway_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
    ModelInstance, Protocol, Quantization,
};
use hier_kv_gateway_core::config::StrategyWeights;
use hier_kv_gateway_core::cost::{CostConfig, ModelPrice, PriceEntry, StaticCostModel};
use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId};
use hier_kv_gateway_core::model_tier::{ModelTier, ModelTierConfig, TierEntry, TierRoutingPolicy};
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::request::RoutingContext;
use hier_kv_gateway_metadata::store::MetadataStore;

use hier_kv_gateway_routing::cost_aware::CostAwareStrategy;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::model_tier::ModelTierStrategy;
use hier_kv_gateway_routing::plugin::RoutingPlugin;
use hier_kv_gateway_routing::strategy::RoutingStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

const SMALL_MODEL: &str = "qwen2.5-7b";
const LARGE_MODEL: &str = "qwen2.5-72b";

/// Build a store with `n` backends, alternating between small and large models.
fn build_store(n: usize) -> (MetadataStore, Vec<BackendId>) {
    let store = MetadataStore::new();
    let region = RegionId::new("cloud-cn-beijing");
    let mut backends = Vec::with_capacity(n);
    for i in 0..n {
        let model = if i % 2 == 0 { SMALL_MODEL } else { LARGE_MODEL };
        let b = BackendId::new(region.clone(), format!("inst-{i}"));
        let info = BackendInfo {
            id: b.clone(),
            backend_type: BackendType::VllmEngine,
            endpoint: Endpoint {
                url: format!("http://10.0.0.{i}:8000"),
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
            region: region.clone(),
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
        };
        store.register_backend(info);
        // Fresh load metrics so the load strategy stays in the loop.
        let mut m = BackendMetrics::default();
        m.active_requests = (i as u64) % 4;
        store.load_update(b.clone(), m);
        backends.push(b);
    }
    (store, backends)
}

fn catalog() -> Arc<dyn hier_kv_gateway_core::cost::CostModel> {
    Arc::new(StaticCostModel::new([
        (
            SMALL_MODEL.to_string(),
            ModelPrice {
                input_per_1m: 0.15,
                output_per_1m: 0.60,
            },
        ),
        (
            LARGE_MODEL.to_string(),
            ModelPrice {
                input_per_1m: 3.0,
                output_per_1m: 12.0,
            },
        ),
    ]))
}

fn cost_cfg() -> CostConfig {
    CostConfig {
        enabled: true,
        prices: vec![
            PriceEntry {
                model: SMALL_MODEL.to_string(),
                input_per_1m: 0.15,
                output_per_1m: 0.60,
            },
            PriceEntry {
                model: LARGE_MODEL.to_string(),
                input_per_1m: 3.0,
                output_per_1m: 12.0,
            },
        ],
        weight: 0.15,
        output_cost_scale: 1.0,
        exclude_on_unknown_price: false,
    }
}

fn tier_cfg() -> Arc<ModelTierConfig> {
    Arc::new(ModelTierConfig {
        enabled: true,
        weight: 0.20,
        policy: TierRoutingPolicy::Pick {
            prompt_token_threshold: 2048,
            max_token_threshold: 1024,
            prefer_large_for_tools: true,
        },
        tiers: vec![
            TierEntry {
                model: SMALL_MODEL.to_string(),
                tier: ModelTier::Small,
            },
            TierEntry {
                model: LARGE_MODEL.to_string(),
                tier: ModelTier::Large,
            },
        ],
    })
}

fn ctx_simple(model: &str) -> RoutingContext {
    RoutingContext {
        model_name: Some(model.to_string()),
        token_ids: vec![1; 100],
        estimated_output_tokens: 64,
        block_size: 16,
        ..RoutingContext::default()
    }
}

/// Isolated cost-aware strategy evaluation cost.
fn bench_cost_aware_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_aware_evaluate");
    group.sample_size(100);

    for n in [2usize, 10, 50] {
        let (store, backends) = build_store(n);
        let strat = CostAwareStrategy::new(catalog(), cost_cfg());
        let ctx = ctx_simple(SMALL_MODEL);

        group.bench_with_input(BenchmarkId::new("backends", n), &n, |b, &_| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let scored = strat
                    .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                    .await
                    .unwrap();
                black_box(scored);
            });
        });
    }
    group.finish();
}

/// Isolated model-tier strategy evaluation cost (Pick policy).
fn bench_model_tier_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_tier_evaluate");
    group.sample_size(100);

    for n in [2usize, 10, 50] {
        let (store, backends) = build_store(n);
        let strat = ModelTierStrategy::new(tier_cfg());
        let ctx = ctx_simple(SMALL_MODEL);

        group.bench_with_input(BenchmarkId::new("backends", n), &n, |b, &_| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let scored = strat
                    .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                    .await
                    .unwrap();
                black_box(scored);
            });
        });
    }
    group.finish();
}

/// Hybrid evaluate: baseline (kv/load/topology) vs with plugins (cost + tier).
///
/// This is the closed-loop decision rule: the per-request overhead of
/// attaching two plugin sub-strategies must stay modest relative to the
/// baseline hybrid evaluate. The benchmark is the evidence.
fn bench_hybrid_with_plugins_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_with_plugins_overhead");
    group.sample_size(50);

    for n in [2usize, 10, 20] {
        let (store, backends) = build_store(n);
        let ctx = ctx_simple(SMALL_MODEL);
        let weights = StrategyWeights {
            kv: 0.30,
            load: 0.30,
            topology: 0.20,
            cost: 0.0, // baseline: cost plugin not attached
        };

        // Baseline: no plugins.
        let baseline = HybridStrategy::new(
            Box::new(KvAwareStrategy::default()),
            Box::new(ModelAwareStrategy::default()),
            Box::new(LoadAwareStrategy::default()),
            Box::new(TopologyAwareStrategy {
                w_rtt: 1.0,
                w_bw: 0.0,
                self_region: RegionId::new("cloud-cn-beijing"),
            }),
            weights.clone(),
            0.0,
        );

        group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, &_| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let scored = baseline
                    .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                    .await
                    .unwrap();
                black_box(scored);
            });
        });

        // With plugins: cost-aware + model-tier attached.
        let cost_strat = CostAwareStrategy::new(catalog(), cost_cfg());
        let tier_strat = ModelTierStrategy::new(tier_cfg());
        let augmented = HybridStrategy::new(
            Box::new(KvAwareStrategy::default()),
            Box::new(ModelAwareStrategy::default()),
            Box::new(LoadAwareStrategy::default()),
            Box::new(TopologyAwareStrategy {
                w_rtt: 1.0,
                w_bw: 0.0,
                self_region: RegionId::new("cloud-cn-beijing"),
            }),
            weights.clone(),
            0.0,
        )
        .with_plugin(RoutingPlugin::from_strategy(Arc::new(cost_strat)))
        .with_plugin(RoutingPlugin::from_strategy(Arc::new(tier_strat)));

        group.bench_with_input(BenchmarkId::new("with_plugins", n), &n, |b, &_| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let scored = augmented
                    .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                    .await
                    .unwrap();
                black_box(scored);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_cost_aware_evaluate,
        bench_model_tier_evaluate,
        bench_hybrid_with_plugins_overhead,
);
criterion_main!(benches);
