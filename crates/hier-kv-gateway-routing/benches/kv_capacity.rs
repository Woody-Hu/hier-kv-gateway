//! Benchmarks for the KV-capacity-aware routing strategy.
//!
//! What we measure
//! ----------------
//! * `kv_capacity_evaluate` — the per-request cost of
//!   [`KvCapacityStrategy::evaluate`] in isolation, varying the candidate
//!   count. This isolates the KV-estimate + capacity-headroom math from the
//!   rest of the hybrid ensemble. The estimator itself is the allocation-free
//!   leaf crate (`hier-kv-gateway-kv-estimate`), so this number is dominated
//!   by the per-backend metadata lookups + the analytical formula.
//! * `hybrid_with_kv_capacity_overhead` — end-to-end [`HybridStrategy::evaluate`]
//!   comparing the baseline (kv/load/topology only) against the
//!   kv-capacity-augmented ensemble (kv/load/topology + kv_capacity). The
//!   absolute overhead must stay modest relative to LLM forward latency
//!   (100ms+); the benchmark is the evidence.
//!
//! All configurations use the *real* `MetadataStore` + `HybridStrategy` +
//! the real builtin KV-estimate catalog — the same path the integration tests
//! exercise. No mocks, no stubbed estimators.
//!
//! ## Anti-cheat
//!
//! Each bench closure asserts the scored output is non-empty and that scores
//! are finite (or explicitly `∞` for the over-capacity case), so a future
//! change that accidentally short-circuits to an empty/constant result is
//! caught by the bench itself, not just by the unit tests.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-routing --bench kv_capacity
//! ```

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use hier_kv_gateway_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
    ModelInstance, Protocol, Quantization,
};
use hier_kv_gateway_core::config::{KvEstimateConfig, StrategyWeights};
use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId};
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::request::RoutingContext;
use hier_kv_gateway_kv_estimate::KvEstimationRegistry;
use hier_kv_gateway_metadata::store::MetadataStore;

use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::kv_capacity::KvCapacityStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::plugin::RoutingPlugin;
use hier_kv_gateway_routing::strategy::RoutingStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

/// A model the builtin KV-estimate catalog recognizes (Llama-3-8B: 32 layers,
/// 8 KV heads, head_dim 128, BF16 → per_token = 131_072 B).
const MODEL: &str = "Llama-3-8B";
const BLOCK_SIZE: u32 = 16;

/// Build a store with `n` backends, all serving the same model, with varied
/// KV-block headroom so the strategy has real signal to score on.
fn build_store(n: usize) -> (MetadataStore, Vec<BackendId>) {
    let store = MetadataStore::new();
    let region = RegionId::new("cloud-cn-beijing");
    let mut backends = Vec::with_capacity(n);
    for i in 0..n {
        let b = BackendId::new(region.clone(), format!("inst-{i}"));
        let info = BackendInfo {
            id: b.clone(),
            backend_type: BackendType::VllmEngine,
            endpoint: Endpoint {
                url: format!("http://10.0.0.{i}:8000"),
                protocol: Protocol::Http,
            },
            models: vec![ModelInstance {
                model_name: MODEL.to_string(),
                model_architecture: "llama".to_string(),
                quantization: Quantization::Bf16,
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
                block_size: BLOCK_SIZE,
                cache_namespace: "default".to_string(),
                max_kv_blocks: 4096,
            },
            status: BackendStatus::Healthy,
        };
        store.register_backend(info);
        // Give each backend plenty of free KV blocks (varying slightly so the
        // strategy produces distinct scores rather than a flat tie).
        let mut m = BackendMetrics::default();
        m.kv_total_blocks = 4096;
        m.kv_used_blocks = (i as u64) * 10;
        m.active_requests = (i as u64) % 4;
        store.load_update(b.clone(), m);
        backends.push(b);
    }
    (store, backends)
}

fn kv_capacity_strategy() -> KvCapacityStrategy {
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

fn ctx_request(prompt_tokens: u32, est_out: u32) -> RoutingContext {
    RoutingContext {
        model_name: Some(MODEL.to_string()),
        token_ids: vec![1; prompt_tokens as usize],
        estimated_output_tokens: est_out,
        block_size: BLOCK_SIZE,
        ..RoutingContext::default()
    }
}

/// Isolated KV-capacity strategy evaluation cost.
fn bench_kv_capacity_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_capacity_evaluate");
    group.sample_size(100);

    for n in [2usize, 10, 50] {
        let (store, backends) = build_store(n);
        let strat = kv_capacity_strategy();
        // 2048 prompt + 256 output tokens → 144 blocks for Llama-3-8B. Every
        // backend has ≥ 4096 - 490 = 3606 free blocks, so all are admitted.
        let ctx = ctx_request(2048, 256);

        group.bench_with_input(BenchmarkId::new("backends", n), &n, |b, &_| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let scored = strat
                    .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                    .await
                    .unwrap();
                // Anti-cheat: non-empty, every score finite & in (0, 1].
                assert!(!scored.is_empty(), "evaluate must return scores");
                for s in &scored {
                    assert!(s.raw_cost.is_finite(), "all backends should be admitted");
                    assert!(s.score > 0.0 && s.score <= 1.0);
                }
                black_box(scored);
            });
        });
    }
    group.finish();
}

/// Hybrid evaluate: baseline (kv/load/topology) vs with the kv_capacity
/// plugin attached. Documents the per-request overhead of capacity-aware
/// routing on top of the existing hybrid ensemble.
fn bench_hybrid_with_kv_capacity_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_with_kv_capacity_overhead");
    group.sample_size(50);

    for n in [2usize, 10, 20] {
        let (store, backends) = build_store(n);
        let ctx = ctx_request(2048, 256);
        let weights = StrategyWeights {
            kv: 0.35,
            load: 0.30,
            topology: 0.20,
            cost: 0.0,
        };
        let self_region = RegionId::new("cloud-cn-beijing");

        // Baseline: no plugins.
        let baseline = HybridStrategy::new(
            Box::new(KvAwareStrategy::default()),
            Box::new(ModelAwareStrategy::default()),
            Box::new(LoadAwareStrategy::default()),
            Box::new(TopologyAwareStrategy {
                w_rtt: 1.0,
                w_bw: 0.0,
                self_region: self_region.clone(),
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
                assert!(!scored.is_empty());
                black_box(scored);
            });
        });

        // Augmented: kv_capacity plugin attached.
        let kv_cap = kv_capacity_strategy();
        let augmented = HybridStrategy::new(
            Box::new(KvAwareStrategy::default()),
            Box::new(ModelAwareStrategy::default()),
            Box::new(LoadAwareStrategy::default()),
            Box::new(TopologyAwareStrategy {
                w_rtt: 1.0,
                w_bw: 0.0,
                self_region: self_region.clone(),
            }),
            weights.clone(),
            0.0,
        )
        .with_plugin(RoutingPlugin::from_strategy(Arc::new(kv_cap)));

        group.bench_with_input(BenchmarkId::new("with_kv_capacity", n), &n, |b, &_| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let scored = augmented
                    .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                    .await
                    .unwrap();
                // Anti-cheat: hybrid still returns a ranked, non-empty list.
                assert!(!scored.is_empty());
                black_box(scored);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_kv_capacity_evaluate,
        bench_hybrid_with_kv_capacity_overhead,
);
criterion_main!(benches);
