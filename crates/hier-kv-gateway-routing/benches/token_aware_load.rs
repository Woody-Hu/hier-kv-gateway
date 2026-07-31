//! Latency benchmark for the token-aware load strategy.
//!
//! What we measure
//! ----------------
//! The token-aware `LoadAwareStrategy` adds two terms to the per-candidate
//! cost (`w_decode * projected_decode_blocks + w_prefill * active_prefill_tokens`).
//! This bench isolates the **routing hot-path overhead** of those extra terms
//! against the count-blind baseline (`w_decode = 0`, `w_prefill = 0`), varying
//! the candidate count.
//!
//! The closed-loop decision rule for introducing token-awareness requires the
//! routing-latency overhead to stay under 10%. This bench is the evidence.
//!
//! Both configurations use the *real* `MetadataStore` + `HybridStrategy` +
//! `RoutingEngine` — the same path the quality replay in
//! `tests/hier-kv-gateway-integration/tests/token_aware_load.rs` exercises.
//! No mocks.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-routing --bench token_aware_load
//! ```

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use hier_kv_gateway_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig, ModelInstance,
    Protocol, Quantization,
};
use hier_kv_gateway_core::config::StrategyWeights;
use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId};
use hier_kv_gateway_core::metrics::{BackendMetrics, LatencyStats};
use hier_kv_gateway_core::request::RoutingContext;
use hier_kv_gateway_metadata::store::MetadataStore;

use hier_kv_gateway_routing::engine::RoutingEngine;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

const MODEL_NAME: &str = "qwen2.5-7b";

/// Build a store with `n` backends, each carrying a realistic load snapshot
/// (mixed active_requests + active_decode_blocks + active_prefill_tokens) so
/// the token-aware terms have real numbers to multiply rather than zeros.
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
                model_name: MODEL_NAME.to_string(),
                model_architecture: "qwen".to_string(),
                quantization: Quantization::Fp16,
                max_context_len: 32_768,
                supports_tool_calling: true,
                supports_streaming: true,
            }],
            region: region.clone(),
            indexer_domain: IndexerDomainId::new(0),
            capabilities: BackendCapabilities {
                supports_kv_events: true,
                supports_batching: true,
                max_batch_size: 32,
                gpu_count: 1,
                gpu_memory_gb: 24,
            },
            kv_config: KvConfig {
                block_size: 16,
                cache_namespace: "default".to_string(),
                max_kv_blocks: 8192,
            },
            status: BackendStatus::Healthy,
        };
        store.register_backend(info);

        // Realistic mixed load: some requests, some decode pressure, some
        // prefill pressure — so the token-aware terms are exercised.
        let metrics = BackendMetrics {
            active_requests: (i as u64) % 4 + 1,
            queue_depth: 0,
            active_decode_blocks: ((i as u64) % 8) * 32, // 0..=224
            active_prefill_tokens: ((i as u64) % 5) * 256, // 0..=1024
            kv_used_blocks: 200,
            kv_total_blocks: 8192,
            gpu_utilization: 0.3,
            gpu_memory_used_mb: 10_000,
            gpu_memory_total_mb: 24_000,
            latency: LatencyStats {
                p50_ms: 10.0,
                p99_ms: 50.0,
                p999_ms: 80.0,
                sample_count: 1000,
            },
            timestamp: chrono::Utc::now().timestamp(),
        };
        store.load_update(b.clone(), metrics);
        backends.push(b);
    }
    (store, backends)
}

fn build_engine(load: LoadAwareStrategy) -> RoutingEngine {
    let kv = Box::new(KvAwareStrategy::default());
    let model = Box::new(ModelAwareStrategy::default());
    let topology = Box::new(TopologyAwareStrategy {
        w_rtt: 1.0,
        w_bw: 0.0,
        self_region: RegionId::new("cloud-cn-beijing"),
    });
    // Pure-load weights (kv/topology zeroed) to isolate the load-strategy cost.
    let weights = StrategyWeights {
        kv: 0.0,
        load: 1.0,
        topology: 0.0,
    };
    let hybrid = HybridStrategy::new(kv, model, Box::new(load), topology, weights, 0.0);
    RoutingEngine::new(hybrid, Duration::from_secs(300), 3, RegionId::new("cloud-cn-beijing"))
}

fn baseline_load() -> LoadAwareStrategy {
    LoadAwareStrategy {
        w_decode: 0.0,
        w_prefill: 0.0,
        ..LoadAwareStrategy::default()
    }
}

fn token_aware_load() -> LoadAwareStrategy {
    LoadAwareStrategy::default()
}

fn ctx_with_output_budget(estimated_output_tokens: u32) -> RoutingContext {
    RoutingContext {
        request_id: None,
        session_id: None,
        model_name: Some(MODEL_NAME.to_string()),
        token_ids: Vec::new(),
        block_hashes: Vec::new(),
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens,
        requires_tool_calling: false,
    }
}

fn bench_route_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_aware_route");
    group.sample_size(100);

    for n in [1usize, 6, 12, 20] {
        let (store, backends) = build_store(n);
        // Sanity: routing must succeed for every iteration.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let ctx = ctx_with_output_budget(1024);

        let baseline = build_engine(baseline_load());
        rt.block_on(async {
            let _ = baseline.route(&ctx, &store).await.unwrap();
        });
        let token_aware = build_engine(token_aware_load());
        rt.block_on(async {
            let _ = token_aware.route(&ctx, &store).await.unwrap();
        });
        let _ = backends; // keep scope

        group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, &_| {
            let baseline = build_engine(baseline_load());
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let d = baseline
                    .route(black_box(&ctx), black_box(&store))
                    .await
                    .unwrap();
                black_box(d);
            });
        });

        group.bench_with_input(BenchmarkId::new("token_aware", n), &n, |b, &_| {
            let token_aware = build_engine(token_aware_load());
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let d = token_aware
                    .route(black_box(&ctx), black_box(&store))
                    .await
                    .unwrap();
                black_box(d);
            });
        });
    }
    group.finish();
}

criterion_group!(name = benches; config = Criterion::default(); targets = bench_route_latency);
criterion_main!(benches);
