//! Benchmarks for the routing hot path.
//!
//! What we measure
//! ----------------
//! * `hybrid_evaluate` — the [`HybridStrategy::evaluate`] call, which is the
//!   central per-request cost. Varies the candidate count to expose the
//!   `O(N)` per-candidate `kv_find_local_overlap` round-trips (defect #③)
//!   and the `normalize_costs` HashMap allocations (defect #④).
//! * `engine_route_full` — end-to-end [`RoutingEngine::route`] including
//!   session-affinity miss → hybrid → select → trace-score recomputation
//!   (defects #① #②).
//! * `select_with_temperature` — the softmax/greedy selection over a scored
//!   list, isolated from the metadata-store cost.
//!
//! These benchmarks establish the *baseline* before any optimization.
//! After fixing defects #① #② #③ we re-run and compare via the criterion
//! HTML report under `target/criterion/`.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-routing
//! ```

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use hier_kv_gateway_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig, ModelInstance,
    Protocol, Quantization,
};
use hier_kv_gateway_core::config::StrategyWeights;
use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId, WorkerWithRank};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_core::metrics::{BackendMetrics, LatencyStats};
use hier_kv_gateway_core::request::RoutingContext;
use hier_kv_gateway_metadata::store::MetadataStore;

use hier_kv_gateway_routing::engine::RoutingEngine;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::strategy::RoutingStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

// --------------------------------------------------------------------------
// Test-data builders
// --------------------------------------------------------------------------

const PREFIX_LEN: usize = 16;
const MODEL_NAME: &str = "qwen2.5-7b";

/// Build a MetadataStore with `n_backends` registered backends, each:
///   * serving `MODEL_NAME`
///   * owning the same `PREFIX_LEN`-long block-hash prefix (so KV overlap = 16
///     for every candidate — exposes the cost of N round-trips when overlap
///     could be obtained in a single `find_all_matches` call)
///   * carrying fresh load metrics so the load strategy stays in the loop
///
/// Note: we keep a single tokio Runtime for all `kv_apply_event` calls.
/// Creating a fresh `Runtime` per iteration and dropping it (which drops the
/// RadixTree clone moved into the closure) triggers the `Drop` impl's
/// `try_send(Shutdown)` best-effort signal — the background worker thread
/// would then exit before subsequent setup calls complete.
fn build_store(n_backends: usize) -> (MetadataStore, Vec<BackendId>, Vec<u64>) {
    let store = MetadataStore::new();
    let region = RegionId::new("cloud-cn-beijing");
    let prefix: Vec<u64> = (1..=PREFIX_LEN as u64).collect();
    let mut backends = Vec::with_capacity(n_backends);

    let rt = tokio::runtime::Runtime::new().expect("failed to build tokio runtime for setup");
    for i in 0..n_backends {
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
                max_context_len: 32768,
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
                max_kv_blocks: 1024,
            },
            status: BackendStatus::Healthy,
        };
        store.register_backend(info);

        // Apply a KV event for this backend so the RadixTree has overlap data.
        let event = KvCacheEvent::Stored {
            worker: WorkerWithRank::from_worker_id(i as u64),
            block_hashes: prefix.clone(),
            parent_hash: None,
            num_block_tokens: Vec::new(),
        };
        let b_for_event = b.clone();
        rt.block_on(async {
            store.kv_apply_event(event, b_for_event).await.unwrap();
        });

        // Fresh load metrics — keeps the load strategy out of the stale-discount path.
        let metrics = BackendMetrics {
            active_requests: (i as u64) % 4,
            queue_depth: 0,
            active_decode_blocks: (i as u64) % 8,
            active_prefill_tokens: 0,
            kv_used_blocks: 100,
            kv_total_blocks: 1024,
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

    (store, backends, prefix)
}

/// Build a HybridStrategy mirroring the one wired up in
/// [hier-kv-gateway/src/main.rs::build_routing_engine].
fn build_hbrid(self_region: RegionId) -> HybridStrategy {
    let kv = Box::new(KvAwareStrategy {
        overlap_score_credit: 1.0,
        prefill_load_scale: 1.0,
        ckf_false_positive_penalty: 0.0,
    });
    let model = Box::new(ModelAwareStrategy::default());
    let load = Box::new(LoadAwareStrategy::default());
    let topology = Box::new(TopologyAwareStrategy {
        w_rtt: 1.0,
        w_bw: 0.0,
        self_region,
    });
    let weights = StrategyWeights {
        kv: 0.35,
        load: 0.30,
        topology: 0.20,
    };
    HybridStrategy::new(kv, model, load, topology, weights, 0.0)
}

fn build_routing_engine(self_region: RegionId) -> RoutingEngine {
    let hybrid = build_hbrid(self_region.clone());
    RoutingEngine::new(hybrid, Duration::from_secs(300), 3, self_region)
}

fn build_routing_ctx(prefix: Vec<u64>) -> RoutingContext {
    RoutingContext {
        request_id: None,
        session_id: None, // force the hybrid path, not session affinity
        model_name: Some(MODEL_NAME.to_string()),
        token_ids: Vec::new(),
        block_hashes: prefix,
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 128,
        requires_tool_calling: false,
    }
}

// --------------------------------------------------------------------------
// Benchmarks
// --------------------------------------------------------------------------

fn bench_hybrid_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_evaluate");
    group.sample_size(50);

    for n in [1usize, 5, 10, 20] {
        let (store, backends, prefix) = build_store(n);
        let hybrid = build_hbrid(RegionId::new("cloud-cn-beijing"));
        let ctx = build_routing_ctx(prefix);

        group.bench_with_input(
            BenchmarkId::new("candidates", n),
            &n,
            |b, &_n| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(&rt).iter(|| async {
                    let scored = hybrid
                        .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                        .await
                        .unwrap();
                    black_box(scored);
                });
            },
        );
    }
    group.finish();
}

fn bench_engine_route_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_route_full");
    group.sample_size(50);

    for n in [1usize, 5, 10, 20] {
        let (store, backends, prefix) = build_store(n);
        let engine = build_routing_engine(RegionId::new("cloud-cn-beijing"));
        let ctx = build_routing_ctx(prefix);

        // Sanity: there must be at least one candidate.
        assert!(!backends.is_empty());

        group.bench_with_input(
            BenchmarkId::new("candidates", n),
            &n,
            |b, &_n| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(&rt).iter(|| async {
                    let decision = engine
                        .route(black_box(&ctx), black_box(&store))
                        .await
                        .unwrap();
                    black_box(decision);
                });
            },
        );
    }
    group.finish();
}

fn bench_select_with_temperature(c: &mut Criterion) {
    use hier_kv_gateway_core::request::ScoredBackend;
    use hier_kv_gateway_routing::engine::select_with_temperature;

    let mut group = c.benchmark_group("select_with_temperature");
    group.sample_size(200);

    for n in [1usize, 5, 10, 20] {
        let scored: Vec<ScoredBackend> = (0..n)
            .map(|i| ScoredBackend {
                backend_id: BackendId::new("r1", format!("i{i}")),
                score: 1.0 / (i as f64 + 1.0),
                raw_cost: i as f64,
                meta_version: 0,
            })
            .collect();

        // Greedy path (temperature = 0)
        group.bench_with_input(
            BenchmarkId::new("greedy", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    let chosen = select_with_temperature(black_box(&scored), 0.0);
                    black_box(chosen);
                });
            },
        );

        // Softmax path (temperature = 0.2)
        group.bench_with_input(
            BenchmarkId::new("softmax_t_0.2", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    let chosen = select_with_temperature(black_box(&scored), 0.2);
                    black_box(chosen);
                });
            },
        );
    }
    group.finish();
}

// Silence unused-import warning for `Arc` — kept for future shared-store benches.
#[allow(dead_code)]
fn _arc_anchor() -> Arc<()> {
    Arc::new(())
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_hybrid_evaluate, bench_engine_route_full, bench_select_with_temperature,
);
criterion_main!(benches);
