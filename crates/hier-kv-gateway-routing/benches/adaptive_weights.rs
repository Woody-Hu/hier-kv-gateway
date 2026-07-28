//! Benchmarks for the adaptive weight controller and its interaction with the
//! hybrid strategy hot path.
//!
//! What we measure
//! ----------------
//! * `effective_weights_cached` — the per-request cost of
//!   [`AdaptiveWeightController::effective_weights`] between recomputations:
//!   one mutex + clock check + clone. This is what the routing hot path pays
//!   when adaptive mode is enabled.
//! * `controller_compute` — a full weight recomputation, varying the number
//!   of backends contributing load snapshots (the `load_spread` scan is
//!   `O(N)` over the fleet).
//! * `record_outcome` — the per-attempt feedback cost
//!   (`record_success` / `record_failure` / `record_kv_overlap`) paid by the
//!   forwarding loop.
//! * `hybrid_evaluate_static_vs_adaptive` — end-to-end
//!   [`HybridStrategy::evaluate`] with static weights vs an attached
//!   adaptive controller, isolating the controller overhead on the decision
//!   path.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-routing --bench adaptive_weights
//! ```

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use hier_kv_gateway_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig, ModelInstance,
    Protocol, Quantization,
};
use hier_kv_gateway_core::config::{AdaptiveConfig, StrategyWeights};
use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId};
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::request::RoutingContext;
use hier_kv_gateway_metadata::store::MetadataStore;

use hier_kv_gateway_routing::adaptive::AdaptiveWeightController;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::strategy::RoutingStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

const MODEL_NAME: &str = "qwen2.5-7b";

fn base_weights() -> StrategyWeights {
    StrategyWeights {
        kv: 0.35,
        load: 0.30,
        topology: 0.20,
    }
}

fn adaptive_cfg(interval_secs: u64) -> AdaptiveConfig {
    AdaptiveConfig {
        enabled: true,
        ema_alpha: 0.2,
        max_adjustment: 0.25,
        min_weight: 0.05,
        adjust_interval_secs: interval_secs,
    }
}

/// Register `n` backends with load metrics in the store; returns their ids.
fn build_store(n: usize) -> (MetadataStore, Vec<BackendId>) {
    let store = MetadataStore::new();
    let region = RegionId::new("cloud-cn-beijing");
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = BackendId::new(region.clone(), format!("inst-{i}"));
        store.register_backend(BackendInfo {
            id: id.clone(),
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
        });
        // Uneven load so `load_spread` does real work.
        let mut m = BackendMetrics::default();
        m.active_requests = (i as u64 * 7) % 23;
        store.load_update(id.clone(), m);
        ids.push(id);
    }
    (store, ids)
}

fn build_hybrid(adaptive: Option<Arc<AdaptiveWeightController>>) -> HybridStrategy {
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
        self_region: RegionId::new("cloud-cn-beijing"),
    });
    let hybrid = HybridStrategy::new(kv, model, load, topology, base_weights(), 0.0);
    match adaptive {
        Some(ctl) => hybrid.with_adaptive(ctl),
        None => hybrid,
    }
}

fn routing_ctx() -> RoutingContext {
    RoutingContext {
        request_id: None,
        session_id: None,
        model_name: Some(MODEL_NAME.to_string()),
        token_ids: Vec::new(),
        block_hashes: (1..=16u64).collect(),
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 128,
        requires_tool_calling: false,
    }
}

/// Per-request cost of the cached (non-recomputing) adaptive path.
fn bench_effective_weights_cached(c: &mut Criterion) {
    let (store, _ids) = build_store(10);
    let ctl = AdaptiveWeightController::new(base_weights(), adaptive_cfg(3600));
    // Prime the cache once.
    let _ = ctl.effective_weights(&store);

    c.bench_function("effective_weights_cached", |b| {
        b.iter(|| {
            let w = ctl.effective_weights(black_box(&store));
            black_box(w);
        });
    });
}

/// Full recompute cost, scaling with fleet size.
fn bench_controller_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("controller_compute");
    group.sample_size(100);

    for n in [2usize, 10, 50, 200] {
        let (store, ids) = build_store(n);
        let ctl = AdaptiveWeightController::new(base_weights(), adaptive_cfg(0));
        // Feed outcome/kv signals so every branch runs.
        for (i, id) in ids.iter().enumerate() {
            if i % 3 == 0 {
                ctl.record_failure(id);
            } else {
                ctl.record_success(id, Duration::from_millis(15));
            }
        }
        ctl.record_kv_overlap(12, 16);

        group.bench_with_input(BenchmarkId::new("backends", n), &n, |b, &_n| {
            b.iter(|| {
                let w = ctl.compute(black_box(&store));
                black_box(w);
            });
        });
    }
    group.finish();
}

/// Feedback-recording cost paid by the forwarding loop per attempt.
fn bench_record_outcome(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_outcome");
    group.sample_size(200);

    let ctl = AdaptiveWeightController::new(base_weights(), adaptive_cfg(0));
    let backend = BackendId::new("r1", "inst-0");

    group.bench_function("record_success", |b| {
        b.iter(|| ctl.record_success(black_box(&backend), Duration::from_millis(20)));
    });
    group.bench_function("record_failure", |b| {
        b.iter(|| ctl.record_failure(black_box(&backend)));
    });
    group.bench_function("record_kv_overlap", |b| {
        b.iter(|| ctl.record_kv_overlap(black_box(12), black_box(16)));
    });
    group.finish();
}

/// Hybrid evaluate with static weights vs with an adaptive controller
/// attached (cached path), isolating the controller's per-request overhead.
fn bench_hybrid_evaluate_static_vs_adaptive(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_evaluate_static_vs_adaptive");
    group.sample_size(50);

    let (store, backends) = build_store(10);
    let ctx = routing_ctx();

    let static_hybrid = build_hybrid(None);
    group.bench_function("static", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.to_async(&rt).iter(|| async {
            let scored = static_hybrid
                .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                .await
                .unwrap();
            black_box(scored);
        });
    });

    let ctl = Arc::new(AdaptiveWeightController::new(
        base_weights(),
        adaptive_cfg(3600),
    ));
    let adaptive_hybrid = build_hybrid(Some(ctl));
    group.bench_function("adaptive_cached", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.to_async(&rt).iter(|| async {
            let scored = adaptive_hybrid
                .evaluate(black_box(&ctx), black_box(&backends), black_box(&store))
                .await
                .unwrap();
            black_box(scored);
        });
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_effective_weights_cached,
        bench_controller_compute,
        bench_record_outcome,
        bench_hybrid_evaluate_static_vs_adaptive,
);
criterion_main!(benches);
