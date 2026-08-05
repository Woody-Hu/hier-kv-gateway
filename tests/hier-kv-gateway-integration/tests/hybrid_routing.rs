//! Real hybrid routing strategy integration test.
//!
//! This test constructs two real backends in different Regions (cloud-beijing and
//! edge-shanghai), registers them into the `MetadataStore`, applies real KV events to
//! cloud-beijing to build the local exact index, and then calls
//! `RoutingEngine::route` to verify:
//! a) The selected backend is cloud-beijing (because the local KV overlap is higher)
//! b) kv_overlap > 0
//! c) The strategy name is "session_affinity" or "hybrid"
//!
//! Real components are used throughout; no mocks are introduced.

use std::time::Duration;

use hier_kv_gateway_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
    ModelInstance, Protocol, Quantization,
};
use hier_kv_gateway_core::config::StrategyWeights;
use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId, WorkerWithRank};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_core::request::RoutingContext;
use hier_kv_gateway_metadata::store::MetadataStore;
use hier_kv_gateway_routing::engine::RoutingEngine;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

/// Construct a backend info struct carrying the given model name.
fn make_backend_info(
    region: &str,
    instance: &str,
    domain: u64,
    model_name: &str,
) -> BackendInfo {
    BackendInfo {
        id: BackendId::new(region, instance),
        backend_type: BackendType::VllmEngine,
        endpoint: Endpoint {
            url: format!("http://{}.example:8000", instance),
            protocol: Protocol::Http,
        },
        models: vec![ModelInstance {
            model_name: model_name.to_string(),
            model_architecture: "llama".to_string(),
            quantization: Quantization::Fp16,
            max_context_len: 4096,
            supports_tool_calling: false,
            supports_streaming: true,
        }],
        region: RegionId::new(region),
        indexer_domain: IndexerDomainId::new(domain),
        capabilities: BackendCapabilities {
            supports_kv_events: true,
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

/// Construct a Stored event.
fn stored(hashes: Vec<u64>) -> KvCacheEvent {
    KvCacheEvent::Stored {
        worker: WorkerWithRank::from_worker_id(1),
        block_hashes: hashes,
        parent_hash: None,
        num_block_tokens: Vec::new(),
    }
}

#[tokio::test]
async fn hybrid_routing_prefers_backend_with_higher_kv_overlap() {
    let store = MetadataStore::new();

    // 1) Register two backends in different Regions, both loading "test-model"
    let cloud_region = "cloud-beijing";
    let edge_region = "edge-shanghai";
    let cloud_backend_id = BackendId::new(cloud_region, "worker-0");
    let edge_backend_id = BackendId::new(edge_region, "worker-1");

    let cloud_info = make_backend_info(cloud_region, "worker-0", 1, "test-model");
    let edge_info = make_backend_info(edge_region, "worker-1", 2, "test-model");
    store.register_backend(cloud_info);
    store.register_backend(edge_info);

    // 2) Apply a KV event to the cloud-beijing backend, storing block hashes [1, 2, 3]
    store
        .kv_apply_event(stored(vec![1, 2, 3]), cloud_backend_id.clone())
        .await
        .expect("applying the KV event to cloud-beijing should succeed");

    // Directly verify the local exact index has been built
    let cloud_overlap = store
        .kv_find_local_overlap(&[1, 2, 3], cloud_backend_id.clone())
        .await;
    assert_eq!(cloud_overlap, 3, "cloud-beijing should hit all 3 blocks");
    let edge_overlap = store
        .kv_find_local_overlap(&[1, 2, 3], edge_backend_id.clone())
        .await;
    assert_eq!(edge_overlap, 0, "edge-shanghai should not hit any blocks");

    // 3) Construct the RoutingEngine, self_region = cloud-beijing
    let hybrid = HybridStrategy::new(
        Box::new(KvAwareStrategy::default()),
        Box::new(ModelAwareStrategy::default()),
        Box::new(LoadAwareStrategy::default()),
        Box::new(TopologyAwareStrategy {
            w_rtt: 1.0,
            w_bw: 0.0,
            self_region: RegionId::new(cloud_region),
        }),
        StrategyWeights {
            kv: 0.35,
            load: 0.30,
            topology: 0.20,
            cost: 0.0,
        },
        0.0, // temperature = 0 -> greedily select the highest score
    );
    let engine = RoutingEngine::new(
        hybrid,
        Duration::from_secs(300),
        3,
        RegionId::new(cloud_region),
    );

    // 4) Construct the RoutingContext: the first 3 block_hashes overlap with the stored hashes
    let ctx = RoutingContext {
        request_id: None,
        session_id: None,
        tenant_id: None,
        model_name: Some("test-model".to_string()),
        token_ids: vec![1, 2, 3, 4, 5],
        block_hashes: vec![1, 2, 3, 4, 5],
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 32,
        requires_tool_calling: false,
    };

    // 5) Call route() and assert
    let decision = engine
        .route(&ctx, &store)
        .await
        .expect("routing should successfully return a decision");

    // a) The selected backend should be cloud-beijing (because the KV overlap is higher)
    assert_eq!(
        decision.backend, cloud_backend_id,
        "should select the cloud-beijing backend, actual: {}",
        decision.backend
    );

    // b) kv_overlap > 0
    assert!(
        decision.kv_overlap > 0,
        "kv_overlap should be > 0, actual: {}",
        decision.kv_overlap
    );
    assert_eq!(
        decision.kv_overlap, 3,
        "kv_overlap should equal 3 (first 3 blocks hit), actual: {}",
        decision.kv_overlap
    );

    // c) strategy should be "hybrid" or "session_affinity"
    assert!(
        decision.strategy == "hybrid" || decision.strategy == "session_affinity",
        "strategy should be hybrid or session_affinity, actual: {}",
        decision.strategy
    );
    assert_eq!(
        decision.strategy, "hybrid",
        "without a session_id, should go through the hybrid strategy, actual: {}",
        decision.strategy
    );
}

#[tokio::test]
async fn hybrid_routing_session_affinity_reuses_previous_backend() {
    // This case verifies: after the first routing selects a backend, session affinity reuses
    // it on the second request with the same session.
    let store = MetadataStore::new();
    let cloud_region = "cloud-beijing";
    let cloud_backend_id = BackendId::new(cloud_region, "worker-0");
    let edge_backend_id = BackendId::new("edge-shanghai", "worker-1");

    store.register_backend(make_backend_info(cloud_region, "worker-0", 1, "m"));
    store.register_backend(make_backend_info("edge-shanghai", "worker-1", 2, "m"));

    // Only build high KV overlap for cloud-beijing
    store
        .kv_apply_event(stored(vec![1, 2, 3]), cloud_backend_id.clone())
        .await
        .unwrap();

    let hybrid = HybridStrategy::new(
        Box::new(KvAwareStrategy::default()),
        Box::new(ModelAwareStrategy::default()),
        Box::new(LoadAwareStrategy::default()),
        Box::new(TopologyAwareStrategy {
            w_rtt: 1.0,
            w_bw: 0.0,
            self_region: RegionId::new(cloud_region),
        }),
        StrategyWeights {
            kv: 0.35,
            load: 0.30,
            topology: 0.20,
            cost: 0.0,
        },
        0.0,
    );
    let engine = RoutingEngine::new(
        hybrid,
        Duration::from_secs(300),
        3,
        RegionId::new(cloud_region),
    );

    let session_id = hier_kv_gateway_core::ids::SessionId::new("session-xyz");

    // First request: with session_id, should go through the hybrid path and write back affinity
    let ctx_first = RoutingContext {
        request_id: None,
        session_id: Some(session_id.clone()),
        tenant_id: None,
        model_name: Some("m".to_string()),
        token_ids: vec![1, 2, 3, 4, 5],
        block_hashes: vec![1, 2, 3, 4, 5],
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 32,
        requires_tool_calling: false,
    };
    let first = engine.route(&ctx_first, &store).await.expect("first routing should succeed");
    assert_eq!(first.backend, cloud_backend_id, "first should select cloud-beijing");
    assert_eq!(first.strategy, "hybrid", "first should go through hybrid");

    // Second request: same session_id, should directly reuse the backend selected on the first request
    let ctx_second = RoutingContext {
        request_id: None,
        session_id: Some(session_id.clone()),
        tenant_id: None,
        model_name: Some("m".to_string()),
        token_ids: vec![10, 20, 30],
        block_hashes: vec![10, 20, 30],
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 16,
        requires_tool_calling: false,
    };
    let second = engine.route(&ctx_second, &store).await.expect("second routing should succeed");
    assert_eq!(
        second.backend, cloud_backend_id,
        "session affinity should reuse the backend selected on the first request"
    );
    assert_eq!(
        second.strategy, "session_affinity",
        "second request should go through the session_affinity strategy, actual: {}",
        second.strategy
    );
    assert_ne!(
        second.backend, edge_backend_id,
        "session affinity should not jump to another backend"
    );
}
