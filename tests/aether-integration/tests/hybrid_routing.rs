//! 真实混合路由策略集成测试。
//!
//! 该测试构造两个不同 Region 的真实后端（cloud-beijing 与 edge-shanghai），
//! 注册到 `MetadataStore`，对 cloud-beijing 应用真实 KV 事件以建立本地精确索引，
//! 然后调用 `RoutingEngine::route` 验证：
//! a) 选中后端是 cloud-beijing（因为本地 KV overlap 更高）
//! b) kv_overlap > 0
//! c) 策略名为 "session_affinity" 或 "hybrid"
//!
//! 全程使用真实组件，不引入任何 mock。

use std::time::Duration;

use aether_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
    ModelInstance, Protocol, Quantization,
};
use aether_core::config::StrategyWeights;
use aether_core::ids::{BackendId, IndexerDomainId, RegionId, WorkerWithRank};
use aether_core::kv_event::KvCacheEvent;
use aether_core::request::RoutingContext;
use aether_metadata::store::MetadataStore;
use aether_routing::engine::RoutingEngine;
use aether_routing::hybrid::HybridStrategy;
use aether_routing::kv_aware::KvAwareStrategy;
use aether_routing::load_aware::LoadAwareStrategy;
use aether_routing::model_aware::ModelAwareStrategy;
use aether_routing::topology_aware::TopologyAwareStrategy;

/// 构造一个 backend 信息结构，承载给定模型名。
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

/// 构造一个 Stored 事件。
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

    // 1) 注册两个不同 Region 的后端，均加载 "test-model"
    let cloud_region = "cloud-beijing";
    let edge_region = "edge-shanghai";
    let cloud_backend_id = BackendId::new(cloud_region, "worker-0");
    let edge_backend_id = BackendId::new(edge_region, "worker-1");

    let cloud_info = make_backend_info(cloud_region, "worker-0", 1, "test-model");
    let edge_info = make_backend_info(edge_region, "worker-1", 2, "test-model");
    store.register_backend(cloud_info);
    store.register_backend(edge_info);

    // 2) 对 cloud-beijing backend 应用 KV 事件，存入块哈希 [1, 2, 3]
    store
        .kv_apply_event(stored(vec![1, 2, 3]), cloud_backend_id.clone())
        .await
        .expect("应用 KV 事件到 cloud-beijing 应成功");

    // 直接验证本地精确索引已建立
    let cloud_overlap = store
        .kv_find_local_overlap(&[1, 2, 3], cloud_backend_id.clone())
        .await;
    assert_eq!(cloud_overlap, 3, "cloud-beijing 应命中全部 3 个块");
    let edge_overlap = store
        .kv_find_local_overlap(&[1, 2, 3], edge_backend_id.clone())
        .await;
    assert_eq!(edge_overlap, 0, "edge-shanghai 不应命中任何块");

    // 3) 构造 RoutingEngine，self_region = cloud-beijing
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
        },
        0.0, // temperature = 0 → 贪心选最高分
    );
    let engine = RoutingEngine::new(
        hybrid,
        Duration::from_secs(300),
        3,
        RegionId::new(cloud_region),
    );

    // 4) 构造 RoutingContext：block_hashes 前 3 位与已存 hash 重合
    let ctx = RoutingContext {
        request_id: None,
        session_id: None,
        model_name: Some("test-model".to_string()),
        token_ids: vec![1, 2, 3, 4, 5],
        block_hashes: vec![1, 2, 3, 4, 5],
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 32,
        requires_tool_calling: false,
    };

    // 5) 调用 route() 并断言
    let decision = engine
        .route(&ctx, &store)
        .await
        .expect("路由应成功返回决策");

    // a) 选中的后端应为 cloud-beijing（因为 KV overlap 更高）
    assert_eq!(
        decision.backend, cloud_backend_id,
        "应选中 cloud-beijing 后端，实际: {}",
        decision.backend
    );

    // b) kv_overlap > 0
    assert!(
        decision.kv_overlap > 0,
        "kv_overlap 应 > 0，实际: {}",
        decision.kv_overlap
    );
    assert_eq!(
        decision.kv_overlap, 3,
        "kv_overlap 应等于 3（前 3 块命中），实际: {}",
        decision.kv_overlap
    );

    // c) strategy 应为 "hybrid" 或 "session_affinity"
    assert!(
        decision.strategy == "hybrid" || decision.strategy == "session_affinity",
        "strategy 应为 hybrid 或 session_affinity，实际: {}",
        decision.strategy
    );
    assert_eq!(
        decision.strategy, "hybrid",
        "无 session_id 时应走 hybrid 策略，实际: {}",
        decision.strategy
    );
}

#[tokio::test]
async fn hybrid_routing_session_affinity_reuses_previous_backend() {
    // 该用例验证：首次路由选中后端后，会话亲和会在第二次同 session 请求时复用。
    let store = MetadataStore::new();
    let cloud_region = "cloud-beijing";
    let cloud_backend_id = BackendId::new(cloud_region, "worker-0");
    let edge_backend_id = BackendId::new("edge-shanghai", "worker-1");

    store.register_backend(make_backend_info(cloud_region, "worker-0", 1, "m"));
    store.register_backend(make_backend_info("edge-shanghai", "worker-1", 2, "m"));

    // 仅给 cloud-beijing 建立高 KV overlap
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
        },
        0.0,
    );
    let engine = RoutingEngine::new(
        hybrid,
        Duration::from_secs(300),
        3,
        RegionId::new(cloud_region),
    );

    let session_id = aether_core::ids::SessionId::new("session-xyz");

    // 第一次请求：带 session_id，应走 hybrid 路径并写回亲和
    let ctx_first = RoutingContext {
        request_id: None,
        session_id: Some(session_id.clone()),
        model_name: Some("m".to_string()),
        token_ids: vec![1, 2, 3, 4, 5],
        block_hashes: vec![1, 2, 3, 4, 5],
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 32,
        requires_tool_calling: false,
    };
    let first = engine.route(&ctx_first, &store).await.expect("首次路由应成功");
    assert_eq!(first.backend, cloud_backend_id, "首次应选中 cloud-beijing");
    assert_eq!(first.strategy, "hybrid", "首次应走 hybrid");

    // 第二次请求：同 session_id，应直接复用首次选中的后端
    let ctx_second = RoutingContext {
        request_id: None,
        session_id: Some(session_id.clone()),
        model_name: Some("m".to_string()),
        token_ids: vec![10, 20, 30],
        block_hashes: vec![10, 20, 30],
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 16,
        requires_tool_calling: false,
    };
    let second = engine.route(&ctx_second, &store).await.expect("二次路由应成功");
    assert_eq!(
        second.backend, cloud_backend_id,
        "会话亲和应复用首次选中的后端"
    );
    assert_eq!(
        second.strategy, "session_affinity",
        "二次请求应走 session_affinity 策略，实际: {}",
        second.strategy
    );
    assert_ne!(
        second.backend, edge_backend_id,
        "会话亲和不应跳到其他后端"
    );
}
