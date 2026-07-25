//! 端到端集成测试。
//!
//! 启动一个真实的 axum HTTP 服务器模拟 OpenAI 兼容推理后端，
//! 然后通过 OpenAICompatConnector 发现后端、注册到 MetadataStore、
//! 调用 RoutingEngine 路由并 forward 请求，验证流式响应的完整链路。
//!
//! 全程使用真实组件，不引入任何 mock。

use std::sync::Arc;
use std::time::Duration;

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

use aether_connector::connector::BackendConnector;
use aether_connector::openai_compat::OpenAICompatConnector;
use aether_connector::registry::ConnectorRegistry;
use aether_core::backend::{BackendType, Endpoint, Protocol};
use aether_core::config::{RoutingConfig, StrategyType, StrategyWeights};
use aether_core::ids::RegionId;
use aether_core::request::InferenceChunk;
use aether_metadata::store::MetadataStore;
use aether_routing::engine::RoutingEngine;

use futures::StreamExt;

/// 模拟后端的 SSE 响应内容。
const FAKE_SSE_RESPONSE: &str = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

/// 模拟后端: POST /v1/chat/completions 返回固定 SSE 流。
async fn mock_chat_completions() -> (axum::http::StatusCode, String) {
    (
        axum::http::StatusCode::OK,
        FAKE_SSE_RESPONSE.to_string(),
    )
}

/// 模拟后端: GET /v1/models 返回模型列表。
async fn mock_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [
            {"id": "test-model", "object": "model", "created": 1700000000, "owned_by": "aether-test"}
        ]
    }))
}

/// 模拟后端: GET /health 返回健康状态。
async fn mock_health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

/// 启动一个真实的 HTTP 后端服务器，返回其监听地址。
async fn start_backend_server() -> String {
    use axum::routing::get;

    let app = Router::new()
        .route("/v1/chat/completions", post(|| async {
            let (status, body) = mock_chat_completions().await;
            (status, [("content-type", "text/event-stream")], body)
        }))
        .route("/v1/models", get(|| async { mock_models().await }))
        .route("/health", get(|| async { mock_health().await }));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let port = addr.port();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // 等待服务器就绪
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://127.0.0.1:{}", port)
}

#[tokio::test]
async fn end_to_end_discover_route_forward() {
    // 1. 启动真实 HTTP 后端
    let backend_url = start_backend_server().await;
    println!("后端服务器启动于 {}", backend_url);

    // 2. 创建连接器指向真实后端
    let connector = OpenAICompatConnector::new(
        &backend_url,
        BackendType::VllmEngine,
        RegionId::new("test-region"),
        "test-instance",
        vec!["test-model".to_string()],
        16,
    );

    // 3. discover() 获取后端信息
    let backends = connector
        .discover()
        .await
        .expect("discover 应该成功从真实后端获取模型列表");
    assert_eq!(backends.len(), 1);
    let backend_info = &backends[0];
    assert_eq!(backend_info.backend_type, BackendType::VllmEngine);
    assert!(
        !backend_info.models.is_empty(),
        "应该从 /v1/models 获取到模型"
    );
    assert_eq!(backend_info.models[0].model_name, "test-model");

    // 4. 健康检查
    let health = connector
        .health_check(&backend_info.id)
        .await
        .expect("健康检查应该成功");
    assert_eq!(health.status, aether_core::backend::BackendStatus::Healthy);

    // 5. 注册到 MetadataStore
    let meta = Arc::new(MetadataStore::new());
    for b in &backends {
        meta.register_backend(b.clone());
    }
    assert_eq!(meta.backends_all().len(), 1);

    // 6. 创建路由引擎（需要构造 HybridStrategy）
    use aether_routing::hybrid::HybridStrategy;
    use aether_routing::kv_aware::KvAwareStrategy;
    use aether_routing::load_aware::LoadAwareStrategy;
    use aether_routing::model_aware::ModelAwareStrategy;
    use aether_routing::topology_aware::TopologyAwareStrategy;

    let hybrid = HybridStrategy::new(
        Box::new(KvAwareStrategy {
            overlap_score_credit: 1.0,
            prefill_load_scale: 1.0,
            ckf_false_positive_penalty: 0.3,
        }),
        Box::new(ModelAwareStrategy),
        Box::new(LoadAwareStrategy::default()),
        Box::new(TopologyAwareStrategy {
            w_rtt: 1.0,
            w_bw: 0.3,
            self_region: RegionId::new("test-region"),
        }),
        StrategyWeights {
            kv: 0.35,
            load: 0.30,
            topology: 0.20,
        },
        0.0,
    );
    let routing_engine = RoutingEngine::new(
        hybrid,
        Duration::from_secs(300),
        3,
        RegionId::new("test-region"),
    );

    // 7. 创建连接器注册表
    let registry = ConnectorRegistry::new();
    registry.register(Arc::new(connector));

    // 8. 构建推理请求
    let request = aether_core::request::InferenceRequest {
        request_id: aether_core::ids::RequestId::new("test-req-1"),
        model: "test-model".to_string(),
        messages: vec![aether_core::request::ChatMessage {
            role: "user".to_string(),
            content: "Hello, world!".to_string(),
        }],
        token_ids: vec![1, 2, 3, 4, 5],
        max_tokens: 100,
        temperature: 0.7,
        stream: true,
        tools: vec![],
        lora_name: None,
    };

    // 9. 路由决策
    let ctx = aether_core::request::RoutingContext {
        request_id: Some(request.request_id.clone()),
        session_id: None,
        model_name: Some(request.model.clone()),
        token_ids: request.token_ids.clone(),
        block_hashes: aether_core::kv_event::compute_block_hashes(
            &aether_core::kv_event::BlockHashInput {
                tokens: &request.token_ids,
                kv_block_size: 16,
                cache_namespace: None,
                lora_name: None,
            },
        ),
        block_size: 16,
        lora_name: None,
        cache_namespace: None,
        estimated_output_tokens: 100,
        requires_tool_calling: false,
    };

    let decision = routing_engine
        .route(&ctx, &meta)
        .await
        .expect("路由决策应该成功");
    println!("路由决策: backend={}, strategy={}", decision.backend, decision.strategy);

    // 10. 通过连接器转发请求
    let connector = registry.get(&BackendType::VllmEngine).expect("应能找到连接器");
    let mut stream = connector
        .forward(&decision.backend, &request)
        .await
        .expect("转发请求应该成功");

    // 11. 收集流式响应并验证
    let mut deltas = Vec::new();
    let mut got_done = false;
    while let Some(chunk) = stream.next().await {
        match chunk {
            InferenceChunk::Delta { text, finish_reason } => {
                println!("收到 Delta: text={:?}, finish_reason={:?}", text, finish_reason);
                deltas.push(text);
            }
            InferenceChunk::Done { backend_id, latency_ms } => {
                println!(
                    "收到 Done: backend={}, latency={}ms",
                    backend_id, latency_ms
                );
                got_done = true;
                break;
            }
            InferenceChunk::Error { code, message } => {
                panic!("收到错误块: code={}, message={}", code, message);
            }
            _ => {}
        }
    }

    assert!(got_done, "流应该以 Done 块结束");
    assert!(
        !deltas.is_empty(),
        "应该收到至少一个 Delta 块"
    );
    let full_text: String = deltas.concat();
    assert!(
        full_text.contains("Hello") || full_text.contains("world"),
        "响应文本应包含 'Hello' 或 'world', 实际: {}",
        full_text
    );
}

#[tokio::test]
async fn end_to_end_collect_metrics() {
    // 启动真实后端
    let backend_url = start_backend_server().await;

    let connector = OpenAICompatConnector::new(
        &backend_url,
        BackendType::VllmEngine,
        RegionId::new("test-region"),
        "test-instance",
        vec!["test-model".to_string()],
        16,
    );

    let backend_id = aether_core::ids::BackendId::new("test-region", "test-instance");

    // collect_metrics 应该成功（即使后端无 /metrics 端点，也应返回默认值）
    let metrics = connector
        .collect_metrics(&backend_id)
        .await
        .expect("指标采集应该成功");

    // 默认值验证（后端无 /metrics 端点，所有字段应为 0 或默认）
    assert_eq!(metrics.active_requests, 0);
    assert_eq!(metrics.queue_depth, 0);
    assert!(metrics.timestamp > 0);
}
