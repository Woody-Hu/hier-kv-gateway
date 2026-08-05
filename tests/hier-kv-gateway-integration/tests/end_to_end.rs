//! End-to-end integration test.
//!
//! Starts a real axum HTTP server simulating an OpenAI-compatible inference backend,
//! then uses OpenAICompatConnector to discover the backend, register it into the
//! MetadataStore, call RoutingEngine to route and forward the request, and verify the
//! streaming response over the full link.
//!
//! Real components are used throughout; no mocks are introduced.

use std::sync::Arc;
use std::time::Duration;

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

use hier_kv_gateway_connector::connector::BackendConnector;
use hier_kv_gateway_connector::openai_compat::OpenAICompatConnector;
use hier_kv_gateway_connector::registry::ConnectorRegistry;
use hier_kv_gateway_core::backend::{BackendType, Endpoint, Protocol};
use hier_kv_gateway_core::config::StrategyWeights;
use hier_kv_gateway_core::ids::RegionId;
use hier_kv_gateway_core::request::InferenceChunk;
use hier_kv_gateway_metadata::store::MetadataStore;
use hier_kv_gateway_routing::engine::RoutingEngine;

use futures::StreamExt;

/// SSE response content for the mock backend.
const FAKE_SSE_RESPONSE: &str = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

/// Mock backend: POST /v1/chat/completions returns a fixed SSE stream.
async fn mock_chat_completions() -> (axum::http::StatusCode, String) {
    (
        axum::http::StatusCode::OK,
        FAKE_SSE_RESPONSE.to_string(),
    )
}

/// Mock backend: GET /v1/models returns the model list.
async fn mock_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [
            {"id": "test-model", "object": "model", "created": 1700000000, "owned_by": "hier-kv-gateway-test"}
        ]
    }))
}

/// Mock backend: GET /health returns the health status.
async fn mock_health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

/// Start a real HTTP backend server and return its listen address.
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

    // Wait for the server to be ready
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://127.0.0.1:{}", port)
}

#[tokio::test]
async fn end_to_end_discover_route_forward() {
    // 1. Start the real HTTP backend
    let backend_url = start_backend_server().await;
    println!("backend server started at {}", backend_url);

    // 2. Create a connector pointing at the real backend
    let connector = OpenAICompatConnector::new(
        &backend_url,
        BackendType::VllmEngine,
        RegionId::new("test-region"),
        "test-instance",
        vec!["test-model".to_string()],
        16,
    );

    // 3. discover() to get backend info
    let backends = connector
        .discover()
        .await
        .expect("discover should successfully fetch the model list from the real backend");
    assert_eq!(backends.len(), 1);
    let backend_info = &backends[0];
    assert_eq!(backend_info.backend_type, BackendType::VllmEngine);
    assert!(
        !backend_info.models.is_empty(),
        "should fetch models from /v1/models"
    );
    assert_eq!(backend_info.models[0].model_name, "test-model");

    // 4. Health check
    let health = connector
        .health_check(&backend_info.id)
        .await
        .expect("health check should succeed");
    assert_eq!(health.status, hier_kv_gateway_core::backend::BackendStatus::Healthy);

    // 5. Register into the MetadataStore
    let meta = Arc::new(MetadataStore::new());
    for b in &backends {
        meta.register_backend(b.clone());
    }
    assert_eq!(meta.backends_all().len(), 1);

    // 6. Create the routing engine (need to construct HybridStrategy)
    use hier_kv_gateway_routing::hybrid::HybridStrategy;
    use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
    use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
    use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
    use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

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

    // 7. Create the connector registry
    let registry = ConnectorRegistry::new();
    registry.register(Arc::new(connector));

    // 8. Build the inference request
    let request = hier_kv_gateway_core::request::InferenceRequest {
        request_id: hier_kv_gateway_core::ids::RequestId::new("test-req-1"),
        model: "test-model".to_string(),
        messages: vec![hier_kv_gateway_core::request::ChatMessage {
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

    // 9. Routing decision
    let ctx = hier_kv_gateway_core::request::RoutingContext {
        request_id: Some(request.request_id.clone()),
        session_id: None,
        tenant_id: None,
        model_name: Some(request.model.clone()),
        token_ids: request.token_ids.clone(),
        block_hashes: hier_kv_gateway_core::kv_event::compute_block_hashes(
            &hier_kv_gateway_core::kv_event::BlockHashInput {
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
        .expect("routing decision should succeed");
    println!("routing decision: backend={}, strategy={}", decision.backend, decision.strategy);

    // 10. Forward the request via the connector (addressed by backend id)
    let connector = registry.get(&decision.backend).expect("should be able to find the connector");
    let mut stream = connector
        .forward(&decision.backend, &request)
        .await
        .expect("forwarding the request should succeed");

    // 11. Collect the streaming response and verify
    let mut deltas = Vec::new();
    let mut got_done = false;
    while let Some(chunk) = stream.next().await {
        match chunk {
            InferenceChunk::Delta { text, finish_reason } => {
                println!("received Delta: text={:?}, finish_reason={:?}", text, finish_reason);
                deltas.push(text);
            }
            InferenceChunk::Done { backend_id, latency_ms } => {
                println!(
                    "received Done: backend={}, latency={}ms",
                    backend_id, latency_ms
                );
                got_done = true;
                break;
            }
            InferenceChunk::Error { code, message } => {
                panic!("received error chunk: code={}, message={}", code, message);
            }
            _ => {}
        }
    }

    assert!(got_done, "the stream should end with a Done chunk");
    assert!(
        !deltas.is_empty(),
        "should receive at least one Delta chunk"
    );
    let full_text: String = deltas.concat();
    assert!(
        full_text.contains("Hello") || full_text.contains("world"),
        "the response text should contain 'Hello' or 'world', actual: {}",
        full_text
    );
}

#[tokio::test]
async fn end_to_end_collect_metrics() {
    // Start the real backend
    let backend_url = start_backend_server().await;

    let connector = OpenAICompatConnector::new(
        &backend_url,
        BackendType::VllmEngine,
        RegionId::new("test-region"),
        "test-instance",
        vec!["test-model".to_string()],
        16,
    );

    let backend_id = hier_kv_gateway_core::ids::BackendId::new("test-region", "test-instance");

    // collect_metrics should succeed (even if the backend has no /metrics endpoint, it should return default values)
    let metrics = connector
        .collect_metrics(&backend_id)
        .await
        .expect("metrics collection should succeed");

    // Default value verification (backend has no /metrics endpoint, all fields should be 0 or default)
    assert_eq!(metrics.active_requests, 0);
    assert_eq!(metrics.queue_depth, 0);
    assert!(metrics.timestamp > 0);
}
