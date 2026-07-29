//! Integration tests for the new telemetry / adaptive / SGLang features.
//!
//! Covered scenarios (all against real components, no mocks inside the
//! gateway; only the downstream inference servers are stub HTTP servers):
//!
//! 1. **SGLang token-id forwarding** — a stub `sglang.launch_server` exposes
//!    `/generate` (accumulated-text SSE), `/get_server_info`, `/v1/models`
//!    and `/health`; the [`SglangConnector`] must forward pre-tokenized
//!    requests to `/generate` and parse the accumulated-text stream back
//!    into deltas.
//! 2. **SGLang metrics** — `collect_metrics` must prefer `/get_server_info`
//!    and map scheduler state into [`BackendMetrics`].
//! 3. **Decision events through the HTTP stack** — one
//!    `POST /v1/chat/completions` must produce exactly one
//!    [`DecisionEvent`] readable from `GET /admin/decision_events`, with the
//!    winning backend, weight snapshot and attempt list populated.
//! 4. **Failover event** — with the first candidate down, the event must
//!    record both attempts (error on the first, success on the second) and
//!    still terminate in `success`.
//! 5. **Adaptive feedback loop** — after a successful request the attached
//!    [`AdaptiveWeightController`] must show one success sample for the
//!    serving backend and a recorded KV hit ratio.

use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use hier_kv_gateway_api::handlers::AppState;
use hier_kv_gateway_api::server::create_router;
use hier_kv_gateway_api::telemetry::{DecisionEventBuffer, RingBufferSink};
use hier_kv_gateway_connector::connector::BackendConnector;
use hier_kv_gateway_connector::registry::ConnectorRegistry;
use hier_kv_gateway_connector::resilience::{CircuitBreakerRegistry, RetryPolicy};
use hier_kv_gateway_connector::sglang::SglangConnector;
use hier_kv_gateway_core::backend::{BackendType, Endpoint, Protocol};
use hier_kv_gateway_core::config::{
    AdaptiveConfig, BackendConfig, ForwardingConfig, ResilienceConfig, RoutingConfig, StrategyType,
    StrategyWeights,
};
use hier_kv_gateway_core::decision_event::{DecisionEvent, DecisionEventSink};
use hier_kv_gateway_core::ids::{BackendId, RegionId, RequestId};
use hier_kv_gateway_core::kv_event::compute_block_hashes;
use hier_kv_gateway_core::kv_event::BlockHashInput;
use hier_kv_gateway_core::request::{ChatMessage, InferenceChunk, InferenceRequest};
use hier_kv_gateway_metadata::store::MetadataStore;
use hier_kv_gateway_routing::adaptive::AdaptiveWeightController;
use hier_kv_gateway_routing::engine::RoutingEngine;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

use futures::StreamExt;

const MODEL: &str = "qwen2.5-7b";
const REGION: &str = "test-region";

// --------------------------------------------------------------------------
// Stub SGLang server
// --------------------------------------------------------------------------

/// Accumulated-text SSE stream returned by the stub `/generate`.
const FAKE_GENERATE_SSE: &str = "data: {\"text\":\"Hello\",\"meta_info\":{\"finish_reason\":null}}\n\ndata: {\"text\":\"Hello world\",\"meta_info\":{\"finish_reason\":null}}\n\ndata: {\"text\":\"Hello world!\",\"meta_info\":{\"finish_reason\":{\"type\":\"stop\"}}}\n\n";

/// Whether the last `/generate` request carried `input_ids` (checked via a
/// shared flag because the gateway must use the token-id path).
async fn start_stub_sglang_server() -> (String, Arc<std::sync::atomic::AtomicBool>) {
    let saw_input_ids = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = saw_input_ids.clone();

    let app = Router::new()
        .route(
            "/generate",
            post(move |Json(body): Json<Value>| {
                let flag = flag.clone();
                async move {
                    if body.get("input_ids").and_then(|v| v.as_array()).is_some() {
                        flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    (
                        axum::http::StatusCode::OK,
                        [("content-type", "text/event-stream")],
                        FAKE_GENERATE_SSE,
                    )
                }
            }),
        )
        .route(
            "/get_server_info",
            get(|| async {
                Json(json!({
                    "internal_states": [{
                        "num_running_reqs": 5,
                        "num_queue_reqs": 2,
                        "num_used_tokens": 16000,
                        "max_total_num_tokens": 1_000_000
                    }]
                }))
            }),
        )
        .route(
            "/v1/models",
            get(|| async {
                Json(json!({
                    "object": "list",
                    "data": [{"id": MODEL, "object": "model"}]
                }))
            }),
        )
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (format!("http://127.0.0.1:{port}"), saw_input_ids)
}

fn tokenized_request() -> InferenceRequest {
    InferenceRequest {
        request_id: RequestId::new("req-sglang-1"),
        model: MODEL.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }],
        token_ids: vec![101, 102, 103, 104],
        max_tokens: 32,
        temperature: 0.5,
        stream: true,
        tools: vec![],
        lora_name: None,
    }
}

#[tokio::test]
async fn sglang_generate_token_id_forwarding() {
    let (url, saw_input_ids) = start_stub_sglang_server().await;
    let connector = SglangConnector::new(&url, RegionId::new(REGION), "sglang-0", vec![MODEL.to_string()], 16)
        .with_emit_token_ids(true);
    let backend = connector.backend_id();

    let mut stream = connector
        .forward(&backend, &tokenized_request())
        .await
        .expect("forward should succeed");

    let mut text = String::new();
    let mut done = false;
    while let Some(chunk) = stream.next().await {
        match chunk {
            InferenceChunk::Delta { text: t, .. } => text.push_str(&t),
            InferenceChunk::Done { .. } => {
                done = true;
                break;
            }
            InferenceChunk::Error { message, .. } => panic!("stream error: {message}"),
            _ => {}
        }
    }

    assert!(saw_input_ids.load(std::sync::atomic::Ordering::SeqCst), "backend must receive input_ids");
    assert_eq!(text, "Hello world!");
    assert!(done, "stream must terminate with Done");
}

#[tokio::test]
async fn sglang_server_info_metrics_preferred() {
    let (url, _flag) = start_stub_sglang_server().await;
    let connector = SglangConnector::new(&url, RegionId::new(REGION), "sglang-0", vec![MODEL.to_string()], 16);
    let backend = connector.backend_id();

    let m = connector
        .collect_metrics(&backend)
        .await
        .expect("metrics should come from /get_server_info");
    assert_eq!(m.active_requests, 5);
    assert_eq!(m.queue_depth, 2);
    assert_eq!(m.kv_used_blocks, 1000); // 16000 / 16
    assert_eq!(m.kv_total_blocks, 62500); // 1_000_000 / 16
}

// --------------------------------------------------------------------------
// Gateway-level harness (real HTTP server + AppState)
// --------------------------------------------------------------------------

/// Wire an AppState with one registered backend connector and, optionally, an
/// adaptive controller; returns the state plus the controller handle.
async fn build_gateway_state(
    backend_url: &str,
    instance: &str,
    adaptive: Option<AdaptiveConfig>,
    buffer: &DecisionEventBuffer,
) -> (AppState, Option<Arc<AdaptiveWeightController>>) {
    let metadata = Arc::new(MetadataStore::new());

    // Discover the backend through the connector registry path.
    let cfg = BackendConfig {
        backend_type: BackendType::SglangEngine,
        endpoint: Endpoint {
            url: backend_url.to_string(),
            protocol: Protocol::Http,
        },
        models: vec![MODEL.to_string()],
        region: RegionId::new(REGION),
        kv_block_size: 16,
        quantization: None,
    };
    let forwarding = ForwardingConfig { emit_token_ids: true };
    let registry = ConnectorRegistry::from_configs(&[cfg], &RegionId::new(REGION), &forwarding);
    let connector = registry
        .get(&BackendId::new(REGION, instance))
        .expect("connector registered");

    let infos = connector.discover().await.expect("discover succeeds");
    for info in infos {
        metadata.register_backend(info);
    }

    let weights = StrategyWeights {
        kv: 0.35,
        load: 0.30,
        topology: 0.20,
    };
    let mut hybrid = HybridStrategy::new(
        Box::new(KvAwareStrategy::default()),
        Box::new(ModelAwareStrategy::default()),
        Box::new(LoadAwareStrategy::default()),
        Box::new(TopologyAwareStrategy {
            w_rtt: 1.0,
            w_bw: 0.0,
            self_region: RegionId::new(REGION),
        }),
        weights.clone(),
        0.0,
    );
    let controller = adaptive.map(|cfg| Arc::new(AdaptiveWeightController::new(weights, cfg)));
    if let Some(ctl) = &controller {
        hybrid = hybrid.with_adaptive(ctl.clone());
    }
    let routing = Arc::new(RoutingEngine::new(
        hybrid,
        Duration::from_secs(300),
        2,
        RegionId::new(REGION),
    ));

    let routing_config = RoutingConfig {
        strategy: StrategyType::Hybrid,
        kv_block_size: 16,
        overlap_score_credit: 1.0,
        prefill_load_scale: 1.0,
        temperature: 0.0,
        session_affinity_ttl_secs: 300,
        max_retries: 2,
        weights: StrategyWeights {
            kv: 0.35,
            load: 0.30,
            topology: 0.20,
        },
        adaptive: AdaptiveConfig::default(),
    };

    let sink: Arc<dyn DecisionEventSink> = Arc::new(RingBufferSink::new(buffer.clone()));
    let state = AppState {
        metadata,
        routing,
        connectors: Arc::new(registry),
        routing_config,
        breakers: Arc::new(CircuitBreakerRegistry::new(&ResilienceConfig::default())),
        retry_policy: RetryPolicy::new(Duration::ZERO, Duration::ZERO),
        peer_registrar: None,
        decision_sink: sink,
        decision_buffer: Some(buffer.clone()),
        gateway_instance: "gw-test".to_string(),
        gateway_region: REGION.to_string(),
    };
    (state, controller)
}

/// Start the gateway HTTP server on an ephemeral port.
async fn start_gateway(state: AppState) -> String {
    let router = create_router(Arc::new(state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    format!("http://127.0.0.1:{port}")
}

/// POST a streaming chat completion and drain the SSE body.
async fn post_chat_completions(gateway_url: &str, token_ids: Vec<u32>) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{gateway_url}/v1/chat/completions"))
        .json(&json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "token_ids": token_ids,
            "max_tokens": 16,
            "stream": true
        }))
        .send()
        .await
        .expect("chat completions request")
}

async fn fetch_decision_events(gateway_url: &str) -> Vec<DecisionEvent> {
    let resp = reqwest::Client::new()
        .get(format!("{gateway_url}/admin/decision_events"))
        .send()
        .await
        .expect("decision events request");
    assert_eq!(resp.status(), 200);
    resp.json().await.expect("decision events must be JSON")
}

#[tokio::test]
async fn decision_event_emitted_through_http_stack() {
    let (backend_url, _flag) = start_stub_sglang_server().await;
    let instance = backend_url.replace("http://", "");
    let buffer = DecisionEventBuffer::new(16);
    let (state, _ctl) = build_gateway_state(&backend_url, &instance, None, &buffer).await;
    let gateway = start_gateway(state).await;

    // Prime the KV index is not required here; the request's block hashes
    // simply yield zero overlap, which the event must still report.
    let token_ids = vec![101u32, 102, 103, 104];

    let resp = post_chat_completions(&gateway, token_ids).await;
    let status = resp.status();
    // Drain the SSE stream so the handler completes before we read events.
    let body = resp.bytes().await.expect("drain response body");
    assert_eq!(status, 200, "response: {:?}", String::from_utf8_lossy(&body));

    // The ring buffer is written synchronously at the end of the handler.
    let events = fetch_decision_events(&gateway).await;
    assert_eq!(events.len(), 1, "exactly one event per request");
    let ev = &events[0];
    assert_eq!(ev.gateway_instance, "gw-test");
    assert_eq!(ev.gateway_region, REGION);
    assert_eq!(ev.model, MODEL);
    assert_eq!(ev.outcome, hier_kv_gateway_core::decision_event::DecisionOutcome::Success);
    assert_eq!(ev.selected_backend.as_deref(), Some(format!("{REGION}/{instance}").as_str()));
    assert_eq!(ev.attempts.len(), 1);
    assert!(ev.attempts[0].success);
    // Hybrid strategy ran → weight snapshot present, and candidates ranked.
    assert!(ev.weights.is_some());
    assert!(!ev.candidates.is_empty());
    assert!(ev.routing_latency_us > 0 || ev.total_latency_us > 0);
}

#[tokio::test]
async fn failover_event_records_all_attempts() {
    // First candidate is a dead port (connection refused), second is healthy.
    // Routing ranks both; the forwarding loop must fail over.
    let (backend_url, _flag) = start_stub_sglang_server().await;
    let live_instance = backend_url.replace("http://", "");
    let dead_url = "http://127.0.0.1:9"; // port 9 (discard) is closed
    let dead_instance = "127.0.0.1:9";

    let buffer = DecisionEventBuffer::new(16);
    let metadata = Arc::new(MetadataStore::new());
    let forwarding = ForwardingConfig { emit_token_ids: true };
    let configs = vec![
        BackendConfig {
            backend_type: BackendType::SglangEngine,
            endpoint: Endpoint { url: dead_url.to_string(), protocol: Protocol::Http },
            models: vec![MODEL.to_string()],
            region: RegionId::new(REGION),
            kv_block_size: 16,
            quantization: None,
        },
        BackendConfig {
            backend_type: BackendType::SglangEngine,
            endpoint: Endpoint { url: backend_url.clone(), protocol: Protocol::Http },
            models: vec![MODEL.to_string()],
            region: RegionId::new(REGION),
            kv_block_size: 16,
            quantization: None,
        },
    ];
    let registry = ConnectorRegistry::from_configs(&configs, &RegionId::new(REGION), &forwarding);

    // Register both backends into the metadata store (skip network discover:
    // register the discover-shape manually so the dead backend needs no server).
    for instance in [dead_instance, live_instance.as_str()] {
        let id = BackendId::new(REGION, instance);
        metadata.register_backend(hier_kv_gateway_core::backend::BackendInfo {
            id: id.clone(),
            backend_type: BackendType::SglangEngine,
            endpoint: Endpoint {
                url: format!("http://{instance}"),
                protocol: Protocol::Http,
            },
            models: vec![hier_kv_gateway_core::backend::ModelInstance {
                model_name: MODEL.to_string(),
                model_architecture: "qwen".to_string(),
                quantization: hier_kv_gateway_core::backend::Quantization::Fp16,
                max_context_len: 32768,
                supports_tool_calling: false,
                supports_streaming: true,
            }],
            region: RegionId::new(REGION),
            indexer_domain: hier_kv_gateway_core::ids::IndexerDomainId::new(0),
            capabilities: hier_kv_gateway_core::backend::BackendCapabilities {
                supports_kv_events: false,
                supports_batching: true,
                max_batch_size: 32,
                gpu_count: 1,
                gpu_memory_gb: 24,
            },
            kv_config: hier_kv_gateway_core::backend::KvConfig {
                block_size: 16,
                cache_namespace: "default".to_string(),
                max_kv_blocks: 1024,
            },
            status: hier_kv_gateway_core::backend::BackendStatus::Healthy,
        });
    }

    let weights = StrategyWeights { kv: 0.35, load: 0.30, topology: 0.20 };
    let hybrid = HybridStrategy::new(
        Box::new(KvAwareStrategy::default()),
        Box::new(ModelAwareStrategy::default()),
        Box::new(LoadAwareStrategy::default()),
        Box::new(TopologyAwareStrategy {
            w_rtt: 1.0,
            w_bw: 0.0,
            self_region: RegionId::new(REGION),
        }),
        weights,
        0.0,
    );
    let routing = Arc::new(RoutingEngine::new(hybrid, Duration::from_secs(300), 3, RegionId::new(REGION)));
    let routing_config = RoutingConfig {
        strategy: StrategyType::Hybrid,
        kv_block_size: 16,
        overlap_score_credit: 1.0,
        prefill_load_scale: 1.0,
        temperature: 0.0,
        session_affinity_ttl_secs: 300,
        max_retries: 3,
        weights: StrategyWeights { kv: 0.35, load: 0.30, topology: 0.20 },
        adaptive: AdaptiveConfig::default(),
    };
    let state = AppState {
        metadata,
        routing,
        connectors: Arc::new(registry),
        routing_config,
        breakers: Arc::new(CircuitBreakerRegistry::new(&ResilienceConfig::default())),
        retry_policy: RetryPolicy::new(Duration::ZERO, Duration::ZERO),
        peer_registrar: None,
        decision_sink: Arc::new(RingBufferSink::new(buffer.clone())),
        decision_buffer: Some(buffer.clone()),
        gateway_instance: "gw-test".to_string(),
        gateway_region: REGION.to_string(),
    };
    let gateway = start_gateway(state).await;

    let resp = post_chat_completions(&gateway, vec![1, 2, 3]).await;
    let status = resp.status();
    let body = resp.bytes().await.expect("drain response body");
    assert_eq!(status, 200, "failover must still succeed: {:?}", String::from_utf8_lossy(&body));

    let events = fetch_decision_events(&gateway).await;
    assert_eq!(events.len(), 1);
    let ev = &events[0];
    assert_eq!(ev.outcome, hier_kv_gateway_core::decision_event::DecisionOutcome::Success);
    assert!(ev.attempts.len() >= 2, "must record failed + winning attempts: {:?}", ev.attempts);
    let first = &ev.attempts[0];
    assert!(!first.success);
    assert!(first.error.is_some());
    let last = ev.attempts.last().unwrap();
    assert!(last.success);
    assert_eq!(last.backend, format!("{REGION}/{live_instance}"));
}

#[tokio::test]
async fn adaptive_controller_receives_forward_feedback() {
    let (backend_url, _flag) = start_stub_sglang_server().await;
    let instance = backend_url.replace("http://", "");
    let buffer = DecisionEventBuffer::new(16);
    let adaptive_cfg = AdaptiveConfig {
        enabled: true,
        ema_alpha: 0.5,
        max_adjustment: 0.25,
        min_weight: 0.05,
        adjust_interval_secs: 0,
    };
    let (state, ctl) = build_gateway_state(&backend_url, &instance, Some(adaptive_cfg), &buffer).await;
    let ctl = ctl.expect("adaptive controller attached");
    let gateway = start_gateway(state).await;

    // 32 token ids = 2 full KV blocks at block size 16, so the controller
    // receives a non-trivial hit ratio (empty prompts are skipped).
    let token_ids: Vec<u32> = (0..32).collect();
    let resp = post_chat_completions(&gateway, token_ids.clone()).await;
    let status = resp.status();
    let body = resp.bytes().await.expect("drain response body");
    assert_eq!(status, 200, "response: {:?}", String::from_utf8_lossy(&body));

    // One success sample with latency for the serving backend.
    let backend = BackendId::new(REGION, instance.as_str());
    let stats = ctl.outcome_stats(&backend).expect("outcome recorded");
    assert_eq!(stats.samples, 1);
    assert!((stats.ema_success - 1.0).abs() < 1e-9);
    assert!(stats.ema_latency_ms >= 0.0);

    // The winning request's KV hit ratio was fed back (32 token ids at
    // block size 16 → 2 prompt blocks).
    let expected_blocks =
        compute_block_hashes(&BlockHashInput {
            tokens: &token_ids,
            kv_block_size: 16,
            cache_namespace: None,
            lora_name: None,
        })
        .len() as u32;
    assert!(expected_blocks >= 1);
    let ratio = ctl.kv_hit_ratio().expect("kv hit ratio recorded");
    assert!((0.0..=1.0).contains(&ratio));
}
