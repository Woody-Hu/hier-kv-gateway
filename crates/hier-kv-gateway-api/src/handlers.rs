//! HTTP route handler functions.
//!
//! Each handler is an axum-compatible async function that takes inputs from [`AppState`]
//! and the HTTP request, invokes routing/connector to do the actual work, and returns a
//! JSON or SSE response.
//!
//! Routing decision information (selected backend, strategy name, KV overlap) is exposed
//! to the client via the response headers `X-Hier-KV-Gateway-Backend` /
//! `X-Hier-KV-Gateway-Strategy` / `X-Hier-KV-Gateway-KV-Overlap`, for troubleshooting and
//! observability.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::stream::{BoxStream, StreamExt};
use serde_json::json;
use tracing::{debug, error, warn};

use hier_kv_gateway_core::backend::BackendInfo;
use hier_kv_gateway_core::config::RoutingConfig;
use hier_kv_gateway_core::error::HierKvGatewayError;
use hier_kv_gateway_core::ids::{BackendId, RequestId, SessionId};
use hier_kv_gateway_core::kv_event::{compute_block_hashes, BlockHashInput};
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::request::{InferenceChunk, InferenceRequest, RoutingContext};

use hier_kv_gateway_connector::registry::ConnectorRegistry;
use hier_kv_gateway_metadata::store::MetadataStore;
use hier_kv_gateway_routing::engine::{RouteDecision, RoutingEngine};

use crate::openai_types::{
    OpenAIChatChunk, OpenAIChatRequest, OpenAIChatResponse, OpenAIModelList,
};

/// Application state shared by HTTP handlers.
pub struct AppState {
    /// Metadata store (KV index, model registry, load statistics, etc.).
    pub metadata: Arc<MetadataStore>,
    /// Routing engine.
    pub routing: Arc<RoutingEngine>,
    /// Connector registry, indexed by BackendType.
    pub connectors: Arc<ConnectorRegistry>,
    /// Routing configuration (provides parameters such as kv_block_size).
    pub routing_config: RoutingConfig,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("backends", &self.metadata.backends_len())
            .field("routing_strategy", &self.routing_config.strategy)
            .finish()
    }
}

/// Metadata derived from the routing decision that needs to be carried in the response.
struct RoutingMeta {
    /// Selected backend identifier string.
    backend: String,
    /// Name of the strategy that triggered the decision.
    strategy: String,
    /// KV overlap length.
    kv_overlap: u32,
}

impl RoutingMeta {
    fn from_decision(d: &RouteDecision) -> Self {
        Self {
            backend: d.backend.to_string(),
            strategy: d.strategy.clone(),
            kv_overlap: d.kv_overlap,
        }
    }

    /// Write this metadata into response headers.
    fn apply_to_headers(&self, headers: &mut HeaderMap) {
        if let Ok(v) = HeaderValue::from_str(&self.backend) {
            headers.insert("X-Hier-KV-Gateway-Backend", v);
        }
        if let Ok(v) = HeaderValue::from_str(&self.strategy) {
            headers.insert("X-Hier-KV-Gateway-Strategy", v);
        }
        let overlap = self.kv_overlap.to_string();
        if let Ok(v) = HeaderValue::from_str(&overlap) {
            headers.insert("X-Hier-KV-Gateway-KV-Overlap", v);
        }
    }
}

/// `POST /v1/chat/completions`
///
/// Handles OpenAI-compatible Chat Completions requests, returning a streaming SSE or
/// non-streaming JSON response based on the `stream` field. Response headers carry the
/// routing decision information.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAIChatRequest>,
) -> Response {
    let stream_mode = req.stream;
    let session_id = req.session.as_ref().map(SessionId::new);
    let model_name = req.model.clone();

    // 1) Convert to internal InferenceRequest
    let inference: InferenceRequest = req.to_inference_request();
    let request_id = inference.request_id.clone();

    // 2) Build the RoutingContext; compute block_hashes when needed
    let block_hashes = if !inference.token_ids.is_empty() {
        compute_block_hashes(&BlockHashInput {
            tokens: &inference.token_ids,
            kv_block_size: state.routing_config.kv_block_size,
            cache_namespace: None,
            lora_name: inference.lora_name.as_deref(),
        })
    } else {
        Vec::new()
    };
    let ctx = RoutingContext {
        request_id: Some(request_id.clone()),
        session_id,
        model_name: Some(model_name.clone()),
        token_ids: inference.token_ids.clone(),
        block_hashes,
        block_size: state.routing_config.kv_block_size,
        lora_name: inference.lora_name.clone(),
        cache_namespace: None,
        estimated_output_tokens: inference.max_tokens,
        requires_tool_calling: !inference.tools.is_empty(),
    };

    // 3) Routing decision
    let decision = match state.routing.route(&ctx, &state.metadata).await {
        Ok(d) => d,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "routing failed");
            return error_response(StatusCode::SERVICE_UNAVAILABLE, &e);
        }
    };
    let routing_meta = RoutingMeta::from_decision(&decision);
    debug!(
        request_id = %request_id,
        backend = %routing_meta.backend,
        strategy = %routing_meta.strategy,
        kv_overlap = routing_meta.kv_overlap,
        "routing decision completed"
    );

    // 4) Take out the connector
    let backend_info = match state.metadata.backend_get(&decision.backend) {
        Some(info) => info,
        None => {
            error!(backend = %decision.backend, "backend selected by routing is not in MetadataStore");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &HierKvGatewayError::NotFound(format!("backend {} not registered", decision.backend)),
            );
        }
    };
    let backend_type = backend_info.backend_type.clone();
    let connector = match state.connectors.get(&backend_type) {
        Some(c) => c,
        None => {
            error!(backend_type = ?backend_type, "no connector registered for this backend type");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &HierKvGatewayError::ConnectorError(format!(
                    "no connector for backend_type {:?}",
                    backend_type
                )),
            );
        }
    };

    // 5) Forward the request and obtain the chunk stream
    let chunk_stream: BoxStream<'static, InferenceChunk> =
        match connector.forward(&decision.backend, &inference).await {
            Ok(s) => s,
            Err(e) => {
                error!(request_id = %request_id, error = %e, "backend forwarding failed");
                return error_response(StatusCode::BAD_GATEWAY, &e);
            }
        };

    // 6) Build the response based on the stream field
    if stream_mode {
        build_sse_response(chunk_stream, &request_id, &model_name, &routing_meta)
    } else {
        build_non_stream_response(
            chunk_stream,
            &request_id,
            &model_name,
            &routing_meta,
            &inference,
        )
        .await
    }
}

/// Convert a [`HierKvGatewayError`] into an HTTP JSON error response.
fn error_response(status: StatusCode, err: &HierKvGatewayError) -> Response {
    let body = Json(json!({
        "error": {
            "message": err.to_string(),
            "type": error_type_name(err),
        }
    }));
    (status, body).into_response()
}

/// Returns a short classification name for HierKvGatewayError, allowing clients to
/// distinguish error types.
fn error_type_name(err: &HierKvGatewayError) -> &'static str {
    match err {
        HierKvGatewayError::BackendUnavailable => "backend_unavailable",
        HierKvGatewayError::RoutingFailed(_) => "routing_failed",
        HierKvGatewayError::ConnectorError(_) => "connector_error",
        HierKvGatewayError::MetricsError(_) => "metrics_error",
        HierKvGatewayError::ConfigError(_) => "config_error",
        HierKvGatewayError::ClusterError(_) => "cluster_error",
        HierKvGatewayError::NotFound(_) => "not_found",
        HierKvGatewayError::RateLimited => "rate_limited",
        HierKvGatewayError::Internal(_) => "internal_error",
    }
}

/// Build a streaming SSE response.
///
/// The output format follows OpenAI conventions:
/// - The start chunk carries `role: "assistant"`;
/// - Each subsequent text increment is sent as a `data: {json}\n\n` event;
/// - The final chunk of the stream carries `finish_reason`;
/// - Finally `data: [DONE]\n\n` is sent.
fn build_sse_response(
    chunk_stream: BoxStream<'static, InferenceChunk>,
    request_id: &RequestId,
    model: &str,
    routing_meta: &RoutingMeta,
) -> Response {
    // Each chunk shares the same (rid, model) copy, so clone them in advance.
    let rid_for_first = request_id.as_str().to_string();
    let model_for_first = model.to_string();
    let rid_for_deltas = rid_for_first.clone();
    let model_for_deltas = model_for_first.clone();
    let rid_for_finish = rid_for_first.clone();
    let model_for_finish = model_for_first.clone();

    // Start chunk
    let first = OpenAIChatChunk::role_chunk(&rid_for_first, &model_for_first);

    let sse_stream = futures::stream::once(async move { first })
        .chain(
            chunk_stream.map(move |chunk| {
                chunk_to_openai(&rid_for_deltas, &model_for_deltas, chunk)
            }),
        )
        .chain(futures::stream::once(async move {
            OpenAIChatChunk::finish_chunk(&rid_for_finish, &model_for_finish, "stop")
        }))
        .map(|c| {
            let json = serde_json::to_string(&c).unwrap_or_else(|_| "{}".to_string());
            Ok::<Event, std::convert::Infallible>(Event::default().data(json))
        })
        .chain(futures::stream::once(async move {
            Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
        }));

    let mut headers = HeaderMap::new();
    routing_meta.apply_to_headers(&mut headers);

    let sse = Sse::new(sse_stream);
    (headers, sse).into_response()
}

/// Convert an [`InferenceChunk`] into an [`OpenAIChatChunk`].
///
/// - `Delta`: returns `delta_chunk` when carrying text; returns `finish_chunk` when
///   carrying `finish_reason`; also returns `finish_chunk` when text is empty but
///   `finish_reason` is present;
/// - `ToolCall`: currently simplified to be ignored (OpenAIChatChunk does not yet carry
///   tool_calls delta); only logged;
/// - `Done`: returns `finish_chunk("stop")` as the end-of-stream signal (the caller's
///   chain tail then appends `[DONE]`);
/// - `Error`: returns `finish_chunk("error")` and logs the error.
fn chunk_to_openai(rid: &str, model: &str, chunk: InferenceChunk) -> OpenAIChatChunk {
    match chunk {
        InferenceChunk::Delta {
            text,
            finish_reason,
        } => {
            if let Some(reason) = finish_reason {
                if text.is_empty() {
                    OpenAIChatChunk::finish_chunk(rid, model, &reason)
                } else {
                    // When both text and finish_reason are present, send delta first, then finish.
                    // Since one chunk can only produce one OpenAIChatChunk, finish is prioritized here.
                    OpenAIChatChunk::finish_chunk(rid, model, &reason)
                }
            } else if !text.is_empty() {
                OpenAIChatChunk::delta_chunk(rid, model, text)
            } else {
                // Empty content with no finish_reason: return an empty delta chunk to keep the stream alive
                OpenAIChatChunk::delta_chunk(rid, model, String::new())
            }
        }
        InferenceChunk::ToolCall {
            id: _,
            function,
            args: _,
        } => {
            warn!(function = %function, "ToolCall chunk is currently not emitted in SSE");
            OpenAIChatChunk::delta_chunk(rid, model, String::new())
        }
        InferenceChunk::Done {
            backend_id: _,
            latency_ms: _,
        } => OpenAIChatChunk::finish_chunk(rid, model, "stop"),
        InferenceChunk::Error { code, message } => {
            warn!(code, %message, "backend returned an error chunk");
            OpenAIChatChunk::finish_chunk(rid, model, "error")
        }
    }
}

/// Build a non-streaming response: consume the entire chunk stream, merge all text, and
/// return the complete JSON.
async fn build_non_stream_response(
    chunk_stream: BoxStream<'static, InferenceChunk>,
    request_id: &RequestId,
    model: &str,
    routing_meta: &RoutingMeta,
    inference: &InferenceRequest,
) -> Response {
    let mut content = String::new();
    let mut finish_reason: Option<String> = None;
    let mut completion_tokens: u64 = 0;
    let mut error_chunk: Option<(u16, String)> = None;

    let mut stream = chunk_stream;
    while let Some(chunk) = stream.next().await {
        match chunk {
            InferenceChunk::Delta {
                text,
                finish_reason: fr,
            } => {
                if !text.is_empty() {
                    content.push_str(&text);
                    // Rough estimate: count tokens by character count for incremental text (fallback when no tokenizer is available)
                    completion_tokens += approx_token_count(&text);
                }
                if let Some(reason) = fr {
                    finish_reason = Some(reason);
                }
            }
            InferenceChunk::ToolCall { .. } => {
                // tool_calls are not processed during non-streaming aggregation
            }
            InferenceChunk::Done { .. } => {
                if finish_reason.is_none() {
                    finish_reason = Some("stop".to_string());
                }
                break;
            }
            InferenceChunk::Error { code, message } => {
                error_chunk = Some((code, message));
                break;
            }
        }
    }

    if let Some((code, message)) = error_chunk {
        let status = StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY);
        let err = HierKvGatewayError::ConnectorError(format!("backend error: {}", message));
        return error_response(status, &err);
    }

    // Estimate prompt_tokens: use the rough ratio of total message characters / 4
    let prompt_chars: usize = inference
        .messages
        .iter()
        .map(|m| m.content.chars().count())
        .sum();
    let prompt_tokens = (prompt_chars as f64 / 4.0).ceil() as u64;

    let resp = OpenAIChatResponse::from_text(
        request_id.as_str(),
        model,
        content,
        finish_reason,
        prompt_tokens,
        completion_tokens,
    );

    let mut headers = HeaderMap::new();
    routing_meta.apply_to_headers(&mut headers);
    (headers, Json(resp)).into_response()
}

/// Use a rough character count -> token count estimate.
///
/// For CJK characters, 1 char ≈ 1 token; for others, 4 chars ≈ 1 token (mixed approximation).
fn approx_token_count(text: &str) -> u64 {
    let mut cjk = 0u64;
    let mut other = 0u64;
    for ch in text.chars() {
        if (ch as u32) >= 0x4E00 && (ch as u32) <= 0x9FFF {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    cjk + (other as f64 / 4.0).ceil() as u64
}

/// `GET /v1/models`
///
/// Lists all models served by registered backends, de-duplicated by
/// `(model_name, backend_id)`.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<OpenAIModelList> {
    let backends: Vec<BackendInfo> = state.metadata.backends_all();
    // Use a HashSet to de-duplicate model names, keeping the first backend that serves the model as owned_by.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut entries: Vec<(String, Option<String>)> = Vec::new();
    for b in &backends {
        for m in &b.models {
            if seen.insert(m.model_name.clone()) {
                entries.push((m.model_name.clone(), Some(b.id.to_string())));
            }
        }
    }
    Json(OpenAIModelList::from_model_ids(entries))
}

/// `GET /health`
///
/// Simple health check; returns `{"status":"ok"}`.
pub async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

/// `GET /admin/backends`
///
/// Returns the [`BackendInfo`] list of all registered backends.
pub async fn admin_backends(State(state): State<Arc<AppState>>) -> Json<Vec<BackendInfo>> {
    Json(state.metadata.backends_all())
}

/// `GET /admin/backends/:id/metrics`
///
/// Queries the load metrics of the specified backend. The `id` path parameter is in the
/// format `region/instance`.
pub async fn admin_metrics(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    let backend_id = match parse_backend_id(&id) {
        Some(bid) => bid,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("invalid backend id: {}, expected '<region>/<instance>'", id),
                        "type": "invalid_argument"
                    }
                })),
            )
                .into_response();
        }
    };

    let metrics: Option<BackendMetrics> = state.metadata.load_get_metrics(&backend_id);
    match metrics {
        Some(m) => Json(m).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("no metrics for backend {}", id),
                    "type": "not_found"
                }
            })),
        )
            .into_response(),
    }
}

/// Parse a `region/instance` formatted string into a [`BackendId`].
///
/// Splits only on the first `/`; the `instance` portion may contain subsequent `/`, but
/// usually does not.
fn parse_backend_id(s: &str) -> Option<BackendId> {
    let slash = s.find('/')?;
    let region = &s[..slash];
    let instance = &s[slash + 1..];
    if region.is_empty() || instance.is_empty() {
        return None;
    }
    Some(BackendId::new(region, instance))
}

/// Used to construct a minimal AppState in tests.
///
/// Only compiled under `cfg(test)`, to avoid exposing unnecessary dependencies in
/// production code.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use hier_kv_gateway_core::config::{StrategyType, StrategyWeights};
    use hier_kv_gateway_routing::hybrid::HybridStrategy;
    use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
    use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
    use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
    use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;
    use std::time::Duration;

    /// Build an AppState using the default hybrid strategy, only for unit tests.
    pub fn build_test_app_state(self_region: &str) -> Arc<AppState> {
        let metadata = Arc::new(MetadataStore::new());
        let routing_config = RoutingConfig {
            strategy: StrategyType::Hybrid,
            kv_block_size: 16,
            overlap_score_credit: 1.0,
            prefill_load_scale: 1.0,
            temperature: 0.0,
            session_affinity_ttl_secs: 300,
            max_retries: 3,
            weights: StrategyWeights {
                kv: 0.35,
                load: 0.30,
                topology: 0.20,
            },
        };
        let hybrid = HybridStrategy::new(
            Box::new(KvAwareStrategy::default()),
            Box::new(ModelAwareStrategy::default()),
            Box::new(LoadAwareStrategy::default()),
            Box::new(TopologyAwareStrategy {
                w_rtt: 1.0,
                w_bw: 0.0,
                self_region: self_region.into(),
            }),
            routing_config.weights.clone(),
            routing_config.temperature,
        );
        let routing = Arc::new(RoutingEngine::new(
            hybrid,
            Duration::from_secs(routing_config.session_affinity_ttl_secs),
            routing_config.max_retries,
            self_region.into(),
        ));
        let connectors = Arc::new(ConnectorRegistry::new());
        Arc::new(AppState {
            metadata,
            routing,
            connectors,
            routing_config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_id_simple() {
        let bid = parse_backend_id("us-east-1/worker-0").unwrap();
        assert_eq!(bid.region.as_str(), "us-east-1");
        assert_eq!(bid.instance.as_str(), "worker-0");
    }

    #[test]
    fn parse_backend_id_rejects_empty_parts() {
        assert!(parse_backend_id("/worker-0").is_none());
        assert!(parse_backend_id("us-east-1/").is_none());
        assert!(parse_backend_id("us-east-1").is_none());
    }

    #[test]
    fn approx_token_count_cjk_and_ascii() {
        assert!(approx_token_count("你好世界") >= 4);
        // "hello" 5 chars -> 2 tokens
        assert_eq!(approx_token_count("hello"), 2);
    }

    #[test]
    fn routing_meta_writes_headers() {
        let decision = RouteDecision {
            backend: BackendId::new("r1", "i1"),
            strategy: "hybrid".to_string(),
            kv_overlap: 7,
            scores: Vec::new(),
        };
        let meta = RoutingMeta::from_decision(&decision);
        let mut headers = HeaderMap::new();
        meta.apply_to_headers(&mut headers);
        assert_eq!(headers.get("X-Hier-KV-Gateway-Backend").unwrap(), "r1/i1");
        assert_eq!(headers.get("X-Hier-KV-Gateway-Strategy").unwrap(), "hybrid");
        assert_eq!(headers.get("X-Hier-KV-Gateway-KV-Overlap").unwrap(), "7");
    }

    #[test]
    fn chunk_to_openai_delta_text() {
        let chunk = InferenceChunk::Delta {
            text: "hi".to_string(),
            finish_reason: None,
        };
        let out = chunk_to_openai("rid", "m", chunk);
        assert_eq!(out.choices[0].delta.content.as_deref(), Some("hi"));
        assert!(out.choices[0].finish_reason.is_none());
    }

    #[test]
    fn chunk_to_openai_delta_finish() {
        let chunk = InferenceChunk::Delta {
            text: String::new(),
            finish_reason: Some("stop".to_string()),
        };
        let out = chunk_to_openai("rid", "m", chunk);
        assert_eq!(out.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn chunk_to_openai_done() {
        let chunk = InferenceChunk::Done {
            backend_id: BackendId::new("r1", "i1"),
            latency_ms: 42,
        };
        let out = chunk_to_openai("rid", "m", chunk);
        assert_eq!(out.choices[0].finish_reason.as_deref(), Some("stop"));
    }
}
