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

use async_trait::async_trait;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, error, warn};

use hier_kv_gateway_core::backend::BackendInfo;
use hier_kv_gateway_core::config::RoutingConfig;
use hier_kv_gateway_core::decision_event::{
    CandidateScore, DecisionEvent, DecisionEventSink, DecisionOutcome, ForwardAttempt,
    WeightSnapshot,
};
use hier_kv_gateway_core::error::HierKvGatewayError;
use hier_kv_gateway_core::ids::{BackendId, RequestId, SessionId};
use hier_kv_gateway_core::kv_event::{compute_block_hashes, BlockHashInput};
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::request::{InferenceChunk, InferenceRequest, RoutingContext};

use hier_kv_gateway_connector::registry::ConnectorRegistry;
use hier_kv_gateway_connector::resilience::{CircuitBreakerRegistry, RetryPolicy};
use hier_kv_gateway_metadata::store::MetadataStore;
use hier_kv_gateway_routing::engine::{RouteDecision, RoutingEngine};
use hier_kv_gateway_routing::strategy::RoutingStrategy;

use crate::openai_types::{
    OpenAIChatChunk, OpenAIChatRequest, OpenAIChatResponse, OpenAIModelList,
};
use crate::telemetry::DecisionEventBuffer;

/// Optional trait allowing the HTTP layer to dynamically register a new peer
/// (typically an external-Region gateway) into the running gossip mesh.
///
/// Defined here (rather than in `hier-kv-gateway-cluster`) so the API crate
/// stays decoupled from the cluster crate; the main binary wires up a concrete
/// implementation backed by [`GossipEngine::meet_peer`].
#[async_trait]
pub trait PeerRegistrar: Send + Sync {
    /// Send a `Meet` to `peer_addr` (host:port). Returns `Ok(())` on success
    /// or an error message describing why the registration failed.
    async fn meet_peer(&self, peer_addr: &str) -> std::result::Result<(), String>;
}

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
    /// Per-backend circuit breakers consulted by the forwarding loop: a
    /// backend with an open circuit is skipped instead of retried.
    pub breakers: Arc<CircuitBreakerRegistry>,
    /// Exponential backoff policy applied between two forward attempts.
    pub retry_policy: RetryPolicy,
    /// Optional peer registrar (backed by `GossipEngine` when cluster mode is enabled).
    ///
    /// When `None`, the `POST /cluster/peers` endpoint returns `503 Service Unavailable`,
    /// indicating the gateway was started without a cluster transport.
    pub peer_registrar: Option<Arc<dyn PeerRegistrar>>,
    /// Decision telemetry sink: one [`DecisionEvent`] is emitted per request
    /// (fan-out to the admin ring buffer, tracing, and/or NDJSON file).
    pub decision_sink: Arc<dyn DecisionEventSink>,
    /// In-memory decision-event ring buffer backing
    /// `GET /admin/decision_events`; `None` when `telemetry.buffer_size = 0`.
    pub decision_buffer: Option<DecisionEventBuffer>,
    /// Gateway instance identifier stamped onto every decision event.
    pub gateway_instance: String,
    /// Gateway region stamped onto every decision event.
    pub gateway_region: String,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("backends", &self.metadata.backends_len())
            .field("routing_strategy", &self.routing_config.strategy)
            .field("cluster_enabled", &self.peer_registrar.is_some())
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
///
/// Exactly one [`DecisionEvent`] is emitted per request — on routing failure,
/// forwarding exhaustion, or success — via [`AppState::decision_sink`].
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAIChatRequest>,
) -> Response {
    let total_start = std::time::Instant::now();
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
    let prompt_blocks = block_hashes.len() as u32;
    let ctx = RoutingContext {
        request_id: Some(request_id.clone()),
        session_id: session_id.clone(),
        tenant_id: None,
        model_name: Some(model_name.clone()),
        token_ids: inference.token_ids.clone(),
        block_hashes,
        block_size: state.routing_config.kv_block_size,
        lora_name: inference.lora_name.clone(),
        cache_namespace: None,
        estimated_output_tokens: inference.max_tokens,
        requires_tool_calling: !inference.tools.is_empty(),
    };

    // 3) Routing decision: fetch up to `1 + max_retries` ranked candidates so the
    //    forwarding loop can fail over to the next-best backend on errors.
    let max_attempts = state.routing_config.max_retries.saturating_add(1) as usize;
    let route_start = std::time::Instant::now();
    let routed = state
        .routing
        .route_candidates(&ctx, &state.metadata, max_attempts)
        .await;
    let routing_latency_us = route_start.elapsed().as_micros() as u64;
    let decisions = match routed {
        Ok(d) if !d.is_empty() => d,
        Ok(_) => {
            let e = HierKvGatewayError::BackendUnavailable;
            error!(request_id = %request_id, error = %e, "routing failed");
            emit_decision_event(DecisionEventParams {
                state: &state,
                request_id: &request_id,
                model: &model_name,
                session_id: session_id.as_ref(),
                decisions: &[],
                attempts: Vec::new(),
                selected: None,
                prompt_blocks,
                routing_latency_us,
                total_start,
                outcome: DecisionOutcome::RoutingFailed,
            });
            return error_response(StatusCode::SERVICE_UNAVAILABLE, &e);
        }
        Err(e) => {
            error!(request_id = %request_id, error = %e, "routing failed");
            emit_decision_event(DecisionEventParams {
                state: &state,
                request_id: &request_id,
                model: &model_name,
                session_id: session_id.as_ref(),
                decisions: &[],
                attempts: Vec::new(),
                selected: None,
                prompt_blocks,
                routing_latency_us,
                total_start,
                outcome: DecisionOutcome::RoutingFailed,
            });
            return error_response(StatusCode::SERVICE_UNAVAILABLE, &e);
        }
    };

    // 4) Forward along the ranked candidates with circuit-breaker gating and
    //    exponential backoff between attempts. The first candidate that yields
    //    a chunk stream wins; its decision metadata is reported to the client.
    let mut attempts: Vec<ForwardAttempt> = Vec::with_capacity(decisions.len());
    match forward_with_retries(&state, &decisions, &inference, &request_id, prompt_blocks, &mut attempts).await {
        Ok((chunk_stream, routing_meta)) => {
            debug!(
                request_id = %request_id,
                backend = %routing_meta.backend,
                strategy = %routing_meta.strategy,
                kv_overlap = routing_meta.kv_overlap,
                "routing decision completed"
            );
            emit_decision_event(DecisionEventParams {
                state: &state,
                request_id: &request_id,
                model: &model_name,
                session_id: session_id.as_ref(),
                decisions: &decisions,
                attempts,
                selected: Some(&routing_meta),
                prompt_blocks,
                routing_latency_us,
                total_start,
                outcome: DecisionOutcome::Success,
            });
            // 5) Build the response based on the stream field
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
        Err(e) => {
            error!(request_id = %request_id, error = %e, "backend forwarding failed");
            emit_decision_event(DecisionEventParams {
                state: &state,
                request_id: &request_id,
                model: &model_name,
                session_id: session_id.as_ref(),
                decisions: &decisions,
                attempts,
                selected: None,
                prompt_blocks,
                routing_latency_us,
                total_start,
                outcome: DecisionOutcome::AllCandidatesFailed,
            });
            error_response(StatusCode::BAD_GATEWAY, &e)
        }
    }
}

/// Inputs for building one [`DecisionEvent`]; see [`emit_decision_event`].
struct DecisionEventParams<'a> {
    state: &'a AppState,
    request_id: &'a RequestId,
    model: &'a str,
    session_id: Option<&'a SessionId>,
    decisions: &'a [RouteDecision],
    attempts: Vec<ForwardAttempt>,
    selected: Option<&'a RoutingMeta>,
    prompt_blocks: u32,
    routing_latency_us: u64,
    total_start: std::time::Instant,
    outcome: DecisionOutcome,
}

/// Build one decision event from the request lifecycle and emit it through
/// the configured sink. Never fails: telemetry must not affect the request.
fn emit_decision_event(p: DecisionEventParams) {
    let candidates: Vec<CandidateScore> = p
        .decisions
        .iter()
        .map(|d| CandidateScore {
            backend: d.backend.to_string(),
            // The engine pushes the selecting strategy's final score last.
            score: d.scores.last().map(|(_, s)| *s).unwrap_or(0.0),
            kv_overlap: d.kv_overlap,
        })
        .collect();

    // Attach the effective hybrid weights only when the hybrid strategy made
    // the winning decision (not for round_robin / affinity / degradation).
    let hybrid_name = p.state.routing.hybrid.name();
    let won_by_hybrid = p
        .selected
        .map(|m| m.strategy == hybrid_name)
        .unwrap_or(false)
        || p
            .decisions
            .first()
            .map(|d| d.strategy == hybrid_name)
            .unwrap_or(false);
    let weights = if won_by_hybrid {
        let w = p.state.routing.weight_snapshot();
        Some(WeightSnapshot {
            kv: w.kv,
            load: w.load,
            topology: w.topology,
        })
    } else {
        None
    };

    let event = DecisionEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        timestamp_unix_ms: chrono::Utc::now().timestamp_millis(),
        gateway_instance: p.state.gateway_instance.clone(),
        gateway_region: p.state.gateway_region.clone(),
        request_id: p.request_id.as_str().to_string(),
        model: p.model.to_string(),
        session_id: p.session_id.map(|s| s.as_str().to_string()),
        strategy: p
            .decisions
            .first()
            .map(|d| d.strategy.clone())
            .unwrap_or_else(|| "none".to_string()),
        weights,
        candidates,
        attempts: p.attempts,
        selected_backend: p.selected.map(|m| m.backend.clone()),
        kv_overlap: p.selected.map(|m| m.kv_overlap).unwrap_or(0),
        prompt_blocks: p.prompt_blocks,
        routing_latency_us: p.routing_latency_us,
        total_latency_us: p.total_start.elapsed().as_micros() as u64,
        outcome: p.outcome,
    };
    p.state.decision_sink.emit(&event);
}

/// Attempt to forward `inference` to the ranked `decisions` in order.
///
/// For each candidate:
/// 1. Skip backends whose circuit is currently open ([`CircuitBreakerRegistry::allow`]).
/// 2. Look up the connector for the candidate's backend type (a missing
///    connector counts as a candidate-local failure, not a global one).
/// 3. Forward; on success record `on_success` and return the stream together
///    with the routing metadata of the winning candidate.
/// 4. On failure record `on_failure`, sleep `retry_policy.backoff(failures)`
///    and move on to the next candidate.
///
/// Every considered candidate (including circuit-skipped ones) appends one
/// [`ForwardAttempt`] to `attempts` for the decision event. When an adaptive
/// weight controller is attached to the routing engine, per-attempt
/// success/failure + latency and the winning request's KV hit ratio are fed
/// back into it.
///
/// Returns the last forwarding error when every allowed candidate failed, or
/// [`HierKvGatewayError::BackendUnavailable`] when every candidate was
/// short-circuited.
async fn forward_with_retries(
    state: &Arc<AppState>,
    decisions: &[RouteDecision],
    inference: &InferenceRequest,
    request_id: &RequestId,
    prompt_blocks: u32,
    attempts: &mut Vec<ForwardAttempt>,
) -> std::result::Result<(BoxStream<'static, InferenceChunk>, RoutingMeta), HierKvGatewayError> {
    let mut failures: u32 = 0;
    let mut last_err: Option<HierKvGatewayError> = None;
    let adaptive = state.routing.adaptive_controller().cloned();

    for decision in decisions {
        // 1. Circuit-breaker gate.
        if !state.breakers.allow(&decision.backend) {
            debug!(
                request_id = %request_id,
                backend = %decision.backend,
                "skipping candidate with open circuit"
            );
            attempts.push(ForwardAttempt {
                backend: decision.backend.to_string(),
                success: false,
                skipped_open_circuit: true,
                error: None,
            });
            continue;
        }

        // Backoff between attempts: after failure #k the next attempt sleeps
        // `backoff(k - 1)` — with the defaults (50 ms base) the first retry
        // waits 50 ms, then 100, 200, ...
        if failures > 0 {
            tokio::time::sleep(state.retry_policy.backoff(failures - 1)).await;
        }

        // 2. Resolve the connector anchored to this candidate backend.
        let Some(connector) = state.connectors.get(&decision.backend) else {
            warn!(backend = %decision.backend, "no connector registered for this backend");
            failures += 1;
            let e = HierKvGatewayError::ConnectorError(format!(
                "no connector for backend {}",
                decision.backend
            ));
            attempts.push(ForwardAttempt {
                backend: decision.backend.to_string(),
                success: false,
                skipped_open_circuit: false,
                error: Some(e.to_string()),
            });
            if let Some(ctl) = adaptive.as_ref() {
                ctl.record_failure(&decision.backend);
            }
            last_err = Some(e);
            continue;
        };

        // 3. Attempt the forward.
        let attempt_start = std::time::Instant::now();
        match connector.forward(&decision.backend, inference).await {
            Ok(stream) => {
                state.breakers.on_success(&decision.backend);
                if let Some(ctl) = adaptive.as_ref() {
                    ctl.record_success(&decision.backend, attempt_start.elapsed());
                    ctl.record_kv_overlap(decision.kv_overlap, prompt_blocks);
                }
                attempts.push(ForwardAttempt {
                    backend: decision.backend.to_string(),
                    success: true,
                    skipped_open_circuit: false,
                    error: None,
                });
                if failures > 0 {
                    debug!(
                        request_id = %request_id,
                        backend = %decision.backend,
                        attempt = failures + 1,
                        "forward succeeded after retry"
                    );
                }
                let routing_meta = RoutingMeta::from_decision(decision);
                return Ok((stream, routing_meta));
            }
            Err(e) => {
                state.breakers.on_failure(&decision.backend);
                if let Some(ctl) = adaptive.as_ref() {
                    ctl.record_failure(&decision.backend);
                }
                warn!(
                    request_id = %request_id,
                    backend = %decision.backend,
                    attempt = failures + 1,
                    error = %e,
                    "forward attempt failed"
                );
                attempts.push(ForwardAttempt {
                    backend: decision.backend.to_string(),
                    success: false,
                    skipped_open_circuit: false,
                    error: Some(e.to_string()),
                });
                failures += 1;
                last_err = Some(e);
            }
        }
    }

    match last_err {
        Some(e) => Err(e),
        None => {
            // Every candidate was short-circuited by the breaker.
            Err(HierKvGatewayError::BackendUnavailable)
        }
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
    let backend_id = match BackendId::parse(&id) {
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

/// Query parameters for `GET /admin/decision_events`.
#[derive(Clone, Debug, Deserialize)]
pub struct DecisionEventsQuery {
    /// Return at most the newest `limit` buffered events; absent/0 returns
    /// everything currently buffered.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /admin/decision_events?limit=N`
///
/// Returns the in-memory ring buffer of recent routing decision events
/// (newest last). External analysis systems can poll this endpoint for
/// low-volume introspection; high-volume pipelines should consume the
/// tracing/NDJSON sinks configured via `[telemetry]` instead.
///
/// Returns `404` when the buffer is disabled (`telemetry.buffer_size = 0`).
pub async fn admin_decision_events(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<DecisionEventsQuery>,
) -> Response {
    match &state.decision_buffer {
        Some(buf) => Json(buf.snapshot(params.limit.unwrap_or(0))).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": "decision event buffer is disabled (telemetry.buffer_size = 0)",
                    "type": "not_found"
                }
            })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Cluster peer management
// ---------------------------------------------------------------------------

/// Request body for `POST /cluster/peers`.
///
/// Used to dynamically register an external-Region gateway into the running
/// gossip mesh after startup. The local gateway sends a `Meet` message to
/// `peer_addr`; the remote gateway replies with a `Pong`, and the standard
/// gossip loop then propagates the new member to the rest of the cluster.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterPeersRequest {
    /// Address of the peer gateway to register, in `host:port` form (matching
    /// the cluster transport's bind address).
    pub peer_addr: String,
}

/// Response body for `POST /cluster/peers`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterPeersResponse {
    /// Whether the Meet message was sent successfully.
    pub ok: bool,
    /// Human-readable status / error message.
    pub message: String,
}

/// `POST /cluster/peers` — dynamically register an external-Region gateway.
///
/// Returns:
/// - `200 OK` with `ok: true` when the Meet was sent successfully.
/// - `200 OK` with `ok: false` and an error message when the transport
///   rejected the send (e.g. peer unreachable).
/// - `503 Service Unavailable` when the gateway was started without a cluster
///   transport (no `PeerRegistrar` wired up).
pub async fn cluster_peers(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ClusterPeersRequest>,
) -> Response {
    let Some(registrar) = state.peer_registrar.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ClusterPeersResponse {
                ok: false,
                message: "cluster transport is not enabled on this gateway".to_string(),
            }),
        )
            .into_response();
    };

    match registrar.meet_peer(&req.peer_addr).await {
        Ok(()) => Json(ClusterPeersResponse {
            ok: true,
            message: format!("Meet sent to {}", req.peer_addr),
        })
        .into_response(),
        Err(e) => Json(ClusterPeersResponse {
            ok: false,
            message: format!("Failed to send Meet to {}: {}", req.peer_addr, e),
        })
        .into_response(),
    }
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
    ///
    /// Returns the bare state (not `Arc`-wrapped) so tests can adjust fields
    /// (retry policy, breakers, connectors) before sharing it.
    pub fn build_test_app_state(self_region: &str) -> AppState {
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
            adaptive: hier_kv_gateway_core::config::AdaptiveConfig::default(),
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
        AppState {
            metadata,
            routing,
            connectors,
            routing_config,
            breakers: Arc::new(CircuitBreakerRegistry::new(
                &hier_kv_gateway_core::config::ResilienceConfig::default(),
            )),
            retry_policy: RetryPolicy::default(),
            peer_registrar: None,
            decision_sink: Arc::new(hier_kv_gateway_core::decision_event::NoopSink),
            decision_buffer: Some(DecisionEventBuffer::new(64)),
            gateway_instance: "test-gw".to_string(),
            gateway_region: self_region.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id_parse_via_core() {
        // The admin endpoint delegates to the canonical core parser.
        let bid = BackendId::parse("us-east-1/worker-0").unwrap();
        assert_eq!(bid.region.as_str(), "us-east-1");
        assert_eq!(bid.instance.as_str(), "worker-0");
        assert!(BackendId::parse("/worker-0").is_none());
        assert!(BackendId::parse("us-east-1/").is_none());
        assert!(BackendId::parse("us-east-1").is_none());
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

    // ---------------------------------------------------------------------
    // forward_with_retries: failover / circuit-breaker behavior
    // ---------------------------------------------------------------------

    mod failover {
        use super::super::test_support::build_test_app_state;
        use super::super::*;
        use hier_kv_gateway_connector::connector::{BackendConnector, HealthStatus};
        use hier_kv_gateway_core::backend::BackendType;
        use hier_kv_gateway_core::config::ResilienceConfig;
        use hier_kv_gateway_core::kv_event::KvCacheEvent;
        use hier_kv_gateway_core::metrics::BackendMetrics;
        use std::time::Duration;

        /// Stub connector whose `forward` either succeeds with an immediate
        /// `Done` stream or fails with a `ConnectorError`.
        struct StubConnector {
            id: BackendId,
            succeed: bool,
        }

        impl StubConnector {
            fn succeeding(instance: &str) -> Arc<dyn BackendConnector> {
                Arc::new(Self {
                    id: BackendId::new("r1", instance),
                    succeed: true,
                })
            }

            fn failing(instance: &str) -> Arc<dyn BackendConnector> {
                Arc::new(Self {
                    id: BackendId::new("r1", instance),
                    succeed: false,
                })
            }
        }

        #[async_trait]
        impl BackendConnector for StubConnector {
            fn backend_type(&self) -> BackendType {
                BackendType::GenericOpenAI
            }

            fn backend_id(&self) -> BackendId {
                self.id.clone()
            }

            async fn discover(&self) -> hier_kv_gateway_core::error::Result<Vec<BackendInfo>> {
                Ok(Vec::new())
            }

            async fn health_check(
                &self,
                _backend: &BackendId,
            ) -> hier_kv_gateway_core::error::Result<HealthStatus> {
                Ok(HealthStatus::default())
            }

            async fn forward(
                &self,
                backend: &BackendId,
                _request: &InferenceRequest,
            ) -> hier_kv_gateway_core::error::Result<BoxStream<'static, InferenceChunk>> {
                if self.succeed {
                    let chunk = InferenceChunk::Done {
                        backend_id: backend.clone(),
                        latency_ms: 0,
                    };
                    Ok(Box::pin(futures::stream::iter(vec![chunk])))
                } else {
                    Err(HierKvGatewayError::ConnectorError(format!(
                        "stub backend {} is down",
                        backend
                    )))
                }
            }

            fn supports_kv_events(&self) -> bool {
                false
            }

            async fn subscribe_kv_events(
                &self,
                _backend: &BackendId,
            ) -> hier_kv_gateway_core::error::Result<BoxStream<'static, KvCacheEvent>> {
                Err(HierKvGatewayError::ConnectorError(
                    "stub does not support KV events".to_string(),
                ))
            }

            async fn collect_metrics(
                &self,
                _backend: &BackendId,
            ) -> hier_kv_gateway_core::error::Result<BackendMetrics> {
                Ok(BackendMetrics::default())
            }
        }

        /// AppState with zero retry backoff (tests must not sleep) and a
        /// failure-threshold-1 breaker registry, plus the given connectors.
        fn build_state(connectors: Vec<Arc<dyn BackendConnector>>) -> Arc<AppState> {
            let mut state = build_test_app_state("r1");
            // Zero-delay retry policy: backoff shape is covered by the
            // connector crate's unit tests; here we only exercise the loop.
            state.retry_policy = RetryPolicy::new(Duration::ZERO, Duration::ZERO);
            state.breakers = Arc::new(CircuitBreakerRegistry::new(&ResilienceConfig {
                retry_backoff_ms: 0,
                retry_max_backoff_ms: 0,
                circuit_breaker_failure_threshold: 1,
                circuit_breaker_cooldown_secs: 3600,
                half_open_success_threshold: 1,
            }));
            let registry = ConnectorRegistry::new();
            for c in connectors {
                registry.register(c);
            }
            state.connectors = Arc::new(registry);
            Arc::new(state)
        }
        fn decision(instance: &str) -> RouteDecision {
            RouteDecision {
                backend: BackendId::new("r1", instance),
                strategy: "test".to_string(),
                kv_overlap: 0,
                scores: Vec::new(),
            }
        }

        fn inference() -> InferenceRequest {
            InferenceRequest {
                request_id: RequestId::new("req-failover"),
                model: "m".to_string(),
                messages: Vec::new(),
                token_ids: vec![1, 2, 3],
                max_tokens: 8,
                temperature: 0.0,
                stream: true,
                tools: Vec::new(),
                lora_name: None,
            }
        }

        #[tokio::test]
        async fn fails_over_to_second_candidate() {
            let state = build_state(vec![StubConnector::failing("a"), StubConnector::succeeding("b")]);
            let decisions = vec![decision("a"), decision("b")];
            let req = inference();
            let mut attempts = Vec::new();

            let (stream, meta) =
                forward_with_retries(&state, &decisions, &req, &req.request_id, 1, &mut attempts)
                    .await
                    .expect("second candidate should succeed");
            assert_eq!(meta.backend, "r1/b");
            // The failure on "a" opened its circuit (threshold = 1).
            assert!(!state.breakers.allow(&BackendId::new("r1", "a")));
            // The stream really came from the winning backend.
            let chunks: Vec<InferenceChunk> = stream.collect().await;
            assert!(matches!(chunks.first(), Some(InferenceChunk::Done { .. })));
            // One failed attempt + one successful attempt were recorded.
            assert_eq!(attempts.len(), 2);
            assert!(!attempts[0].success && attempts[0].error.is_some());
            assert!(attempts[1].success);
        }

        #[tokio::test]
        async fn returns_last_error_when_all_candidates_fail() {
            let state = build_state(vec![StubConnector::failing("a"), StubConnector::failing("b")]);
            let decisions = vec![decision("a"), decision("b")];
            let req = inference();
            let mut attempts = Vec::new();

            let result =
                forward_with_retries(&state, &decisions, &req, &req.request_id, 1, &mut attempts).await;
            match result {
                Err(HierKvGatewayError::ConnectorError(msg)) => {
                    assert!(msg.contains("r1/b"), "last error should be from b: {msg}");
                }
                Err(other) => panic!("expected ConnectorError, got {other:?}"),
                Ok(_) => panic!("all candidates failing must error"),
            }
            assert_eq!(attempts.len(), 2);
            assert!(attempts.iter().all(|a| !a.success));
        }

        #[tokio::test]
        async fn open_circuit_skips_candidate() {
            let state = build_state(vec![StubConnector::succeeding("a"), StubConnector::succeeding("b")]);
            // Pre-open a's circuit.
            state.breakers.on_failure(&BackendId::new("r1", "a"));
            assert!(!state.breakers.allow(&BackendId::new("r1", "a")));

            let decisions = vec![decision("a"), decision("b")];
            let req = inference();
            let mut attempts = Vec::new();
            let (_stream, meta) =
                forward_with_retries(&state, &decisions, &req, &req.request_id, 1, &mut attempts)
                    .await
                    .expect("b should serve while a is short-circuited");
            assert_eq!(meta.backend, "r1/b");
            assert_eq!(attempts.len(), 2);
            assert!(attempts[0].skipped_open_circuit);
            assert!(attempts[1].success);
        }

        #[tokio::test]
        async fn all_circuits_open_returns_unavailable() {
            let state = build_state(vec![StubConnector::succeeding("a")]);
            state.breakers.on_failure(&BackendId::new("r1", "a"));

            let decisions = vec![decision("a")];
            let req = inference();
            let mut attempts = Vec::new();
            let result =
                forward_with_retries(&state, &decisions, &req, &req.request_id, 1, &mut attempts).await;
            match result {
                Err(e) => assert!(matches!(e, HierKvGatewayError::BackendUnavailable)),
                Ok(_) => panic!("short-circuited candidates must yield BackendUnavailable"),
            }
            assert_eq!(attempts.len(), 1);
            assert!(attempts[0].skipped_open_circuit);
        }

        #[tokio::test]
        async fn missing_connector_counts_as_candidate_failure() {
            // Only "b" is registered; "a" resolves to no connector.
            let state = build_state(vec![StubConnector::succeeding("b")]);
            let decisions = vec![decision("a"), decision("b")];
            let req = inference();
            let mut attempts = Vec::new();
            let (_stream, meta) =
                forward_with_retries(&state, &decisions, &req, &req.request_id, 1, &mut attempts)
                    .await
                    .expect("b should serve after a resolves to nothing");
            assert_eq!(meta.backend, "r1/b");
            assert_eq!(attempts.len(), 2);
            assert!(!attempts[0].success);
            assert!(attempts[0].error.is_some());
        }
    }
}
