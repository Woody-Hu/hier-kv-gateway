//! SGLang connector.
//!
//! Integrates [SGLang](https://github.com/sgl-project/sglang) servers
//! (`python -m sglang.launch_server ...`) with the gateway:
//!
//! * **Forwarding (text)** — delegated to the OpenAI-compatible path
//!   (`POST /v1/chat/completions`, SSE streaming), which SGLang's HTTP server
//!   implements natively.
//! * **Forwarding (token ids)** — when `forwarding.emit_token_ids = true` and
//!   the request carries `token_ids`, the gateway forwards to SGLang's native
//!   `POST /generate` with `input_ids`. This is the radix-cache-friendly path:
//!   the backend skips re-tokenization and the KV block hashes the gateway
//!   routed on match the prefill blocks SGLang actually builds.
//! * **Metrics** — read from `GET /get_server_info` (`internal_states[0]`):
//!   `num_running_reqs` → `active_requests`, `num_queue_reqs` → `queue_depth`,
//!   `num_used_tokens` / `max_total_num_tokens` → KV block usage. Falls back
//!   to the Prometheus `/metrics` parser when the endpoint is unavailable.
//! * **Health / discovery** — `/health` and `/v1/models`, same as the
//!   OpenAI-compatible path.
//!
//! SGLang does not publish a KV-cache event stream, so
//! [`BackendConnector::supports_kv_events`] returns `false`; the gateway keeps
//! its own radix-tree KV index from the block hashes it computes at routing
//! time, which mirrors SGLang's radix-cache prefix semantics as long as
//! `kv_block_size` matches on both sides.

use std::time::Instant;

use hier_kv_gateway_core::backend::{BackendInfo, BackendType, Endpoint};
use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::{BackendId, BackendInstanceId, RegionId};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::request::{InferenceChunk, InferenceRequest};

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::Serialize;

use crate::connector::{BackendConnector, HealthStatus};
use crate::openai_compat::OpenAICompatConnector;

/// SGLang engine connector.
///
/// Anchored to one `sglang.launch_server` HTTP endpoint. See the module docs
/// for the endpoint mapping.
pub struct SglangConnector {
    /// OpenAI-compatible inner connector (text chat / health / discovery /
    /// Prometheus metrics fallback).
    inner: OpenAICompatConnector,
    /// HTTP client for the SGLang-native endpoints.
    client: reqwest::Client,
    /// Backend base URL, e.g. `http://10.0.0.1:30000`.
    base_url: String,
    /// KV block size used to convert token counts to block counts.
    kv_block_size: u32,
    /// When `true`, requests carrying `token_ids` are forwarded to the native
    /// `/generate` endpoint with `input_ids` instead of the chat endpoint.
    emit_token_ids: bool,
}

impl SglangConnector {
    /// Create a new SGLang connector.
    pub fn new(
        base_url: impl Into<String>,
        region: RegionId,
        instance_id: impl Into<BackendInstanceId>,
        models: Vec<String>,
        kv_block_size: u32,
    ) -> Self {
        let base_url = base_url.into();
        let inner = OpenAICompatConnector::new(
            base_url.clone(),
            BackendType::SglangEngine,
            region,
            instance_id,
            models,
            kv_block_size,
        );
        Self {
            inner,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build reqwest client"),
            base_url,
            kv_block_size,
            emit_token_ids: false,
        }
    }

    /// Enable or disable token-id forwarding via `/generate` (builder style).
    pub fn with_emit_token_ids(mut self, emit: bool) -> Self {
        self.emit_token_ids = emit;
        self
    }

    /// Construct a connector from an endpoint URL and configuration.
    pub fn from_endpoint(
        endpoint: &Endpoint,
        region: &RegionId,
        models: Vec<String>,
        kv_block_size: u32,
    ) -> Self {
        Self::new(
            &endpoint.url,
            region.clone(),
            endpoint.url.replace("http://", "").replace("https://", ""),
            models,
            kv_block_size,
        )
    }

    /// Native `/generate` URL.
    fn generate_url(&self) -> String {
        format!("{}/generate", self.base_url)
    }

    /// `/get_server_info` URL.
    fn server_info_url(&self) -> String {
        format!("{}/get_server_info", self.base_url)
    }

    /// Forward via the native `/generate` endpoint with `input_ids`.
    async fn forward_generate(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<BoxStream<'static, InferenceChunk>> {
        let start = Instant::now();
        let body = GenerateRequest::from(request);

        let resp = self
            .client
            .post(self.generate_url())
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                HierKvGatewayError::ConnectorError(format!("sglang /generate failed: {}", e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(HierKvGatewayError::ConnectorError(format!(
                "sglang /generate returned HTTP {}: {}",
                status, text
            )));
        }

        Ok(Box::pin(SglangGenerateParser::new(
            resp.bytes_stream(),
            backend.clone(),
            start,
        )))
    }

    /// Fetch load metrics from `/get_server_info`.
    ///
    /// Returns `None` when the endpoint errors or carries no usable scheduler
    /// state (older servers), letting the caller fall back to Prometheus.
    async fn fetch_server_info_metrics(&self) -> Option<BackendMetrics> {
        let resp = self
            .client
            .get(self.server_info_url())
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let value: serde_json::Value = resp.json().await.ok()?;
        let now = chrono::Utc::now().timestamp_millis();
        Some(map_server_info_metrics(&value, self.kv_block_size, now))
    }
}

#[async_trait]
impl BackendConnector for SglangConnector {
    fn backend_type(&self) -> BackendType {
        BackendType::SglangEngine
    }

    fn backend_id(&self) -> BackendId {
        self.inner.backend_id()
    }

    async fn discover(&self) -> Result<Vec<BackendInfo>> {
        self.inner.discover().await
    }

    async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus> {
        self.inner.health_check(backend).await
    }

    async fn forward(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<BoxStream<'static, InferenceChunk>> {
        if self.emit_token_ids && !request.token_ids.is_empty() {
            return self.forward_generate(backend, request).await;
        }
        self.inner.forward(backend, request).await
    }

    fn supports_kv_events(&self) -> bool {
        // SGLang exposes no KV-cache event stream; the gateway tracks prefix
        // residency in its own radix-tree index built from routing-time block
        // hashes (same semantics as SGLang's radix cache).
        false
    }

    async fn subscribe_kv_events(
        &self,
        _backend: &BackendId,
    ) -> Result<BoxStream<'static, KvCacheEvent>> {
        Err(HierKvGatewayError::ConnectorError(
            "SGLang connector does not support KV cache events".to_string(),
        ))
    }

    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics> {
        // Prefer the scheduler-accurate /get_server_info snapshot; fall back
        // to the Prometheus /metrics parser (SGLang exposes it when started
        // with --enable-metrics).
        if let Some(m) = self.fetch_server_info_metrics().await {
            return Ok(m);
        }
        self.inner.collect_metrics(backend).await
    }
}

// ===== /generate wire format =====

/// Native `/generate` request body (token-id form).
#[derive(Serialize)]
struct GenerateRequest {
    /// Pre-tokenized prompt.
    input_ids: Vec<u32>,
    /// Sampling parameters; SGLang accepts a subset of OpenAI sampling knobs.
    sampling_params: GenerateSamplingParams,
    /// Always stream: the gateway pipeline is chunk-oriented.
    stream: bool,
}

/// Sampling parameters accepted by SGLang's `/generate`.
#[derive(Serialize)]
struct GenerateSamplingParams {
    /// Maximum number of new tokens.
    max_new_tokens: u32,
    /// Sampling temperature.
    temperature: f64,
}

impl From<&InferenceRequest> for GenerateRequest {
    fn from(req: &InferenceRequest) -> Self {
        Self {
            input_ids: req.token_ids.clone(),
            sampling_params: GenerateSamplingParams {
                max_new_tokens: req.max_tokens,
                temperature: req.temperature,
            },
            stream: true,
        }
    }
}

/// One streamed `/generate` chunk: `{"text": ..., "meta_info": {...}}`.
///
/// SGLang streams the *accumulated* text on every chunk; the parser converts
/// it back into deltas so the gateway's chunk semantics stay incremental.
#[derive(serde::Deserialize)]
struct GenerateStreamChunk {
    /// Accumulated output text so far.
    #[serde(default)]
    text: String,
    /// Metadata; `finish_reason` is null until the terminal chunk.
    #[serde(default)]
    meta_info: Option<GenerateMetaInfo>,
}

/// `meta_info` of a streamed `/generate` chunk.
#[derive(serde::Deserialize)]
struct GenerateMetaInfo {
    /// Terminal reason, e.g. `{"type": "stop"}` / `{"type": "length"}`.
    #[serde(default)]
    finish_reason: Option<GenerateFinishReason>,
}

/// `finish_reason` object inside `meta_info`.
#[derive(serde::Deserialize)]
struct GenerateFinishReason {
    /// Reason kind (`stop`, `length`, `abort`, ...).
    #[serde(rename = "type")]
    kind: String,
}

/// SSE parser for SGLang's `/generate` streaming format.
///
/// Wire format: `data: {"text": "<accumulated>", "meta_info": {...}}` lines;
/// the stream simply closes at the end (no `data: [DONE]` sentinel).
///
/// Exposed for wire-format benchmarks; not part of the stable public API.
#[doc(hidden)]
pub struct SglangGenerateParser<S> {
    inner: S,
    buffer: String,
    backend_id: BackendId,
    start: Instant,
    /// Bytes of accumulated text already emitted as deltas.
    emitted_len: usize,
    /// Whether the terminal (finish_reason) chunk has been seen.
    finished: bool,
    /// Whether the terminal `Done` chunk has been returned to the consumer.
    done_emitted: bool,
}

impl<S> SglangGenerateParser<S>
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
{
    /// Create a parser over a raw byte stream.
    pub fn new(stream: S, backend_id: BackendId, start: Instant) -> Self {
        Self {
            inner: stream,
            buffer: String::new(),
            backend_id,
            start,
            emitted_len: 0,
            finished: false,
            done_emitted: false,
        }
    }

    fn done_chunk(&self) -> InferenceChunk {
        InferenceChunk::Done {
            backend_id: self.backend_id.clone(),
            latency_ms: self.start.elapsed().as_millis() as u64,
        }
    }

    /// Extract one chunk from the buffer, or `None` when more data is needed.
    fn try_parse(&mut self) -> Option<InferenceChunk> {
        loop {
            let newline_pos = self.buffer.find('\n')?;
            let line = self.buffer[..newline_pos].trim_end_matches('\r').to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data.trim() == "[DONE]" {
                self.finished = true;
                self.done_emitted = true;
                return Some(self.done_chunk());
            }

            let chunk = match serde_json::from_str::<GenerateStreamChunk>(data) {
                Ok(c) => c,
                Err(e) => {
                    tracing::trace!("sglang stream JSON parse failed (skipped): {} - {}", e, data);
                    continue;
                }
            };

            // Convert accumulated text to a delta. If the server ever sends
            // deltas directly (text shorter than what we emitted), fall back
            // to emitting the payload as-is.
            let delta_text = if chunk.text.len() >= self.emitted_len {
                let d = chunk.text[self.emitted_len..].to_string();
                self.emitted_len = chunk.text.len();
                d
            } else {
                self.emitted_len += chunk.text.len();
                chunk.text.clone()
            };

            let finish_reason = chunk
                .meta_info
                .and_then(|m| m.finish_reason)
                .map(|f| f.kind);
            if finish_reason.is_some() {
                self.finished = true;
            }

            if delta_text.is_empty() && finish_reason.is_none() {
                continue; // keep-alive chunk; parse the next line
            }
            return Some(InferenceChunk::Delta {
                text: delta_text,
                finish_reason,
            });
        }
    }
}

impl<S> futures::Stream for SglangGenerateParser<S>
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
{
    type Item = InferenceChunk;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use futures::StreamExt;
        let this = self.get_mut();

        if this.finished {
            // After the terminal Delta, report Done exactly once; then end.
            if this.done_emitted {
                return std::task::Poll::Ready(None);
            }
            this.done_emitted = true;
            return std::task::Poll::Ready(Some(this.done_chunk()));
        }

        loop {
            if let Some(chunk) = this.try_parse() {
                return std::task::Poll::Ready(Some(chunk));
            }
            match this.inner.poll_next_unpin(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(InferenceChunk::Error {
                        code: 502,
                        message: format!("sglang stream read error: {}", e),
                    }));
                }
                std::task::Poll::Ready(None) => {
                    // EOF: drain remaining buffer, then synthesize Done.
                    if !this.buffer.is_empty() {
                        this.buffer.push('\n');
                        if let Some(chunk) = this.try_parse() {
                            return std::task::Poll::Ready(Some(chunk));
                        }
                    }
                    this.finished = true;
                    this.done_emitted = true;
                    return std::task::Poll::Ready(Some(this.done_chunk()));
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

// ===== /get_server_info mapping =====

/// Map a `/get_server_info` JSON payload to [`BackendMetrics`].
///
/// The scheduler state lives in `internal_states[0]` (an object on some
/// versions). Token counts are converted to KV blocks with `kv_block_size`.
/// Unknown or missing fields degrade to zero, matching the "structured
/// default" contract of [`BackendConnector::collect_metrics`].
///
/// Exposed for wire-format benchmarks; not part of the stable public API.
#[doc(hidden)]
pub fn map_server_info_metrics(value: &serde_json::Value, kv_block_size: u32, now_ms: i64) -> BackendMetrics {
    let mut m = BackendMetrics {
        timestamp: now_ms,
        ..Default::default()
    };

    // Locate the scheduler state object.
    let state = value
        .get("internal_states")
        .and_then(|s| {
            if let Some(arr) = s.as_array() {
                arr.first().cloned()
            } else {
                Some(s.clone())
            }
        })
        .unwrap_or_else(|| value.clone());

    let get_u64 = |keys: &[&str]| -> Option<u64> {
        keys.iter()
            .find_map(|k| state.get(k).and_then(|v| v.as_f64()))
            .map(|v| v.max(0.0) as u64)
    };

    m.active_requests = get_u64(&["num_running_reqs", "running_reqs"]).unwrap_or(0);
    m.queue_depth = get_u64(&["num_queue_reqs", "num_waiting_reqs"]).unwrap_or(0);

    let block = u64::from(kv_block_size.max(1));
    if let Some(used_tokens) = get_u64(&["num_used_tokens"]) {
        m.kv_used_blocks = used_tokens / block;
    }
    // max_total_num_tokens may live at the top level on some versions.
    let total_tokens = get_u64(&["max_total_num_tokens"]).or_else(|| {
        value
            .get("max_total_num_tokens")
            .and_then(|v| v.as_f64())
            .map(|v| v.max(0.0) as u64)
    });
    if let Some(total) = total_tokens {
        m.kv_total_blocks = total / block;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use hier_kv_gateway_core::ids::RequestId;
    use hier_kv_gateway_core::request::ChatMessage;

    fn tokenized_request() -> InferenceRequest {
        InferenceRequest {
            request_id: RequestId::new("r1"),
            model: "qwen2.5-7b".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            token_ids: vec![101, 102, 103],
            max_tokens: 32,
            temperature: 0.5,
            stream: true,
            tools: vec![],
            lora_name: None,
        }
    }

    #[test]
    fn generate_request_uses_input_ids_and_sampling_params() {
        let body = GenerateRequest::from(&tokenized_request());
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["input_ids"], serde_json::json!([101, 102, 103]));
        assert_eq!(json["sampling_params"]["max_new_tokens"], 32);
        assert_eq!(json["sampling_params"]["temperature"], 0.5);
        assert_eq!(json["stream"], true);
        assert!(json.get("messages").is_none());
    }

    #[test]
    fn server_info_maps_scheduler_state_array() {
        let info = serde_json::json!({
            "internal_states": [{
                "num_running_reqs": 7,
                "num_queue_reqs": 3,
                "num_used_tokens": 16000,
                "max_total_num_tokens": 1_000_000,
                "token_usage": 0.016
            }]
        });
        let m = map_server_info_metrics(&info, 16, 42);
        assert_eq!(m.active_requests, 7);
        assert_eq!(m.queue_depth, 3);
        assert_eq!(m.kv_used_blocks, 1000); // 16000 / 16
        assert_eq!(m.kv_total_blocks, 62500); // 1_000_000 / 16
        assert_eq!(m.timestamp, 42);
    }

    #[test]
    fn server_info_tolerates_missing_fields_and_top_level_total() {
        let info = serde_json::json!({
            "internal_states": [{"num_running_reqs": 2}],
            "max_total_num_tokens": 8192
        });
        let m = map_server_info_metrics(&info, 16, 0);
        assert_eq!(m.active_requests, 2);
        assert_eq!(m.queue_depth, 0);
        assert_eq!(m.kv_used_blocks, 0);
        assert_eq!(m.kv_total_blocks, 512);
    }

    /// Feed a byte stream through the parser and collect all chunks.
    async fn collect_chunks(lines: &[&str]) -> Vec<InferenceChunk> {
        let payload = lines
            .iter()
            .map(|l| format!("data: {l}\n\n"))
            .collect::<String>();
        let byte_stream = futures::stream::iter(vec![Ok::<_, reqwest::Error>(
            bytes::Bytes::from(payload),
        )]);
        let parser = SglangGenerateParser::new(
            byte_stream,
            BackendId::new("r1", "sglang-0"),
            Instant::now(),
        );
        parser.collect().await
    }

    #[tokio::test]
    async fn generate_stream_accumulated_text_becomes_deltas() {
        let chunks = collect_chunks(&[
            r#"{"text": "Hello", "meta_info": {"finish_reason": null}}"#,
            r#"{"text": "Hello world", "meta_info": {"finish_reason": null}}"#,
            r#"{"text": "Hello world!", "meta_info": {"finish_reason": {"type": "stop"}}}"#,
        ])
        .await;

        let deltas: Vec<(String, Option<String>)> = chunks
            .iter()
            .filter_map(|c| match c {
                InferenceChunk::Delta { text, finish_reason } => {
                    Some((text.clone(), finish_reason.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 3);
        assert_eq!(deltas[0], ("Hello".to_string(), None));
        assert_eq!(deltas[1], (" world".to_string(), None));
        assert_eq!(deltas[2], ("!".to_string(), Some("stop".to_string())));
        // Stream terminates with Done.
        assert!(matches!(chunks.last(), Some(InferenceChunk::Done { .. })));
    }

    #[tokio::test]
    async fn generate_stream_eof_without_finish_reason_yields_done() {
        let chunks = collect_chunks(&[r#"{"text": "partial", "meta_info": {"finish_reason": null}}"#]).await;
        assert!(matches!(chunks.first(), Some(InferenceChunk::Delta { .. })));
        assert!(matches!(chunks.last(), Some(InferenceChunk::Done { .. })));
    }

    #[test]
    fn backend_type_is_sglang() {
        let c = SglangConnector::new(
            "http://localhost:30000",
            RegionId::new("edge-1"),
            "sglang-0",
            vec!["qwen2.5-7b".to_string()],
            16,
        );
        assert_eq!(c.backend_type(), BackendType::SglangEngine);
        assert_eq!(
            c.backend_id(),
            BackendId::new("edge-1", "sglang-0")
        );
        assert!(!c.supports_kv_events());
    }

    #[test]
    fn backend_type_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&BackendType::SglangEngine).unwrap(),
            r#""sglang_engine""#
        );
        let back: BackendType = serde_json::from_str(r#""sglang_engine""#).unwrap();
        assert_eq!(back, BackendType::SglangEngine);
    }
}
