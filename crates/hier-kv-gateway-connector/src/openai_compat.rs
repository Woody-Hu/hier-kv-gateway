//! OpenAI-compatible connector implementation.
//!
//! Suitable for vLLM, llama.cpp, and other inference engines compatible with the OpenAI
//! Chat Completions API. Forwards requests via HTTP POST `/v1/chat/completions`
//! (`stream: true`), and parses the SSE (Server-Sent Events) stream to return a sequence
//! of [`InferenceChunk`].

use std::sync::Arc;
use std::time::Instant;

use hier_kv_gateway_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
    ModelInstance, Protocol, Quantization,
};
use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::{BackendId, BackendInstanceId, IndexerDomainId, RegionId};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_core::metrics::{BackendMetrics, LatencyStats};
use hier_kv_gateway_core::request::{InferenceChunk, InferenceRequest};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::connector::{BackendConnector, HealthStatus};

/// OpenAI-compatible connector.
///
/// Suitable for vLLM, llama.cpp (server mode), and any inference service compatible with
/// the OpenAI API. Communicates over HTTP and supports streaming SSE parsing.
pub struct OpenAICompatConnector {
    /// HTTP client (reusable connection pool).
    client: reqwest::Client,
    /// Backend base URL, e.g. `http://10.0.0.1:8000`.
    base_url: String,
    /// Backend type (distinguishes vLLM / llama.cpp / generic).
    backend_type: BackendType,
    /// Region this backend resides in.
    region: RegionId,
    /// Backend instance identifier.
    instance_id: BackendInstanceId,
    /// List of models served by this backend (injected from config, returned by discover).
    models: Vec<String>,
    /// KV block size (used to construct BackendInfo).
    kv_block_size: u32,
}

impl OpenAICompatConnector {
    /// Create a new OpenAI-compatible connector.
    pub fn new(
        base_url: impl Into<String>,
        backend_type: BackendType,
        region: RegionId,
        instance_id: impl Into<BackendInstanceId>,
        models: Vec<String>,
        kv_block_size: u32,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build reqwest client"),
            base_url: base_url.into(),
            backend_type,
            region,
            instance_id: instance_id.into(),
            models,
            kv_block_size,
        }
    }

    /// Construct a connector from an endpoint URL and configuration.
    pub fn from_endpoint(
        endpoint: &Endpoint,
        backend_type: BackendType,
        region: &RegionId,
        models: Vec<String>,
        kv_block_size: u32,
    ) -> Self {
        Self::new(
            &endpoint.url,
            backend_type,
            region.clone(),
            // Use the host:port of the URL as the instance identifier
            endpoint.url.replace("http://", "").replace("https://", ""),
            models,
            kv_block_size,
        )
    }

    /// Construct the BackendId for this connector.
    fn backend_id(&self) -> BackendId {
        BackendId::new(self.region.clone(), self.instance_id.clone())
    }

    /// Health check URL.
    fn health_url(&self) -> String {
        format!("{}/health", self.base_url)
    }

    /// Model list URL.
    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url)
    }

    /// Chat Completions URL.
    fn completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }
}

#[async_trait]
impl BackendConnector for OpenAICompatConnector {
    fn backend_type(&self) -> BackendType {
        self.backend_type.clone()
    }

    async fn discover(&self) -> Result<Vec<BackendInfo>> {
        // Try to fetch the model list from /v1/models
        let models = self.fetch_models().await.unwrap_or_default();

        let model_instances: Vec<ModelInstance> = if models.is_empty() {
            // If /v1/models is unavailable, use the model names injected from config
            self.models
                .iter()
                .map(|name| ModelInstance {
                    model_name: name.clone(),
                    model_architecture: "unknown".to_string(),
                    quantization: Quantization::Fp16,
                    max_context_len: 32768,
                    supports_tool_calling: false,
                    supports_streaming: true,
                })
                .collect()
        } else {
            models
        };

        let backend_info = BackendInfo {
            id: self.backend_id(),
            backend_type: self.backend_type.clone(),
            endpoint: Endpoint {
                url: self.base_url.clone(),
                protocol: Protocol::Http,
            },
            models: model_instances,
            region: self.region.clone(),
            indexer_domain: IndexerDomainId(0),
            capabilities: BackendCapabilities {
                supports_kv_events: false,
                supports_batching: true,
                max_batch_size: 0,
                gpu_count: 0,
                gpu_memory_gb: 0,
            },
            kv_config: KvConfig {
                block_size: self.kv_block_size,
                cache_namespace: String::new(),
                max_kv_blocks: 0,
            },
            status: BackendStatus::Healthy,
        };

        Ok(vec![backend_info])
    }

    async fn health_check(&self, _backend: &BackendId) -> Result<HealthStatus> {
        let now = chrono::Utc::now().timestamp() as u64;
        let resp = self
            .client
            .get(self.health_url())
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => Ok(HealthStatus::healthy(now)),
            Ok(r) => {
                tracing::warn!("health check returned non-200 status: {}", r.status());
                Ok(HealthStatus {
                    status: BackendStatus::Unhealthy,
                    healthy_since_unix: now,
                    error_count: 1,
                })
            }
            Err(e) => {
                tracing::warn!("health check failed: {}", e);
                Ok(HealthStatus::unhealthy(now, 1))
            }
        }
    }

    async fn forward(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<BoxStream<'static, InferenceChunk>> {
        let start = Instant::now();
        let backend_id = backend.clone();

        // Build the OpenAI Chat Completions request body
        let body = ChatCompletionRequest::from(request);
        let body_json = serde_json::to_value(&body)
            .map_err(|e| HierKvGatewayError::ConnectorError(format!("request serialization failed: {}", e)))?;

        let resp = self
            .client
            .post(self.completions_url())
            .json(&body_json)
            .send()
            .await
            .map_err(|e| HierKvGatewayError::ConnectorError(format!("request forwarding failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(HierKvGatewayError::ConnectorError(format!(
                "backend returned HTTP {}: {}",
                status, text
            )));
        }

        // Get the byte stream and parse SSE
        let byte_stream = resp.bytes_stream();
        let stream = SseParser::new(byte_stream, backend_id, start);

        Ok(Box::pin(stream))
    }

    fn supports_kv_events(&self) -> bool {
        false
    }

    async fn subscribe_kv_events(
        &self,
        _backend: &BackendId,
    ) -> Result<BoxStream<'static, KvCacheEvent>> {
        Err(HierKvGatewayError::ConnectorError(
            "OpenAI-compatible connector does not support KV cache events".to_string(),
        ))
    }

    async fn collect_metrics(&self, _backend: &BackendId) -> Result<BackendMetrics> {
        // Try to fetch the /metrics endpoint (supported by vLLM)
        let now = chrono::Utc::now().timestamp_millis() as u64;

        let resp = self
            .client
            .get(format!("{}/metrics", self.base_url))
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await;

        let mut metrics = BackendMetrics {
            timestamp: now as i64,
            ..Default::default()
        };

        if let Ok(resp) = resp {
            if resp.status().is_success() {
                let text = resp.text().await.unwrap_or_default();
                parse_prometheus_metrics(&text, &mut metrics);
            }
        }

        Ok(metrics)
    }
}

/// vLLM connector (inherits OpenAI compatibility; KV event support can be extended).
pub struct VllmConnector(OpenAICompatConnector);

impl VllmConnector {
    pub fn new(
        base_url: impl Into<String>,
        region: RegionId,
        instance_id: impl Into<BackendInstanceId>,
        models: Vec<String>,
        kv_block_size: u32,
    ) -> Self {
        Self(OpenAICompatConnector::new(
            base_url,
            BackendType::VllmEngine,
            region,
            instance_id,
            models,
            kv_block_size,
        ))
    }
}

#[async_trait]
impl BackendConnector for VllmConnector {
    fn backend_type(&self) -> BackendType {
        self.0.backend_type()
    }
    async fn discover(&self) -> Result<Vec<BackendInfo>> {
        self.0.discover().await
    }
    async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus> {
        self.0.health_check(backend).await
    }
    async fn forward(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<BoxStream<'static, InferenceChunk>> {
        self.0.forward(backend, request).await
    }
    fn supports_kv_events(&self) -> bool {
        false // Can later be extended to true with subscribe_kv_events implemented
    }
    async fn subscribe_kv_events(
        &self,
        _backend: &BackendId,
    ) -> Result<BoxStream<'static, KvCacheEvent>> {
        Err(HierKvGatewayError::ConnectorError(
            "vLLM connector does not yet support KV cache events".to_string(),
        ))
    }
    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics> {
        self.0.collect_metrics(backend).await
    }
}

// ===== OpenAI API request/response structures =====

/// OpenAI Chat Completions request body.
#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessageSerde>,
    max_tokens: u32,
    temperature: f64,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolSerde>>,
}

#[derive(Serialize)]
struct ChatMessageSerde {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ToolSerde {
    #[serde(rename = "type")]
    tool_type: String,
    function: ToolFunctionSerde,
}

#[derive(Serialize)]
struct ToolFunctionSerde {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl From<&InferenceRequest> for ChatCompletionRequest {
    fn from(req: &InferenceRequest) -> Self {
        let messages: Vec<ChatMessageSerde> = req
            .messages
            .iter()
            .map(|m| ChatMessageSerde {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        let tools = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|t| ToolSerde {
                        tool_type: "function".to_string(),
                        function: ToolFunctionSerde {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: serde_json::from_str(&t.parameters_schema)
                                .unwrap_or(serde_json::json!({})),
                        },
                    })
                    .collect(),
            )
        };

        Self {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: true, // Always use streaming
            tools,
        }
    }
}

/// A single SSE data chunk of an OpenAI streaming response.
#[derive(Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    id: Option<String>,
    function: Option<ToolCallFunction>,
}

#[derive(Deserialize)]
struct ToolCallFunction {
    name: Option<String>,
    arguments: Option<String>,
}

// ===== SSE parser =====

/// SSE stream parser that converts an HTTP byte stream into a sequence of InferenceChunk.
///
/// Parsing rules:
/// - Lines prefixed with `data: ` are data lines
/// - `data: [DONE]` indicates end of stream
/// - Other lines (empty lines, event lines) are ignored
struct SseParser<S> {
    inner: S,
    buffer: String,
    backend_id: BackendId,
    start: Instant,
}

impl<S> SseParser<S>
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
{
    fn new(stream: S, backend_id: BackendId, start: Instant) -> Self {
        Self {
            inner: stream,
            buffer: String::new(),
            backend_id,
            start,
        }
    }

    /// Extract and parse a complete SSE event from the buffer.
    ///
    /// Returns `Some(chunk)` to indicate a produced data block, or `None` to indicate more
    /// data is needed.
    fn try_parse(&mut self) -> Option<InferenceChunk> {
        loop {
            // Find a newline-delimited line
            let newline_pos = self.buffer.find('\n')?;
            let line = self.buffer[..newline_pos].trim_end_matches('\r').to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    let latency = self.start.elapsed().as_millis() as u64;
                    return Some(InferenceChunk::Done {
                        backend_id: self.backend_id.clone(),
                        latency_ms: latency,
                    });
                }

                // Parse JSON
                match serde_json::from_str::<ChatCompletionChunk>(data) {
                    Ok(chunk) => {
                        for choice in &chunk.choices {
                            // Tool calls
                            for tc in &choice.delta.tool_calls {
                                if let (Some(id), Some(func)) = (&tc.id, &tc.function) {
                                    return Some(InferenceChunk::ToolCall {
                                        id: id.clone(),
                                        function: func.name.clone().unwrap_or_default(),
                                        args: func.arguments.clone().unwrap_or_default(),
                                    });
                                }
                            }

                            // Text content
                            if let Some(content) = &choice.delta.content {
                                if !content.is_empty() {
                                    return Some(InferenceChunk::Delta {
                                        text: content.clone(),
                                        finish_reason: choice.finish_reason.clone(),
                                    });
                                }
                            }

                            // finish_reason present but no content (end of stream)
                            if let Some(reason) = &choice.finish_reason {
                                if choice.delta.content.is_none() {
                                    return Some(InferenceChunk::Delta {
                                        text: String::new(),
                                        finish_reason: Some(reason.clone()),
                                    });
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::trace!("SSE JSON parsing failed (skipped): {} - data: {}", e, data);
                    }
                }
            }
            // Non-data: lines are ignored
        }
    }
}

impl<S> futures::Stream for SseParser<S>
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
{
    type Item = InferenceChunk;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // First try to parse existing data from the buffer
            if let Some(chunk) = this.try_parse() {
                return std::task::Poll::Ready(Some(chunk));
            }

            // Pull more data from the underlying stream
            use futures::StreamExt;
            match this.inner.poll_next_unpin(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&bytes));
                    // Continue looping to try parsing
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    tracing::warn!("SSE stream read error: {}", e);
                    return std::task::Poll::Ready(Some(InferenceChunk::Error {
                        code: 502,
                        message: format!("stream read error: {}", e),
                    }));
                }
                std::task::Poll::Ready(None) => {
                    // Stream ended, but [DONE] was not received
                    if this.buffer.is_empty() {
                        let latency = this.start.elapsed().as_millis() as u64;
                        return std::task::Poll::Ready(Some(InferenceChunk::Done {
                            backend_id: this.backend_id.clone(),
                            latency_ms: latency,
                        }));
                    }
                    // Try to parse the remaining buffer
                    if let Some(chunk) = this.try_parse() {
                        return std::task::Poll::Ready(Some(chunk));
                    }
                    // Actually finish
                    let latency = this.start.elapsed().as_millis() as u64;
                    return std::task::Poll::Ready(Some(InferenceChunk::Done {
                        backend_id: this.backend_id.clone(),
                        latency_ms: latency,
                    }));
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

// ===== Helper functions =====

impl OpenAICompatConnector {
    /// Fetch the model list from /v1/models.
    async fn fetch_models(&self) -> Result<Vec<ModelInstance>> {
        let resp = self
            .client
            .get(self.models_url())
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| HierKvGatewayError::ConnectorError(format!("failed to fetch model list: {}", e)))?;

        if !resp.status().is_success() {
            return Ok(vec![]);
        }

        let models_resp: ModelsResponse = resp
            .json()
            .await
            .map_err(|e| HierKvGatewayError::ConnectorError(format!("failed to parse model list: {}", e)))?;

        let instances = models_resp
            .data
            .into_iter()
            .map(|m| ModelInstance {
                model_name: m.id,
                model_architecture: "unknown".to_string(),
                quantization: Quantization::Fp16,
                max_context_len: 32768,
                supports_tool_calling: false,
                supports_streaming: true,
            })
            .collect();

        Ok(instances)
    }
}

/// /v1/models response.
#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    id: String,
}

/// Parse Prometheus-format metrics text and extract key metrics into BackendMetrics.
fn parse_prometheus_metrics(text: &str, metrics: &mut BackendMetrics) {
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        if let Some((name, value)) = parse_prom_line(line) {
            match name.as_str() {
                "vllm:num_requests_running" | "hier_kv_gateway:active_requests" => {
                    metrics.active_requests = value as u64;
                }
                "vllm:num_requests_waiting" | "hier_kv_gateway:queue_depth" => {
                    metrics.queue_depth = value as u64;
                }
                "vllm:gpu_cache_usage_perc" | "hier_kv_gateway:kv_cache_usage" => {
                    metrics.kv_used_blocks = (value * 1000.0) as u64;
                    if metrics.kv_total_blocks == 0 {
                        metrics.kv_total_blocks = 1000;
                    }
                }
                "vllm:gpu_utilization" | "hier_kv_gateway:gpu_utilization" => {
                    metrics.gpu_utilization = value;
                }
                _ => {}
            }
        }
    }
}

/// Parse a single Prometheus metric line and return (metric_name, value).
fn parse_prom_line(line: &str) -> Option<(String, f64)> {
    // Format: metric_name{labels} value OR metric_name value
    let space_pos = line.rfind(' ')?;
    let name_part = &line[..space_pos];
    let value_str = &line[space_pos + 1..];

    let value: f64 = value_str.parse().ok()?;

    // Strip labels
    let name = if let Some(brace_pos) = name_part.find('{') {
        &name_part[..brace_pos]
    } else {
        name_part
    };

    Some((name.to_string(), value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_prom_line_simple() {
        let (name, val) = parse_prom_line("vllm:num_requests_running 5").unwrap();
        assert_eq!(name, "vllm:num_requests_running");
        assert!((val - 5.0).abs() < 1e-9);
    }

    #[test]
    fn parse_prom_line_with_labels() {
        let (name, val) =
            parse_prom_line("vllm:gpu_cache_usage_perc{model=\"test\"} 0.75").unwrap();
        assert_eq!(name, "vllm:gpu_cache_usage_perc");
        assert!((val - 0.75).abs() < 1e-9);
    }

    #[test]
    fn parse_prom_metrics_extracts_values() {
        let mut m = BackendMetrics::default();
        let text = r#"
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
vllm:num_requests_waiting 2
vllm:gpu_cache_usage_perc 0.5
"#;
        parse_prometheus_metrics(text, &mut m);
        assert_eq!(m.active_requests, 3);
        assert_eq!(m.queue_depth, 2);
        assert_eq!(m.kv_used_blocks, 500);
    }

    #[test]
    fn chat_completion_request_serialization() {
        let req = InferenceRequest {
            request_id: RequestId::new("r1"),
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            token_ids: vec![],
            max_tokens: 100,
            temperature: 0.7,
            stream: true,
            tools: vec![],
            lora_name: None,
        };
        let body = ChatCompletionRequest::from(&req);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["stream"], true);
        assert_eq!(json["max_tokens"], 100);
    }

    use hier_kv_gateway_core::ids::RequestId;
    use hier_kv_gateway_core::request::ChatMessage;
}
