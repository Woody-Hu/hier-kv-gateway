//! OpenAI 兼容连接器实现。
//!
//! 适用于 vLLM、llama.cpp 及其他兼容 OpenAI Chat Completions API 的推理引擎。
//! 通过 HTTP POST `/v1/chat/completions`（`stream: true`）转发请求，
//! 并解析 SSE（Server-Sent Events）流返回 [`InferenceChunk`] 序列。

use std::sync::Arc;
use std::time::Instant;

use aether_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
    ModelInstance, Protocol, Quantization,
};
use aether_core::error::{AetherError, Result};
use aether_core::ids::{BackendId, BackendInstanceId, IndexerDomainId, RegionId};
use aether_core::kv_event::KvCacheEvent;
use aether_core::metrics::{BackendMetrics, LatencyStats};
use aether_core::request::{InferenceChunk, InferenceRequest};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::connector::{BackendConnector, HealthStatus};

/// OpenAI 兼容连接器。
///
/// 适用于 vLLM、llama.cpp（server 模式）及任何兼容 OpenAI API 的推理服务。
/// 通过 HTTP 通信，支持流式 SSE 解析。
pub struct OpenAICompatConnector {
    /// HTTP 客户端（可复用连接池）。
    client: reqwest::Client,
    /// 后端基础 URL，例如 `http://10.0.0.1:8000`。
    base_url: String,
    /// 后端类型（区分 vLLM / llama.cpp / 通用）。
    backend_type: BackendType,
    /// 所在区域。
    region: RegionId,
    /// 后端实例标识。
    instance_id: BackendInstanceId,
    /// 该后端承载的模型列表（从配置注入，discover 时返回）。
    models: Vec<String>,
    /// KV 块大小（用于构造 BackendInfo）。
    kv_block_size: u32,
}

impl OpenAICompatConnector {
    /// 创建一个新的 OpenAI 兼容连接器。
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
                .expect("reqwest client 构建失败"),
            base_url: base_url.into(),
            backend_type,
            region,
            instance_id: instance_id.into(),
            models,
            kv_block_size,
        }
    }

    /// 从端点 URL 和配置构造连接器。
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
            // 用 URL 的 host:port 作为实例标识
            endpoint.url.replace("http://", "").replace("https://", ""),
            models,
            kv_block_size,
        )
    }

    /// 构造该连接器对应的 BackendId。
    fn backend_id(&self) -> BackendId {
        BackendId::new(self.region.clone(), self.instance_id.clone())
    }

    /// 健康检查 URL。
    fn health_url(&self) -> String {
        format!("{}/health", self.base_url)
    }

    /// 模型列表 URL。
    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url)
    }

    /// Chat Completions URL。
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
        // 尝试从 /v1/models 获取模型列表
        let models = self.fetch_models().await.unwrap_or_default();

        let model_instances: Vec<ModelInstance> = if models.is_empty() {
            // 若 /v1/models 不可用，使用配置注入的模型名
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
                tracing::warn!("健康检查返回非 200 状态: {}", r.status());
                Ok(HealthStatus {
                    status: BackendStatus::Unhealthy,
                    healthy_since_unix: now,
                    error_count: 1,
                })
            }
            Err(e) => {
                tracing::warn!("健康检查失败: {}", e);
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

        // 构造 OpenAI Chat Completions 请求体
        let body = ChatCompletionRequest::from(request);
        let body_json = serde_json::to_value(&body)
            .map_err(|e| AetherError::ConnectorError(format!("请求序列化失败: {}", e)))?;

        let resp = self
            .client
            .post(self.completions_url())
            .json(&body_json)
            .send()
            .await
            .map_err(|e| AetherError::ConnectorError(format!("请求转发失败: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(AetherError::ConnectorError(format!(
                "后端返回 HTTP {}: {}",
                status, text
            )));
        }

        // 获取字节流并解析 SSE
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
        Err(AetherError::ConnectorError(
            "OpenAI 兼容连接器不支持 KV 缓存事件".to_string(),
        ))
    }

    async fn collect_metrics(&self, _backend: &BackendId) -> Result<BackendMetrics> {
        // 尝试获取 /metrics 端点（vLLM 支持）
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

/// vLLM 连接器（继承 OpenAI 兼容，可扩展 KV 事件支持）。
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
        false // 后续可扩展为 true 并实现 subscribe_kv_events
    }
    async fn subscribe_kv_events(
        &self,
        _backend: &BackendId,
    ) -> Result<BoxStream<'static, KvCacheEvent>> {
        Err(AetherError::ConnectorError(
            "vLLM 连接器暂不支持 KV 缓存事件".to_string(),
        ))
    }
    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics> {
        self.0.collect_metrics(backend).await
    }
}

// ===== OpenAI API 请求/响应结构 =====

/// OpenAI Chat Completions 请求体。
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
            stream: true, // 始终使用流式
            tools,
        }
    }
}

/// OpenAI 流式响应的一个 SSE 数据块。
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

// ===== SSE 解析器 =====

/// SSE 流解析器，将 HTTP 字节流转换为 InferenceChunk 序列。
///
/// 解析规则：
/// - 以 `data: ` 前缀的行是数据行
/// - `data: [DONE]` 表示流结束
/// - 其他行（空行、事件行）忽略
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

    /// 从缓冲区中提取并解析完整的 SSE 事件。
    ///
    /// 返回 Some(chunk) 表示产出一个数据块，None 表示需要更多数据。
    fn try_parse(&mut self) -> Option<InferenceChunk> {
        loop {
            // 查找换行符分隔的行
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

                // 解析 JSON
                match serde_json::from_str::<ChatCompletionChunk>(data) {
                    Ok(chunk) => {
                        for choice in &chunk.choices {
                            // 工具调用
                            for tc in &choice.delta.tool_calls {
                                if let (Some(id), Some(func)) = (&tc.id, &tc.function) {
                                    return Some(InferenceChunk::ToolCall {
                                        id: id.clone(),
                                        function: func.name.clone().unwrap_or_default(),
                                        args: func.arguments.clone().unwrap_or_default(),
                                    });
                                }
                            }

                            // 文本内容
                            if let Some(content) = &choice.delta.content {
                                if !content.is_empty() {
                                    return Some(InferenceChunk::Delta {
                                        text: content.clone(),
                                        finish_reason: choice.finish_reason.clone(),
                                    });
                                }
                            }

                            // finish_reason 存在但无 content（流末）
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
                        tracing::trace!("SSE JSON 解析失败（跳过）: {} - data: {}", e, data);
                    }
                }
            }
            // 非 data: 行忽略
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
            // 先尝试从缓冲区解析已有数据
            if let Some(chunk) = this.try_parse() {
                return std::task::Poll::Ready(Some(chunk));
            }

            // 从底层流拉取更多数据
            use futures::StreamExt;
            match this.inner.poll_next_unpin(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&bytes));
                    // 继续循环尝试解析
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    tracing::warn!("SSE 流读取错误: {}", e);
                    return std::task::Poll::Ready(Some(InferenceChunk::Error {
                        code: 502,
                        message: format!("流读取错误: {}", e),
                    }));
                }
                std::task::Poll::Ready(None) => {
                    // 流结束，但未收到 [DONE]
                    if this.buffer.is_empty() {
                        let latency = this.start.elapsed().as_millis() as u64;
                        return std::task::Poll::Ready(Some(InferenceChunk::Done {
                            backend_id: this.backend_id.clone(),
                            latency_ms: latency,
                        }));
                    }
                    // 尝试解析剩余缓冲区
                    if let Some(chunk) = this.try_parse() {
                        return std::task::Poll::Ready(Some(chunk));
                    }
                    // 真正结束
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

// ===== 辅助函数 =====

impl OpenAICompatConnector {
    /// 从 /v1/models 获取模型列表。
    async fn fetch_models(&self) -> Result<Vec<ModelInstance>> {
        let resp = self
            .client
            .get(self.models_url())
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| AetherError::ConnectorError(format!("获取模型列表失败: {}", e)))?;

        if !resp.status().is_success() {
            return Ok(vec![]);
        }

        let models_resp: ModelsResponse = resp
            .json()
            .await
            .map_err(|e| AetherError::ConnectorError(format!("解析模型列表失败: {}", e)))?;

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

/// /v1/models 响应。
#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    id: String,
}

/// 解析 Prometheus 格式的 metrics 文本，提取关键指标到 BackendMetrics。
fn parse_prometheus_metrics(text: &str, metrics: &mut BackendMetrics) {
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        if let Some((name, value)) = parse_prom_line(line) {
            match name.as_str() {
                "vllm:num_requests_running" | "aether:active_requests" => {
                    metrics.active_requests = value as u64;
                }
                "vllm:num_requests_waiting" | "aether:queue_depth" => {
                    metrics.queue_depth = value as u64;
                }
                "vllm:gpu_cache_usage_perc" | "aether:kv_cache_usage" => {
                    metrics.kv_used_blocks = (value * 1000.0) as u64;
                    if metrics.kv_total_blocks == 0 {
                        metrics.kv_total_blocks = 1000;
                    }
                }
                "vllm:gpu_utilization" | "aether:gpu_utilization" => {
                    metrics.gpu_utilization = value;
                }
                _ => {}
            }
        }
    }
}

/// 解析单行 Prometheus metric，返回 (metric_name, value)。
fn parse_prom_line(line: &str) -> Option<(String, f64)> {
    // 格式: metric_name{labels} value 或 metric_name value
    let space_pos = line.rfind(' ')?;
    let name_part = &line[..space_pos];
    let value_str = &line[space_pos + 1..];

    let value: f64 = value_str.parse().ok()?;

    // 去除 labels
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

    use aether_core::ids::RequestId;
    use aether_core::request::ChatMessage;
}
