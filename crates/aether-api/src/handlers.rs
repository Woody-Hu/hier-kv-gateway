//! HTTP 路由处理函数。
//!
//! 每个 handler 是一个 axum 兼容的 async 函数，从 [`AppState`] 与 HTTP 请求中
//! 获取输入，调用 routing/connector 完成实际工作后返回 JSON 或 SSE 响应。
//!
//! 路由决策信息（选中后端、策略名、KV 重叠）会通过响应头
//! `X-Aether-Backend` / `X-Aether-Strategy` / `X-Aether-KV-Overlap` 暴露给客户端，
//! 便于排查与可观测。

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::stream::{BoxStream, StreamExt};
use serde_json::json;
use tracing::{debug, error, warn};

use aether_core::backend::BackendInfo;
use aether_core::config::RoutingConfig;
use aether_core::error::AetherError;
use aether_core::ids::{BackendId, RequestId, SessionId};
use aether_core::kv_event::{compute_block_hashes, BlockHashInput};
use aether_core::metrics::BackendMetrics;
use aether_core::request::{InferenceChunk, InferenceRequest, RoutingContext};

use aether_connector::registry::ConnectorRegistry;
use aether_metadata::store::MetadataStore;
use aether_routing::engine::{RouteDecision, RoutingEngine};

use crate::openai_types::{
    OpenAIChatChunk, OpenAIChatRequest, OpenAIChatResponse, OpenAIModelList,
};

/// HTTP handler 共享的应用状态。
pub struct AppState {
    /// 元数据存储（KV 索引、模型注册表、负载统计等）。
    pub metadata: Arc<MetadataStore>,
    /// 路由引擎。
    pub routing: Arc<RoutingEngine>,
    /// 连接器注册表，按 BackendType 索引。
    pub connectors: Arc<ConnectorRegistry>,
    /// 路由配置（提供 kv_block_size 等参数）。
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

/// 由路由决策派生出的、需要在响应中携带的元信息。
struct RoutingMeta {
    /// 选中的后端标识字符串。
    backend: String,
    /// 触发决策的策略名。
    strategy: String,
    /// KV 重叠长度。
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

    /// 把这些元信息写入响应头。
    fn apply_to_headers(&self, headers: &mut HeaderMap) {
        if let Ok(v) = HeaderValue::from_str(&self.backend) {
            headers.insert("X-Aether-Backend", v);
        }
        if let Ok(v) = HeaderValue::from_str(&self.strategy) {
            headers.insert("X-Aether-Strategy", v);
        }
        let overlap = self.kv_overlap.to_string();
        if let Ok(v) = HeaderValue::from_str(&overlap) {
            headers.insert("X-Aether-KV-Overlap", v);
        }
    }
}

/// `POST /v1/chat/completions`
///
/// 处理 OpenAI 兼容的 Chat Completions 请求，根据 `stream` 字段返回流式 SSE
/// 或非流式 JSON。响应头会携带路由决策信息。
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAIChatRequest>,
) -> Response {
    let stream_mode = req.stream;
    let session_id = req.session.as_ref().map(SessionId::new);
    let model_name = req.model.clone();

    // 1) 转换为内部 InferenceRequest
    let inference: InferenceRequest = req.to_inference_request();
    let request_id = inference.request_id.clone();

    // 2) 构造 RoutingContext，必要时计算 block_hashes
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

    // 3) 路由决策
    let decision = match state.routing.route(&ctx, &state.metadata).await {
        Ok(d) => d,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "路由失败");
            return error_response(StatusCode::SERVICE_UNAVAILABLE, &e);
        }
    };
    let routing_meta = RoutingMeta::from_decision(&decision);
    debug!(
        request_id = %request_id,
        backend = %routing_meta.backend,
        strategy = %routing_meta.strategy,
        kv_overlap = routing_meta.kv_overlap,
        "路由决策完成"
    );

    // 4) 取出连接器
    let backend_info = match state.metadata.backend_get(&decision.backend) {
        Some(info) => info,
        None => {
            error!(backend = %decision.backend, "路由选中的后端不在 MetadataStore 中");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &AetherError::NotFound(format!("backend {} not registered", decision.backend)),
            );
        }
    };
    let backend_type = backend_info.backend_type.clone();
    let connector = match state.connectors.get(&backend_type) {
        Some(c) => c,
        None => {
            error!(backend_type = ?backend_type, "未注册该后端类型的连接器");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &AetherError::ConnectorError(format!(
                    "no connector for backend_type {:?}",
                    backend_type
                )),
            );
        }
    };

    // 5) 转发请求并取得 chunk 流
    let chunk_stream: BoxStream<'static, InferenceChunk> =
        match connector.forward(&decision.backend, &inference).await {
            Ok(s) => s,
            Err(e) => {
                error!(request_id = %request_id, error = %e, "后端转发失败");
                return error_response(StatusCode::BAD_GATEWAY, &e);
            }
        };

    // 6) 根据 stream 字段构造响应
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

/// 把一个 [`AetherError`] 转换为 HTTP JSON 错误响应。
fn error_response(status: StatusCode, err: &AetherError) -> Response {
    let body = Json(json!({
        "error": {
            "message": err.to_string(),
            "type": error_type_name(err),
        }
    }));
    (status, body).into_response()
}

/// 返回 AetherError 的简短分类名，便于客户端区分错误种类。
fn error_type_name(err: &AetherError) -> &'static str {
    match err {
        AetherError::BackendUnavailable => "backend_unavailable",
        AetherError::RoutingFailed(_) => "routing_failed",
        AetherError::ConnectorError(_) => "connector_error",
        AetherError::MetricsError(_) => "metrics_error",
        AetherError::ConfigError(_) => "config_error",
        AetherError::ClusterError(_) => "cluster_error",
        AetherError::NotFound(_) => "not_found",
        AetherError::RateLimited => "rate_limited",
        AetherError::Internal(_) => "internal_error",
    }
}

/// 构造流式 SSE 响应。
///
/// 输出格式遵循 OpenAI 约定：
/// - 起始 chunk 携带 `role: "assistant"`；
/// - 后续每个文本增量作为一个 `data: {json}\n\n` 事件；
/// - 流末携带 `finish_reason` 的 chunk；
/// - 最后发送 `data: [DONE]\n\n`。
fn build_sse_response(
    chunk_stream: BoxStream<'static, InferenceChunk>,
    request_id: &RequestId,
    model: &str,
    routing_meta: &RoutingMeta,
) -> Response {
    // 各 chunk 共享同一个 (rid, model) 副本，因此提前 clone。
    let rid_for_first = request_id.as_str().to_string();
    let model_for_first = model.to_string();
    let rid_for_deltas = rid_for_first.clone();
    let model_for_deltas = model_for_first.clone();
    let rid_for_finish = rid_for_first.clone();
    let model_for_finish = model_for_first.clone();

    // 起始 chunk
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

/// 把一个 [`InferenceChunk`] 转换为 [`OpenAIChatChunk`]。
///
/// - `Delta`：携带文本时返回 `delta_chunk`；带 `finish_reason` 时返回 `finish_chunk`；
///   文本为空但带 `finish_reason` 时也返回 `finish_chunk`；
/// - `ToolCall`：当前简化为忽略（OpenAIChatChunk 暂未承载 tool_calls delta），仅记录日志；
/// - `Done`：返回 `finish_chunk("stop")`，作为流末信号（由调用方的链尾再补 `[DONE]`）；
/// - `Error`：返回 `finish_chunk("error")` 并记录日志。
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
                    // 文本与 finish_reason 同时出现时，先发 delta，再发 finish。
                    // 由于一个 chunk 只能产出一条 OpenAIChatChunk，这里优先发 finish。
                    OpenAIChatChunk::finish_chunk(rid, model, &reason)
                }
            } else if !text.is_empty() {
                OpenAIChatChunk::delta_chunk(rid, model, text)
            } else {
                // 空内容且无 finish_reason：返回空 delta chunk 以维持流活跃
                OpenAIChatChunk::delta_chunk(rid, model, String::new())
            }
        }
        InferenceChunk::ToolCall {
            id: _,
            function,
            args: _,
        } => {
            warn!(function = %function, "ToolCall chunk 当前未在 SSE 中输出");
            OpenAIChatChunk::delta_chunk(rid, model, String::new())
        }
        InferenceChunk::Done {
            backend_id: _,
            latency_ms: _,
        } => OpenAIChatChunk::finish_chunk(rid, model, "stop"),
        InferenceChunk::Error { code, message } => {
            warn!(code, %message, "后端返回错误 chunk");
            OpenAIChatChunk::finish_chunk(rid, model, "error")
        }
    }
}

/// 构造非流式响应：把 chunk 流消费完，合并所有文本，返回完整 JSON。
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
                    // 粗略估算：一个增量文本按字符数估算 token 数（无 tokenizer 时的兜底）
                    completion_tokens += approx_token_count(&text);
                }
                if let Some(reason) = fr {
                    finish_reason = Some(reason);
                }
            }
            InferenceChunk::ToolCall { .. } => {
                // 非流式聚合时暂时不处理 tool_calls
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
        let err = AetherError::ConnectorError(format!("backend error: {}", message));
        return error_response(status, &err);
    }

    // 估算 prompt_tokens：使用消息总字符数 / 4 的粗略比例
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

/// 用一个粗略的字符数→token 数估算。
///
/// 中文按 1 字 ≈ 1 token，英文按 4 字符 ≈ 1 token 的混合近似。
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
/// 列出所有已注册后端承载的模型，按 `(model_name, backend_id)` 去重后返回。
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<OpenAIModelList> {
    let backends: Vec<BackendInfo> = state.metadata.backends_all();
    // 用 HashSet 去重模型名，保留首个承载该模型的后端作为 owned_by。
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
/// 简单健康检查，返回 `{"status":"ok"}`。
pub async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

/// `GET /admin/backends`
///
/// 返回所有已注册后端的 [`BackendInfo`] 列表。
pub async fn admin_backends(State(state): State<Arc<AppState>>) -> Json<Vec<BackendInfo>> {
    Json(state.metadata.backends_all())
}

/// `GET /admin/backends/:id/metrics`
///
/// 查询指定后端的负载指标。`id` 路径参数格式为 `region/instance`。
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

/// 把 `region/instance` 格式的字符串解析为 [`BackendId`]。
///
/// 仅按第一个 `/` 切分；`instance` 部分允许包含后续 `/`，但通常不含。
fn parse_backend_id(s: &str) -> Option<BackendId> {
    let slash = s.find('/')?;
    let region = &s[..slash];
    let instance = &s[slash + 1..];
    if region.is_empty() || instance.is_empty() {
        return None;
    }
    Some(BackendId::new(region, instance))
}

/// 用于在测试中构造一个最小的 AppState。
///
/// 仅在 `cfg(test)` 下编译，避免在生产代码中暴露不必要的依赖。
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use aether_core::config::{StrategyType, StrategyWeights};
    use aether_routing::hybrid::HybridStrategy;
    use aether_routing::kv_aware::KvAwareStrategy;
    use aether_routing::load_aware::LoadAwareStrategy;
    use aether_routing::model_aware::ModelAwareStrategy;
    use aether_routing::topology_aware::TopologyAwareStrategy;
    use std::time::Duration;

    /// 构造一个使用默认混合策略的 AppState，仅用于单元测试。
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
        // "hello" 5 字符 → 2 tokens
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
        assert_eq!(headers.get("X-Aether-Backend").unwrap(), "r1/i1");
        assert_eq!(headers.get("X-Aether-Strategy").unwrap(), "hybrid");
        assert_eq!(headers.get("X-Aether-KV-Overlap").unwrap(), "7");
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
