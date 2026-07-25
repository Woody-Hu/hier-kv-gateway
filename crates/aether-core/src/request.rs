//! 推理请求与响应类型。
//!
//! [`InferenceRequest`] 描述客户端发给网关的请求结构，
//! [`InferenceChunk`] 描述流式响应中可消费的数据块；
//! [`RoutingContext`] 是网关路由层在选路阶段需要使用的上下文。

use serde::{Deserialize, Serialize};

use crate::ids::{BackendId, RequestId, SessionId};

/// 一条聊天消息。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    /// 角色，例如 `system`、`user`、`assistant`。
    pub role: String,
    /// 消息文本内容。
    pub content: String,
}

/// 工具（function calling）定义。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    /// 工具名。
    pub name: String,
    /// 工具描述。
    pub description: String,
    /// 参数 JSON Schema 文本。
    pub parameters_schema: String,
}

/// 推理请求。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// 请求唯一标识。
    pub request_id: RequestId,
    /// 目标模型名。
    pub model: String,
    /// 聊天消息列表。
    pub messages: Vec<ChatMessage>,
    /// 已分词的 token 序列，与 messages 二选一或并行提供。
    #[serde(default)]
    pub token_ids: Vec<u32>,
    /// 最大生成 token 数。
    pub max_tokens: u32,
    /// 采样温度。
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// 是否启用流式输出。
    #[serde(default)]
    pub stream: bool,
    /// 可用工具列表。
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// 可选 LoRA 适配器名。
    #[serde(default)]
    pub lora_name: Option<String>,
}

fn default_temperature() -> f64 {
    1.0
}

/// 流式推理输出的一个数据块。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceChunk {
    /// 增量文本块。
    Delta {
        /// 本次增量文本。
        text: String,
        /// 结束原因，仅在流末块出现。
        finish_reason: Option<String>,
    },
    /// 工具调用块。
    ToolCall {
        /// 工具调用 ID。
        id: String,
        /// 工具函数名。
        function: String,
        /// 调用参数（JSON 文本）。
        args: String,
    },
    /// 流结束块，携带最终路由信息与延迟统计。
    Done {
        /// 实际处理该请求的后端标识。
        backend_id: BackendId,
        /// 端到端延迟（毫秒）。
        latency_ms: u64,
    },
    /// 错误块。
    Error {
        /// 错误码（兼容 HTTP 状态码语义）。
        code: u16,
        /// 错误信息。
        message: String,
    },
}

/// 路由策略对单个候选后端的评分结果。
///
/// `score` 越大表示越优先；`raw_cost` 是该策略对后端的原始代价（越低越好），
/// 便于在混合策略中按代价做归一化加权。`meta_version` 标记本次评分所基于的
/// 元数据版本，可用于缓存失效判断。
#[derive(Clone, Debug)]
pub struct ScoredBackend {
    /// 被评分的后端标识。
    pub backend_id: BackendId,
    /// 综合评分，越大越优先。
    pub score: f64,
    /// 原始代价（越低越好），用于跨策略归一化。
    pub raw_cost: f64,
    /// 评分所基于的元数据版本。
    pub meta_version: u64,
}

/// 路由层选路时使用的上下文。
#[derive(Clone, Debug, Default)]
pub struct RoutingContext {
    /// 请求标识。
    pub request_id: Option<RequestId>,
    /// 会话标识，用于会话亲和。
    pub session_id: Option<SessionId>,
    /// 目标模型名。
    pub model_name: Option<String>,
    /// 已分词的 token 序列。
    pub token_ids: Vec<u32>,
    /// 已知的块哈希列表。
    pub block_hashes: Vec<u64>,
    /// KV 块大小。
    pub block_size: u32,
    /// LoRA 适配器名。
    pub lora_name: Option<String>,
    /// 缓存命名空间。
    pub cache_namespace: Option<String>,
    /// 估算的输出 token 数。
    pub estimated_output_tokens: u32,
    /// 是否需要后端支持工具调用。
    pub requires_tool_calling: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_round_trip() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        };
        let s = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn inference_request_defaults_temperature() {
        let s = r#"{
            "request_id": "r1",
            "model": "m",
            "messages": [],
            "max_tokens": 16,
            "stream": false
        }"#;
        let req: InferenceRequest = serde_json::from_str(s).unwrap();
        assert!((req.temperature - 1.0).abs() < 1e-9);
        assert!(req.token_ids.is_empty());
        assert!(req.tools.is_empty());
        assert!(req.lora_name.is_none());
    }

    #[test]
    fn inference_request_with_tools_and_lora() {
        let s = r#"{
            "request_id": "r2",
            "model": "m",
            "messages": [{"role":"user","content":"hi"}],
            "token_ids": [1,2,3],
            "max_tokens": 32,
            "temperature": 0.3,
            "stream": true,
            "tools": [
                {"name":"t","description":"d","parameters_schema":"{}"}
            ],
            "lora_name": "adapter-a"
        }"#;
        let req: InferenceRequest = serde_json::from_str(s).unwrap();
        assert_eq!(req.token_ids, vec![1, 2, 3]);
        assert!(req.stream);
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.lora_name.as_deref(), Some("adapter-a"));
    }

    #[test]
    fn inference_chunk_delta_tag() {
        let c = InferenceChunk::Delta {
            text: "hi".to_string(),
            finish_reason: Some("stop".to_string()),
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.contains(r#""type":"delta""#));
        let back: InferenceChunk = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn inference_chunk_tool_call_tag() {
        let c = InferenceChunk::ToolCall {
            id: "call-1".to_string(),
            function: "f".to_string(),
            args: "{}".to_string(),
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.contains(r#""type":"tool_call""#));
    }

    #[test]
    fn inference_chunk_done_tag() {
        let c = InferenceChunk::Done {
            backend_id: BackendId::new("us-east-1", "worker-0"),
            latency_ms: 42,
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.contains(r#""type":"done""#));
        let back: InferenceChunk = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn inference_chunk_error_tag() {
        let c = InferenceChunk::Error {
            code: 503,
            message: "backend gone".to_string(),
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.contains(r#""type":"error""#));
    }

    #[test]
    fn routing_context_default_empty() {
        let rc = RoutingContext::default();
        assert!(rc.token_ids.is_empty());
        assert_eq!(rc.block_size, 0);
        assert!(!rc.requires_tool_calling);
    }
}
