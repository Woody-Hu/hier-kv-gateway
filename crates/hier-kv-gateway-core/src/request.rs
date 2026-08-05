//! Inference request and response types.
//!
//! [`InferenceRequest`] describes the request structure sent by clients to the gateway,
//! [`InferenceChunk`] describes consumable data chunks in a streaming response,
//! and [`RoutingContext`] is the context used by the gateway routing layer during path selection.

use serde::{Deserialize, Serialize};

use crate::ids::{BackendId, RequestId, SessionId, TenantId};

/// A chat message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    /// Role, e.g. `system`, `user`, `assistant`.
    pub role: String,
    /// Message text content.
    pub content: String,
}

/// Tool (function calling) definition.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Parameter JSON Schema text.
    pub parameters_schema: String,
}

/// Inference request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request identifier.
    pub request_id: RequestId,
    /// Target model name.
    pub model: String,
    /// List of chat messages.
    pub messages: Vec<ChatMessage>,
    /// Tokenized token sequence; either provided alongside `messages` or instead of them.
    #[serde(default)]
    pub token_ids: Vec<u32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Whether streaming output is enabled.
    #[serde(default)]
    pub stream: bool,
    /// List of available tools.
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// Optional LoRA adapter name.
    #[serde(default)]
    pub lora_name: Option<String>,
}

fn default_temperature() -> f64 {
    1.0
}

/// A data chunk in streaming inference output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceChunk {
    /// Incremental text chunk.
    Delta {
        /// Incremental text for this chunk.
        text: String,
        /// Finish reason, only present on the final stream chunk.
        finish_reason: Option<String>,
    },
    /// Tool call chunk.
    ToolCall {
        /// Tool call ID.
        id: String,
        /// Tool function name.
        function: String,
        /// Call arguments (JSON text).
        args: String,
    },
    /// Stream-end chunk, carrying final routing info and latency statistics.
    Done {
        /// Identifier of the backend that actually handled the request.
        backend_id: BackendId,
        /// End-to-end latency (milliseconds).
        latency_ms: u64,
    },
    /// Error chunk.
    Error {
        /// Error code (compatible with HTTP status code semantics).
        code: u16,
        /// Error message.
        message: String,
    },
}

/// Scoring result of a routing strategy for a single candidate backend.
///
/// A higher `score` means higher priority; `raw_cost` is the strategy's raw cost
/// for the backend (lower is better), useful for normalized weighting across
/// strategies in hybrid routing. `meta_version` records the metadata version the
/// score was based on, useful for cache invalidation.
#[derive(Clone, Debug)]
pub struct ScoredBackend {
    /// Identifier of the backend being scored.
    pub backend_id: BackendId,
    /// Combined score; higher is preferred.
    pub score: f64,
    /// Raw cost (lower is better), used for cross-strategy normalization.
    pub raw_cost: f64,
    /// Metadata version the score is based on.
    pub meta_version: u64,
}

/// Context used by the routing layer during path selection.
#[derive(Clone, Debug, Default)]
pub struct RoutingContext {
    /// Request identifier.
    pub request_id: Option<RequestId>,
    /// Session identifier, used for session affinity.
    pub session_id: Option<SessionId>,
    /// Tenant identifier, used for multi-tenant admission control.
    pub tenant_id: Option<TenantId>,
    /// Target model name.
    pub model_name: Option<String>,
    /// Tokenized token sequence.
    pub token_ids: Vec<u32>,
    /// Known block hash list.
    pub block_hashes: Vec<u64>,
    /// KV block size.
    pub block_size: u32,
    /// LoRA adapter name.
    pub lora_name: Option<String>,
    /// Cache namespace.
    pub cache_namespace: Option<String>,
    /// Estimated number of output tokens.
    pub estimated_output_tokens: u32,
    /// Whether the backend must support tool calling.
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
