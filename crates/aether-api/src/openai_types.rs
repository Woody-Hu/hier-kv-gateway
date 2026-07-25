//! OpenAI API 兼容的请求/响应 JSON 类型。
//!
//! 本模块定义了与 OpenAI Chat Completions API 一致的外部 JSON 协议，
//! 并提供与 Aether 内部 [`aether_core::request::InferenceRequest`] 之间的双向转换：
//!
//! - [`OpenAIChatRequest`]：客户端 POST `/v1/chat/completions` 的请求体。
//! - [`OpenAIChatResponse`]：非流式响应体，与 OpenAI API 一致。
//! - [`OpenAIChatChunk`]：流式 SSE 响应中的单条 chunk。
//! - [`OpenAIModelList`]：GET `/v1/models` 的响应体。
//!
//! 转换函数 [`OpenAIChatRequest::to_inference_request`] 会生成新的 [`RequestId`]，
//! 并把 OpenAI 风格的 `tools`（嵌套 `function` 字段）展平为 Aether 内部使用的
//! [`ToolDefinition`]。

use serde::{Deserialize, Serialize};

use aether_core::ids::RequestId;
use aether_core::request::{ChatMessage, InferenceRequest, ToolDefinition};

/// 默认采样温度，与 OpenAI 一致。
const DEFAULT_TEMPERATURE: f64 = 1.0;
/// 默认最大生成 token 数。
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// OpenAI Chat Completions 请求体。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIChatRequest {
    /// 目标模型名。
    pub model: String,
    /// 聊天消息列表。
    pub messages: Vec<OpenAIMessage>,
    /// 最大生成 token 数，缺省时取 [`DEFAULT_MAX_TOKENS`]。
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// 采样温度，缺省时取 [`DEFAULT_TEMPERATURE`]。
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// 是否启用流式输出。
    #[serde(default)]
    pub stream: bool,
    /// 可用工具列表（OpenAI 风格的 `function` 工具）。
    #[serde(default)]
    pub tools: Vec<OpenAITool>,
    /// 可选 LoRA 适配器名（Aether 扩展字段）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_name: Option<String>,
    /// 可选会话 ID，用于会话亲和路由（Aether 扩展字段）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    /// 可选的已分词 token 序列（Aether 扩展字段，用于 KV 路由）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<u32>,
}

fn default_temperature() -> f64 {
    DEFAULT_TEMPERATURE
}

fn default_max_tokens() -> u32 {
    DEFAULT_MAX_TOKENS
}

/// OpenAI 风格的聊天消息。
///
/// `content` 仅支持纯文本，与 Aether 内部 [`ChatMessage`] 一致；
/// 多模态内容（图像等）暂不支持。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAIMessage {
    /// 角色，例如 `system`、`user`、`assistant`。
    pub role: String,
    /// 消息文本内容。
    pub content: String,
}

/// OpenAI 风格的工具定义。
///
/// 序列化后形如：
/// ```json
/// { "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAITool {
    /// 工具类型，目前仅支持 `function`。
    #[serde(rename = "type")]
    pub tool_type: String,
    /// 函数定义。
    pub function: OpenAIToolFunction,
}

/// OpenAI 风格的工具函数定义。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIToolFunction {
    /// 函数名。
    pub name: String,
    /// 函数描述。
    #[serde(default)]
    pub description: String,
    /// 参数 JSON Schema（任意 JSON 值）。
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// OpenAI Chat Completions 非流式响应体。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIChatResponse {
    /// 响应唯一 ID（与请求 `id` 一致或服务端生成）。
    pub id: String,
    /// 对象类型，固定为 `"chat.completion"`。
    pub object: String,
    /// 创建时间戳（Unix 秒）。
    pub created: i64,
    /// 生成该响应所用的模型名。
    pub model: String,
    /// 候选列表（非流式通常只有一项）。
    pub choices: Vec<OpenAIChoice>,
    /// 用量统计。
    pub usage: OpenAIUsage,
}

/// 非流式响应中的单个候选。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIChoice {
    /// 候选序号。
    pub index: u32,
    /// 候选消息。
    pub message: OpenAIChoiceMessage,
    /// 结束原因，例如 `stop`、`length`、`tool_calls`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// 非流式响应中的消息。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIChoiceMessage {
    /// 角色，固定为 `assistant`。
    pub role: String,
    /// 文本内容。
    #[serde(default)]
    pub content: String,
}

/// Token 用量统计。
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpenAIUsage {
    /// 提示 token 数。
    pub prompt_tokens: u64,
    /// 生成 token 数。
    pub completion_tokens: u64,
    /// 总 token 数。
    pub total_tokens: u64,
}

/// OpenAI Chat Completions 流式响应中的单个 SSE chunk。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIChatChunk {
    /// 响应唯一 ID。
    pub id: String,
    /// 对象类型，固定为 `"chat.completion.chunk"`。
    pub object: String,
    /// 创建时间戳（Unix 秒）。
    pub created: i64,
    /// 生成该 chunk 所用的模型名。
    pub model: String,
    /// 候选列表（流式通常只有一项）。
    pub choices: Vec<OpenAIChunkChoice>,
}

/// 流式 chunk 中的单个候选。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIChunkChoice {
    /// 候选序号。
    pub index: u32,
    /// 增量 delta。
    pub delta: OpenAIDelta,
    /// 结束原因，仅在流末块出现。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// 流式 chunk 中的增量内容。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIDelta {
    /// 角色，仅首个 chunk 携带。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// 增量文本。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// GET `/v1/models` 响应体。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIModelList {
    /// 对象类型，固定为 `"list"`。
    pub object: String,
    /// 模型条目列表。
    pub data: Vec<OpenAIModel>,
}

/// 单个模型条目。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIModel {
    /// 模型 ID。
    pub id: String,
    /// 对象类型，固定为 `"model"`。
    pub object: String,
    /// 创建时间戳（Unix 秒）。
    pub created: i64,
    /// 模型归属（OpenAI 兼容字段，Aether 中填后端标识）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
}

impl OpenAIChatRequest {
    /// 将 OpenAI 风格请求转换为 Aether 内部 [`InferenceRequest`]。
    ///
    /// - 自动生成新的 [`RequestId`]（基于 UUID v4）。
    /// - 将 [`OpenAIMessage`] 映射为 [`ChatMessage`]。
    /// - 将 [`OpenAITool`] 展平为 [`ToolDefinition`]，`parameters` 字段以 JSON 文本形式存储。
    pub fn to_inference_request(&self) -> InferenceRequest {
        let request_id = RequestId::new(uuid::Uuid::new_v4().to_string());
        let messages: Vec<ChatMessage> = self
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        let tools: Vec<ToolDefinition> = self
            .tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                parameters_schema: serde_json::to_string(&t.function.parameters)
                    .unwrap_or_else(|_| "{}".to_string()),
            })
            .collect();
        InferenceRequest {
            request_id,
            model: self.model.clone(),
            messages,
            token_ids: self.token_ids.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream,
            tools,
            lora_name: self.lora_name.clone(),
        }
    }
}

impl OpenAIChatResponse {
    /// 用请求 ID、模型名、合并后的文本内容与结束原因构造非流式响应。
    ///
    /// `prompt_tokens` / `completion_tokens` 由调用方提供；`total_tokens` 自动求和。
    pub fn from_text(
        request_id: &str,
        model: &str,
        content: String,
        finish_reason: Option<String>,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) -> Self {
        let created = chrono::Utc::now().timestamp();
        Self {
            id: request_id.to_string(),
            object: "chat.completion".to_string(),
            created,
            model: model.to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIChoiceMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason,
            }],
            usage: OpenAIUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }
}

impl OpenAIChatChunk {
    /// 构造流式起始 chunk（携带 `role: "assistant"`）。
    pub fn role_chunk(request_id: &str, model: &str) -> Self {
        Self {
            id: request_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![OpenAIChunkChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        }
    }

    /// 构造携带增量文本的 chunk。
    pub fn delta_chunk(request_id: &str, model: &str, content: String) -> Self {
        Self {
            id: request_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![OpenAIChunkChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: Some(content),
                },
                finish_reason: None,
            }],
        }
    }

    /// 构造流末 chunk（携带 `finish_reason`）。
    pub fn finish_chunk(request_id: &str, model: &str, finish_reason: &str) -> Self {
        Self {
            id: request_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![OpenAIChunkChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
        }
    }
}

impl OpenAIModelList {
    /// 用一组模型 ID 构造模型列表响应。
    pub fn from_model_ids<I>(into_iter: I) -> Self
    where
        I: IntoIterator<Item = (String, Option<String>)>,
    {
        let created = chrono::Utc::now().timestamp();
        let data: Vec<OpenAIModel> = into_iter
            .into_iter()
            .map(|(id, owned_by)| OpenAIModel {
                id,
                object: "model".to_string(),
                created,
                owned_by,
            })
            .collect();
        Self {
            object: "list".to_string(),
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_openai_chat_request_minimal() {
        let s = r#"{
            "model": "gpt-4",
            "messages": [{"role":"user","content":"hi"}]
        }"#;
        let req: OpenAIChatRequest = serde_json::from_str(s).unwrap();
        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream);
        assert!((req.temperature - 1.0).abs() < 1e-9);
        assert_eq!(req.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn parse_openai_chat_request_full() {
        let s = r#"{
            "model": "qwen2.5-7b",
            "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 256,
            "temperature": 0.3,
            "stream": true,
            "tools": [
                {"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}
            ],
            "lora_name": "adapter-a",
            "session": "sess-1",
            "token_ids": [1,2,3]
        }"#;
        let req: OpenAIChatRequest = serde_json::from_str(s).unwrap();
        assert!(req.stream);
        assert!((req.temperature - 0.3).abs() < 1e-9);
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.lora_name.as_deref(), Some("adapter-a"));
        assert_eq!(req.token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn to_inference_request_maps_fields() {
        let req = OpenAIChatRequest {
            model: "m".to_string(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            max_tokens: 64,
            temperature: 0.5,
            stream: true,
            tools: vec![OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIToolFunction {
                    name: "f".to_string(),
                    description: "d".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }],
            lora_name: Some("adapter".to_string()),
            session: Some("sess".to_string()),
            token_ids: vec![1, 2],
        };
        let infer = req.to_inference_request();
        assert_eq!(infer.model, "m");
        assert_eq!(infer.messages.len(), 1);
        assert_eq!(infer.max_tokens, 64);
        assert!((infer.temperature - 0.5).abs() < 1e-9);
        assert!(infer.stream);
        assert_eq!(infer.tools.len(), 1);
        assert_eq!(infer.tools[0].name, "f");
        assert_eq!(infer.lora_name.as_deref(), Some("adapter"));
        assert_eq!(infer.token_ids, vec![1, 2]);
        // request_id 应为 UUID 形式的非空字符串
        assert!(!infer.request_id.as_str().is_empty());
    }

    #[test]
    fn openai_chat_response_serializes() {
        let resp = OpenAIChatResponse::from_text(
            "req-1",
            "m",
            "hello".to_string(),
            Some("stop".to_string()),
            5,
            3,
        );
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("\"object\":\"chat.completion\""));
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(s.contains("\"content\":\"hello\""));
        assert!(s.contains("\"finish_reason\":\"stop\""));
        assert!(s.contains("\"total_tokens\":8"));
    }

    #[test]
    fn openai_chat_chunk_role_chunk() {
        let chunk = OpenAIChatChunk::role_chunk("req-1", "m");
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn openai_chat_chunk_delta_chunk() {
        let chunk = OpenAIChatChunk::delta_chunk("req-1", "m", "hi".to_string());
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hi"));
        assert!(chunk.choices[0].delta.role.is_none());
    }

    #[test]
    fn openai_chat_chunk_finish_chunk() {
        let chunk = OpenAIChatChunk::finish_chunk("req-1", "m", "stop");
        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("stop")
        );
    }

    #[test]
    fn openai_model_list_from_ids() {
        let list = OpenAIModelList::from_model_ids(
            [("m1".to_string(), None), ("m2".to_string(), Some("backend-1".to_string()))],
        );
        assert_eq!(list.object, "list");
        assert_eq!(list.data.len(), 2);
        assert_eq!(list.data[0].id, "m1");
        assert_eq!(list.data[1].owned_by.as_deref(), Some("backend-1"));
    }
}
