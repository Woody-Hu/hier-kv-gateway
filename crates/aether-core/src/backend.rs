//! 后端信息模型。
//!
//! 描述一个推理后端实例的静态元数据：身份、类型、连接端点、模型、区域、
//! 索引域、能力、KV 缓存配置与运行状态。这些字段由连接器在握手阶段填充，
//! 并被路由层、集群层与 API 层读取使用。

use serde::{Deserialize, Serialize};

use crate::ids::{BackendId, IndexerDomainId, RegionId};

/// 后端类型枚举，决定了该后端使用何种引擎/协议。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// Dynamo 集群后端。
    DynamoCluster,
    /// LLM-D 集群后端。
    LlmDCluster,
    /// vLLM 引擎后端。
    VllmEngine,
    /// llama.cpp 引擎后端。
    LlamaCppEngine,
    /// 兼容 OpenAI API 的通用后端。
    GenericOpenAI,
}

/// 网络协议。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// HTTP/HTTPS。
    Http,
    /// gRPC。
    Grpc,
    /// NATS 消息总线。
    Nats,
}

/// 后端连接端点。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Endpoint {
    /// 后端服务 URL，例如 `http://10.0.0.1:8080`。
    pub url: String,
    /// 访问该端点使用的协议。
    pub protocol: Protocol,
}

/// 量化方式。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    /// 半精度浮点。
    Fp16,
    /// Brain 浮点。
    Bf16,
    /// 8 位整数量化。
    Int8,
    /// 4 位整数量化。
    Int4,
    /// AWQ 量化。
    Awq,
    /// GPTQ 量化。
    Gptq,
}

/// 单个模型实例的元数据。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInstance {
    /// 模型名称，例如 `Meta-Llama-3-8B-Instruct`。
    pub model_name: String,
    /// 模型架构，例如 `llama`、`mixtral`。
    pub model_architecture: String,
    /// 量化方式。
    pub quantization: Quantization,
    /// 模型最大上下文长度（token 数）。
    pub max_context_len: u32,
    /// 是否支持工具调用。
    pub supports_tool_calling: bool,
    /// 是否支持流式输出。
    pub supports_streaming: bool,
}

/// 后端能力描述。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// 是否支持发布 KV 缓存事件。
    pub supports_kv_events: bool,
    /// 是否支持请求批处理。
    pub supports_batching: bool,
    /// 最大批大小（0 表示不支持批处理）。
    pub max_batch_size: u32,
    /// GPU 数量。
    pub gpu_count: u32,
    /// GPU 总显存（GB）。
    pub gpu_memory_gb: u32,
}

/// KV 缓存配置。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvConfig {
    /// 单个 KV 缓存块的 token 数。
    pub block_size: u32,
    /// 缓存命名空间，用于隔离不同租户/模型。
    pub cache_namespace: String,
    /// 后端最多可用的 KV 块数。
    pub max_kv_blocks: u64,
}

/// 后端运行状态。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BackendStatus {
    /// 健康，可正常路由。
    Healthy,
    /// 降级，仍可路由但应优先选择其他实例。
    Degraded,
    /// 不健康，不应路由新请求。
    Unhealthy,
    /// 状态未知（尚未上报或心跳丢失）。
    Unknown,
}

/// 后端完整信息模型。
///
/// 由连接器在握手时构造，并随集群成员变更广播给路由层。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendInfo {
    /// 后端唯一标识。
    pub id: BackendId,
    /// 后端类型。
    pub backend_type: BackendType,
    /// 连接端点。
    pub endpoint: Endpoint,
    /// 该后端承载的模型列表。
    pub models: Vec<ModelInstance>,
    /// 所在区域。
    pub region: RegionId,
    /// 所属索引器域。
    pub indexer_domain: IndexerDomainId,
    /// 后端能力。
    pub capabilities: BackendCapabilities,
    /// KV 缓存配置。
    pub kv_config: KvConfig,
    /// 当前运行状态。
    pub status: BackendStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_json_round_trip() {
        let ep = Endpoint {
            url: "http://10.0.0.1:8080".to_string(),
            protocol: Protocol::Http,
        };
        let s = serde_json::to_string(&ep).unwrap();
        let back: Endpoint = serde_json::from_str(&s).unwrap();
        assert_eq!(ep, back);
    }

    #[test]
    fn backend_type_serde_snake_case() {
        let s = serde_json::to_string(&BackendType::VllmEngine).unwrap();
        assert_eq!(s, r#""vllm_engine""#);
        let back: BackendType = serde_json::from_str(&s).unwrap();
        assert_eq!(back, BackendType::VllmEngine);
    }

    #[test]
    fn quantization_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&Quantization::Fp16).unwrap(),
            r#""fp16""#
        );
        assert_eq!(
            serde_json::to_string(&Quantization::Bf16).unwrap(),
            r#""bf16""#
        );
    }

    #[test]
    fn backend_status_serde_snake_case() {
        let s = serde_json::to_string(&BackendStatus::Unhealthy).unwrap();
        assert_eq!(s, r#""unhealthy""#);
    }

    #[test]
    fn protocol_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&Protocol::Grpc).unwrap(),
            r#""grpc""#
        );
        assert_eq!(
            serde_json::to_string(&Protocol::Nats).unwrap(),
            r#""nats""#
        );
    }
}
