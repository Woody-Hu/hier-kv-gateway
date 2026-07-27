//! Backend info model.
//!
//! Describes the static metadata of an inference backend instance: identity, type,
//! connection endpoint, models, region, indexer domain, capabilities, KV cache
//! configuration, and runtime status. These fields are populated by the connector
//! during the handshake phase and read by the routing, cluster, and API layers.

use serde::{Deserialize, Serialize};

use crate::ids::{BackendId, IndexerDomainId, RegionId};

/// Backend type enum, determines which engine/protocol the backend uses.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// LLM-D cluster backend.
    LlmDCluster,
    /// vLLM engine backend.
    VllmEngine,
    /// llama.cpp engine backend.
    LlamaCppEngine,
    /// Generic backend compatible with the OpenAI API.
    GenericOpenAI,
    /// NVIDIA Dynamo backend (NATS-based component bus).
    ///
    /// Enabled when the `dynamo` connector feature is turned on in the
    /// `hier_kv_gateway_connector` crate. Routing behaves identically to the
    /// OpenAI-compatible connector when the feature is disabled; the variant
    /// is always present so that configs and serialized state remain stable.
    DynamoEngine,
}

/// Network protocol.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// HTTP/HTTPS.
    Http,
    /// gRPC.
    Grpc,
    /// NATS message bus.
    Nats,
}

/// Backend connection endpoint.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Endpoint {
    /// Backend service URL, e.g. `http://10.0.0.1:8080`.
    pub url: String,
    /// Protocol used to access this endpoint.
    pub protocol: Protocol,
}

/// Quantization method.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    /// Half-precision floating point.
    Fp16,
    /// Brain floating point.
    Bf16,
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit integer quantization.
    Int4,
    /// AWQ quantization.
    Awq,
    /// GPTQ quantization.
    Gptq,
}

/// Metadata for a single model instance.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInstance {
    /// Model name, e.g. `Meta-Llama-3-8B-Instruct`.
    pub model_name: String,
    /// Model architecture, e.g. `llama`, `mixtral`.
    pub model_architecture: String,
    /// Quantization method.
    pub quantization: Quantization,
    /// Maximum model context length (token count).
    pub max_context_len: u32,
    /// Whether tool calling is supported.
    pub supports_tool_calling: bool,
    /// Whether streaming output is supported.
    pub supports_streaming: bool,
}

/// Backend capability description.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Whether publishing KV cache events is supported.
    pub supports_kv_events: bool,
    /// Whether request batching is supported.
    pub supports_batching: bool,
    /// Maximum batch size (0 means batching is not supported).
    pub max_batch_size: u32,
    /// Number of GPUs.
    pub gpu_count: u32,
    /// Total GPU memory (GB).
    pub gpu_memory_gb: u32,
}

/// KV cache configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvConfig {
    /// Number of tokens per KV cache block.
    pub block_size: u32,
    /// Cache namespace, used to isolate tenants/models.
    pub cache_namespace: String,
    /// Maximum number of KV blocks available to the backend.
    pub max_kv_blocks: u64,
}

/// Backend runtime status.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BackendStatus {
    /// Healthy and routable.
    Healthy,
    /// Degraded; still routable but other instances should be preferred.
    Degraded,
    /// Unhealthy; new requests should not be routed.
    Unhealthy,
    /// Status unknown (no report yet or heartbeat lost).
    Unknown,
}

/// Full backend info model.
///
/// Constructed by the connector during handshake and broadcast to the routing
/// layer on cluster membership changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Backend unique identifier.
    pub id: BackendId,
    /// Backend type.
    pub backend_type: BackendType,
    /// Connection endpoint.
    pub endpoint: Endpoint,
    /// List of models hosted by this backend.
    pub models: Vec<ModelInstance>,
    /// Region the backend resides in.
    pub region: RegionId,
    /// Indexer domain the backend belongs to.
    pub indexer_domain: IndexerDomainId,
    /// Backend capabilities.
    pub capabilities: BackendCapabilities,
    /// KV cache configuration.
    pub kv_config: KvConfig,
    /// Current runtime status.
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
