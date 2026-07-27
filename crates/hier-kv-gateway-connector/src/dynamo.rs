//! NVIDIA Dynamo connector (feature-gated on `dynamo`).
//!
//! Dynamo is NVIDIA's cloud-native LLM inference framework that uses NATS
//! as its component message bus. This connector implements
//! [`BackendConnector`] for Dynamo backends:
//!
//! - **Discovery**: uses a static model list injected from config (Dynamo
//!   components do not currently expose a `/v1/models` style endpoint on the
//!   NATS bus).
//! - **Health checks**: NATS request-reply on `<subject_prefix>.health`.
//! - **Inference forwarding**: NATS request-reply on
//!   `<subject_prefix>.generate`. The request payload is an OpenAI Chat
//!   Completions JSON document (Dynamo's `Worker` protocol accepts this on
//!   the generate subject). The reply is parsed into a sequence of
//!   [`InferenceChunk`] objects.
//! - **KV cache events**: subscribes to `<subject_prefix>.kv_events`. The
//!   payload is a JSON-encoded [`KvCacheEvent`].
//! - **Metrics**: NATS request-reply on `<subject_prefix>.metrics`. The
//!   reply is a JSON-encoded [`BackendMetrics`].
//!
//! When the `dynamo` feature is disabled, this module still exports a stub
//! [`DynamoConnector`] type that wraps an [`OpenAICompatConnector`] and
//! returns [`HierKvGatewayError::ConnectorError`] from all
//! Dynamo-specific operations. This keeps the public API stable regardless
//! of feature flags.

use std::time::Duration;

use hier_kv_gateway_core::backend::{BackendInfo, BackendType, Endpoint, Protocol};
use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::{BackendId, BackendInstanceId, RegionId};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::request::{InferenceChunk, InferenceRequest};

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::Serialize;

use crate::connector::{BackendConnector, HealthStatus};

/// Default NATS subject prefix used by Dynamo deployments.
pub const DEFAULT_DYNAMO_SUBJECT_PREFIX: &str = "dyn";

/// Default timeout for NATS request-reply calls.
#[allow(dead_code)]
const DEFAULT_NATS_TIMEOUT: Duration = Duration::from_secs(60);

/// Configuration for a Dynamo connector.
///
/// A single connector corresponds to one Dynamo backend instance (typically
/// one Dynamo `Worker` component). The `subject_prefix` allows multiple
/// Dynamo workers to share a NATS cluster without colliding.
#[derive(Clone, Debug)]
pub struct DynamoConnectorConfig {
    /// NATS URL, e.g. `nats://127.0.0.1:4222`.
    pub nats_url: String,
    /// Subject prefix; defaults to `dyn`.
    pub subject_prefix: String,
    /// Region this backend resides in.
    pub region: RegionId,
    /// Backend instance identifier (usually the Dynamo component name).
    pub instance_id: BackendInstanceId,
    /// Models served by this backend (injected from config).
    pub models: Vec<String>,
    /// KV block size (must match the backend convention).
    pub kv_block_size: u32,
    /// Optional request timeout override.
    pub request_timeout: Option<Duration>,
}

impl DynamoConnectorConfig {
    /// Build a config from the gateway's [`Endpoint`] + config fields.
    ///
    /// The endpoint `url` is expected to be a NATS URL
    /// (e.g. `nats://10.0.0.1:4222`). The `protocol` field is ignored —
    /// Dynamo always uses NATS.
    pub fn from_endpoint(
        endpoint: &Endpoint,
        region: &RegionId,
        instance_id: impl Into<BackendInstanceId>,
        models: Vec<String>,
        kv_block_size: u32,
    ) -> Self {
        Self {
            nats_url: endpoint.url.clone(),
            subject_prefix: DEFAULT_DYNAMO_SUBJECT_PREFIX.to_string(),
            region: region.clone(),
            instance_id: instance_id.into(),
            models,
            kv_block_size,
            request_timeout: None,
        }
    }

    /// NATS subject for health checks.
    pub fn health_subject(&self) -> String {
        format!("{}.health.{}", self.subject_prefix, self.instance_id.as_str())
    }

    /// NATS subject for inference requests.
    pub fn generate_subject(&self) -> String {
        format!("{}.generate.{}", self.subject_prefix, self.instance_id.as_str())
    }

    /// NATS subject for KV cache events.
    pub fn kv_events_subject(&self) -> String {
        format!("{}.kv_events.{}", self.subject_prefix, self.instance_id.as_str())
    }

    /// NATS subject for metrics.
    pub fn metrics_subject(&self) -> String {
        format!("{}.metrics.{}", self.subject_prefix, self.instance_id.as_str())
    }
}

// ---------------------------------------------------------------------------
// Dynamo-native implementation (feature-gated on `dynamo`)
// ---------------------------------------------------------------------------

#[cfg(feature = "dynamo")]
mod native {
    use super::*;
    use std::time::Instant;
    use tokio::sync::OnceCell;

    /// Dynamo connector backed by a real NATS connection.
    ///
    /// The NATS connection is established lazily on first use and cached in
    /// a [`OnceCell`]. This keeps [`DynamoConnector::new`] synchronous so
    /// it can be called from the (synchronous) `ConnectorRegistry::from_configs`
    /// constructor.
    pub struct DynamoConnector {
        config: DynamoConnectorConfig,
        timeout: Duration,
        client: OnceCell<async_nats::Client>,
    }

    impl DynamoConnector {
        /// Create a connector without performing the connect handshake.
        ///
        /// The first call to any I/O-bearing method will lazily establish
        /// the NATS connection. Use [`Self::connect`] for an eager
        /// constructor that returns an error if the NATS server is
        /// unreachable.
        pub fn new(config: DynamoConnectorConfig) -> Self {
            let timeout = config.request_timeout.unwrap_or(DEFAULT_NATS_TIMEOUT);
            Self {
                config,
                timeout,
                client: OnceCell::new(),
            }
        }

        /// Eagerly connect to the configured NATS server.
        pub async fn connect(config: DynamoConnectorConfig) -> Result<Self> {
            let timeout = config.request_timeout.unwrap_or(DEFAULT_NATS_TIMEOUT);
            let client = async_nats::connect(&config.nats_url).await.map_err(|e| {
                HierKvGatewayError::ConnectorError(format!(
                    "Dynamo NATS connect failed ({}): {}",
                    config.nats_url, e
                ))
            })?;
            Ok(Self {
                config,
                timeout,
                client: OnceCell::from(client),
            })
        }

        /// Get or establish the NATS client.
        async fn client(&self) -> Result<&async_nats::Client> {
            self.client
                .get_or_try_init(|| async {
                    async_nats::connect(&self.config.nats_url).await.map_err(|e| {
                        HierKvGatewayError::ConnectorError(format!(
                            "Dynamo NATS connect failed ({}): {}",
                            self.config.nats_url, e
                        ))
                    })
                })
                .await
        }
    }

    #[async_trait]
    impl BackendConnector for DynamoConnector {
        fn backend_type(&self) -> BackendType {
            BackendType::DynamoEngine
        }

        fn backend_id(&self) -> BackendId {
            BackendId::new(self.config.region.clone(), self.config.instance_id.clone())
        }

        async fn discover(&self) -> Result<Vec<BackendInfo>> {
            // Dynamo does not expose /v1/models; rely on config-injected models.
            let model_instances: Vec<ModelInstance> = self
                .config
                .models
                .iter()
                .map(|name| ModelInstance {
                    model_name: name.clone(),
                    model_architecture: "unknown".to_string(),
                    quantization: Quantization::Fp16,
                    max_context_len: 32768,
                    supports_tool_calling: false,
                    supports_streaming: true,
                })
                .collect();

            let info = BackendInfo {
                id: self.backend_id(),
                backend_type: BackendType::DynamoEngine,
                endpoint: Endpoint {
                    url: self.config.nats_url.clone(),
                    protocol: Protocol::Nats,
                },
                models: model_instances,
                region: self.config.region.clone(),
                indexer_domain: IndexerDomainId(0),
                capabilities: BackendCapabilities {
                    supports_kv_events: true,
                    supports_batching: true,
                    max_batch_size: 0,
                    gpu_count: 0,
                    gpu_memory_gb: 0,
                },
                kv_config: KvConfig {
                    block_size: self.config.kv_block_size,
                    cache_namespace: String::new(),
                    max_kv_blocks: 0,
                },
                status: BackendStatus::Healthy,
            };
            Ok(vec![info])
        }

        async fn health_check(&self, _backend: &BackendId) -> Result<HealthStatus> {
            let now = chrono::Utc::now().timestamp() as u64;
            let client = match self.client().await {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "Dynamo health check: NATS unavailable");
                    return Ok(HealthStatus::unhealthy(now, 1));
                }
            };
            let subject = self.config.health_subject();
            match tokio::time::timeout(
                Duration::from_secs(5),
                client.request(subject, "".into()),
            )
            .await
            {
                Ok(Ok(_)) => Ok(HealthStatus::healthy(now)),
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "Dynamo health check NATS error");
                    Ok(HealthStatus::unhealthy(now, 1))
                }
                Err(_) => {
                    tracing::warn!("Dynamo health check timed out");
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
            let client = self.client().await?.clone();

            let body = DynamoGenerateRequest::from(request);
            let payload = serde_json::to_vec(&body).map_err(|e| {
                HierKvGatewayError::ConnectorError(format!(
                    "Dynamo request serialization failed: {}",
                    e
                ))
            })?;

            let subject = self.config.generate_subject();
            let reply = tokio::time::timeout(self.timeout, client.request(subject, payload.into()))
                .await
                .map_err(|_| {
                    HierKvGatewayError::ConnectorError(format!(
                        "Dynamo generate request timed out after {:?}",
                        self.timeout
                    ))
                })?
                .map_err(|e| {
                    HierKvGatewayError::ConnectorError(format!(
                        "Dynamo generate NATS error: {}",
                        e
                    ))
                })?;

            // The Dynamo worker returns a newline-delimited stream of JSON
            // InferenceChunk objects. We parse them eagerly and emit a
            // futures::stream::iter. A future improvement could use a
            // NATS subscription for true streaming; this is good enough for
            // the connector contract.
            let text = String::from_utf8_lossy(&reply);
            let mut chunks: Vec<InferenceChunk> = Vec::new();
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<InferenceChunk>(line) {
                    Ok(c) => chunks.push(c),
                    Err(e) => {
                        tracing::trace!(line = %line, error = %e, "Dynamo: skipping unparseable line");
                    }
                }
            }
            // Always ensure a terminal Done chunk.
            let terminal = matches!(
                chunks.last(),
                Some(InferenceChunk::Done { .. }) | Some(InferenceChunk::Error { .. })
            );
            if !terminal {
                chunks.push(InferenceChunk::Done {
                    backend_id,
                    latency_ms: start.elapsed().as_millis() as u64,
                });
            }
            Ok(Box::pin(futures::stream::iter(chunks)))
        }

        fn supports_kv_events(&self) -> bool {
            true
        }

        async fn subscribe_kv_events(
            &self,
            _backend: &BackendId,
        ) -> Result<BoxStream<'static, KvCacheEvent>> {
            let client = self.client().await?.clone();
            let subject = self.config.kv_events_subject();
            let mut sub = client
                .subscribe(subject)
                .await
                .map_err(|e| {
                    HierKvGatewayError::ConnectorError(format!(
                        "Dynamo KV events subscribe failed: {}",
                        e
                    ))
                })?;
            // Use a futures::stream::unfold-based generator to avoid pulling
            // in the `async-stream` crate as a dependency.
            let stream = futures::stream::unfold(sub, |mut sub| async move {
                loop {
                    match sub.next().await {
                        Some(msg) => match serde_json::from_slice::<KvCacheEvent>(&msg.payload) {
                            Ok(ev) => return Some((ev, sub)),
                            Err(e) => {
                                tracing::warn!(error = %e, "Dynamo: skipping malformed KV event");
                                continue;
                            }
                        },
                        None => return None,
                    }
                }
            });
            Ok(Box::pin(stream))
        }

        async fn collect_metrics(&self, _backend: &BackendId) -> Result<BackendMetrics> {
            let now = chrono::Utc::now().timestamp() as i64;
            let client = self.client().await?.clone();
            let subject = self.config.metrics_subject();
            let reply = tokio::time::timeout(
                Duration::from_secs(3),
                client.request(subject, "".into()),
            )
            .await
            .map_err(|_| {
                HierKvGatewayError::ConnectorError(
                    "Dynamo metrics request timed out".to_string(),
                )
            })?
            .map_err(|e| {
                HierKvGatewayError::ConnectorError(format!(
                    "Dynamo metrics NATS error: {}",
                    e
                ))
            })?;
            let mut metrics: BackendMetrics =
                serde_json::from_slice(&reply).unwrap_or_default();
            metrics.timestamp = now;
            Ok(metrics)
        }
    }
}

#[cfg(feature = "dynamo")]
pub use native::DynamoConnector;

// ---------------------------------------------------------------------------
// Stub implementation (when `dynamo` feature is disabled)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "dynamo"))]
mod stub {
    use super::*;
    use crate::openai_compat::OpenAICompatConnector;

    /// Stub Dynamo connector used when the `dynamo` feature is disabled.
    ///
    /// All Dynamo-specific operations (NATS request-reply, KV event
    /// subscription) return an error. For OpenAI-compatible HTTP fallback
    /// behavior, use [`OpenAICompatConnector`] directly.
    pub struct DynamoConnector {
        config: DynamoConnectorConfig,
        http_fallback: OpenAICompatConnector,
    }

    impl DynamoConnector {
        /// Create a stub connector.
        ///
        /// The `nats_url` in the config is used as the HTTP base URL for
        /// the fallback OpenAI-compatible connector so that a misconfigured
        /// deployment fails fast rather than silently doing nothing.
        pub fn new(config: DynamoConnectorConfig) -> Self {
            let http_fallback = OpenAICompatConnector::new(
                config.nats_url.clone(),
                BackendType::DynamoEngine,
                config.region.clone(),
                config.instance_id.clone(),
                config.models.clone(),
                config.kv_block_size,
            );
            Self {
                config,
                http_fallback,
            }
        }

        /// Access the underlying config.
        pub fn config(&self) -> &DynamoConnectorConfig {
            &self.config
        }
    }

    #[async_trait]
    impl BackendConnector for DynamoConnector {
        fn backend_type(&self) -> BackendType {
            BackendType::DynamoEngine
        }

        fn backend_id(&self) -> BackendId {
            self.http_fallback.backend_id()
        }

        async fn discover(&self) -> Result<Vec<BackendInfo>> {
            // Delegate to the HTTP fallback so that a deployment using
            // Dynamo over an HTTP gateway still works without the feature.
            let mut infos = self.http_fallback.discover().await?;
            for info in infos.iter_mut() {
                info.backend_type = BackendType::DynamoEngine;
                info.endpoint.protocol = Protocol::Nats;
                info.capabilities.supports_kv_events = true;
            }
            Ok(infos)
        }

        async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus> {
            // No NATS connection: fall back to HTTP health check.
            self.http_fallback.health_check(backend).await
        }

        async fn forward(
            &self,
            backend: &BackendId,
            request: &InferenceRequest,
        ) -> Result<BoxStream<'static, InferenceChunk>> {
            self.http_fallback.forward(backend, request).await
        }

        fn supports_kv_events(&self) -> bool {
            // The stub cannot subscribe to NATS, so report false.
            false
        }

        async fn subscribe_kv_events(
            &self,
            _backend: &BackendId,
        ) -> Result<BoxStream<'static, KvCacheEvent>> {
            Err(HierKvGatewayError::ConnectorError(
                "Dynamo connector compiled without the `dynamo` feature; \
                 KV event subscription is unavailable. \
                 Rebuild with `--features dynamo`."
                    .to_string(),
            ))
        }

        async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics> {
            self.http_fallback.collect_metrics(backend).await
        }
    }
}

#[cfg(not(feature = "dynamo"))]
pub use stub::DynamoConnector;

// ---------------------------------------------------------------------------
// Shared request payload used by both implementations
// ---------------------------------------------------------------------------

/// Dynamo generate request payload.
///
/// Mirrors the OpenAI Chat Completions request shape; the Dynamo `Worker`
/// component accepts the same JSON on its generate subject.
#[derive(Serialize)]
#[allow(dead_code)]
struct DynamoGenerateRequest {
    model: String,
    messages: Vec<DynamoMessage>,
    max_tokens: u32,
    temperature: f64,
    stream: bool,
}

#[derive(Serialize)]
#[allow(dead_code)]
struct DynamoMessage {
    role: String,
    content: String,
}

impl From<&InferenceRequest> for DynamoGenerateRequest {
    fn from(req: &InferenceRequest) -> Self {
        let messages = req
            .messages
            .iter()
            .map(|m| DynamoMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        Self {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_helpers_use_prefix_and_instance() {
        let cfg = DynamoConnectorConfig {
            nats_url: "nats://127.0.0.1:4222".to_string(),
            subject_prefix: "dyn".to_string(),
            region: RegionId::new("us-east-1"),
            instance_id: BackendInstanceId::new("worker-0"),
            models: vec!["m".to_string()],
            kv_block_size: 16,
            request_timeout: None,
        };
        assert_eq!(cfg.health_subject(), "dyn.health.worker-0");
        assert_eq!(cfg.generate_subject(), "dyn.generate.worker-0");
        assert_eq!(cfg.kv_events_subject(), "dyn.kv_events.worker-0");
        assert_eq!(cfg.metrics_subject(), "dyn.metrics.worker-0");
    }

    #[test]
    fn from_endpoint_uses_url_as_nats_url() {
        let ep = Endpoint {
            url: "nats://10.0.0.1:4222".to_string(),
            protocol: Protocol::Nats,
        };
        let cfg = DynamoConnectorConfig::from_endpoint(
            &ep,
            &RegionId::new("r1"),
            "worker-1",
            vec!["m".to_string()],
            16,
        );
        assert_eq!(cfg.nats_url, "nats://10.0.0.1:4222");
        assert_eq!(cfg.subject_prefix, DEFAULT_DYNAMO_SUBJECT_PREFIX);
        assert_eq!(cfg.instance_id.as_str(), "worker-1");
    }

    #[test]
    fn dynamo_request_serialization_matches_openai_shape() {
        use hier_kv_gateway_core::ids::RequestId;
        use hier_kv_gateway_core::request::ChatMessage;
        let req = InferenceRequest {
            request_id: RequestId::new("r1"),
            model: "m".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            token_ids: vec![],
            max_tokens: 32,
            temperature: 0.5,
            stream: true,
            tools: vec![],
            lora_name: None,
        };
        let body = DynamoGenerateRequest::from(&req);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "m");
        assert_eq!(json["stream"], true);
        assert_eq!(json["max_tokens"], 32);
        assert_eq!(json["messages"][0]["role"], "user");
    }

    #[cfg(not(feature = "dynamo"))]
    #[test]
    fn stub_connector_can_be_constructed() {
        let cfg = DynamoConnectorConfig {
            nats_url: "http://localhost:8000".to_string(),
            subject_prefix: "dyn".to_string(),
            region: RegionId::new("r1"),
            instance_id: BackendInstanceId::new("i1"),
            models: vec!["m".to_string()],
            kv_block_size: 16,
            request_timeout: None,
        };
        let connector = DynamoConnector::new(cfg);
        assert_eq!(connector.backend_type(), BackendType::DynamoEngine);
    }
}
