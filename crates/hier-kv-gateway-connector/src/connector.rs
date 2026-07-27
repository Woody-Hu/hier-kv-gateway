//! Backend connector trait and common type definitions.
//!
//! [`BackendConnector`] abstracts capabilities common to a collection of inference backend
//! instances: discovery, health checks, streaming inference forwarding, KV cache event
//! subscription, and metrics collection. Different engines (vLLM / llama.cpp / generic
//! OpenAI-compatible services) each implement this trait.
//!
//! [`HealthStatus`] describes a snapshot result of a health check, used by the routing
//! layer to exclude unhealthy instances.

use hier_kv_gateway_core::backend::{BackendInfo, BackendStatus, BackendType};
use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::request::{InferenceChunk, InferenceRequest};

use async_trait::async_trait;

/// Backend connector abstraction.
///
/// A `BackendConnector` typically corresponds to an access entry for a class of
/// ([`BackendType`]) backend instances. Objects implementing this trait are registered to
/// [`crate::registry::ConnectorRegistry`] as `Arc<dyn BackendConnector>`.
#[async_trait]
pub trait BackendConnector: Send + Sync {
    /// Returns the backend type proxied by this connector.
    fn backend_type(&self) -> BackendType;

    /// Discover the list of backend instances managed by this connector.
    ///
    /// Usually executed once during the handshake phase to obtain a set of static metadata
    /// [`BackendInfo`], and re-triggered by the cluster layer on member changes.
    async fn discover(&self) -> Result<Vec<BackendInfo>>;

    /// Perform a health check on the specified backend and return the current snapshot
    /// [`HealthStatus`].
    async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus>;

    /// Forward the inference request to the specified backend and return a stream of
    /// [`InferenceChunk`] sequences.
    ///
    /// The stream must terminate with [`InferenceChunk::Done`] or [`InferenceChunk::Error`],
    /// allowing the caller to determine request completion or failure.
    async fn forward(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<futures::stream::BoxStream<'static, InferenceChunk>>;

    /// Whether this connector supports subscribing to KV cache events.
    ///
    /// OpenAI-compatible services typically do not provide KV events and should return
    /// `false`.
    fn supports_kv_events(&self) -> bool;

    /// Subscribe to the KV cache event stream of the specified backend.
    ///
    /// When [`BackendConnector::supports_kv_events`] returns `false`, the implementation
    /// should directly return [`hier_kv_gateway_core::error::HierKvGatewayError::ConnectorError`].
    async fn subscribe_kv_events(
        &self,
        backend: &BackendId,
    ) -> Result<futures::stream::BoxStream<'static, KvCacheEvent>>;

    /// Collect real-time load metrics [`BackendMetrics`] for the specified backend.
    ///
    /// When the backend does not expose a metrics endpoint, the implementation can return
    /// a structured default value (zero-value snapshot), ensuring the routing layer can
    /// continue working without real metrics.
    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics>;
}

/// Snapshot result of a health check.
///
/// Returned by [`BackendConnector::health_check`], recording the current status,
/// continuous health duration, and cumulative error count, for the cluster layer to
/// decide whether to remove the instance from the routing pool.
#[derive(Clone, Debug)]
pub struct HealthStatus {
    /// Current backend operational status.
    pub status: BackendStatus,
    /// Unix timestamp (seconds) since entering the current status.
    ///
    /// When `status` is [`BackendStatus::Healthy`], this field indicates when the backend
    /// became continuously healthy; for other statuses, it represents the status
    /// transition time.
    pub healthy_since_unix: u64,
    /// Number of consecutive health check failures; should be zero in healthy status.
    pub error_count: u32,
}

impl HealthStatus {
    /// Construct a healthy snapshot with `error_count` 0.
    pub fn healthy(since_unix: u64) -> Self {
        Self {
            status: BackendStatus::Healthy,
            healthy_since_unix: since_unix,
            error_count: 0,
        }
    }

    /// Construct an unhealthy snapshot, recording the cumulative error count.
    pub fn unhealthy(since_unix: u64, error_count: u32) -> Self {
        Self {
            status: BackendStatus::Unhealthy,
            healthy_since_unix: since_unix,
            error_count,
        }
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            status: BackendStatus::Unknown,
            healthy_since_unix: 0,
            error_count: 0,
        }
    }
}
