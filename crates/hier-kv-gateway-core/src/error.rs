//! Unified error type definitions for the Hier KV Gateway.
//!
//! Uses [`thiserror`] to derive [`std::error::Error`] and [`Display`](std::fmt::Display),
//! making it easy to propagate and contextualize errors across crate boundaries.

use thiserror::Error;

/// All error kinds that may occur during Hier KV Gateway operation.
///
/// `Clone` is derived so the error can be shared across concurrent awaiters
/// of an in-flight request (single-flight / request coalescing), where one
/// leader's terminal error is propagated to every follower awaiting the same
/// shared future. All variants hold only `String` / unit data, so `Clone` is
/// trivially sound.
#[derive(Clone, Debug, Error)]
pub enum HierKvGatewayError {
    /// All known backends are unavailable; no target can be selected.
    #[error("No available backend instance")]
    BackendUnavailable,

    /// Routing decision failed, e.g. no backend satisfies the constraints or all scores are invalid.
    #[error("Routing failed: {0}")]
    RoutingFailed(String),

    /// Connector error communicating with the backend, e.g. connection refused or protocol parsing failed.
    #[error("Connector error: {0}")]
    ConnectorError(String),

    /// Metrics collection or computation error.
    #[error("Metrics error: {0}")]
    MetricsError(String),

    /// Configuration loading or validation error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Cluster membership protocol (gossip/liveness) related error.
    #[error("Cluster error: {0}")]
    ClusterError(String),

    /// The requested resource (backend, model, config entry, etc.) was not found.
    #[error("Not found: {0}")]
    NotFound(String),

    /// Rate limit triggered; the caller should back off and retry.
    #[error("Rate limited")]
    RateLimited,

    /// Other internal errors, used for uncategorized failure scenarios.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Common Result alias for the Hier KV Gateway crate.
pub type Result<T> = std::result::Result<T, HierKvGatewayError>;
