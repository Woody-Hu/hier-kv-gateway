//! Routing strategy abstraction.
//!
//! All concrete strategies (KV-aware, model-aware, load-aware, topology-aware, hybrid)
//! implement [`RoutingStrategy`]. A strategy receives a list of candidate backends and
//! a metadata store handle, and returns a [`ScoredBackend`] score for each candidate.

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

/// Common interface for routing strategies.
///
/// Implementors must ensure:
/// - [`evaluate`](RoutingStrategy::evaluate) is safe to call concurrently;
/// - [`is_available`](RoutingStrategy::is_available) is a cheap synchronous probe used by the hybrid strategy to decide whether to enable it;
/// - [`weight`](RoutingStrategy::weight) returns the strategy's static weight in the hybrid score.
#[async_trait]
pub trait RoutingStrategy: Send + Sync {
    /// Strategy name, used for logging and decision tracing.
    fn name(&self) -> &'static str;

    /// Score each candidate backend one by one, returning an order-irrelevant list of [`ScoredBackend`].
    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>>;

    /// Whether the strategy is currently available (e.g. disable the KV strategy when the KV index is not started).
    fn is_available(&self, meta: &MetadataStore) -> bool;

    /// The strategy's static weight in the hybrid score.
    fn weight(&self) -> f64;
}
