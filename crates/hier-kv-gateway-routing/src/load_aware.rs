//! Load-aware routing strategy.
//!
//! Combines the candidate backend's active request count, queue depth, P99 latency,
//! GPU utilization, and KV cache usage, weighted to obtain `load_cost`. Backends
//! with insufficient available capacity are excluded directly (score = 0).

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// Load-aware routing strategy.
pub struct LoadAwareStrategy {
    /// Weight of the active request count.
    pub w_req: f64,
    /// Weight of the queue depth.
    pub w_queue: f64,
    /// Weight of the P99 latency (latency is in milliseconds and is first divided by 100).
    pub w_lat: f64,
    /// Weight of the GPU utilization.
    pub w_gpu: f64,
    /// Weight of the KV cache usage.
    pub w_kv: f64,
}

impl Default for LoadAwareStrategy {
    fn default() -> Self {
        Self {
            w_req: 1.0,
            w_queue: 1.0,
            w_lat: 0.01,
            w_gpu: 1.0,
            w_kv: 1.0,
        }
    }
}

#[async_trait]
impl RoutingStrategy for LoadAwareStrategy {
    fn name(&self) -> &'static str {
        "load_aware"
    }

    async fn evaluate(
        &self,
        _ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        let mut out = Vec::with_capacity(candidates.len());
        for cand in candidates {
            let Some(m) = meta.load_get_metrics(cand) else {
                // No metrics: exclude
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 0.0,
                    raw_cost: f64::MAX,
                    meta_version: 0,
                });
                continue;
            };

            // Exclude when available capacity <= 0
            if m.available_capacity() <= 0 {
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 0.0,
                    raw_cost: f64::MAX,
                    meta_version: 0,
                });
                continue;
            }

            // Weighted load cost
            let load_cost = self.w_req * m.active_requests as f64
                + self.w_queue * m.queue_depth as f64
                + self.w_lat * (m.latency.p99_ms / 100.0)
                + self.w_gpu * m.gpu_utilization
                + self.w_kv * m.kv_cache_usage();

            let score = 1.0 / (1.0 + load_cost);
            out.push(ScoredBackend {
                backend_id: cand.clone(),
                score,
                raw_cost: load_cost,
                meta_version: 0,
            });
        }
        Ok(out)
    }

    fn is_available(&self, meta: &MetadataStore) -> bool {
        // Enabled only when at least one backend has reported metrics
        meta.backends_all()
            .iter()
            .any(|b| meta.load_get_metrics(&b.id).is_some())
    }

    fn weight(&self) -> f64 {
        0.30
    }
}
