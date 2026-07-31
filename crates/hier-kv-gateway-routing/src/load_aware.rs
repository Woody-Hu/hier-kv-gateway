//! Load-aware routing strategy.
//!
//! Combines the candidate backend's active request count, queue depth, P99 latency,
//! GPU utilization, and KV cache usage, weighted to obtain `load_cost`. Backends
//! with insufficient available capacity are excluded directly (score = 0).
//!
//! ## Token-budget awareness (decode / prefill pressure)
//!
//! In addition to the count-based signals above, the cost optionally includes two
//! *token-budget* terms that close the gap left by count-blind scoring:
//!
//! - **Projected decode pressure** (`w_decode`): the backend's current
//!   `active_decode_blocks` *plus* the decode blocks this request would add if
//!   routed here, derived from [`RoutingContext::estimated_output_tokens`]
//!   (a conservative upper bound — it is sourced from the client's `max_tokens`,
//!   which the generation cannot exceed). This prevents a long-generation
//!   request from being piled onto a backend whose `active_requests` count is
//!   momentarily low but whose decode capacity is already saturated.
//! - **Prefill pressure** (`w_prefill`): the backend's `active_prefill_tokens`,
//!   a signal that was already collected by [`BackendMetrics`] but previously
//!   unused in the soft cost. It is *not* projected with the incoming prompt
//!   because the load strategy does not have the KV-overlap signal needed to
//!   estimate uncached prompt blocks (that belongs to the KV-aware strategy);
//!   keeping the two strategies independent preserves the hybrid normalization.
//!
//! Both terms are gated by their weights: setting `w_decode = 0` and
//! `w_prefill = 0` reproduces the historical count-blind behaviour exactly, so
//! the feature is opt-out for configurations that need byte-for-byte parity.
//!
//! The conservative-upper-bound choice (rather than a point estimate of output
//! length) follows the finding that pessimistic estimates avoid starvation
//! while still capturing the load-distribution benefit.

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
    /// Weight of the *projected decode pressure* (in KV blocks): the backend's
    /// current `active_decode_blocks` plus the blocks this request's
    /// `estimated_output_tokens` would add. `0.0` disables the term and
    /// reproduces the historical count-blind cost.
    pub w_decode: f64,
    /// Weight of the backend's `active_prefill_tokens` (prefill pressure).
    /// `0.0` disables the term.
    pub w_prefill: f64,
}

impl Default for LoadAwareStrategy {
    fn default() -> Self {
        Self {
            w_req: 1.0,
            w_queue: 1.0,
            w_lat: 0.01,
            w_gpu: 1.0,
            w_kv: 1.0,
            // Conservative defaults: a "busy" backend typically carries
            // `active_requests ≈ 4` (cost ≈ 4) and `active_decode_blocks` in
            // the low hundreds. `w_decode = 0.02` makes a 200-block decode
            // footprint contribute ≈ 4 — comparable to the request-count term
            // without dominating it. `w_prefill = 0.001` keeps the prefill
            // term a tie-breaker (1000 prefill tokens ≈ 1.0 cost).
            w_decode: 0.02,
            w_prefill: 0.001,
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
        ctx: &RoutingContext,
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

            // Projected decode pressure: existing decode blocks + the blocks this
            // request's output budget would add. `estimated_output_tokens` is a
            // conservative upper bound (sourced from the client's `max_tokens`),
            // so the projection never underestimates the decode footprint.
            let req_decode_blocks = if ctx.block_size > 0 {
                ((ctx.estimated_output_tokens as u64 + ctx.block_size as u64 - 1)
                    / ctx.block_size as u64) as f64
            } else {
                0.0
            };
            let projected_decode = m.active_decode_blocks as f64 + req_decode_blocks;

            // Weighted load cost
            let load_cost = self.w_req * m.active_requests as f64
                + self.w_queue * m.queue_depth as f64
                + self.w_lat * (m.latency.p99_ms / 100.0)
                + self.w_gpu * m.gpu_utilization
                + self.w_kv * m.kv_cache_usage()
                + self.w_decode * projected_decode
                + self.w_prefill * m.active_prefill_tokens as f64;

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
