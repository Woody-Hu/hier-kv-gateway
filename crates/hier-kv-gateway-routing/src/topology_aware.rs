//! Topology-aware routing strategy.
//!
//! Using the Region where this gateway resides as the baseline, the network cost
//! is estimated from the RTT to the Region of each candidate backend. The smaller
//! the RTT, the higher the score; when the RTT is unknown (returns `f64::INFINITY`),
//! the score approaches 0.

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::{BackendId, RegionId};
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// Topology-aware routing strategy.
pub struct TopologyAwareStrategy {
    /// RTT weight.
    pub w_rtt: f64,
    /// Bandwidth weight (currently reserved, not used in the scoring formula).
    pub w_bw: f64,
    /// The Region where this gateway resides.
    pub self_region: RegionId,
}

#[async_trait]
impl RoutingStrategy for TopologyAwareStrategy {
    fn name(&self) -> &'static str {
        "topology_aware"
    }

    async fn evaluate(
        &self,
        _ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        let mut out = Vec::with_capacity(candidates.len());
        for cand in candidates {
            let rtt = meta.topo_rtt_ms(&self.self_region, &cand.region);
            // Network cost: the larger the RTT, the higher the cost
            let network_cost = self.w_rtt * rtt;
            // Score: normalized using 100ms as the reference baseline
            let score = 1.0 / (1.0 + network_cost / 100.0);
            out.push(ScoredBackend {
                backend_id: cand.clone(),
                score,
                raw_cost: network_cost,
                meta_version: 0,
            });
        }
        Ok(out)
    }

    fn is_available(&self, _meta: &MetadataStore) -> bool {
        // The topology strategy is always available (missing data degrades to f64::INFINITY)
        true
    }

    fn weight(&self) -> f64 {
        0.20
    }
}
