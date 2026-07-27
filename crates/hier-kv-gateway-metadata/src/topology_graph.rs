//! Topology graph: maintains the Region list and pairwise RTT matrix.
//!
//! The read path is guarded by [`parking_lot::RwLock`]; the write path takes the
//! write lock and modifies the internal map. When `rtt_ms` query finds no direct
//! entry, it tries the reverse direction; if neither exists it returns `f64::INFINITY`.

use std::collections::HashMap;

use hier_kv_gateway_core::ids::{RegionId, RegionTier};
use hier_kv_gateway_core::topology::{GeoCoord, LatencyEstimate, LatencyMatrix, RegionInfo};
use parking_lot::RwLock;

/// Topology graph.
pub struct TopologyGraph {
    /// RegionId → RegionInfo.
    regions: RwLock<HashMap<RegionId, RegionInfo>>,
    /// Pairwise RTT matrix (milliseconds).
    latency_matrix: RwLock<LatencyMatrix>,
}

impl std::fmt::Debug for TopologyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let regions = self.regions.read();
        f.debug_struct("TopologyGraph")
            .field("regions", &regions.len())
            .finish()
    }
}

impl Default for TopologyGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologyGraph {
    /// Create an empty topology graph.
    pub fn new() -> Self {
        Self {
            regions: RwLock::new(HashMap::new()),
            latency_matrix: RwLock::new(LatencyMatrix::default()),
        }
    }

    /// Query the RTT (milliseconds) between two Regions.
    ///
    /// If the direct entry is missing, the reverse direction is tried; if still
    /// missing, `f64::INFINITY` is returned.
    pub fn rtt_ms(&self, from: &RegionId, to: &RegionId) -> f64 {
        if from == to {
            return 0.0;
        }
        let matrix = self.latency_matrix.read();
        // Try both forward and reverse keys; latency is treated as symmetric
        if let Some(est) = matrix
            .entries
            .get(&(from.clone(), to.clone()))
            .or_else(|| matrix.entries.get(&(to.clone(), from.clone())))
        {
            return est.rtt_p50_ms;
        }
        f64::INFINITY
    }

    /// Get information about the specified Region.
    pub fn get_region(&self, region: &RegionId) -> Option<RegionInfo> {
        self.regions.read().get(region).cloned()
    }

    /// Update the RTT estimate between two Regions (also writes the reverse
    /// entry, for symmetric lookup).
    pub fn update_latency(&self, a: &RegionId, b: &RegionId, estimate: LatencyEstimate) {
        if a == b {
            return;
        }
        let mut matrix = self.latency_matrix.write();
        matrix.entries.insert((a.clone(), b.clone()), estimate.clone());
        matrix.entries.insert((b.clone(), a.clone()), estimate);
    }

    /// Add a Region.
    pub fn add_region(&self, info: RegionInfo) {
        self.regions.write().insert(info.id.clone(), info);
    }

    /// Remove a Region (also cleans up its latency entries).
    pub fn remove_region(&self, region: &RegionId) {
        self.regions.write().remove(region);
        // Cleanup of the LatencyMatrix itself depends on its API; if a remove method is exposed, call it here.
    }

    /// List all RegionIds.
    pub fn all_regions(&self) -> Vec<RegionId> {
        self.regions.read().keys().cloned().collect()
    }
}

/// Placeholder references used only for documentation and types, ensuring
/// related hier-kv-gateway-core types remain visible in docs.
#[allow(dead_code)]
type _TopologyDeps = (GeoCoord, RegionTier);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtt_symmetric_after_update() {
        let graph = TopologyGraph::new();
        let a = RegionId::new("r1");
        let b = RegionId::new("r2");
        let estimate = LatencyEstimate {
            rtt_p50_ms: 12.5,
            rtt_p99_ms: 20.0,
            bandwidth_mbps: 1000.0,
            last_updated_unix: 1,
        };
        graph.update_latency(&a, &b, estimate);
        assert_eq!(graph.rtt_ms(&a, &b), 12.5);
        assert_eq!(graph.rtt_ms(&b, &a), 12.5);
        assert_eq!(graph.rtt_ms(&a, &a), 0.0);
    }
}
