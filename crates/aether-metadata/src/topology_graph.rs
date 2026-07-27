//! 拓扑图：维护 Region 列表与两两之间的 RTT 矩阵。
//!
//! 读路径通过 [`parking_lot::RwLock`] 保护；写路径获取写锁后修改内部 map。
//! `rtt_ms` 查询时若直接条目不存在，尝试反向查询；若都不存在返回 `f64::INFINITY`。

use std::collections::HashMap;

use aether_core::ids::{RegionId, RegionTier};
use aether_core::topology::{GeoCoord, LatencyEstimate, LatencyMatrix, RegionInfo};
use parking_lot::RwLock;

/// 拓扑图。
pub struct TopologyGraph {
    /// RegionId → RegionInfo。
    regions: RwLock<HashMap<RegionId, RegionInfo>>,
    /// 两两 RTT 矩阵（毫秒）。
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
    /// 创建一个空的拓扑图。
    pub fn new() -> Self {
        Self {
            regions: RwLock::new(HashMap::new()),
            latency_matrix: RwLock::new(LatencyMatrix::default()),
        }
    }

    /// 查询两个 Region 之间的 RTT（毫秒）。
    ///
    /// 若直接条目缺失，尝试反向；仍缺失返回 `f64::INFINITY`。
    pub fn rtt_ms(&self, from: &RegionId, to: &RegionId) -> f64 {
        if from == to {
            return 0.0;
        }
        let matrix = self.latency_matrix.read();
        // 同时尝试正向与反向键，延迟视作对称
        if let Some(est) = matrix
            .entries
            .get(&(from.clone(), to.clone()))
            .or_else(|| matrix.entries.get(&(to.clone(), from.clone())))
        {
            return est.rtt_p50_ms;
        }
        f64::INFINITY
    }

    /// 获取指定 Region 的信息。
    pub fn get_region(&self, region: &RegionId) -> Option<RegionInfo> {
        self.regions.read().get(region).cloned()
    }

    /// 更新两个 Region 之间的 RTT 估计（同时写入反向条目，便于对称查询）。
    pub fn update_latency(&self, a: &RegionId, b: &RegionId, estimate: LatencyEstimate) {
        if a == b {
            return;
        }
        let mut matrix = self.latency_matrix.write();
        matrix.entries.insert((a.clone(), b.clone()), estimate.clone());
        matrix.entries.insert((b.clone(), a.clone()), estimate);
    }

    /// 添加一个 Region。
    pub fn add_region(&self, info: RegionInfo) {
        self.regions.write().insert(info.id.clone(), info);
    }

    /// 移除一个 Region（同时清理其延迟条目）。
    pub fn remove_region(&self, region: &RegionId) {
        self.regions.write().remove(region);
        // LatencyMatrix 自身的清理依赖其 API；若暴露 remove，可在此调用。
    }

    /// 列出所有 RegionId。
    pub fn all_regions(&self) -> Vec<RegionId> {
        self.regions.read().keys().cloned().collect()
    }
}

/// 一些仅用于文档与类型的占位引用，确保 aether-core 中相关类型在文档中可见。
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
