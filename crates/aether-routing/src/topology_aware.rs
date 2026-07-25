//! 拓扑感知路由策略。
//!
//! 以本网关所在区域为基准，按到候选后端所在区域的 RTT 估算网络成本。
//! RTT 越小分数越高；当 RTT 未知（返回 `f64::INFINITY`）时分数趋近 0。

use async_trait::async_trait;

use aether_core::error::Result;
use aether_core::ids::{BackendId, RegionId};
use aether_core::request::{RoutingContext, ScoredBackend};
use aether_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// 拓扑感知路由策略。
pub struct TopologyAwareStrategy {
    /// RTT 权重。
    pub w_rtt: f64,
    /// 带宽权重（当前预留，未参与评分公式）。
    pub w_bw: f64,
    /// 本网关所在区域。
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
            // 网络成本：RTT 越大成本越高
            let network_cost = self.w_rtt * rtt;
            // 评分：以 100ms 作为参考基准做归一化
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
        // 拓扑策略始终可用（缺失数据退化为 f64::INFINITY）
        true
    }

    fn weight(&self) -> f64 {
        0.20
    }
}
