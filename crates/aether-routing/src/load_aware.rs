//! 负载感知路由策略。
//!
//! 综合考虑候选后端的活跃请求数、队列深度、P99 延迟、GPU 利用率与 KV 缓存使用率，
//! 加权求和得到 `load_cost`。剩余可用容量不足的后端被直接排除（score = 0）。

use async_trait::async_trait;

use aether_core::error::Result;
use aether_core::ids::BackendId;
use aether_core::request::{RoutingContext, ScoredBackend};
use aether_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// 负载感知路由策略。
pub struct LoadAwareStrategy {
    /// 活跃请求数权重。
    pub w_req: f64,
    /// 队列深度权重。
    pub w_queue: f64,
    /// P99 延迟权重（延迟以毫秒为单位，会先除以 100）。
    pub w_lat: f64,
    /// GPU 利用率权重。
    pub w_gpu: f64,
    /// KV 缓存使用率权重。
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
                // 无指标：排除
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 0.0,
                    raw_cost: f64::MAX,
                    meta_version: 0,
                });
                continue;
            };

            // 可用容量 <= 0 时排除
            if m.available_capacity() <= 0 {
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 0.0,
                    raw_cost: f64::MAX,
                    meta_version: 0,
                });
                continue;
            }

            // 加权负载成本
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
        // 至少一个 backend 上报了指标才启用
        meta.backends_all()
            .iter()
            .any(|b| meta.load_get_metrics(&b.id).is_some())
    }

    fn weight(&self) -> f64 {
        0.30
    }
}
