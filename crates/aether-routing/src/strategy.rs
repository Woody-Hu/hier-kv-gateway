//! 路由策略抽象。
//!
//! 所有具体策略（KV 感知、模型感知、负载感知、拓扑感知、混合）都实现 [`RoutingStrategy`]。
//! 策略接收一个候选后端列表与元数据存储句柄，返回每个候选的 [`ScoredBackend`] 评分。

use async_trait::async_trait;

use aether_core::error::Result;
use aether_core::ids::BackendId;
use aether_core::request::{RoutingContext, ScoredBackend};
use aether_metadata::store::MetadataStore;

/// 路由策略通用接口。
///
/// 实现者需保证：
/// - [`evaluate`](RoutingStrategy::evaluate) 在并发场景下可安全调用；
/// - [`is_available`](RoutingStrategy::is_available) 是廉价同步的探针，用于混合策略判断是否启用；
/// - [`weight`](RoutingStrategy::weight) 返回策略在混合评分中的静态权重。
#[async_trait]
pub trait RoutingStrategy: Send + Sync {
    /// 策略名，用于日志与决策追踪。
    fn name(&self) -> &'static str;

    /// 对候选后端逐个评分，返回排序无关的 [`ScoredBackend`] 列表。
    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>>;

    /// 该策略当前是否可用（例如 KV 索引未启动时禁用 KV 策略）。
    fn is_available(&self, meta: &MetadataStore) -> bool;

    /// 策略在混合评分中的静态权重。
    fn weight(&self) -> f64;
}
