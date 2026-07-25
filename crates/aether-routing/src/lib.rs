//! Aether LLM Gateway 路由引擎。
//!
//! 该 crate 提供路由策略抽象与一组可组合的具体策略实现：
//!
//! - [`strategy::RoutingStrategy`]：所有策略共有的 trait。
//! - [`kv_aware::KvAwareStrategy`]：基于 KV 缓存命中重叠（参考 Dynamo 成本函数）。
//! - [`model_aware::ModelAwareStrategy`]：基于模型匹配与能力约束的硬性过滤器。
//! - [`load_aware::LoadAwareStrategy`]：基于后端负载指标的策略。
//! - [`topology_aware::TopologyAwareStrategy`]：基于区域间 RTT 的拓扑感知策略。
//! - [`hybrid::HybridStrategy`]：默认的混合策略，融合上述子策略并按权重评分。
//! - [`engine::RoutingEngine`]：上层路由引擎，整合会话亲和与混合策略产出 [`engine::RouteDecision`]。

pub mod strategy;
pub mod kv_aware;
pub mod model_aware;
pub mod load_aware;
pub mod topology_aware;
pub mod hybrid;
pub mod engine;
