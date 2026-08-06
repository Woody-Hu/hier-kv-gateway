//! Hier KV Gateway routing engine.
//!
//! This crate provides the routing strategy abstraction and a set of composable
//! concrete strategy implementations:
//!
//! - [`strategy::RoutingStrategy`]: the trait shared by all strategies.
//! - [`kv_aware::KvAwareStrategy`]: cost function based on KV cache hit overlap.
//! - [`model_aware::ModelAwareStrategy`]: hard filter based on model matching and capability constraints.
//! - [`load_aware::LoadAwareStrategy`]: strategy based on backend load metrics.
//! - [`topology_aware::TopologyAwareStrategy`]: topology-aware strategy based on inter-Region RTT.
//! - [`kv_capacity::KvCapacityStrategy`]: capacity-aware strategy that estimates a request's KV-cache footprint and scores backends by available KV / GPU-memory headroom.
//! - [`hybrid::HybridStrategy`]: the default hybrid strategy, fusing the sub-strategies above and scoring by weight.
//! - [`round_robin::RoundRobinStrategy`]: metadata-free baseline that rotates through the candidate set in order.
//! - [`adaptive::AdaptiveWeightController`]: EMA-based feedback loop that adjusts hybrid weights from execution metrics and broadcast load state.
//! - [`engine::RoutingEngine`]: the upper-level routing engine, integrating session affinity and the hybrid strategy to produce [`engine::RouteDecision`].
//! - [`prefix_history::PrefixReuseHistory`]: local prefix reuse history, recording dispatch decisions for degradation routing replay.
//! - [`degradation::DegradationStrategy`]: degradation routing strategy based on prefix reuse history when metadata is missing or stale.

pub mod strategy;
pub mod kv_aware;
pub mod model_aware;
pub mod load_aware;
pub mod topology_aware;
pub mod cost_aware;
pub mod kv_capacity;
pub mod model_tier;
pub mod hybrid;
pub mod round_robin;
pub mod adaptive;
pub mod prefix_history;
pub mod degradation;
pub mod engine;
pub mod plugin;
pub mod tenant_scheduler;
