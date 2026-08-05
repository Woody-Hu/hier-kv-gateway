//! Routing plugin mechanism.
//!
//! The gateway's routing layer historically hard-coded three sub-strategies
//! (KV / load / topology) plus a model-aware hard filter into
//! [`HybridStrategy`]. As operators ask for more dimensions — cost awareness,
//! large/small model tiering, latency-SLO awareness, etc. — keeping every new
//! dimension as a new constructor argument stops scaling.
//!
//! This module defines [`RoutingPlugin`], a thin extension surface that lets
//! a deployer add a new sub-strategy to the hybrid ensemble *without* forking
//! the engine. A plugin is just a [`RoutingStrategy`] with extra metadata
//! (a stable id, a weight source, an enable predicate) so the engine can
//! normalize, weight, and trace it uniformly.
//!
//! ## What a plugin can do
//!
//! * **Contribute a soft sub-strategy** to the hybrid score (e.g.
//!   [`crate::cost_aware::CostAwareStrategy`] — cheaper backends score
//!   higher).
//! * **Act as the primary scorer** by being installed via
//!   [`RoutingEngine::with_primary_strategy`](crate::engine::RoutingEngine::with_primary_strategy)
//!   (e.g. round-robin, kv-only, load-only, topology-only — these were
//!   previously first-class `StrategyType` variants and are now also
//!   expressible as plugins).
//! * **Wrap the routing decision** (future: tiered fallback, retry
//!   amplification, canary pinning). The trait is intentionally minimal so
//!   these higher-order behaviors can layer on without changing the engine.
//!
//! ## What a plugin cannot do
//!
//! * Replace the model-aware hard filter — that is structural to the hybrid
//!   strategy and gates whether a candidate is even scored. (A future
//!   "filter plugin" extension point could expose this; the current design
//!   keeps it explicit.)
//! * Reorder or suppress the forwarding loop's retry list — that is owned by
//!   the engine and the API layer's circuit-breaker logic.
//!
//! ## Relation to existing strategies
//!
//! Every existing concrete strategy (`KvAwareStrategy`, `LoadAwareStrategy`,
//! `TopologyAwareStrategy`, `RoundRobinStrategy`) already implements
//! [`RoutingStrategy`], so they are trivially wrappable as
//! [`RoutingPlugin`]s via [`RoutingPlugin::from_strategy`]. The
//! `StrategyType::Kv` / `Load` / `Topology` / `RoundRobin` config variants
//! continue to work — they're now internally materialized as plugins passed
//! to `with_primary_strategy` (see `build_routing_engine` in the main
//! binary).
//!
//! ## Example: registering a custom cost-aware sub-strategy
//!
//! ```no_run
//! use std::sync::Arc;
//! use hier_kv_gateway_core::cost::{CostConfig, StaticCostModel};
//! use hier_kv_gateway_routing::cost_aware::CostAwareStrategy;
//! use hier_kv_gateway_routing::plugin::RoutingPlugin;
//!
//! let cfg = CostConfig::default();
//! let model = Arc::new(cfg.build_model());
//! let plugin = RoutingPlugin::from_strategy(
//!     Arc::new(CostAwareStrategy::new(model, cfg)),
//! );
//! // Pass `plugin` to the engine builder; see `build_routing_engine` in
//! // `crates/hier-kv-gateway/src/main.rs` for the wiring.
//! ```

use std::sync::Arc;

use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// A routing plugin: a labeled, weighted, optionally-enabled sub-strategy.
///
/// Plugins are held by the engine as `Arc<dyn RoutingStrategy>` + metadata;
/// the [`RoutingPlugin`] wrapper is a convenience that bundles the strategy
/// with its id/weight/enable-predicate so registration sites stay terse.
#[derive(Clone)]
pub struct RoutingPlugin {
    /// The underlying strategy.
    pub strategy: Arc<dyn RoutingStrategy>,
    /// Stable plugin identifier, used in decision-event tracing and logs.
    pub id: &'static str,
}

impl RoutingPlugin {
    /// Wrap a strategy as a plugin, deriving `id` from
    /// [`RoutingStrategy::name`].
    pub fn from_strategy(strategy: Arc<dyn RoutingStrategy>) -> Self {
        let id = strategy.name();
        Self { strategy, id }
    }

    /// Wrap a strategy as a plugin with an explicit id (useful when the same
    /// strategy type is registered multiple times with different
    /// configurations, e.g. two cost catalogs).
    pub fn with_id(strategy: Arc<dyn RoutingStrategy>, id: &'static str) -> Self {
        Self { strategy, id }
    }

    /// Borrow the underlying strategy trait object.
    pub fn as_strategy(&self) -> &dyn RoutingStrategy {
        self.strategy.as_ref()
    }

    /// Whether the strategy is currently available; delegates to
    /// [`RoutingStrategy::is_available`].
    pub fn is_available(&self, meta: &MetadataStore) -> bool {
        self.strategy.is_available(meta)
    }

    /// The strategy's static weight; delegates to
    /// [`RoutingStrategy::weight`].
    pub fn weight(&self) -> f64 {
        self.strategy.weight()
    }

    /// The plugin's stable id.
    pub fn id(&self) -> &'static str {
        self.id
    }
}

impl std::fmt::Debug for RoutingPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingPlugin")
            .field("id", &self.id)
            .field("weight", &self.weight())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::round_robin::RoundRobinStrategy;
    use hier_kv_gateway_metadata::store::MetadataStore;

    #[test]
    fn from_strategy_derives_id_from_name() {
        let plugin = RoutingPlugin::from_strategy(Arc::new(RoundRobinStrategy::new()));
        assert_eq!(plugin.id(), "round_robin");
        assert!((plugin.weight() - 1.0).abs() < 1e-9);
        assert!(plugin.is_available(&MetadataStore::new()));
    }

    #[test]
    fn with_id_overrides_name() {
        let plugin = RoutingPlugin::with_id(Arc::new(RoundRobinStrategy::new()), "rr-canary");
        assert_eq!(plugin.id(), "rr-canary");
        // Underlying strategy name is unchanged.
        assert_eq!(plugin.strategy.name(), "round_robin");
    }
}
