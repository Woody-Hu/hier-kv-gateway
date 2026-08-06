//! Estimator registry: combines the builtin [`StandardEstimator`] with
//! operator-provided custom specs and user plugin estimators.
//!
//! The registry is the single entry point the gateway uses. It is constructed
//! once at startup (from `[kv_estimate]` config + any programmatic plugins),
//! wrapped in an `Arc`, and read concurrently on the routing hot path.
//!
//! ## Resolution order
//!
//! For a model name, [`KvEstimationRegistry::spec_for`] asks each estimator in
//! registration order; the first that recognizes the model wins. The builtin
//! [`StandardEstimator`] is always registered first (so builtin models work
//! out of the box), followed by user plugins. A user plugin can therefore
//! *override* a builtin by recognizing the same name first only if it is
//! registered *before* the builtin — use [`with_estimator_front`] for that.
//!
//! By default custom specs configured via `[[kv_estimate.models]]` are folded
//! into the builtin `StandardEstimator` (so they take precedence over builtin
//! pattern matches but defer to an explicitly-registered front plugin).
//!
//! [`with_estimator_front`]: KvEstimationRegistry::with_estimator_front

use std::sync::Arc;

use crate::estimate::{EstimateInput, KvEstimate};
use crate::plugin::{KvEstimator, SpecCatalog, StandardEstimator};
use crate::spec::ModelSpec;

/// Composite estimator: builtin catalog + custom specs + user plugins.
#[derive(Clone)]
pub struct KvEstimationRegistry {
    estimators: Arc<[Arc<dyn KvEstimator>]>,
}

impl KvEstimationRegistry {
    /// Build a registry with just the builtin catalog.
    pub fn with_builtins() -> Self {
        Self::new(StandardEstimator::with_builtins())
    }

    /// Build a registry with a builtin `StandardEstimator` layered over an
    /// operator-provided custom-spec catalog (the common config-driven path).
    pub fn with_catalog(catalog: SpecCatalog) -> Self {
        Self::new(StandardEstimator::with_catalog(catalog))
    }

    /// Build a registry whose first estimator is `builtin`, then push any
    /// user plugins via [`with_estimator`](Self::with_estimator).
    pub fn new(builtin: StandardEstimator) -> Self {
        let est: Arc<dyn KvEstimator> = Arc::new(builtin);
        Self {
            estimators: Arc::from([est]),
        }
    }

    /// Append a user plugin estimator at the back (lowest priority).
    #[must_use]
    pub fn with_estimator(self, plugin: Arc<dyn KvEstimator>) -> Self {
        let mut v: Vec<Arc<dyn KvEstimator>> = self.estimators.iter().cloned().collect();
        v.push(plugin);
        Self {
            estimators: Arc::from(v),
        }
    }

    /// Prepend a user plugin estimator at the front (highest priority), so it
    /// can override the builtin for specific model names.
    #[must_use]
    pub fn with_estimator_front(self, plugin: Arc<dyn KvEstimator>) -> Self {
        let mut v: Vec<Arc<dyn KvEstimator>> = Vec::with_capacity(self.estimators.len() + 1);
        v.push(plugin);
        v.extend(self.estimators.iter().cloned());
        Self {
            estimators: Arc::from(v),
        }
    }

    /// Number of registered estimators (builtin + plugins).
    pub fn estimator_count(&self) -> usize {
        self.estimators.len()
    }

    /// Resolve a spec for `model` by asking each estimator in order.
    pub fn spec_for(&self, model: &str) -> Option<ModelSpec> {
        for est in self.estimators.iter() {
            if let Some(s) = est.spec_for(model) {
                return Some(s);
            }
        }
        None
    }

    /// Estimate the KV footprint of `input` for `model`. The estimator that
    /// recognizes the model provides both the spec and the (possibly custom)
    /// formula. Returns `None` when no estimator recognizes the model.
    pub fn estimate(&self, model: &str, input: &EstimateInput) -> Option<KvEstimate> {
        for est in self.estimators.iter() {
            if let Some(spec) = est.spec_for(model) {
                return Some(est.estimate(&spec, input));
            }
        }
        None
    }
}

impl Default for KvEstimationRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl std::fmt::Debug for KvEstimationRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvEstimationRegistry")
            .field("estimators", &self.estimators.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{KvDtype, ModelSpec};

    #[test]
    fn registry_resolves_builtin() {
        let r = KvEstimationRegistry::with_builtins();
        let spec = r.spec_for("Llama-3-8B").unwrap();
        assert_eq!(spec.num_layers, 32);
    }

    #[test]
    fn registry_estimate_builtin() {
        let r = KvEstimationRegistry::with_builtins();
        let est = r.estimate("Llama-3-8B", &EstimateInput::new(4096, 0)).unwrap();
        assert_eq!(est.bytes, 536_870_912);
    }

    #[test]
    fn registry_unknown_model_returns_none() {
        let r = KvEstimationRegistry::with_builtins();
        assert!(r.spec_for("nope").is_none());
        assert!(r.estimate("nope", &EstimateInput::new(16, 0)).is_none());
    }

    #[test]
    fn registry_custom_catalog_overrides_builtin() {
        let custom = ModelSpec::standard(99, 4, 128, KvDtype::Bf16);
        let r = KvEstimationRegistry::with_catalog(SpecCatalog::new().insert("Qwen2.5-7B", custom));
        let spec = r.spec_for("Qwen2.5-7B").unwrap();
        assert_eq!(spec.num_layers, 99);
    }

    struct FixedEstimator;
    impl KvEstimator for FixedEstimator {
        fn name(&self) -> &str {
            "fixed"
        }
        fn spec_for(&self, model: &str) -> Option<ModelSpec> {
            if model == "fixed-model" {
                Some(ModelSpec::standard(1, 1, 1, KvDtype::Fp8))
            } else {
                None
            }
        }
        fn estimate(&self, spec: &ModelSpec, input: &EstimateInput) -> KvEstimate {
            // Custom formula: return a fixed sentinel (1 byte) regardless of
            // the standard formula, to prove the plugin's estimate is used.
            let _ = (spec, input);
            crate::estimate::KvEstimate {
                bytes: 1,
                blocks: 0,
                per_token_bytes: 1,
                effective_seq_len: 0,
                batch_size: 1,
            }
        }
    }

    #[test]
    fn user_plugin_estimator_is_used_for_its_models() {
        let r = KvEstimationRegistry::with_builtins()
            .with_estimator(Arc::new(FixedEstimator));
        assert_eq!(r.estimator_count(), 2);
        let est = r.estimate("fixed-model", &EstimateInput::new(999, 999)).unwrap();
        // Plugin's custom estimate (1) wins over the standard formula.
        assert_eq!(est.bytes, 1);
    }

    #[test]
    fn user_plugin_does_not_shadow_builtin_for_other_models() {
        let r = KvEstimationRegistry::with_builtins()
            .with_estimator(Arc::new(FixedEstimator));
        let est = r.estimate("Llama-3-8B", &EstimateInput::new(4096, 0)).unwrap();
        // Standard formula, not the plugin's sentinel.
        assert_eq!(est.bytes, 536_870_912);
    }

    #[test]
    fn front_plugin_overrides_builtin() {
        struct OverrideLlama;
        impl KvEstimator for OverrideLlama {
            fn name(&self) -> &str {
                "override"
            }
            fn spec_for(&self, m: &str) -> Option<ModelSpec> {
                if m.contains("Llama-3-8B") {
                    Some(ModelSpec::standard(7, 8, 128, KvDtype::Bf16))
                } else {
                    None
                }
            }
            fn estimate(&self, spec: &ModelSpec, input: &EstimateInput) -> KvEstimate {
                crate::estimate::estimate_kv(spec, input)
            }
        }
        let r = KvEstimationRegistry::with_builtins()
            .with_estimator_front(Arc::new(OverrideLlama));
        let spec = r.spec_for("Llama-3-8B").unwrap();
        assert_eq!(spec.num_layers, 7); // front plugin's 7, not builtin's 32
    }

    #[test]
    fn registry_clone_shares_arc() {
        let r = KvEstimationRegistry::with_builtins();
        let _r2 = r.clone();
        assert_eq!(r.estimator_count(), _r2.estimator_count());
    }
}
