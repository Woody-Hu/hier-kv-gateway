//! Estimator plugin trait and the default [`StandardEstimator`].
//!
//! ## Two extension paths
//!
//! The KV-estimation module is plugin-driven so new models can be added
//! without forking the codebase:
//!
//! 1. **Add a model spec (data, no code).** Insert a `[[kv_estimate.models]]`
//!    entry in the gateway TOML, or call
//!    [`SpecCatalog::insert`]. The standard formula in
//!    [`crate::estimate::estimate_kv`] covers it. This is the path for the
//!    overwhelming majority of new models — any MHA/GQA/MQA/MLA architecture
//!    is a few config fields, not a new formula.
//!
//! 2. **Add a custom estimator (code).** Implement [`KvEstimator`] for a
//!    genuinely novel attention scheme whose KV footprint cannot be expressed
//!    by the standard formula, and register it via
//!    [`KvEstimationRegistry::with_estimator`](crate::registry::KvEstimationRegistry::with_estimator).
//!    The registry asks each estimator `spec_for(model)` in registration
//!    order; the first one that recognizes the model both provides the spec
//!    *and* computes the estimate, so a custom formula is fully honored.
//!
//! This mirrors how vLLM's KV-size calculation is parametrized by
//! `config.json` fields for standard models but overridden by engine code for
//! exotic cases (e.g. Cross-Attention, Mamba/SSM state).
//!
//! ## Allocation on the hot path
//!
//! [`StandardEstimator::spec_for`] and [`SpecCatalog::lookup`] return
//! `Option<ModelSpec>` *by value*. Because [`ModelSpec`] is `Copy` and
//! [`crate::catalog::lookup_builtin`] is allocation-free, the entire
//! `spec_for` → `estimate` path on the routing hot loop allocates **zero**
//! bytes (proven by `tests/alloc_free.rs`). The custom-spec `HashMap` is read
//! via `Borrow<str>` so the lookup key is a borrowed `&str` — no clone.

use std::sync::Arc;

use crate::catalog::{builtin_specs_raw, lookup_builtin};
use crate::estimate::{estimate_kv, EstimateInput, KvEstimate};
use crate::spec::ModelSpec;

/// Object-safe estimator: resolves a model name to its spec and computes the
/// KV-cache footprint.
///
/// The default implementation ([`StandardEstimator`]) uses the builtin +
/// operator-provided spec catalog and the analytical formula in
/// [`crate::estimate`]. Custom implementations override both methods to
/// support architectures the standard formula does not cover.
pub trait KvEstimator: Send + Sync {
    /// Estimator name, for logging and decision tracing.
    fn name(&self) -> &str;

    /// Resolve the architectural spec for `model`, if recognized.
    ///
    /// Returning `ModelSpec` by value (it is `Copy`) keeps the hot path
    /// allocation-free.
    fn spec_for(&self, model: &str) -> Option<ModelSpec>;

    /// Compute the KV-cache footprint of `input` under `spec`.
    ///
    /// `spec` is guaranteed to have come from this estimator's own
    /// [`spec_for`](KvEstimator::spec_for), so a custom estimator can carry
    /// private state between the two calls if needed.
    fn estimate(&self, spec: &ModelSpec, input: &EstimateInput) -> KvEstimate;
}

/// The default estimator: builtin catalog + operator-provided specs, scored
/// with the standard analytical formula.
///
/// Cheap to clone via the inner `Arc`.
#[derive(Clone, Default)]
pub struct StandardEstimator {
    catalog: Arc<SpecCatalog>,
}

impl StandardEstimator {
    /// Build an estimator with the builtin catalog only.
    pub fn with_builtins() -> Self {
        Self {
            catalog: Arc::new(SpecCatalog::with_builtins()),
        }
    }

    /// Build an estimator from an operator-provided catalog (custom specs
    /// layered on top of the builtins).
    pub fn with_catalog(catalog: SpecCatalog) -> Self {
        Self {
            catalog: Arc::new(catalog),
        }
    }

    /// Borrow the underlying spec catalog.
    pub fn catalog(&self) -> &SpecCatalog {
        &self.catalog
    }
}

impl KvEstimator for StandardEstimator {
    fn name(&self) -> &str {
        "standard"
    }

    fn spec_for(&self, model: &str) -> Option<ModelSpec> {
        self.catalog.lookup(model)
    }

    fn estimate(&self, spec: &ModelSpec, input: &EstimateInput) -> KvEstimate {
        estimate_kv(spec, input)
    }
}

/// Spec catalog: an owned, immutable map from model name → spec, layered over
/// the builtin pattern-matching table.
///
/// Lookup order: exact-name custom spec → builtin pattern match. Custom specs
/// take precedence so operators can override a builtin (e.g. to correct a
/// dtype or add a sliding window the builtin missed).
///
/// The map keys are the model names; the values are nameless [`ModelSpec`]s
/// (`Copy`). [`SpecCatalog::lookup`] returns `Option<ModelSpec>` by copy, so
/// resolving a spec never touches the heap.
#[derive(Clone, Default)]
pub struct SpecCatalog {
    custom: Arc<std::collections::HashMap<String, ModelSpec>>,
}

impl SpecCatalog {
    /// Empty catalog (no custom specs; builtin matching still applies via
    /// [`lookup_builtin`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty catalog + builtin table reference (kept for explicitness; the
    /// builtin table is always consulted regardless).
    pub fn with_builtins() -> Self {
        Self::default()
    }

    /// Build a catalog from an iterator of `(name, spec)` pairs (custom specs
    /// keyed by name). Later entries override earlier ones for the same name.
    pub fn from_specs<I>(specs: I) -> Self
    where
        I: IntoIterator<Item = (String, ModelSpec)>,
    {
        let map: std::collections::HashMap<String, ModelSpec> = specs.into_iter().collect();
        Self {
            custom: Arc::new(map),
        }
    }

    /// Add (or replace) a custom spec. Returns a new catalog (the catalog is
    /// immutable; this rebuilds the inner map).
    pub fn insert(self, name: impl Into<String>, spec: ModelSpec) -> Self {
        let mut map = Arc::try_unwrap(self.custom).unwrap_or_else(|arc| (*arc).clone());
        map.insert(name.into(), spec);
        Self {
            custom: Arc::new(map),
        }
    }

    /// Number of custom (operator-provided) specs.
    pub fn custom_len(&self) -> usize {
        self.custom.len()
    }

    /// Look up a spec by model name: exact custom match first, then builtin
    /// pattern match. Returns a `Copy` of the spec — allocation-free.
    pub fn lookup(&self, model: &str) -> Option<ModelSpec> {
        if let Some(s) = self.custom.get(model) {
            return Some(*s);
        }
        lookup_builtin(model)
    }
}

impl std::fmt::Debug for StandardEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StandardEstimator")
            .field("custom_specs", &self.catalog.custom_len())
            .field("builtin_specs", &builtin_specs_raw().len())
            .finish()
    }
}

impl std::fmt::Debug for SpecCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpecCatalog")
            .field("custom", &self.custom.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{KvDtype, ModelSpec};

    #[test]
    fn standard_estimator_resolves_builtin() {
        let est = StandardEstimator::with_builtins();
        let spec = est.spec_for("Qwen2.5-7B").unwrap();
        assert_eq!(spec.num_layers, 28);
    }

    #[test]
    fn standard_estimator_unknown_is_none() {
        let est = StandardEstimator::with_builtins();
        assert!(est.spec_for("no-such-model").is_none());
    }

    #[test]
    fn standard_estimator_uses_formula() {
        let est = StandardEstimator::with_builtins();
        let spec = est.spec_for("Llama-3-8B").unwrap();
        let r = est.estimate(&spec, &EstimateInput::new(4096, 0));
        // 131_072 B/token * 4096 = 536_870_912
        assert_eq!(r.bytes, 536_870_912);
    }

    #[test]
    fn custom_spec_overrides_builtin() {
        let custom = ModelSpec::standard(99, 4, 128, KvDtype::Bf16);
        let est = StandardEstimator::with_catalog(SpecCatalog::new().insert("Qwen2.5-7B", custom));
        let spec = est.spec_for("Qwen2.5-7B").unwrap();
        assert_eq!(spec.num_layers, 99); // custom wins over builtin's 28
    }

    #[test]
    fn custom_spec_added_for_unknown_model() {
        let custom = ModelSpec::standard(20, 4, 96, KvDtype::Fp16);
        let est =
            StandardEstimator::with_catalog(SpecCatalog::new().insert("my-private-model", custom));
        let spec = est.spec_for("my-private-model").unwrap();
        assert_eq!(spec.num_layers, 20);
    }

    #[test]
    fn catalog_from_specs_builder() {
        let specs = vec![
            ("a".to_string(), ModelSpec::standard(1, 1, 1, KvDtype::Fp16)),
            ("b".to_string(), ModelSpec::standard(2, 2, 2, KvDtype::Bf16)),
        ];
        let cat = SpecCatalog::from_specs(specs);
        assert_eq!(cat.custom_len(), 2);
        assert_eq!(cat.lookup("a").unwrap().num_layers, 1);
        // builtin still reachable for non-custom names
        assert!(cat.lookup("Qwen2.5-7B").is_some());
    }

    #[test]
    fn from_specs_later_entry_overrides() {
        let specs = vec![
            ("m".to_string(), ModelSpec::standard(10, 1, 1, KvDtype::Fp16)),
            ("m".to_string(), ModelSpec::standard(20, 1, 1, KvDtype::Fp16)),
        ];
        let cat = SpecCatalog::from_specs(specs);
        assert_eq!(cat.lookup("m").unwrap().num_layers, 20);
    }

    #[test]
    fn insert_replaces_existing_custom() {
        let cat = SpecCatalog::new()
            .insert("m", ModelSpec::standard(10, 1, 1, KvDtype::Fp16))
            .insert("m", ModelSpec::standard(30, 1, 1, KvDtype::Fp16));
        assert_eq!(cat.lookup("m").unwrap().num_layers, 30);
    }

    #[test]
    fn estimator_name() {
        let est = StandardEstimator::with_builtins();
        assert_eq!(est.name(), "standard");
    }

    #[test]
    fn standard_estimator_clone_is_cheap() {
        let est = StandardEstimator::with_builtins();
        let _cloned = est.clone();
        // Just asserting clone compiles and runs; the Arc makes it cheap.
    }

    #[test]
    fn catalog_lookup_returns_copy_no_aliasing() {
        let cat = SpecCatalog::new().insert("m", ModelSpec::standard(32, 8, 128, KvDtype::Bf16));
        let mut a = cat.lookup("m").unwrap();
        a.num_layers = 999;
        // Local copy reflects the mutation...
        assert_eq!(a.num_layers, 999);
        // ...but a fresh lookup is unaffected.
        let b = cat.lookup("m").unwrap();
        assert_eq!(b.num_layers, 32, "lookup returns an independent copy");
    }
}
