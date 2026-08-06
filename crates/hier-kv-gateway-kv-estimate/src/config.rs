//! Configuration for the KV-estimation module (`[kv_estimate]` in TOML).
//!
//! This section is the *data* half of KV-capacity-aware routing. It lives in
//! the independent `hier-kv-gateway-kv-estimate` crate (so the estimator is
//! reusable with no gateway dependency) and is referenced from
//! `GatewayConfig` in `hier-kv-gateway-core`. The *behaviour* half — the
//! `KvCapacityStrategy` that turns estimates into routing scores — lives in
//! the routing crate.
//!
//! All fields carry defaults so existing configurations keep parsing unchanged
//! (`enabled = false` ⇒ no KV-capacity sub-strategy is attached).
//!
//! ## TOML shape
//!
//! ```toml
//! [kv_estimate]
//! enabled = true
//! weight = 0.25
//! gpu_mem_safety_fraction = 0.5
//! exclude_on_unknown_spec = false
//!
//! [[kv_estimate.models]]
//! name = "my-private-model"
//! num_layers = 20
//! num_kv_heads = 4
//! head_dim = 96
//! dtype = "fp16"
//! ```
//!
//! Each `[[kv_estimate.models]]` entry is a [`NamedModelSpec`]: the `name` is
//! the routing key, and the remaining (flattened) fields are the
//! [`ModelSpec`]. Custom entries override builtins of the same name.

use serde::{Deserialize, Serialize};

use crate::plugin::SpecCatalog;
use crate::spec::{ModelSpec, NamedModelSpec};

/// KV-estimation configuration section (`[kv_estimate]` in TOML).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct KvEstimateConfig {
    /// Master switch. When `false` no KV-capacity sub-strategy is attached
    /// and the estimator is not constructed on the routing path.
    pub enabled: bool,
    /// Hybrid weight of the KV-capacity sub-strategy in `[0.0, 1.0]`. `0.0`
    /// keeps the strategy attached (so its availability probe runs) but it
    /// contributes nothing to the hybrid score — useful for staging the
    /// model table without affecting routing.
    pub weight: f64,
    /// Fraction of *free* GPU memory considered available for KV-cache
    /// growth when a backend reports GPU memory but not KV block totals.
    /// KV is not the only GPU memory user (weights, activations), so this is
    /// a conservative cap. `0.5` means "at most half of the currently free
    /// GPU memory may be claimed by this request's KV growth".
    pub gpu_mem_safety_fraction: f64,
    /// When `true`, backends whose served model has no resolvable spec are
    /// excluded (raw_cost = ∞, score 0). When `false` (default), they are
    /// treated as neutral and the other sub-strategies decide — safer, since
    /// excluding on an unknown model could starve a backend that does in fact
    /// have room.
    pub exclude_on_unknown_spec: bool,
    /// Operator-provided custom model specs, layered over the builtin
    /// catalog. A custom entry overrides a builtin of the same name.
    pub models: Vec<NamedModelSpec>,
}

impl Default for KvEstimateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.20,
            gpu_mem_safety_fraction: 0.5,
            exclude_on_unknown_spec: false,
            models: Vec::new(),
        }
    }
}

impl KvEstimateConfig {
    /// Build the spec catalog (builtin + custom) from this config.
    ///
    /// Construction is a one-time startup cost (the names are cloned into the
    /// catalog map); the returned catalog is then read allocation-free on the
    /// hot path.
    pub fn build_catalog(&self) -> SpecCatalog {
        SpecCatalog::from_specs(
            self.models
                .iter()
                .map(|n| (n.name.clone(), n.spec)),
        )
    }

    /// Whether the config actively enables KV-capacity routing.
    pub fn active(&self) -> bool {
        self.enabled
    }

    /// Iterate over the custom model specs as `(name, spec)` pairs (helper
    /// for tests and for the routing strategy that needs the spec table).
    pub fn custom_specs(&self) -> impl Iterator<Item = (&str, ModelSpec)> {
        self.models.iter().map(|n| (n.name.as_str(), n.spec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{AttentionKind, KvDtype};

    #[test]
    fn default_is_off() {
        let c = KvEstimateConfig::default();
        assert!(!c.enabled);
        assert!((c.weight - 0.20).abs() < 1e-9);
        assert!((c.gpu_mem_safety_fraction - 0.5).abs() < 1e-9);
        assert!(!c.exclude_on_unknown_spec);
        assert!(c.models.is_empty());
        assert!(!c.active());
    }

    #[test]
    fn parses_explicit_values_and_custom_models() {
        let toml_text = r#"
enabled = true
weight = 0.25
gpu_mem_safety_fraction = 0.6
exclude_on_unknown_spec = true

[[models]]
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
"#;
        let c: KvEstimateConfig = toml::from_str(toml_text).unwrap();
        assert!(c.enabled);
        assert!((c.weight - 0.25).abs() < 1e-9);
        assert!((c.gpu_mem_safety_fraction - 0.6).abs() < 1e-9);
        assert!(c.exclude_on_unknown_spec);
        assert_eq!(c.models.len(), 1);
        assert_eq!(c.models[0].name, "my-private-model");
        assert_eq!(c.models[0].spec.num_layers, 20);
        assert_eq!(c.models[0].spec.dtype, KvDtype::Fp16);
    }

    #[test]
    fn absent_section_uses_default() {
        let c: KvEstimateConfig = toml::from_str("").unwrap();
        assert!(!c.enabled);
    }

    #[test]
    fn mla_model_entry_parses() {
        let toml_text = r#"
enabled = true

[[models]]
name = "custom-mla"
num_layers = 30
attention = "mla"
dtype = "bf16"
kv_lora_rank = 384
qk_rope_head_dim = 48
"#;
        let c: KvEstimateConfig = toml::from_str(toml_text).unwrap();
        assert_eq!(c.models.len(), 1);
        assert_eq!(c.models[0].spec.attention, AttentionKind::Mla);
        assert_eq!(c.models[0].spec.kv_lora_rank, 384);
        assert_eq!(c.models[0].spec.qk_rope_head_dim, 48);
    }

    #[test]
    fn build_catalog_layers_custom_over_builtin() {
        let cfg = KvEstimateConfig {
            enabled: true,
            weight: 0.2,
            gpu_mem_safety_fraction: 0.5,
            exclude_on_unknown_spec: false,
            models: vec![NamedModelSpec::new(
                "Qwen2.5-7B",
                ModelSpec::standard(99, 4, 128, KvDtype::Bf16),
            )],
        };
        let cat = cfg.build_catalog();
        let s = cat.lookup("Qwen2.5-7B").unwrap();
        assert_eq!(s.num_layers, 99); // custom overrides builtin's 28
        // builtin still reachable for other names
        assert!(cat.lookup("Llama-3-8B").is_some());
    }

    #[test]
    fn custom_specs_iterator_pairs_name_and_spec() {
        let cfg = KvEstimateConfig {
            enabled: true,
            weight: 0.2,
            gpu_mem_safety_fraction: 0.5,
            exclude_on_unknown_spec: false,
            models: vec![
                NamedModelSpec::new("a", ModelSpec::standard(1, 1, 1, KvDtype::Fp16)),
                NamedModelSpec::new("b", ModelSpec::standard(2, 2, 2, KvDtype::Bf16)),
            ],
        };
        let pairs: Vec<_> = cfg.custom_specs().collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[0].1.num_layers, 1);
        assert_eq!(pairs[1].0, "b");
    }

    #[test]
    fn config_round_trips_toml() {
        let cfg = KvEstimateConfig {
            enabled: true,
            weight: 0.3,
            gpu_mem_safety_fraction: 0.6,
            exclude_on_unknown_spec: true,
            models: vec![NamedModelSpec::new(
                "x",
                ModelSpec::mla(30, 384, 48, KvDtype::Bf16),
            )],
        };
        let toml_text = toml::to_string(&cfg).unwrap();
        let back: KvEstimateConfig = toml::from_str(&toml_text).unwrap();
        assert_eq!(back.enabled, cfg.enabled);
        assert!((back.weight - 0.3).abs() < 1e-9);
        assert_eq!(back.models.len(), 1);
        assert_eq!(back.models[0].spec.attention, AttentionKind::Mla);
    }
}
