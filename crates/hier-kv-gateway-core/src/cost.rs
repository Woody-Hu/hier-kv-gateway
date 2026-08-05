//! Cost model: per-model unit pricing and projected-dollar-cost estimation.
//!
//! This module is the *data* half of cost-aware routing, kept separate from the
//! *strategy* half ([`hier_kv_gateway_routing::cost_aware::CostAwareStrategy`])
//! in the same way LiteLLM separates its `model_prices_and_context_window.json`
//! price table from the `cost-based-routing` router, and Langfuse separates its
//! `/pricing` API from any consumer.
//!
//! ## Unit convention
//!
//! Internally all prices are **USD per 1 million tokens** (the Glide / Langfuse
//! convention), which is human-readable in TOML and avoids `3e-5`-style floats.
//! A future `CostModel` implementation that fetches OpenRouter-style
//! per-token strings would multiply by `1_000_000` before returning
//! [`ModelPrice`]; callers never see the difference.
//!
//! ## Projection
//!
//! [`CostModel::projected_cost`] is a pure function of `(model, prompt_tokens,
//! estimated_output_tokens)`. The `estimated_output_tokens` is exactly the field
//! [`crate::request::RoutingContext`] already carries, so no new context field is
//! required. After a response completes, actual usage should be reconciled via a
//! budget/admission layer (out of scope for this module — see the design doc).

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Per-model unit prices, in **USD per 1 million tokens**.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelPrice {
    /// USD per 1M input (prompt) tokens.
    pub input_per_1m: f64,
    /// USD per 1M output (completion) tokens.
    pub output_per_1m: f64,
}

impl ModelPrice {
    /// Zero price (e.g. for a self-hosted model whose marginal cost is treated
    /// as free). Still counts as "known price" so the strategy does not exclude
    /// the backend on the `exclude_on_unknown_price` policy.
    pub const FREE: ModelPrice = ModelPrice {
        input_per_1m: 0.0,
        output_per_1m: 0.0,
    };
}

/// Price catalog + projection. Implementations may be a static TOML/JSON table
/// (LiteLLM/Langfuse style, see [`StaticCostModel`]) or a future fetched
/// catalog (OpenRouter/Langfuse HTTP API style).
///
/// Object-safe so it can be held as `Arc<dyn CostModel>` and swapped at startup.
pub trait CostModel: Send + Sync {
    /// Resolve the unit price for a model name.
    ///
    /// Returning `None` means "unknown price"; the caller (the cost-aware
    /// strategy) decides whether to exclude the backend or treat it as neutral
    /// based on its configured policy.
    fn price_for(&self, model: &str) -> Option<ModelPrice>;

    /// Projected dollar cost of serving `prompt_tokens` input +
    /// `est_output_tokens` output on `model`.
    ///
    /// Returns `None` when the model's price is unknown. The default
    /// implementation is a pure function of [`CostModel::price_for`]; override
    /// only for non-token pricing (e.g. time-based `*_per_second`).
    fn projected_cost(
        &self,
        model: &str,
        prompt_tokens: u32,
        est_output_tokens: u32,
    ) -> Option<f64> {
        let p = self.price_for(model)?;
        let c_in = (prompt_tokens as f64 / 1_000_000.0) * p.input_per_1m;
        let c_out = (est_output_tokens as f64 / 1_000_000.0) * p.output_per_1m;
        Some(c_in + c_out)
    }
}

/// Static price catalog built from configuration (LiteLLM `model_prices` /
/// Langfuse `model_costs` analogue).
///
/// `Arc<dyn CostModel>`-compatible: cheap to clone via the inner `Arc`.
#[derive(Clone, Debug, Default)]
pub struct StaticCostModel {
    prices: Arc<HashMap<String, ModelPrice>>,
}

impl StaticCostModel {
    /// Build a catalog from an iterator of `(model, price)` pairs.
    pub fn new<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (String, ModelPrice)>,
    {
        let prices = Arc::new(entries.into_iter().collect::<HashMap<_, _>>());
        Self { prices }
    }

    /// Build from the TOML-friendly [`Vec<PriceEntry>`] used by [`CostConfig`].
    pub fn from_entries(entries: &[PriceEntry]) -> Self {
        Self::new(
            entries
                .iter()
                .map(|e| (e.model.clone(), e.to_price())),
        )
    }

    /// Number of priced models in the catalog.
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
}

impl CostModel for StaticCostModel {
    fn price_for(&self, model: &str) -> Option<ModelPrice> {
        self.prices.get(model).copied()
    }
}

/// TOML-friendly price entry, used inside [`CostConfig`].
///
/// ```toml
/// [[cost.prices]]
/// model = "qwen2.5-7b"
/// input_per_1m = 0.15
/// output_per_1m = 0.60
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriceEntry {
    /// Model name (matches `ModelInstance::model_name`).
    pub model: String,
    /// USD per 1M input tokens.
    pub input_per_1m: f64,
    /// USD per 1M output tokens.
    pub output_per_1m: f64,
}

impl PriceEntry {
    /// Convert to the compact [`ModelPrice`].
    pub fn to_price(&self) -> ModelPrice {
        ModelPrice {
            input_per_1m: self.input_per_1m,
            output_per_1m: self.output_per_1m,
        }
    }
}

/// Cost-model configuration section (`[cost]` in TOML).
///
/// All fields carry defaults so existing configurations keep parsing unchanged
/// (`enabled = false` ⇒ cost-aware routing is off).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CostConfig {
    /// Master switch. When `false` no cost sub-strategy is attached.
    pub enabled: bool,
    /// Static price table. A future impl may fetch this from an HTTP API.
    pub prices: Vec<PriceEntry>,
    /// Hybrid weight of the cost sub-strategy in `[0.0, 1.0]`. `0.0` disables
    /// the cost term even when `enabled = true` (useful for staging the price
    /// table without affecting routing).
    pub weight: f64,
    /// Scale on the projected output term (`>= 1.0` = conservative upper
    /// bound, mirroring `LoadAwareStrategy`'s `w_decode` philosophy).
    pub output_cost_scale: f64,
    /// When `true`, backends whose served model has no price entry are excluded
    /// (`raw_cost = f64::MAX`, score 0 — the `ModelAwareStrategy` exclusion
    /// convention). When `false`, unknown prices are treated as neutral
    /// (`raw_cost = 0.0`) and the other sub-strategies decide.
    pub exclude_on_unknown_price: bool,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            prices: Vec::new(),
            weight: 0.15,
            output_cost_scale: 1.0,
            exclude_on_unknown_price: false,
        }
    }
}

impl CostConfig {
    /// Build a [`StaticCostModel`] from the configured `prices` table.
    pub fn build_model(&self) -> StaticCostModel {
        StaticCostModel::from_entries(&self.prices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_cost_basic() {
        let model = StaticCostModel::new([(
            "m".to_string(),
            ModelPrice {
                input_per_1m: 2.0,
                output_per_1m: 8.0,
            },
        )]);
        // 500k input + 100k output => 1.0 + 0.8 = 1.8 USD
        let c = model.projected_cost("m", 500_000, 100_000).unwrap();
        assert!((c - 1.8).abs() < 1e-9);
    }

    #[test]
    fn projected_cost_unknown_is_none() {
        let model = StaticCostModel::default();
        assert!(model.projected_cost("nope", 10, 10).is_none());
    }

    #[test]
    fn from_entries_round_trip() {
        let entries = vec![PriceEntry {
            model: "m".to_string(),
            input_per_1m: 1.0,
            output_per_1m: 2.0,
        }];
        let model = StaticCostModel::from_entries(&entries);
        let p = model.price_for("m").unwrap();
        assert!((p.input_per_1m - 1.0).abs() < 1e-9);
        assert!((p.output_per_1m - 2.0).abs() < 1e-9);
        assert_eq!(model.len(), 1);
    }

    #[test]
    fn cost_config_default_is_off() {
        let c = CostConfig::default();
        assert!(!c.enabled);
        assert!(c.prices.is_empty());
        assert!((c.weight - 0.15).abs() < 1e-9);
        assert!((c.output_cost_scale - 1.0).abs() < 1e-9);
        assert!(!c.exclude_on_unknown_price);
    }

    #[test]
    fn cost_config_parses_explicit_values() {
        let toml_text = r#"
enabled = true
weight = 0.25
output_cost_scale = 1.5
exclude_on_unknown_price = true

[[prices]]
model = "qwen2.5-7b"
input_per_1m = 0.15
output_per_1m = 0.60

[[prices]]
model = "qwen2.5-72b"
input_per_1m = 3.0
output_per_1m = 12.0
"#;
        let c: CostConfig = toml::from_str(toml_text).unwrap();
        assert!(c.enabled);
        assert_eq!(c.prices.len(), 2);
        assert!((c.weight - 0.25).abs() < 1e-9);
        assert!((c.output_cost_scale - 1.5).abs() < 1e-9);
        assert!(c.exclude_on_unknown_price);
        let m = c.build_model();
        let small = m.price_for("qwen2.5-7b").unwrap();
        assert!((small.input_per_1m - 0.15).abs() < 1e-9);
        // Cheapest-capable sanity: 1M input + 0 output, small < large.
        let cs = m.projected_cost("qwen2.5-7b", 1_000_000, 0).unwrap();
        let cl = m
            .projected_cost("qwen2.5-72b", 1_000_000, 0)
            .unwrap();
        assert!(cs < cl);
    }
}
