//! Large/small model tiering configuration.
//!
//! Holds the *data* half of large↔small model coordination (the
//! [`ModelTierConfig`] section). The *strategy* half lives in the routing
//! crate (`hier_kv_gateway_routing::model_tier::ModelTierStrategy`).
//!
//! ## Open-source landscape
//!
//! Mainstream token gateways do coordinate large and small models, but the
//! *mechanism* varies:
//!
//! - **LiteLLM** ships two explicit features:
//!   * `fallbacks` — a *forwarding-time* chain: try model group A, on failure
//!     fall back to group B. This is realized at the retry layer, not the
//!     scoring layer.
//!   * `model_group_alias` + `cost-based-routing` — *routing-time*: pick the
//!     cheapest capable model from a group. Closer to what we implement here.
//! - **OpenRouter** ranks models by price × capability × latency and exposes
//!   a `ranking` field; clients pick a tier. The gateway itself does not
//!     "fall back" — it routes once.
//! - **Portkey** / **Helicone** offer a "conditional routing" rule engine
//!   (if prompt length > N or `tools` present → route to model B). This is
//!   exactly the *Pick* policy we implement below.
//! - **vLLM** / **SGLang** are single-model servers; tiering is a gateway
//!   concern, not an engine concern.
//!
//! ## Two policies, one strategy
//!
//! [`TierRoutingPolicy::Pick`] scores backends by how well their tier matches
//! the request's *complexity* (short prompt + no tools → prefer small;
//! long prompt or tool-calling → prefer large). This is a *soft* sub-strategy
//! that contributes a weighted term to the hybrid score.
//!
//! [`TierRoutingPolicy::Fallback`] ranks *all* small-model backends ahead of
//! large-model backends unconditionally — when used as the **primary**
//! strategy, the engine's ranked candidate list becomes "try small first,
//! then large", and the forwarding loop's existing retry logic realizes the
//! fallback chain for free (no new retry code needed).
//!
//! ## Honesty note
//!
//! "Seamless" large→small fallback on *quality* signals (e.g. the small
//! model produced a low-confidence answer) requires evaluating the response,
//! which is out of scope for a routing-only strategy. We implement
//! *routing-time* tiering (complexity-aware Pick + order-based Fallback);
//! response-quality-driven fallback is documented as future work.

use serde::{Deserialize, Serialize};

/// A model's tier in the large/small coordination scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Small / cheap / fast model (e.g. 7B-class).
    Small,
    /// Large / expensive / capable model (e.g. 72B-class).
    Large,
}

/// How the tier strategy picks between small and large.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum TierRoutingPolicy {
    /// Complexity-aware scoring: prefer small for simple requests, large for
    /// complex ones, based on configurable thresholds. Acts as a soft
    /// sub-strategy in the hybrid ensemble.
    Pick {
        /// Prompt-token count above which a request is considered "complex"
        /// (prefers large). Defaults to 2048.
        #[serde(default = "default_prompt_threshold")]
        prompt_token_threshold: u32,
        /// Requested `max_tokens` above which a request is considered
        /// "complex" (prefers large). Defaults to 1024.
        #[serde(default = "default_max_token_threshold")]
        max_token_threshold: u32,
        /// When `true`, any request carrying `tools` is routed to large
        /// regardless of token counts (tool-calling needs the larger model's
        /// instruction following).
        #[serde(default = "default_prefer_large_for_tools")]
        prefer_large_for_tools: bool,
    },
    /// Unconditional small-first ordering: rank every small-model backend
    /// ahead of every large-model backend. Intended for use as the *primary*
    /// strategy so the forwarding loop's retry realizes "small then large".
    Fallback,
}

fn default_prompt_threshold() -> u32 {
    2048
}
fn default_max_token_threshold() -> u32 {
    1024
}
fn default_prefer_large_for_tools() -> bool {
    true
}

impl Default for TierRoutingPolicy {
    fn default() -> Self {
        TierRoutingPolicy::Pick {
            prompt_token_threshold: default_prompt_threshold(),
            max_token_threshold: default_max_token_threshold(),
            prefer_large_for_tools: default_prefer_large_for_tools(),
        }
    }
}

/// A single model→tier mapping entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TierEntry {
    /// Model name (matches `ModelInstance::model_name`).
    pub model: String,
    /// Tier assigned to this model.
    pub tier: ModelTier,
}

/// Large/small model tiering configuration section (`[model_tier]` in TOML).
///
/// All fields carry defaults so existing configurations keep parsing unchanged
/// (`enabled = false` ⇒ no tier sub-strategy is attached).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelTierConfig {
    /// Master switch. When `false` no tier sub-strategy is attached.
    pub enabled: bool,
    /// Hybrid weight of the tier sub-strategy in `[0.0, 1.0]`. `0.0` keeps
    /// the strategy attached (so its `is_available` runs) but it contributes
    /// nothing to the hybrid score — useful for staging the tier table.
    pub weight: f64,
    /// The routing policy governing how small/large are chosen.
    pub policy: TierRoutingPolicy,
    /// Model→tier mapping table.
    pub tiers: Vec<TierEntry>,
}

impl Default for ModelTierConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.20,
            policy: TierRoutingPolicy::default(),
            tiers: Vec::new(),
        }
    }
}

impl ModelTierConfig {
    /// Resolve the tier for a model name, if listed.
    pub fn tier_for(&self, model: &str) -> Option<ModelTier> {
        self.tiers
            .iter()
            .find(|e| e.model == model)
            .map(|e| e.tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off_with_pick_policy() {
        let c = ModelTierConfig::default();
        assert!(!c.enabled);
        assert!((c.weight - 0.20).abs() < 1e-9);
        assert!(matches!(c.policy, TierRoutingPolicy::Pick { .. }));
        assert!(c.tiers.is_empty());
    }

    #[test]
    fn parses_pick_policy_with_explicit_thresholds() {
        let toml_text = r#"
enabled = true
weight = 0.25

[policy]
type = "pick"
prompt_token_threshold = 4096
max_token_threshold = 2048
prefer_large_for_tools = false

[[tiers]]
model = "qwen2.5-7b"
tier = "small"

[[tiers]]
model = "qwen2.5-72b"
tier = "large"
"#;
        let c: ModelTierConfig = toml::from_str(toml_text).unwrap();
        assert!(c.enabled);
        assert!((c.weight - 0.25).abs() < 1e-9);
        match c.policy {
            TierRoutingPolicy::Pick {
                prompt_token_threshold,
                max_token_threshold,
                prefer_large_for_tools,
            } => {
                assert_eq!(prompt_token_threshold, 4096);
                assert_eq!(max_token_threshold, 2048);
                assert!(!prefer_large_for_tools);
            }
            _ => panic!("expected Pick policy"),
        }
        assert_eq!(c.tier_for("qwen2.5-7b"), Some(ModelTier::Small));
        assert_eq!(c.tier_for("qwen2.5-72b"), Some(ModelTier::Large));
        assert_eq!(c.tier_for("unknown"), None);
    }

    #[test]
    fn parses_fallback_policy() {
        let toml_text = r#"
enabled = true
weight = 0.0

[policy]
type = "fallback"

[[tiers]]
model = "m-small"
tier = "small"
"#;
        let c: ModelTierConfig = toml::from_str(toml_text).unwrap();
        assert!(matches!(c.policy, TierRoutingPolicy::Fallback));
        assert_eq!(c.tier_for("m-small"), Some(ModelTier::Small));
    }

    #[test]
    fn absent_section_uses_default() {
        let c: ModelTierConfig = toml::from_str("").unwrap();
        assert!(!c.enabled);
    }

    #[test]
    fn pick_policy_defaults_thresholds_when_omitted() {
        let toml_text = r#"
enabled = true
[policy]
type = "pick"
"#;
        let c: ModelTierConfig = toml::from_str(toml_text).unwrap();
        match c.policy {
            TierRoutingPolicy::Pick {
                prompt_token_threshold,
                max_token_threshold,
                prefer_large_for_tools,
            } => {
                assert_eq!(prompt_token_threshold, 2048);
                assert_eq!(max_token_threshold, 1024);
                assert!(prefer_large_for_tools);
            }
            _ => panic!("expected Pick policy"),
        }
    }
}
