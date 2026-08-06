//! Standalone, plugin-driven KV-cache memory estimation for LLM inference
//! gateways.
//!
//! This crate answers one question: **given a model's architecture and a
//! request's shape, how much GPU KV-cache memory does the request need?**
//!
//! It is deliberately a *leaf* crate — it depends on nothing gateway-specific
//! (only `serde` + `thiserror`) — so the estimator is reusable by any
//! inference-router, scheduler, or capacity-planning tool, and so the hot
//! path stays allocation-free and nanosecond-fast.
//!
//! ## What it computes
//!
//! The analytical KV-cache formulas used by mainstream engines (vLLM, SGLang,
//! Mooncake, llm-d), not a simulation:
//!
//! - **Standard attention (MHA / GQA / MQA):**
//!   `per_token = 2 * layers * kv_heads * head_dim * dtype_bytes`
//! - **MLA (DeepSeek-V2 / V3 / R1):**
//!   `per_token = layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes`
//!   — a single latent reconstructs both K and V, so there is no factor of 2.
//! - **Sliding-window** attention caps the effective cached sequence length.
//! - **Block paging** (vLLM/SGLang/Mooncake `block_size`) pads the footprint
//!   up to whole blocks.
//!
//! See the [`estimate`] module docs for the full formula derivation and
//! provenance.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                  KvEstimationRegistry                     │
//! │   (builtin StandardEstimator + custom specs + plugins)   │
//! │                                                          │
//! │   spec_for(model) ──► first estimator that matches       │
//! │   estimate(model, input) ──► that estimator's formula    │
//! └──────────────────────────────────────────────────────────┘
//!         ▲                              ▲
//!         │                              │
//!   ┌─────┴──────┐              ┌────────┴────────┐
//!   │  Standard  │              │  user plugin    │
//!   │ Estimator  │              │ KvEstimator impl│
//!   │ (formula)  │              │ (custom formula)│
//!   └─────┬──────┘              └─────────────────┘
//!         │
//!   ┌─────┴──────────────────────┐
//!   │  SpecCatalog               │
//!   │  custom specs (config TOML)│
//!   │  + builtin pattern table   │
//!   │    (Llama/Qwen/Mistral/    │
//!   │     Gemma/DeepSeek/GLM…)   │
//!   └────────────────────────────┘
//! ```
//!
//! ## Two extension paths
//!
//! 1. **Add a model (data):** a `[[kv_estimate.models]]` TOML entry, or
//!    [`SpecCatalog::insert`]. The standard formula covers it. This is the
//!    path for ~all new models.
//! 2. **Add a custom estimator (code):** implement [`KvEstimator`] for a
//!    novel architecture and register via
//!    [`KvEstimationRegistry::with_estimator`].
//!
//! ## Quick start
//!
//! ```
//! use hier_kv_gateway_kv_estimate::{
//!     KvEstimationRegistry, EstimateInput,
//! };
//!
//! let registry = KvEstimationRegistry::with_builtins();
//! let input = EstimateInput::new(4096, 1024).with_block_size(16);
//! let est = registry.estimate("Qwen2.5-7B", &input).unwrap();
//! println!("{} bytes ({} blocks)", est.bytes, est.blocks);
//! ```

pub mod catalog;
pub mod config;
pub mod estimate;
pub mod plugin;
pub mod registry;
pub mod spec;

pub use catalog::{builtin_entries, builtin_specs_raw, lookup_builtin};
pub use config::KvEstimateConfig;
pub use estimate::{estimate_kv, per_block_bytes, per_token_bytes, EstimateInput, KvEstimate};
pub use plugin::{KvEstimator, SpecCatalog, StandardEstimator};
pub use registry::KvEstimationRegistry;
pub use spec::{AttentionKind, KvDtype, ModelSpec, NamedModelSpec};
