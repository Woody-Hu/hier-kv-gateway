//! Model architectural specification for KV-cache estimation.
//!
//! [`ModelSpec`] captures exactly the architectural parameters that determine
//! a model's per-token KV-cache footprint — nothing more. It is deliberately
//! a plain `Copy`-able struct of integers (no `String`, no `Arc`) so the
//! hot-path estimator ([`crate::estimate::estimate_kv`]) and the registry
//! lookup ([`crate::registry::KvEstimationRegistry::spec_for`]) stay
//! **allocation-free** — a property proven by `tests/alloc_free.rs`.
//!
//! The model *name* is not a formula parameter, so it is deliberately kept
//! out of [`ModelSpec`]. Names live as catalog keys alongside specs (see
//! [`NamedModelSpec`] and [`crate::catalog`]).
//!
//! ## Field sources
//!
//! Every field maps 1:1 to a HuggingFace `config.json` entry, so a new model
//! can be added by copying its config values (see the builtin catalog in
//! [`crate::catalog`] for worked examples). The mapping is:
//!
//! | `ModelSpec` field     | HuggingFace `config.json` field            |
//! |-----------------------|--------------------------------------------|
//! | `num_layers`          | `num_hidden_layers`                        |
//! | `num_kv_heads`        | `num_key_value_heads` (= `num_attention_heads` for MHA) |
//! | `head_dim`            | `head_dim` (or `hidden_size / num_attention_heads`) |
//! | `attention`           | `Standard` by default; `Mla` for DeepSeek-V2/V3 |
//! | `dtype`               | `torch_dtype`                              |
//! | `kv_lora_rank`        | `kv_lora_rank` (DeepSeek MLA only)         |
//! | `qk_rope_head_dim`    | `qk_rope_head_dim` (DeepSeek MLA only)     |
//! | `sliding_window`      | `sliding_window` (0 = none)                |

use serde::{Deserialize, Serialize};

/// Attention family, selecting which KV-cache formula applies.
///
/// `Standard` covers MHA, GQA and MQA uniformly — they differ only in
/// `num_kv_heads`, which is already a field of [`ModelSpec`]. `Mla`
/// (Multi-head Latent Attention, DeepSeek-V2/V3) compresses the KV cache
/// into a single latent vector and uses a different formula (see
/// [`crate::estimate`] for the math).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    /// Standard attention (MHA / GQA / MQA). Cache stores full K and V.
    #[default]
    Standard,
    /// Multi-head Latent Attention (DeepSeek-V2 / V3 / R1). Cache stores a
    /// single compressed latent `c_kv` plus a RoPE-decoupled `k_pe`.
    Mla,
}

/// Element dtype of the KV cache tensor.
///
/// `bytes()` returns the per-element size. FP8 and INT8 cache (vLLM/SGLang
/// `--kv-cache-dtype fp8`) halve the footprint relative to the default
/// FP16/BF16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KvDtype {
    /// 32-bit float (4 bytes).
    Fp32,
    /// 16-bit float (2 bytes).
    Fp16,
    /// Brain float 16 (2 bytes).
    #[default]
    Bf16,
    /// 8-bit float (1 byte) — vLLM/SGLang `fp8` KV cache.
    Fp8,
    /// 8-bit int (1 byte) — vLLM `int8` KV cache.
    Int8,
}

impl KvDtype {
    /// Bytes per KV-cache element.
    pub const fn bytes(self) -> u32 {
        match self {
            KvDtype::Fp32 => 4,
            KvDtype::Fp16 | KvDtype::Bf16 => 2,
            KvDtype::Fp8 | KvDtype::Int8 => 1,
        }
    }
}

/// Architectural spec of one model, sufficient to compute its KV-cache size.
///
/// This is **the** value type of the estimator. It is `Copy` (plain integers
/// and two `#[derive(Copy)]` enums), so resolving a spec from the registry
/// and passing it to [`crate::estimate::estimate_kv`] never allocates.
///
/// All optional-ish fields use `u32` with `0` meaning "not applicable" so the
/// type is `Copy` and TOML-friendly (no `Option` tag noise in config files).
/// The model name is *not* here — see [`NamedModelSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSpec {
    /// Number of transformer layers (`num_hidden_layers`).
    pub num_layers: u32,
    /// Number of KV heads. For MHA this equals the query-head count; for GQA
    /// it is smaller (`num_key_value_heads`); for MQA it is 1. Ignored for
    /// [`AttentionKind::Mla`].
    pub num_kv_heads: u32,
    /// Per-head dimension (`head_dim`, or `hidden_size / num_attention_heads`).
    /// Ignored for [`AttentionKind::Mla`].
    pub head_dim: u32,
    /// Attention family — selects the KV formula.
    pub attention: AttentionKind,
    /// KV-cache element dtype.
    pub dtype: KvDtype,
    /// MLA compressed-KV latent rank (`kv_lora_rank`). Used only for
    /// [`AttentionKind::Mla`]; `0` otherwise.
    pub kv_lora_rank: u32,
    /// MLA RoPE-decoupled key head dim (`qk_rope_head_dim`). Used only for
    /// [`AttentionKind::Mla`]; `0` otherwise.
    pub qk_rope_head_dim: u32,
    /// Sliding-window attention size in tokens (`sliding_window`). `0` means
    /// no sliding window (full attention). When > 0, the effective cached
    /// sequence length is capped at this value.
    pub sliding_window: u32,
}

impl ModelSpec {
    /// Build a spec with the minimum Standard-attention fields, leaving MLA /
    /// sliding-window fields at their `0` defaults.
    pub const fn standard(
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u32,
        dtype: KvDtype,
    ) -> Self {
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            attention: AttentionKind::Standard,
            dtype,
            kv_lora_rank: 0,
            qk_rope_head_dim: 0,
            sliding_window: 0,
        }
    }

    /// Build a DeepSeek-style MLA spec.
    pub const fn mla(
        num_layers: u32,
        kv_lora_rank: u32,
        qk_rope_head_dim: u32,
        dtype: KvDtype,
    ) -> Self {
        Self {
            num_layers,
            num_kv_heads: 0,
            head_dim: 0,
            attention: AttentionKind::Mla,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            sliding_window: 0,
        }
    }

    /// Set the sliding window (builder style).
    #[must_use]
    pub const fn with_sliding_window(mut self, window: u32) -> Self {
        self.sliding_window = window;
        self
    }
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self {
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            attention: AttentionKind::Standard,
            dtype: KvDtype::Bf16,
            kv_lora_rank: 0,
            qk_rope_head_dim: 0,
            sliding_window: 0,
        }
    }
}

/// A model name paired with its [`ModelSpec`] — the shape stored in the
/// catalog and parsed from `[[kv_estimate.models]]` TOML entries.
///
/// The `name` is the routing/lookup key; the `spec` (flattened into the same
/// TOML table) holds the architecture. Keeping name out of [`ModelSpec`]
/// itself is what lets the hot path be `Copy` and allocation-free.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedModelSpec {
    /// Model name (matches the name the gateway routes on).
    pub name: String,
    /// Architectural spec, flattened into the same TOML table as `name`.
    #[serde(flatten)]
    pub spec: ModelSpec,
}

impl NamedModelSpec {
    /// Build a named spec pair.
    pub fn new(name: impl Into<String>, spec: ModelSpec) -> Self {
        Self {
            name: name.into(),
            spec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_bytes() {
        assert_eq!(KvDtype::Fp32.bytes(), 4);
        assert_eq!(KvDtype::Fp16.bytes(), 2);
        assert_eq!(KvDtype::Bf16.bytes(), 2);
        assert_eq!(KvDtype::Fp8.bytes(), 1);
        assert_eq!(KvDtype::Int8.bytes(), 1);
    }

    #[test]
    fn attention_kind_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&AttentionKind::Standard).unwrap(),
            r#""standard""#
        );
        assert_eq!(
            serde_json::from_str::<AttentionKind>(r#""mla""#).unwrap(),
            AttentionKind::Mla
        );
    }

    #[test]
    fn dtype_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&KvDtype::Bf16).unwrap(),
            r#""bf16""#
        );
        assert_eq!(
            serde_json::from_str::<KvDtype>(r#""fp8""#).unwrap(),
            KvDtype::Fp8
        );
    }

    #[test]
    fn standard_builder_sets_defaults() {
        let s = ModelSpec::standard(32, 8, 128, KvDtype::Bf16);
        assert_eq!(s.num_layers, 32);
        assert_eq!(s.num_kv_heads, 8);
        assert_eq!(s.attention, AttentionKind::Standard);
        assert_eq!(s.sliding_window, 0);
        assert_eq!(s.kv_lora_rank, 0);
    }

    #[test]
    fn mla_builder_zeroes_standard_fields() {
        let s = ModelSpec::mla(61, 512, 64, KvDtype::Bf16);
        assert_eq!(s.attention, AttentionKind::Mla);
        assert_eq!(s.num_kv_heads, 0);
        assert_eq!(s.head_dim, 0);
        assert_eq!(s.kv_lora_rank, 512);
        assert_eq!(s.qk_rope_head_dim, 64);
    }

    #[test]
    fn model_spec_is_copy_and_clone() {
        let s = ModelSpec::standard(32, 8, 128, KvDtype::Bf16);
        let copied = s; // Copy
        let cloned = s; // Copy again
        assert_eq!(s, copied);
        assert_eq!(s, cloned);
        // const builder with sliding window
        let w = ModelSpec::standard(32, 8, 128, KvDtype::Bf16).with_sliding_window(4096);
        assert_eq!(w.sliding_window, 4096);
    }

    #[test]
    fn spec_round_trips_toml() {
        let s = ModelSpec::standard(32, 8, 128, KvDtype::Bf16).with_sliding_window(4096);
        let toml_text = toml::to_string(&s).unwrap();
        let back: ModelSpec = toml::from_str(&toml_text).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn named_spec_round_trips_toml_with_flattened_fields() {
        let n = NamedModelSpec::new(
            "my-private-model",
            ModelSpec::standard(20, 4, 96, KvDtype::Fp16),
        );
        let toml_text = toml::to_string(&n).unwrap();
        // The name and the spec fields live in the same flat table.
        assert!(toml_text.contains("name = \"my-private-model\""));
        assert!(toml_text.contains("num_layers = 20"));
        let back: NamedModelSpec = toml::from_str(&toml_text).unwrap();
        assert_eq!(back, n);
    }

    #[test]
    fn mla_named_spec_parses_from_toml() {
        let toml_text = r#"
name = "custom-mla"
num_layers = 30
attention = "mla"
dtype = "bf16"
kv_lora_rank = 384
qk_rope_head_dim = 48
"#;
        let n: NamedModelSpec = toml::from_str(toml_text).unwrap();
        assert_eq!(n.name, "custom-mla");
        assert_eq!(n.spec.attention, AttentionKind::Mla);
        assert_eq!(n.spec.kv_lora_rank, 384);
        assert_eq!(n.spec.qk_rope_head_dim, 48);
    }

    #[test]
    fn spec_default_is_empty_standard_bf16() {
        let s = ModelSpec::default();
        assert_eq!(s.attention, AttentionKind::Standard);
        assert_eq!(s.dtype, KvDtype::Bf16);
        assert_eq!(s.num_layers, 0);
    }
}
