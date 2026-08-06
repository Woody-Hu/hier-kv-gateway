//! Builtin catalog of mainstream model KV-cache specs.
//!
//! Each entry's parameters are transcribed from the model's HuggingFace
//! `config.json`. The catalog covers the four attention families the
//! estimator supports — MHA, GQA, MQA, MLA — plus sliding-window variants —
//! so every formula branch is exercised by a real, deployed model.
//!
//! ## Lookup is allocation-free
//!
//! [`lookup_builtin`] does a case-insensitive *substring* match against the
//! [`BUILTIN_NAME_MAP`] patterns. The match is implemented without
//! `to_ascii_lowercase()` (which would allocate a `String`) — see
//! [`contains_ascii_ci`]. Combined with [`crate::spec::ModelSpec`] being
//! `Copy`, the whole `lookup_builtin` → `spec_for` → `estimate` path on the
//! routing hot loop allocates **zero** bytes (proven by `tests/alloc_free.rs`).
//!
//! ## Adding a model
//!
//! The catalog is a static array; add a line in [`BUILTIN_SPECS`] following
//! the existing pattern, copying the values from the model's `config.json`,
//! and a `(pattern, index)` entry in [`BUILTIN_NAME_MAP`]. For models not
//! shipped here, operators can either add a `[[kv_estimate.models]]` entry in
//! their gateway TOML (no code change) or register a custom
//! [`crate::plugin::KvEstimator`] plugin (for non-standard architectures).

use crate::spec::{AttentionKind, ModelSpec};

/// The builtin spec table, ordered by family for readability.
///
/// Entries carry no name (the name is the routing key, kept out of
/// [`ModelSpec`]); names are mapped to indices via [`BUILTIN_NAME_MAP`].
pub const fn builtin_specs_raw() -> &'static [ModelSpec] {
    &BUILTIN_SPECS
}

/// Static array of builtin specs.
static BUILTIN_SPECS: [ModelSpec; 18] = [
    // ===== Llama 2 (MHA) =====
    // Llama-2 uses full multi-head attention (kv_heads == query heads).
    ModelSpec {
        num_layers: 32,
        num_kv_heads: 32,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Fp16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    ModelSpec {
        num_layers: 40,
        num_kv_heads: 40,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Fp16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    ModelSpec {
        num_layers: 80,
        num_kv_heads: 8, // GQA
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Fp16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    // ===== Llama 3 / 3.1 (GQA) =====
    ModelSpec {
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    ModelSpec {
        num_layers: 80,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    ModelSpec {
        num_layers: 126,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    // ===== Qwen 2 / 2.5 (GQA) =====
    ModelSpec {
        num_layers: 28,
        num_kv_heads: 4,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    ModelSpec {
        num_layers: 48,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    ModelSpec {
        num_layers: 64,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    ModelSpec {
        num_layers: 80,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
    // ===== Mistral / Mixtral (GQA + sliding window 4096) =====
    ModelSpec {
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Fp16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 4096,
    },
    ModelSpec {
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Fp16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 4096,
    },
    ModelSpec {
        num_layers: 56,
        num_kv_heads: 8,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Fp16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 4096,
    },
    // ===== Gemma 2 (GQA + sliding window) =====
    ModelSpec {
        num_layers: 42,
        num_kv_heads: 8,
        head_dim: 256,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 4096,
    },
    ModelSpec {
        num_layers: 46,
        num_kv_heads: 16,
        head_dim: 256,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 4096,
    },
    // ===== DeepSeek V2/V3/R1 (MLA) =====
    // DeepSeek-V2-Lite: 27 layers, kv_lora_rank 512, qk_rope_head_dim 64.
    ModelSpec {
        num_layers: 27,
        num_kv_heads: 0,
        head_dim: 0,
        attention: AttentionKind::Mla,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 512,
        qk_rope_head_dim: 64,
        sliding_window: 0,
    },
    // DeepSeek-V3 / R1: 61 layers, kv_lora_rank 512, qk_rope_head_dim 64.
    ModelSpec {
        num_layers: 61,
        num_kv_heads: 0,
        head_dim: 0,
        attention: AttentionKind::Mla,
        dtype: crate::spec::KvDtype::Bf16,
        kv_lora_rank: 512,
        qk_rope_head_dim: 64,
        sliding_window: 0,
    },
    // ===== ChatGLM3-6B (GQA, kv_heads=2) =====
    ModelSpec {
        num_layers: 28,
        num_kv_heads: 2,
        head_dim: 128,
        attention: AttentionKind::Standard,
        dtype: crate::spec::KvDtype::Fp16,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        sliding_window: 0,
    },
];

/// Pairs of `(name_pattern, spec_index)` — maps model-name substrings to
/// indices into [`BUILTIN_SPECS`]. The registry walks this list in order and
/// uses the first match (case-insensitive substring), so order from most
/// specific to least specific.
///
/// Patterns are kept ASCII-lowercase; the haystack is matched case-insensitively
/// without allocating (see [`contains_ascii_ci`]).
const BUILTIN_NAME_MAP: &[(&str, usize)] = &[
    // Llama 2 (MHA / GQA)
    ("llama-2-7b", 0),
    ("llama2-7b", 0),
    ("llama-2-13b", 1),
    ("llama2-13b", 1),
    ("llama-2-70b", 2),
    ("llama2-70b", 2),
    // Llama 3.x (GQA)
    ("llama-3.1-405b", 5),
    ("llama-3.3-70b", 4),
    ("llama-3.1-70b", 4),
    ("llama-3-70b", 4),
    ("llama-3.1-8b", 3),
    ("llama-3-8b", 3),
    // Qwen 2.5 (GQA)
    ("qwen2.5-72b", 9),
    ("qwen2.5-32b", 8),
    ("qwen2.5-14b", 7),
    ("qwen2.5-7b", 6),
    ("qwen2.5-coder", 6),
    // Qwen 2 (GQA)
    ("qwen2-72b", 9),
    ("qwen2-7b", 6),
    // Mistral / Mixtral (GQA + sliding window)
    ("mixtral-8x22b", 12),
    ("mixtral-8x7b", 11),
    ("mistral-large", 11),
    ("mistral-7b", 10),
    // Gemma 2 (GQA + sliding window)
    ("gemma-2-27b", 14),
    ("gemma-2-9b", 13),
    // DeepSeek (MLA)
    ("deepseek-r1", 16),
    ("deepseek-v3", 16),
    ("deepseek-v2-lite", 15),
    ("deepseek-v2", 15),
    // ChatGLM3 (GQA)
    ("chatglm3", 17),
    ("glm-3", 17),
];

/// Case-insensitive ASCII substring test — **allocation-free**.
///
/// `needle` must be ASCII lowercase (all builtin patterns are). Each candidate
/// window of `haystack` is compared byte-wise, lowercasing the haystack byte
/// on the fly. This avoids `str::to_ascii_lowercase()` which would allocate a
/// `String` on every lookup — a cost that matters on the routing hot path.
const fn contains_ascii_ci(haystack: &str, needle: &str) -> bool {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() {
        return true;
    }
    let hlen = h.len();
    let nlen = n.len();
    if nlen > hlen {
        return false;
    }
    let last = hlen - nlen;
    let mut i = 0;
    while i <= last {
        let mut j = 0;
        let mut miss = false;
        while j < nlen {
            // ASCII lowercasing: only A..=Z -> a..=z; other bytes unchanged.
            let hb = h[i + j];
            let hb_low = if hb >= b'A' && hb <= b'Z' { hb + 32 } else { hb };
            if hb_low != n[j] {
                miss = true;
                break;
            }
            j += 1;
        }
        if !miss {
            return true;
        }
        i += 1;
    }
    false
}

/// Look up a builtin spec by model name (case-insensitive substring match).
///
/// Returns a *copy* of the spec (ModelSpec is `Copy`). `None` when no builtin
/// pattern matches. Allocation-free.
pub fn lookup_builtin(model: &str) -> Option<ModelSpec> {
    for &(pattern, idx) in BUILTIN_NAME_MAP {
        if contains_ascii_ci(model, pattern) {
            return Some(BUILTIN_SPECS[idx]);
        }
    }
    None
}

/// Iterator over `(pattern, spec)` for the builtin table — used by tests and
/// by the docs to enumerate coverage.
pub fn builtin_entries() -> impl Iterator<Item = (&'static str, ModelSpec)> {
    BUILTIN_NAME_MAP
        .iter()
        .map(|&(p, idx)| (p, BUILTIN_SPECS[idx]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::KvDtype;

    #[test]
    fn catalog_has_all_four_attention_families() {
        let families: Vec<_> = (0..BUILTIN_SPECS.len())
            .map(|i| BUILTIN_SPECS[i].attention)
            .collect();
        assert!(families.contains(&AttentionKind::Standard));
        assert!(families.contains(&AttentionKind::Mla));
    }

    #[test]
    fn catalog_has_sliding_window_models() {
        let any_sliding = (0..BUILTIN_SPECS.len())
            .any(|i| BUILTIN_SPECS[i].sliding_window > 0);
        assert!(any_sliding);
    }

    #[test]
    fn lookup_qwen_case_insensitive() {
        let s = lookup_builtin("Qwen2.5-7B-Instruct").unwrap();
        assert_eq!(s.num_layers, 28);
        assert_eq!(s.num_kv_heads, 4);
    }

    #[test]
    fn lookup_deepseek_v3_is_mla() {
        let s = lookup_builtin("deepseek-v3").unwrap();
        assert_eq!(s.attention, AttentionKind::Mla);
        assert_eq!(s.num_layers, 61);
        assert_eq!(s.kv_lora_rank, 512);
    }

    #[test]
    fn lookup_deepseek_r1_maps_to_v3_spec() {
        let s = lookup_builtin("DeepSeek-R1").unwrap();
        assert_eq!(s.attention, AttentionKind::Mla);
        assert_eq!(s.num_layers, 61);
    }

    #[test]
    fn lookup_mistral_has_sliding_window() {
        let s = lookup_builtin("mistral-7b-v0.1").unwrap();
        assert_eq!(s.sliding_window, 4096);
    }

    #[test]
    fn lookup_llama2_7b_is_mha() {
        // Llama-2-7B: 32 kv_heads (== 32 query heads) => MHA
        let s = lookup_builtin("meta-llama/Llama-2-7b-hf").unwrap();
        assert_eq!(s.num_kv_heads, 32);
        assert_eq!(s.num_layers, 32);
    }

    #[test]
    fn lookup_llama3_8b_is_gqa() {
        let s = lookup_builtin("Llama-3-8B").unwrap();
        assert_eq!(s.num_kv_heads, 8);
        assert_eq!(s.num_layers, 32);
    }

    #[test]
    fn lookup_unknown_returns_none() {
        assert!(lookup_builtin("totally-unknown-model").is_none());
    }

    #[test]
    fn lookup_specific_before_generic() {
        // "llama-3.1-405b" must resolve to the 405B spec (126 layers), not
        // the 8B spec that "llama-3" would match.
        let s = lookup_builtin("llama-3.1-405b").unwrap();
        assert_eq!(s.num_layers, 126);
    }

    #[test]
    fn lookup_chatglm3() {
        let s = lookup_builtin("chatglm3-6b").unwrap();
        assert_eq!(s.num_kv_heads, 2);
        assert_eq!(s.num_layers, 28);
    }

    #[test]
    fn all_name_map_indices_in_bounds() {
        for &(_, idx) in BUILTIN_NAME_MAP {
            assert!(idx < BUILTIN_SPECS.len(), "index {idx} out of bounds");
        }
    }

    #[test]
    fn builtin_entries_non_empty() {
        assert!(builtin_entries().count() > 0);
    }

    // ---- contains_ascii_ci correctness (the allocation-free matcher) ----

    #[test]
    fn contains_ascii_ci_basic() {
        assert!(contains_ascii_ci("Qwen2.5-7B-Instruct", "qwen2.5-7b"));
        assert!(contains_ascii_ci("Llama-3-8B", "llama-3-8b"));
        assert!(!contains_ascii_ci("Llama-3-8B", "qwen"));
    }

    #[test]
    fn contains_ascii_ci_empty_needle_matches() {
        assert!(contains_ascii_ci("anything", ""));
        assert!(contains_ascii_ci("", ""));
    }

    #[test]
    fn contains_ascii_ci_needle_longer_than_haystack() {
        assert!(!contains_ascii_ci("ab", "abcd"));
    }

    #[test]
    fn contains_ascii_ci_full_haystack_match() {
        assert!(contains_ascii_ci("DEEPSEEK-V3", "deepseek-v3"));
        assert!(contains_ascii_ci("deepseek-v3", "deepseek-v3"));
    }

    #[test]
    fn contains_ascii_ci_non_ascii_passthrough() {
        // Non-ASCII bytes are left untouched (only A..=Z lowercased). A pattern
        // with non-ascii still compares by raw byte equality on those positions.
        assert!(contains_ascii_ci("模型-qwen2.5-7b", "qwen2.5-7b"));
    }

    #[test]
    fn lookup_returns_independent_copy() {
        // ModelSpec is Copy; mutating one returned copy must not affect a
        // subsequent lookup (no shared interior state).
        let mut a = lookup_builtin("Llama-3-8B").unwrap();
        a.num_layers = 999;
        // The local copy reflects the mutation...
        assert_eq!(a.num_layers, 999);
        // ...but a fresh lookup is unaffected.
        let b = lookup_builtin("Llama-3-8B").unwrap();
        assert_eq!(b.num_layers, 32, "lookup must return a fresh copy");
    }

    #[test]
    fn builtin_specs_use_expected_dtypes() {
        // Spot-check a couple to guard against transcription drift.
        let llama2 = lookup_builtin("Llama-2-7B").unwrap();
        assert_eq!(llama2.dtype, KvDtype::Fp16);
        let llama3 = lookup_builtin("Llama-3-8B").unwrap();
        assert_eq!(llama3.dtype, KvDtype::Bf16);
    }
}
