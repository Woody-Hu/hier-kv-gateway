//! Core KV-cache size estimation — pure, allocation-free hot path.
//!
//! This module implements the analytical KV-cache memory formulas used across
//! mainstream inference engines. It is **not** a simulation: given a model's
//! architectural parameters ([`ModelSpec`]) and a request shape
//! ([`EstimateInput`]), it computes the exact KV-cache byte/block footprint
//! with a handful of integer multiplies.
//!
//! ## Formulas (and where they come from)
//!
//! ### Standard attention (MHA / GQA / MQA)
//!
//! Llama, Qwen, Mistral, Gemma, GLM, … all cache full K and V tensors. The
//! per-token footprint is:
//!
//! ```text
//! per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
//! ```
//!
//! - The factor `2` is K and V.
//! - `num_kv_heads` already distinguishes MHA (= query heads), GQA (fewer)
//!   and MQA (= 1); the formula is identical, only the parameter differs.
//!
//! This is the formula in **vLLM** (`vllm/worker/worker.py`:
//! `cache_block_size = 2 * num_layers * num_kv_heads * head_size * block_size
//! * dtype_size`) and **SGLang** (`sglang/srt/configs/model_config.py`:
//! `get_kv_cache_bytes`).
//!
//! ### MLA — Multi-head Latent Attention (DeepSeek-V2 / V3 / R1)
//!
//! MLA compresses the KV cache into a single latent vector `c_kv` of dim
//! `kv_lora_rank`, from which both K and V are reconstructed at attention
//! time via separate up-projections. A small RoPE-decoupled key `k_pe` of dim
//! `qk_rope_head_dim` is cached alongside it. The per-token footprint is:
//!
//! ```text
//! per_token_bytes = num_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
//! ```
//!
//! There is **no factor of 2**: one latent reconstructs both K and V. For
//! DeepSeek-V3 (61 layers, `kv_lora_rank=512`, `qk_rope_head_dim=64`, BF16)
//! this is `61 * 576 * 2 = 70 272` B/token — ~57× smaller than the
//! equivalent full-K/V layout. (DeepSeek-V2 paper §3.1.)
//!
//! ### Sliding-window attention
//!
//! When `sliding_window > 0`, the persistent cache holds at most
//! `sliding_window` tokens per sequence (older tokens' KV is evicted). The
//! effective cached sequence length is therefore
//! `min(seq_len, sliding_window)`. Used by Mistral (4096) and Gemma-2.
//!
//! ### Block paging
//!
//! Paged-attention engines (vLLM, SGLang, Mooncake) allocate KV memory in
//! fixed-size blocks of `block_size` tokens. The request's footprint is
//! padded up to a whole number of blocks:
//!
//! ```text
//! padded_seq_len = ceil(effective_seq_len / block_size) * block_size
//! total_bytes    = per_token_bytes * batch_size * padded_seq_len
//! total_blocks   = ceil(effective_seq_len / block_size) * batch_size
//! ```
//!
//! Passing `block_size = 0` to [`EstimateInput`] disables padding and yields
//! the exact (un-padded) footprint.
//!
//! ## Performance contract
//!
//! [`estimate_kv`] performs a constant number of integer operations and
//! returns a `Copy` struct — **zero heap allocation** on the hot path. The
//! benchmark in `benches/kv_estimate.rs` asserts this.

use crate::spec::{AttentionKind, ModelSpec};

/// Request shape passed to the estimator.
///
/// All fields are plain integers so the struct is `Copy`; the hot-path
/// estimator never needs to allocate one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EstimateInput {
    /// Number of sequences in the batch. The gateway forwards one request at
    /// a time, so it passes `1`; the API is general so batched producers can
    /// pass the real batch size.
    pub batch_size: u32,
    /// Number of prompt (input) tokens.
    pub input_tokens: u32,
    /// Estimated/generated output tokens. Use the client's `max_tokens` for a
    /// conservative upper bound.
    pub output_tokens: u32,
    /// KV-cache block size (tokens per block). `0` disables block padding and
    /// yields the exact footprint; `> 0` pads up to whole blocks, matching
    /// paged-attention engines.
    pub block_size: u32,
}

impl EstimateInput {
    /// Build an input with batch size 1 and no block padding.
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            batch_size: 1,
            input_tokens,
            output_tokens,
            block_size: 0,
        }
    }

    /// Set the batch size (builder style).
    #[must_use]
    pub fn with_batch(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the block size (builder style). `0` = no padding.
    #[must_use]
    pub fn with_block_size(mut self, block_size: u32) -> Self {
        self.block_size = block_size;
        self
    }
}

/// Estimation result. A `Copy` struct of plain numerics — no allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvEstimate {
    /// Total KV-cache bytes for the whole batch (block-padded when
    /// [`EstimateInput::block_size`] > 0).
    pub bytes: u64,
    /// Total KV-cache blocks needed (`ceil(effective_seq_len / block_size) *
    /// batch_size`). `0` when `block_size == 0`.
    pub blocks: u64,
    /// Per-token KV-cache bytes for this model (architecture constant).
    pub per_token_bytes: u64,
    /// Effective cached sequence length after sliding-window capping.
    pub effective_seq_len: u32,
    /// Batch size the estimate was computed for.
    pub batch_size: u32,
}

impl KvEstimate {
    /// Footprint in mebibytes (1 MiB = 2²⁰ bytes).
    pub fn mib(self) -> f64 {
        self.bytes as f64 / (1u64 << 20) as f64
    }

    /// Footprint in megabytes (1 MB = 10⁶ bytes). Use this when comparing
    /// against `BackendMetrics::gpu_memory_*_mb`, which is reported in MB.
    pub fn mb(self) -> f64 {
        self.bytes as f64 / 1_000_000.0
    }
}

/// Per-token KV-cache bytes for `spec` — the architecture-level constant.
///
/// Exposed so callers can convert a backend's available block count into
/// bytes (see `KvCapacityStrategy`).
pub const fn per_token_bytes(spec: &ModelSpec) -> u64 {
    let db = spec.dtype.bytes() as u64;
    match spec.attention {
        AttentionKind::Standard => {
            // 2 (K+V) * layers * kv_heads * head_dim * dtype
            2 * spec.num_layers as u64
                * spec.num_kv_heads as u64
                * spec.head_dim as u64
                * db
        }
        AttentionKind::Mla => {
            // single latent c_kv (kv_lora_rank) reconstructs both K and V,
            // plus the RoPE-decoupled k (qk_rope_head_dim). No factor of 2.
            let latent = spec.kv_lora_rank as u64 + spec.qk_rope_head_dim as u64;
            spec.num_layers as u64 * latent * db
        }
    }
}

/// Per-block KV-cache bytes for `spec` at `block_size` (per batch element).
///
/// `per_block = per_token_bytes * block_size`. Returns 0 when `block_size`
/// is 0.
pub const fn per_block_bytes(spec: &ModelSpec, block_size: u32) -> u64 {
    per_token_bytes(spec) * block_size as u64
}

/// Ceiling division `a / b` for `u64`, assuming `b > 0`.
///
/// Delegates to the std `div_ceil` (const-stable), which — unlike the
/// hand-rolled `(a + b - 1) / b` form — cannot overflow when `a` is near
/// `u64::MAX`.
const fn div_ceil(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

/// Compute the KV-cache footprint of `input` under `spec`.
///
/// Pure, allocation-free, branch-light. See the [module docs](self) for the
/// formulas and their provenance.
pub fn estimate_kv(spec: &ModelSpec, input: &EstimateInput) -> KvEstimate {
    let per_token = per_token_bytes(spec);

    // Sequence length = input + output, saturating at u32::MAX.
    let seq_len = input.input_tokens.saturating_add(input.output_tokens);

    // Apply sliding-window cap on the *cached* length. A window of 0 means
    // full attention (no cap).
    let effective = if spec.sliding_window > 0 && spec.sliding_window < seq_len {
        spec.sliding_window
    } else {
        seq_len
    };

    let batch = if input.batch_size == 0 { 1 } else { input.batch_size };

    let (bytes, blocks) = if input.block_size > 0 {
        let blocks_per_seq = div_ceil(effective as u64, input.block_size as u64);
        let total_blocks = blocks_per_seq * batch as u64;
        let padded_per_seq = blocks_per_seq * input.block_size as u64;
        let bytes = per_token * batch as u64 * padded_per_seq;
        (bytes, total_blocks)
    } else {
        let bytes = per_token * batch as u64 * effective as u64;
        (bytes, 0)
    };

    KvEstimate {
        bytes,
        blocks,
        per_token_bytes: per_token,
        effective_seq_len: effective,
        batch_size: batch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::KvDtype;

    /// Llama-3-8B: 32 layers, 8 KV heads (GQA), head_dim 128, BF16.
    /// per_token = 2 * 32 * 8 * 128 * 2 = 131_072 B = 128 KiB/token.
    fn llama3_8b() -> ModelSpec {
        ModelSpec::standard(32, 8, 128, KvDtype::Bf16)
    }

    #[test]
    fn per_token_bytes_standard_gqa() {
        // 2 * 32 * 8 * 128 * 2 = 131_072
        assert_eq!(per_token_bytes(&llama3_8b()), 131_072);
    }

    #[test]
    fn per_token_bytes_mha_equals_gqa_with_full_heads() {
        // Llama-2-7B is MHA: 32 layers, 32 kv_heads, head_dim 128, fp16.
        let mha = ModelSpec::standard(32, 32, 128, KvDtype::Fp16);
        // 2 * 32 * 32 * 128 * 2 = 524_288
        assert_eq!(per_token_bytes(&mha), 524_288);
    }

    #[test]
    fn per_token_bytes_mla_deepseek_v3() {
        // 61 layers, kv_lora_rank 512, qk_rope_head_dim 64, bf16.
        // 61 * (512 + 64) * 2 = 70_272
        let v3 = ModelSpec::mla(61, 512, 64, KvDtype::Bf16);
        assert_eq!(per_token_bytes(&v3), 70_272);
    }

    #[test]
    fn mla_much_smaller_than_equivalent_full_kv() {
        // A hypothetical full-K/V DeepSeek-V3 (128 KV heads * 128 head_dim):
        let full = ModelSpec::standard(61, 128, 128, KvDtype::Bf16);
        let mla = ModelSpec::mla(61, 512, 64, KvDtype::Bf16);
        let ratio = per_token_bytes(&full) as f64 / per_token_bytes(&mla) as f64;
        // MLA should be >50× smaller — the headline DeepSeek-V2/V3 claim.
        assert!(ratio > 50.0, "MLA compression ratio {ratio} too small");
    }

    #[test]
    fn estimate_basic_no_padding() {
        // 4096 tokens, batch 1, no block padding.
        let est = estimate_kv(&llama3_8b(), &EstimateInput::new(4096, 0));
        // 131_072 B/token * 4096 tokens = 536_870_912 B = 512 MiB
        assert_eq!(est.bytes, 131_072 * 4096);
        assert_eq!(est.blocks, 0);
        assert_eq!(est.effective_seq_len, 4096);
        assert!((est.mib() - 512.0).abs() < 1e-6);
    }

    #[test]
    fn estimate_batch_scales_linearly() {
        let one = estimate_kv(&llama3_8b(), &EstimateInput::new(1024, 0).with_batch(1));
        let eight = estimate_kv(&llama3_8b(), &EstimateInput::new(1024, 0).with_batch(8));
        assert_eq!(eight.bytes, one.bytes * 8);
        assert_eq!(eight.batch_size, 8);
    }

    #[test]
    fn estimate_block_padding_rounds_up() {
        // block_size 16, 4097 tokens -> 257 blocks/seq, padded to 4112 tokens.
        let est = estimate_kv(
            &llama3_8b(),
            &EstimateInput::new(4097, 0).with_block_size(16),
        );
        assert_eq!(est.blocks, 257); // ceil(4097/16) = 257
        assert_eq!(est.bytes, 131_072 * 257 * 16);
        assert_eq!(est.effective_seq_len, 4097);
    }

    #[test]
    fn estimate_block_padding_exact_multiple() {
        // 4096 tokens, block_size 16 -> 256 blocks, no padding overhead.
        let est = estimate_kv(
            &llama3_8b(),
            &EstimateInput::new(4096, 0).with_block_size(16),
        );
        assert_eq!(est.blocks, 256);
        assert_eq!(est.bytes, 131_072 * 4096);
    }

    #[test]
    fn estimate_block_padding_with_batch() {
        // batch 4, 100 tokens, block_size 16 -> ceil(100/16)=7 blocks/seq
        // -> 7*4 = 28 blocks total.
        let est = estimate_kv(
            &llama3_8b(),
            &EstimateInput::new(100, 0).with_batch(4).with_block_size(16),
        );
        assert_eq!(est.blocks, 28);
    }

    #[test]
    fn estimate_input_plus_output() {
        let est = estimate_kv(&llama3_8b(), &EstimateInput::new(1000, 500));
        assert_eq!(est.effective_seq_len, 1500);
        assert_eq!(est.bytes, 131_072 * 1500);
    }

    #[test]
    fn sliding_window_caps_effective_length() {
        // Mistral-7B-style: sliding_window 4096. A 8192-token request only
        // caches the last 4096.
        let spec = llama3_8b().with_sliding_window(4096);
        let est = estimate_kv(&spec, &EstimateInput::new(8192, 0));
        assert_eq!(est.effective_seq_len, 4096);
        assert_eq!(est.bytes, 131_072 * 4096);
    }

    #[test]
    fn sliding_window_no_effect_below_window() {
        let spec = llama3_8b().with_sliding_window(4096);
        let est = estimate_kv(&spec, &EstimateInput::new(2048, 0));
        assert_eq!(est.effective_seq_len, 2048);
    }

    #[test]
    fn fp8_halves_footprint() {
        let bf16 = ModelSpec::standard(32, 8, 128, KvDtype::Bf16);
        let fp8 = ModelSpec::standard(32, 8, 128, KvDtype::Fp8);
        let e_bf16 = estimate_kv(&bf16, &EstimateInput::new(4096, 0));
        let e_fp8 = estimate_kv(&fp8, &EstimateInput::new(4096, 0));
        assert_eq!(e_fp8.bytes * 2, e_bf16.bytes);
    }

    #[test]
    fn zero_batch_treated_as_one() {
        let est = estimate_kv(&llama3_8b(), &EstimateInput {
            batch_size: 0,
            input_tokens: 16,
            output_tokens: 0,
            block_size: 0,
        });
        assert_eq!(est.batch_size, 1);
        assert_eq!(est.bytes, 131_072 * 16);
    }

    #[test]
    fn mla_estimate_uses_latent_formula() {
        let v3 = ModelSpec::mla(61, 512, 64, KvDtype::Bf16);
        let est = estimate_kv(&v3, &EstimateInput::new(4096, 0));
        // 70_272 B/token * 4096 tokens = 287_834_112 B
        assert_eq!(est.bytes, 70_272 * 4096);
        assert_eq!(est.per_token_bytes, 70_272);
    }

    #[test]
    fn mla_block_padding_correct() {
        let v3 = ModelSpec::mla(61, 512, 64, KvDtype::Bf16);
        let est = estimate_kv(&v3, &EstimateInput::new(4097, 0).with_block_size(16));
        assert_eq!(est.blocks, 257);
        // per_token * batch * padded_per_seq = 70_272 * 1 * (257*16)
        assert_eq!(est.bytes, 70_272 * 257 * 16);
    }

    #[test]
    fn saturating_add_for_huge_inputs() {
        // input + output must not panic on overflow.
        let est = estimate_kv(
            &llama3_8b(),
            &EstimateInput::new(u32::MAX, u32::MAX),
        );
        assert_eq!(est.effective_seq_len, u32::MAX);
    }

    #[test]
    fn per_block_bytes_helper() {
        assert_eq!(per_block_bytes(&llama3_8b(), 16), 131_072 * 16);
        assert_eq!(per_block_bytes(&llama3_8b(), 0), 0);
    }

    #[test]
    fn mb_uses_decimal_megabytes() {
        let est = estimate_kv(&llama3_8b(), &EstimateInput::new(1_000_000, 0));
        // 131_072 B/token * 1_000_000 tokens = 131_072_000_000 B = 131_072 MB
        assert!((est.mb() - 131_072.0).abs() < 1e-3);
    }
}
