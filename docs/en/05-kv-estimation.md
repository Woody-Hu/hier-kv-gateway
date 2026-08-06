# KV Memory Estimation Module Architecture

> English | [中文](../05-kv-estimation.md)

> Standalone, plugin-driven, allocation-free-hot-path KV-cache memory estimation for LLM inference

## 1. Background & Goals

### 1.1 Problem domain

To route an inference request to a backend that can *hold* it, the gateway must first answer:

> Given a model's architecture (layers / KV heads / head dim / dtype / attention family) and a request's shape (batch size / input length / output length / block size), how much GPU KV-cache memory will this inference need?

This is the precondition for **capacity-aware routing** (admission control / load shedding): only by knowing the request's KV footprint can it be compared against a backend's remaining KV-block or GPU-memory headroom to exclude backends that cannot fit it.

### 1.2 Design goals

1. **Analytical formulas, not simulation**: byte counts come from integer multiply-adds over model architecture parameters + request shape, identical to how vLLM / SGLang / Mooncake / llm-d compute KV size. No queueing-theory or scheduling simulation.
2. **Standalone leaf crate**: `hier-kv-gateway-kv-estimate` depends on no gateway types, only `serde`. Reusable by any inference router / scheduler / capacity-planning tool.
3. **Plugin extensibility**: new models prefer the **data** path (one TOML line); new attention schemes use the **code** path (implement `KvEstimator`). Both covered.
4. **Prefabricated mainstream models**: builtins for Llama-2/3, Qwen2/2.5, Mistral/Mixtral, Gemma-2, DeepSeek-V2/V3/R1, ChatGLM3 — covering MHA / GQA / MQA / MLA / sliding window.
5. **Allocation-free, nanosecond hot path**: `spec_for → estimate` is fully `Copy`, no `String`, no `HashMap` value clone — proven by a counting-allocator test (§6).
6. **Configurable + off by default**: `[kv_estimate]` is off by default; enabling it does not break existing config parsing.

### 1.3 Non-goals

- **Not** predicting the incremental footprint after a prefix hit: that is `KvAwareStrategy`'s domain (prefix-overlap scoring). This module computes the *from-empty-cache-to-full-length* upper bound, used for capacity admission.
- **Not** simulating scheduler behaviour: no estimation of intra-batch token reuse, preemption, swap-out.
- **Not** replacing backend-reported KV block totals: when a backend reports `kv_total_blocks`/`kv_used_blocks`, routing uses the exact block path; the estimate's GPU-memory fallback is used only when the backend reports GPU memory alone.

---

## 2. Top-level architecture

```
┌──────────────────────────────────────────────────────────┐
│                  KvEstimationRegistry                     │
│   (builtin StandardEstimator + custom specs + plugins)   │
│                                                          │
│   spec_for(model) ──► first estimator (in reg. order)    │
│                       that recognizes model              │
│   estimate(model, input) ──► that estimator's formula    │
└──────────────────────────────────────────────────────────┘
        ▲                              ▲
        │                              │
  ┌─────┴──────┐              ┌────────┴────────┐
  │  Standard  │              │  user plugin    │
  │ Estimator  │              │ KvEstimator impl│
  │ (formula)  │              │ (custom formula)│
  └─────┬──────┘              └─────────────────┘
        │
  ┌─────┴──────────────────────┐
  │  SpecCatalog               │
  │  custom specs (config TOML)│
  │  + builtin pattern table   │
  │    (Llama/Qwen/Mistral/    │
  │     Gemma/DeepSeek/GLM…)   │
  └────────────────────────────┘
```

Module layout:

| File | Responsibility |
|------|----------------|
| [spec.rs](../../crates/hier-kv-gateway-kv-estimate/src/spec.rs) | `ModelSpec` / `AttentionKind` / `KvDtype` / `NamedModelSpec` — architecture params determining KV footprint (`Copy` value type) |
| [catalog.rs](../../crates/hier-kv-gateway-kv-estimate/src/catalog.rs) | builtin model table + allocation-free case-insensitive substring match `lookup_builtin` |
| [estimate.rs](../../crates/hier-kv-gateway-kv-estimate/src/estimate.rs) | pure analytical formulas `estimate_kv` / `per_token_bytes` / `per_block_bytes` |
| [plugin.rs](../../crates/hier-kv-gateway-kv-estimate/src/plugin.rs) | `KvEstimator` trait / `StandardEstimator` / `SpecCatalog` |
| [registry.rs](../../crates/hier-kv-gateway-kv-estimate/src/registry.rs) | `KvEstimationRegistry` — composite estimator, resolution order = registration order |
| [config.rs](../../crates/hier-kv-gateway-kv-estimate/src/config.rs) | `KvEstimateConfig` — the `[kv_estimate]` TOML section |

---

## 3. Analytical formulas (aligned with open-source implementations)

### 3.1 Standard attention (MHA / GQA / MQA)

Llama / Qwen / Mistral / Gemma / GLM all cache full K and V tensors. Per-token footprint:

```
per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
```

- Factor `2` = K + V.
- `num_kv_heads` already distinguishes MHA (= query heads) / GQA (fewer) / MQA (= 1); the formula is identical, only the parameter differs.

This is the formula in **vLLM** (`vllm/worker/worker.py`: `cache_block_size = 2 * num_layers * num_kv_heads * head_size * block_size * dtype_size`) and **SGLang** (`sglang/srt/configs/model_config.py`: `get_kv_cache_bytes`).

### 3.2 MLA — Multi-head Latent Attention (DeepSeek-V2 / V3 / R1)

MLA compresses the KV cache into a single latent vector `c_kv` (dim `kv_lora_rank`), from which both K and V are reconstructed at attention time via separate up-projections; a small RoPE-decoupled key `k_pe` (dim `qk_rope_head_dim`) is cached alongside. Per-token footprint:

```
per_token_bytes = num_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
```

**No factor of 2**: one latent reconstructs both K and V. For DeepSeek-V3 (61 layers, `kv_lora_rank=512`, `qk_rope_head_dim=64`, BF16): `61 * 576 * 2 = 70 272` B/token — ~57× smaller than the equivalent full-K/V layout (DeepSeek-V2 paper §3.1).

### 3.3 Sliding-window attention

When `sliding_window > 0`, the persistent cache holds at most `sliding_window` tokens per sequence (older tokens' KV is evicted). Effective cached sequence length = `min(seq_len, sliding_window)`. Used by Mistral (4096) and Gemma-2.

### 3.4 Block paging

Paged-attention engines (vLLM / SGLang / Mooncake) allocate KV memory in fixed-size blocks of `block_size` tokens; the request footprint is padded up to whole blocks:

```
padded_seq_len = ceil(effective_seq_len / block_size) * block_size
total_bytes    = per_token_bytes * batch_size * padded_seq_len
total_blocks   = ceil(effective_seq_len / block_size) * batch_size
```

`EstimateInput::block_size = 0` disables padding and yields the exact (un-padded) footprint.

### 3.5 End-to-end example

Llama-3-8B (32 layers, 8 KV heads, head_dim 128, BF16), 4096 prompt + 1024 output, block_size 16, batch 1:

```
per_token = 2 * 32 * 8 * 128 * 2 = 131_072 B (128 KiB/token)
effective_seq_len = 4096 + 1024 = 5120
blocks = ceil(5120/16) = 320
padded_seq_len = 320 * 16 = 5120
total_bytes = 131_072 * 5120 = 671_088_640 B (≈ 640 MiB)
```

---

## 4. Plugin extensibility (two paths)

### 4.1 Path A: add a model spec (data, no code)

The path for the overwhelming majority of new models — any MHA/GQA/MQA/MLA architecture is a few config fields, not a new formula:

```toml
[[kv_estimate.models]]
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

Fields map 1:1 to HuggingFace `config.json`:

| `ModelSpec` field | HuggingFace `config.json` field |
|-------------------|-------------------------------|
| `num_layers` | `num_hidden_layers` |
| `num_kv_heads` | `num_key_value_heads` (= `num_attention_heads` for MHA) |
| `head_dim` | `head_dim` (or `hidden_size / num_attention_heads`) |
| `attention` | `standard` by default; `mla` for DeepSeek-V2/V3 |
| `dtype` | `torch_dtype` |
| `kv_lora_rank` | `kv_lora_rank` (MLA only) |
| `qk_rope_head_dim` | `qk_rope_head_dim` (MLA only) |
| `sliding_window` | `sliding_window` (0 = none) |

Custom specs take precedence over builtins of the same name (use to correct a builtin dtype or add a missing sliding window).

### 4.2 Path B: add a custom estimator (code)

When an architecture's KV footprint **cannot** be expressed by the standard formula (e.g. extra Cross-Attention cache, Mamba/SSM state, hybrid schemes), implement the `KvEstimator` trait:

```rust
pub trait KvEstimator: Send + Sync {
    fn name(&self) -> &str;
    fn spec_for(&self, model: &str) -> Option<ModelSpec>;
    fn estimate(&self, spec: &ModelSpec, input: &EstimateInput) -> KvEstimate;
}
```

Register via `KvEstimationRegistry::with_estimator` (append, low priority) or `with_estimator_front` (prepend, high priority — can override builtins). The registry asks each estimator `spec_for(model)` in registration order; the **first** that recognizes the model provides both the spec and the (possibly custom) formula — the custom formula is fully honored.

This mirrors vLLM: standard models are parametrized by `config.json` fields, exotic cases are overridden by engine code.

---

## 5. Builtin catalog

18 specs covering the four attention families; each entry's parameters are transcribed from the model's HuggingFace `config.json`:

| Family | Attention | Representative models |
|--------|-----------|----------------------|
| Llama-2 | MHA / GQA | Llama-2-7B/13B/70B |
| Llama-3 / 3.1 / 3.3 | GQA | Llama-3-8B/70B, 3.1-405B |
| Qwen2 / 2.5 | GQA | Qwen2.5-7B/14B/32B/72B, Qwen2-7B/72B |
| Mistral / Mixtral | GQA + sliding window 4096 | Mistral-7B, Mixtral-8x7B/8x22B |
| Gemma-2 | GQA + sliding window | Gemma-2-9B/27B |
| DeepSeek | MLA | DeepSeek-V2-Lite, V3, R1 |
| ChatGLM3 | GQA (kv_heads=2) | ChatGLM3-6B |

Lookup is a **case-insensitive substring match**, ordered most-specific to least-specific (e.g. `llama-3.1-405b` precedes `llama-3`, so 405B resolves to the 126-layer spec, not the 8B's 32-layer one).

---

## 6. Allocation-free hot-path design

The hot path = "known model name → resolve spec → compute footprint", run once per request, per candidate backend. It is guaranteed to allocate **zero** bytes:

1. **`ModelSpec` is `Copy`**: plain integers + two `#[derive(Copy)]` enums, no `String`, no `Arc`. The model name is deliberately kept out of `ModelSpec` (it lives as a catalog key alongside specs) — this is what lets the hot path be `Copy`.
2. **`lookup_builtin` is allocation-free**: `contains_ascii_ci` does the case-insensitive substring match **without** `to_ascii_lowercase()` (which would allocate a `String` per lookup).
3. **Custom-spec lookup borrows via `Borrow<str>`**: `HashMap<String, ModelSpec>::get` accepts a `&str` key, so the lookup key is a borrowed `&str` — no clone.
4. **`estimate_kv` is pure integer math**: a constant number of integer multiply-adds returning a `Copy` struct.

Proof in [tests/alloc_free.rs](../../crates/hier-kv-gateway-kv-estimate/tests/alloc_free.rs): a counting global allocator is installed, then `estimate_kv` / `per_token_bytes` / `registry.spec_for` (hit/miss) / `registry.estimate` (full hot path) / custom-catalog lookup are each run 10 000 times, asserting the allocation delta inside the window = **0** (zero tolerance). All checks live in a **single** `#[test]` to avoid `cargo test`'s parallel-thread allocator-counter cross-talk.

---

## 7. Configuration interface

```toml
[kv_estimate]
enabled = true            # master switch, default false
weight = 0.20             # hybrid weight
gpu_mem_safety_fraction = 0.5  # claimable fraction of free GPU memory (fallback)
exclude_on_unknown_spec = false  # exclude (true) or neutral (false) on unknown spec

[[kv_estimate.models]]    # optional: custom model spec
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | When off, no `KvCapacityStrategy` is attached and the estimator is not on the routing path |
| `weight` | `0.20` | `[0,1]`; `0.0` keeps it attached (availability probe runs) but contributes nothing |
| `gpu_mem_safety_fraction` | `0.5` | When a backend reports only GPU memory, only "currently free memory × this fraction" is claimable by KV (KV is not the only GPU memory consumer) |
| `exclude_on_unknown_spec` | `false` | `true`: unknown-spec backend gets `raw_cost=∞` (excluded); `false`: neutral, letting other sub-strategies decide (safer — avoids starving a backend that does have room) |

---

## 8. Integration with routing

The estimation module itself only computes byte counts; turning them into routing scores is the job of `KvCapacityStrategy` (routing crate, see [02-routing-algorithms.md §9](../02-routing-algorithms.md)). Integration:

```
[kv_estimate] enabled=true
       │ build_routing_engine
       ▼
KvEstimationRegistry (Arc, constructed once at startup)
       │
       ▼
KvCapacityStrategy (plugin)
       │ HybridStrategy::with_plugin
       ▼
Hybrid scoring (normalized independently from KV/Load/Topology)
```

- **Data half** (`KvEstimateConfig`, spec catalog) lives in this leaf crate, surfaced via `GatewayConfig::kv_estimate`.
- **Behaviour half** (`KvCapacityStrategy`, turning estimates into routing scores) lives in the routing crate, attached to Hybrid as a `RoutingPlugin`.

The two are decoupled: the estimator is independently reusable, the routing strategy independently testable.

---

## 9. Testing & anti-cheat

| Layer | File | Coverage |
|-------|------|----------|
| Formula unit tests | [estimate.rs](../../crates/hier-kv-gateway-kv-estimate/src/estimate.rs) `#[cfg(test)]` | MHA/GQA/MLA/sliding-window/block-padding/batch-scaling/dtype/saturating-add (19) |
| spec/catalog unit tests | [spec.rs](../../crates/hier-kv-gateway-kv-estimate/src/spec.rs) / [catalog.rs](../../crates/hier-kv-gateway-kv-estimate/src/catalog.rs) `#[cfg(test)]` | TOML round-trip, case-insensitive match, four-family coverage, index-bounds check |
| registry/plugin unit tests | [registry.rs](../../crates/hier-kv-gateway-kv-estimate/src/registry.rs) / [plugin.rs](../../crates/hier-kv-gateway-kv-estimate/src/plugin.rs) `#[cfg(test)]` | resolution order, front-override, custom formula honored, clone shares Arc |
| Allocation-free proof | [tests/alloc_free.rs](../../crates/hier-kv-gateway-kv-estimate/tests/alloc_free.rs) | counting allocator asserts 0 bytes on hot path |
| Benchmark anti-cheat | [benches/kv_estimate.rs](../../crates/hier-kv-gateway-kv-estimate/benches/kv_estimate.rs) | every iteration asserts the hand-computed expected byte count; `black_box` prevents elision |

**Anti-cheat principle**: every bench iteration does `assert_eq!(r.bytes, expected)` — if someone "optimizes" the formula into a no-op (or breaks it), the assertion panics and the bench fails; the numbers cannot be faked. The allocation-free test intercepts **every** `alloc` in the test binary via a counting global allocator; a non-zero delta is a genuine regression.

---

## 10. File index

| File | Description |
|------|-------------|
| [lib.rs](../../crates/hier-kv-gateway-kv-estimate/src/lib.rs) | crate entry + architecture overview |
| [spec.rs](../../crates/hier-kv-gateway-kv-estimate/src/spec.rs) | `ModelSpec` value type |
| [catalog.rs](../../crates/hier-kv-gateway-kv-estimate/src/catalog.rs) | builtin table + allocation-free matcher |
| [estimate.rs](../../crates/hier-kv-gateway-kv-estimate/src/estimate.rs) | analytical formulas |
| [plugin.rs](../../crates/hier-kv-gateway-kv-estimate/src/plugin.rs) | `KvEstimator` trait + `StandardEstimator` |
| [registry.rs](../../crates/hier-kv-gateway-kv-estimate/src/registry.rs) | `KvEstimationRegistry` composite estimator |
| [config.rs](../../crates/hier-kv-gateway-kv-estimate/src/config.rs) | `[kv_estimate]` TOML section |
| [kv_capacity.rs](../../crates/hier-kv-gateway-routing/src/kv_capacity.rs) | `KvCapacityStrategy` (behaviour half, routing crate) |
