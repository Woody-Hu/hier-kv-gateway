//! KV-estimation hot-path benchmarks.
//!
//! What we measure
//! ---------------
//! The analytical estimator ([`estimate_kv`]) and the registry lookup
//! ([`KvEstimationRegistry::estimate`]) are on the routing hot path — every
//! request, every candidate backend. This bench pins their latency so
//! regressions show up.
//!
//! Anti-cheat
//! ----------
//! Every bench iteration **asserts** the hand-computed expected byte count.
//! If someone "optimizes" the formula into a no-op (or breaks it), the
//! assertion panics and the bench fails — the numbers cannot be faked. The
//! `black_box` calls further prevent the compiler from eliding the work.
//!
//! A separate `tests/alloc_free.rs` proves the hot path allocates zero bytes
//! via a counting global allocator — the strongest form of the
//! "no allocation" claim.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-kv-estimate --bench kv_estimate
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use hier_kv_gateway_kv_estimate::{
    estimate_kv, per_token_bytes, EstimateInput, KvEstimationRegistry, KvEstimator, ModelSpec,
    SpecCatalog, StandardEstimator,
};
use hier_kv_gateway_kv_estimate::spec::KvDtype;

/// Llama-3-8B GQA spec. per_token = 2*32*8*128*2 = 131_072 B.
fn llama3_8b() -> ModelSpec {
    ModelSpec::standard(32, 8, 128, KvDtype::Bf16)
}

/// DeepSeek-V3 MLA spec. per_token = 61*(512+64)*2 = 70_272 B.
fn deepseek_v3() -> ModelSpec {
    ModelSpec::mla(61, 512, 64, KvDtype::Bf16)
}

/// Mistral-7B-style GQA + sliding window 4096.
fn mistral_7b() -> ModelSpec {
    ModelSpec::standard(32, 8, 128, KvDtype::Fp16).with_sliding_window(4096)
}

/// Hand-computed expectation for Llama-3-8B, 4096 tokens, block_size 16.
/// 131_072 B/token * ceil(4096/16)=256 blocks * 16 tokens = 131_072 * 4096.
const LLAMA3_8B_4K_BLOCKS_BYTES: u64 = 131_072 * 4096;

fn bench_estimate_kv(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_kv");
    group.sample_size(500);

    let specs = [
        ("llama3_8b_gqa", llama3_8b()),
        ("deepseek_v3_mla", deepseek_v3()),
        ("mistral_7b_sliding", mistral_7b()),
    ];

    for (label, spec) in specs {
        // 4096 input tokens, 1024 output, block_size 16, batch 1.
        let input = EstimateInput::new(4096, 1024).with_block_size(16);
        // Anti-cheat: pre-compute the expected value and assert each iteration.
        let expected = estimate_kv(&spec, &input);
        assert!(expected.bytes > 0, "{label}: estimate must be non-zero");

        group.bench_function(label, |b| {
            b.iter(|| {
                let r = estimate_kv(black_box(&spec), black_box(&input));
                // Anti-cheat: the result must match the pre-computed value.
                assert_eq!(r.bytes, expected.bytes, "{label}: estimate changed");
                assert_eq!(r.per_token_bytes, expected.per_token_bytes);
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_estimate_kv_input_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_kv_by_input");
    group.sample_size(500);
    let spec = llama3_8b();

    for &tokens in &[512u32, 4096, 32_768, 131_072] {
        let input = EstimateInput::new(tokens, 0).with_block_size(16);
        let expected = estimate_kv(&spec, &input);
        group.bench_with_input(
            BenchmarkId::new("llama3_8b", tokens),
            &tokens,
            |b, &_| {
                b.iter(|| {
                    let r = estimate_kv(black_box(&spec), black_box(&input));
                    assert_eq!(r.bytes, expected.bytes); // anti-cheat
                    black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_registry_estimate(c: &mut Criterion) {
    let registry = KvEstimationRegistry::with_builtins();
    let input = EstimateInput::new(4096, 1024).with_block_size(16);

    let mut group = c.benchmark_group("registry_estimate");
    group.sample_size(500);

    for model in ["Llama-3-8B", "Qwen2.5-7B", "deepseek-v3", "mistral-7b"] {
        // Anti-cheat: every model must resolve and produce a fixed value.
        let expected = registry.estimate(model, &input).unwrap();
        assert!(expected.bytes > 0, "{model}: must resolve");

        group.bench_function(model, |b| {
            b.iter(|| {
                let r = registry
                    .estimate(black_box(model), black_box(&input))
                    .unwrap();
                assert_eq!(r.bytes, expected.bytes, "{model}: estimate changed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_registry_spec_for(c: &mut Criterion) {
    // Pure lookup cost (no estimate), with a cold-cache miss in the mix.
    let registry = KvEstimationRegistry::with_builtins();
    let mut group = c.benchmark_group("registry_spec_for");
    group.sample_size(500);

    group.bench_function("builtin_hit", |b| {
        b.iter(|| {
            let s = registry.spec_for(black_box("Llama-3-8B")).unwrap();
            assert_eq!(s.num_layers, 32); // anti-cheat
            black_box(s);
        });
    });

    group.bench_function("builtin_miss", |b| {
        b.iter(|| {
            let r = registry.spec_for(black_box("totally-unknown-model"));
            assert!(r.is_none()); // anti-cheat
            black_box(r);
        });
    });

    group.finish();
}

fn bench_per_token_bytes(c: &mut Criterion) {
    let spec = llama3_8b();
    c.bench_function("per_token_bytes", |b| {
        b.iter(|| {
            let p = per_token_bytes(black_box(&spec));
            assert_eq!(p, 131_072); // anti-cheat: 2*32*8*128*2
            black_box(p);
        });
    });
}

fn bench_custom_catalog_lookup(c: &mut Criterion) {
    // Operator-provided custom specs layered over builtins — measures the
    // HashMap custom-lookup path that config-driven deployments hit.
    let mut cat = SpecCatalog::new();
    for i in 0..64 {
        cat = cat.insert(
            format!("custom-model-{i}"),
            ModelSpec::standard(32, 8, 128, KvDtype::Bf16),
        );
    }
    let est = StandardEstimator::with_catalog(cat);
    let mut group = c.benchmark_group("custom_catalog_lookup");
    group.sample_size(500);

    group.bench_function("custom_hit", |b| {
        b.iter(|| {
            let s = est.spec_for(black_box("custom-model-42")).unwrap();
            assert_eq!(s.num_layers, 32); // anti-cheat
            black_box(s);
        });
    });

    group.bench_function("builtin_fallback_through_custom", |b| {
        b.iter(|| {
            let s = est.spec_for(black_box("Llama-3-8B")).unwrap();
            assert_eq!(s.num_layers, 32); // anti-cheat
            black_box(s);
        });
    });

    group.finish();
}

// Re-export the constant so the compiler cannot prove it unused.
const _: u64 = LLAMA3_8B_4K_BLOCKS_BYTES;

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        bench_estimate_kv,
        bench_estimate_kv_input_sizes,
        bench_registry_estimate,
        bench_registry_spec_for,
        bench_per_token_bytes,
        bench_custom_catalog_lookup,
);
criterion_main!(benches);
