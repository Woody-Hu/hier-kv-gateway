//! Proves the KV-estimation hot path allocates **zero** bytes.
//!
//! This is the strongest honest form of the "no allocation on the hot path"
//! claim made in the crate docs and benchmarks. We install a counting global
//! allocator, then run the real per-request code paths — `estimate_kv`, the
//! registry `spec_for` lookup, and the full `registry.estimate` (lookup +
//! formula) — many times inside a window bracketed by allocation-counter
//! snapshots. If any byte is allocated on the hot path, the assertion fails.
//!
//! ## Single-test design (why the measurement is trustworthy)
//!
//! All checks live in ONE `#[test]` function. This is deliberate: the global
//! allocator counter is process-wide, and `cargo test` runs tests in parallel
//! threads. If several `#[test]` functions each opened a measurement window
//! concurrently, sibling tests' allocations would leak into each other's
//! windows and produce false positives. With a single test there are no
//! sibling test threads, so the only allocations that can land inside a
//! window are the ones made by the measured code itself — which must be zero.
//!
//! ## What is and isn't on the hot path
//!
//! - **On the hot path (must not allocate):** resolving a spec for a known
//!   model name and computing its KV footprint. This runs once per request,
//!   per candidate backend, on the routing decision loop.
//! - **Off the hot path (allowed to allocate):** constructing the registry,
//!   building the catalog from config, registering plugins. These happen once
//!   at startup; each check takes its baseline *after* construction + warm-up
//!   so those allocations are not counted.
//!
//! ## Why this isn't cheating
//!
//! The counting allocator intercepts *every* `alloc` in the test binary, not
//! just the crate's. The bracketed loops contain no printing, no formatting
//! (assertion messages are only evaluated on failure), no `Vec`/`String`
//! growth — only `Copy` returns and `black_box` sinks. So a non-zero delta is
//! a genuine regression in the hot path, not harness noise. Each path is
//! warmed up before snapshotting so one-time lazy std allocations don't trip
//! the check.
//!
//! Run with:
//! ```bash
//! cargo test -p hier-kv-gateway-kv-estimate --test alloc_free -- --nocapture
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use hier_kv_gateway_kv_estimate::spec::KvDtype;
use hier_kv_gateway_kv_estimate::{
    estimate_kv, per_token_bytes, EstimateInput, KvEstimationRegistry, KvEstimator, ModelSpec,
    SpecCatalog, StandardEstimator,
};

/// Total bytes allocated since process start (monotonic; never decremented).
static ALLOCED: AtomicUsize = AtomicUsize::new(0);

/// Counting allocator: forwards to `System` while tallying `alloc` sizes.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCED.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Llama-3-8B GQA spec. per_token = 2*32*8*128*2 = 131_072 B.
fn llama3_8b() -> ModelSpec {
    ModelSpec::standard(32, 8, 128, KvDtype::Bf16)
}

/// Snapshot the cumulative allocation counter.
fn alloc_now() -> usize {
    ALLOCED.load(Ordering::Relaxed)
}

/// Assert the hot path between `before` and `after` allocated nothing.
/// Zero means zero — no tolerance, so a real regression cannot hide.
fn assert_zero(label: &str, before: usize, after: usize) {
    assert_eq!(
        before, after,
        "{label}: hot path allocated {} bytes (must be zero)",
        after - before
    );
}

/// Measure `iters` iterations of `body`, asserting zero allocation in between.
fn measure<F: FnMut()>(label: &str, iters: usize, mut body: F) {
    // Warm up twice: the very first call may trigger one-time std/runtime
    // lazy init that is unrelated to the hot path.
    body();
    body();
    let before = alloc_now();
    for _ in 0..iters {
        body();
    }
    let after = alloc_now();
    assert_zero(label, before, after);
}

#[test]
fn hot_path_is_allocation_free() {
    // --- estimate_kv: the raw analytical formula ---
    {
        let spec = llama3_8b();
        let input = EstimateInput::new(4096, 1024).with_block_size(16);
        measure("estimate_kv", 10_000, || {
            let r = estimate_kv(std::hint::black_box(&spec), std::hint::black_box(&input));
            std::hint::black_box(r);
        });
    }

    // --- per_token_bytes: the architecture-level constant ---
    {
        let spec = llama3_8b();
        measure("per_token_bytes", 10_000, || {
            let p = per_token_bytes(std::hint::black_box(&spec));
            std::hint::black_box(p);
        });
    }

    // --- registry.spec_for, builtin hit ---
    {
        let registry = KvEstimationRegistry::with_builtins();
        measure("registry.spec_for (builtin hit)", 10_000, || {
            let s = registry
                .spec_for(std::hint::black_box("Llama-3-8B"))
                .unwrap();
            std::hint::black_box(s);
        });
    }

    // --- registry.spec_for, builtin miss (scans the whole pattern table) ---
    {
        let registry = KvEstimationRegistry::with_builtins();
        measure("registry.spec_for (builtin miss)", 10_000, || {
            let r = registry
                .spec_for(std::hint::black_box("totally-unknown-model"));
            std::hint::black_box(r);
        });
    }

    // --- registry.estimate: the full hot path (name -> spec -> formula) ---
    {
        let registry = KvEstimationRegistry::with_builtins();
        let input = EstimateInput::new(4096, 1024).with_block_size(16);
        measure("registry.estimate (full hot path)", 10_000, || {
            let r = registry
                .estimate(
                    std::hint::black_box("Llama-3-8B"),
                    std::hint::black_box(&input),
                )
                .unwrap();
            std::hint::black_box(r);
        });
    }

    // --- registry.estimate across attention families (GQA / MLA / sliding) ---
    {
        let registry = KvEstimationRegistry::with_builtins();
        let input = EstimateInput::new(8192, 512).with_block_size(16);
        let models = ["Llama-3-8B", "Qwen2.5-7B", "deepseek-v3", "mistral-7b"];
        measure("registry.estimate (rotating models)", 2_500, || {
            for m in models {
                let r = registry.estimate(m, &input).unwrap();
                std::hint::black_box(r);
            }
        });
    }

    // --- StandardEstimator.spec_for (one level below the registry) ---
    {
        let est = StandardEstimator::with_builtins();
        measure("StandardEstimator.spec_for", 10_000, || {
            let s = est
                .spec_for(std::hint::black_box("Llama-3-8B"))
                .unwrap();
            std::hint::black_box(s);
        });
    }

    // --- custom catalog lookup (hit): HashMap get via Borrow<str> ---
    {
        let cat = SpecCatalog::new().insert(
            "custom-model",
            ModelSpec::standard(32, 8, 128, KvDtype::Bf16),
        );
        let est = StandardEstimator::with_catalog(cat);
        measure("custom catalog lookup (hit)", 10_000, || {
            let s = est
                .spec_for(std::hint::black_box("custom-model"))
                .unwrap();
            std::hint::black_box(s);
        });
    }

    // --- custom catalog present, lookup falls through to a builtin pattern ---
    {
        let cat = SpecCatalog::new().insert(
            "custom-model",
            ModelSpec::standard(32, 8, 128, KvDtype::Bf16),
        );
        let est = StandardEstimator::with_catalog(cat);
        measure("custom catalog (builtin fallback)", 10_000, || {
            let s = est
                .spec_for(std::hint::black_box("Llama-3-8B"))
                .unwrap();
            std::hint::black_box(s);
        });
    }
}
