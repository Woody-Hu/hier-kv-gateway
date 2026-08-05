//! Benchmarks for the single-flight request coalescer.
//!
//! What we measure
//! ----------------
//! * `coalesce_concurrent` — end-to-end latency of N concurrent identical
//!   requests through the coalescer, varying N. The producer simulates a
//!   fixed-latency backend forward. This is the headline benchmark: with
//!   coalescing, N concurrent requests should complete in ≈ one forward
//!   latency, not N × forward latency.
//! * `coalesce_distinct` — N concurrent requests with *distinct* keys, as a
//!   control: no dedup should happen, so latency reflects N independent
//!   forwards (interleaved by the runtime).
//! * `request_key_hash` — the per-request cost of computing the semantic key
//!   (canonical-JSON serialization + DefaultHasher), varying the message count.
//!   This is the per-request overhead the coalescer imposes even when no
//!   dedup occurs.
//!
//! ## Anti-cheat
//!
//! Each `coalesce_concurrent` iteration asserts `forwards_saved == N - 1` via
//! the `CoalesceStats` counters. A broken coalescer (e.g. one that forwards
//! every request independently) would show `forwards_saved == 0` and the
//! benchmark would panic, so the numbers cannot be gamed.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-api --bench request_coalescer
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use hier_kv_gateway_api::coalescer::{CoalescedResponse, CoalesceError, RequestCoalescer};
use hier_kv_gateway_core::coalescing::CoalescingConfig;

/// A producer that simulates a fixed-latency backend forward and counts how
/// many times the *actual forward* ran. The counter is the anti-cheat hook.
fn make_producer(
    delay_ms: u64,
    counter: Arc<AtomicUsize>,
) -> impl std::future::Future<Output = Result<CoalescedResponse, CoalesceError>> + Send + 'static
{
    counter.fetch_add(1, Ordering::SeqCst);
    async move {
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        Ok(CoalescedResponse {
            status: 200,
            body: Arc::from(b"{\"ok\":true}" as &[u8]),
            backend: "r1/inst-0".to_string(),
            strategy: "hybrid".to_string(),
            kv_overlap: 3,
        })
    }
}

fn cfg(ttl_ms: u64, max_inflight: usize) -> CoalescingConfig {
    CoalescingConfig {
        enabled: true,
        ttl_ms,
        max_inflight,
    }
}

/// N concurrent identical requests through the coalescer.
///
/// Asserts `forwards_saved == N - 1` so the benchmark cannot be gamed.
fn bench_coalesce_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalesce_concurrent");
    group.sample_size(20); // Each iteration spawns N tasks + sleeps, so keep samples low.

    // Simulated backend forward latency. Long enough that the coalescing
    // benefit dominates the task-spawn overhead.
    const FORWARD_MS: u64 = 50;

    for n in [2usize, 8, 32] {
        group.bench_with_input(BenchmarkId::new("waiters", n), &n, |b, &n| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let coalescer = RequestCoalescer::new(cfg(0, 1024));
                let forward_calls = Arc::new(AtomicUsize::new(0));
                let key = 42u64;

                let mut handles = Vec::with_capacity(n);
                for _ in 0..n {
                    let c = coalescer.clone();
                    let fc = forward_calls.clone();
                    handles.push(tokio::spawn(async move {
                        c.coalesce(key, move || make_producer(FORWARD_MS, fc)).await
                    }));
                }
                let results: Vec<_> = futures::future::join_all(handles).await;

                // Anti-cheat: exactly one forward ran, N-1 were saved.
                assert_eq!(
                    forward_calls.load(Ordering::SeqCst),
                    1,
                    "coalescer must forward exactly once for N={} identical requests",
                    n
                );
                let (_, _, _, saved) = coalescer.stats().snapshot();
                assert_eq!(
                    saved,
                    n as u64 - 1,
                    "forwards_saved must equal N-1 for N={}",
                    n
                );
                // Every caller got a successful response.
                assert_eq!(results.len(), n);
                for r in &results {
                    assert!(r.as_ref().unwrap().is_ok());
                }
                black_box(results);
            });
        });
    }
    group.finish();
}

/// N concurrent requests with *distinct* keys — control group.
///
/// No dedup should happen, so the forward counter should equal N.
fn bench_coalesce_distinct(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalesce_distinct");
    group.sample_size(20);

    const FORWARD_MS: u64 = 5; // Shorter: we're measuring overhead, not dedup benefit.

    for n in [2usize, 8, 32] {
        group.bench_with_input(BenchmarkId::new("distinct_keys", n), &n, |b, &n| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let coalescer = RequestCoalescer::new(cfg(0, 1024));
                let forward_calls = Arc::new(AtomicUsize::new(0));

                let mut handles = Vec::with_capacity(n);
                for i in 0..n as u64 {
                    let c = coalescer.clone();
                    let fc = forward_calls.clone();
                    handles.push(tokio::spawn(async move {
                        c.coalesce(i, move || make_producer(FORWARD_MS, fc)).await
                    }));
                }
                let results: Vec<_> = futures::future::join_all(handles).await;

                // Anti-cheat: N distinct keys ⇒ N forwards, 0 saved.
                assert_eq!(
                    forward_calls.load(Ordering::SeqCst),
                    n,
                    "distinct keys must forward N times for N={}",
                    n
                );
                let (_, _, _, saved) = coalescer.stats().snapshot();
                assert_eq!(saved, 0, "distinct keys must save 0 forwards for N={}", n);
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Per-request cost of computing the semantic request key.
///
/// This is the overhead the coalescer imposes on every non-streaming request
/// (even when no dedup occurs). It must be cheap relative to the forward.
fn bench_request_key_hash(c: &mut Criterion) {
    use hier_kv_gateway_api::openai_types::OpenAIChatRequest;

    let mut group = c.benchmark_group("request_key_hash");
    group.sample_size(200);

    for n_msg in [1usize, 4, 16] {
        let mut messages = Vec::with_capacity(n_msg);
        for i in 0..n_msg {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            messages.push(serde_json::json!({
                "role": role,
                "content": format!("Message number {} with some realistic content length.", i),
            }));
        }
        let req_json = serde_json::json!({
            "model": "qwen2.5-7b",
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
        });
        let mut req: OpenAIChatRequest = serde_json::from_value(req_json).unwrap();
        req.stream = false;

        group.bench_with_input(
            BenchmarkId::new("messages", n_msg),
            &n_msg,
            |b, _| {
                b.iter(|| {
                    let key = hier_kv_gateway_api::coalescer::request_key(black_box(&req));
                    black_box(key);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_coalesce_concurrent,
        bench_coalesce_distinct,
        bench_request_key_hash,
);
criterion_main!(benches);
