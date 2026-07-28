//! Benchmarks for the SGLang connector wire format and the failover
//! (circuit-breaker) hot path.
//!
//! What we measure
//! ----------------
//! * `generate_stream_parse` — the [`SglangGenerateParser`] converting
//!   SGLang's accumulated-text SSE chunks into gateway deltas, at several
//!   stream lengths. This is the per-chunk cost on the token-id forwarding
//!   path (`/generate`).
//! * `server_info_metrics_map` — mapping a `/get_server_info` payload to
//!   [`BackendMetrics`]; runs once per metrics-poll per backend.
//! * `circuit_breaker_allow` — the per-candidate check the forwarding loop
//!   performs before every attempt, with the breaker closed (healthy
//!   fleet) and open (backend under failover).
//! * `circuit_breaker_record` — `on_success` / `on_failure` bookkeeping
//!   after each attempt.
//! * `retry_backoff` — backoff computation per retry attempt.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-connector --bench sglang_wire
//! ```

use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::StreamExt;

use hier_kv_gateway_connector::resilience::{CircuitBreakerRegistry, RetryPolicy};
use hier_kv_gateway_connector::sglang::{map_server_info_metrics, SglangGenerateParser};
use hier_kv_gateway_core::config::ResilienceConfig;
use hier_kv_gateway_core::ids::BackendId;

/// Build one SSE payload of `n_chunks` accumulated-text frames ending with a
/// terminal finish_reason frame.
fn generate_payload(n_chunks: usize) -> Vec<u8> {
    let mut s = String::new();
    let mut text = String::new();
    for i in 0..n_chunks {
        text.push_str(&format!("tok{i} "));
        let frame = if i + 1 == n_chunks {
            format!(
                r#"{{"text": "{}", "meta_info": {{"finish_reason": {{"type": "stop"}}}}}}"#,
                text
            )
        } else {
            format!(r#"{{"text": "{}", "meta_info": {{"finish_reason": null}}}}"#, text)
        };
        s.push_str("data: ");
        s.push_str(&frame);
        s.push_str("\n\n");
    }
    s.into_bytes()
}

fn bench_generate_stream_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_stream_parse");
    group.sample_size(50);

    for n in [8usize, 64, 256] {
        let payload = generate_payload(n);
        group.bench_with_input(BenchmarkId::new("chunks", n), &payload, |b, payload| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.to_async(&rt).iter(|| async {
                let byte_stream = futures::stream::iter(vec![Ok::<_, reqwest::Error>(
                    bytes::Bytes::from(payload.clone()),
                )]);
                let parser = SglangGenerateParser::new(
                    byte_stream,
                    BackendId::new("r1", "sglang-0"),
                    Instant::now(),
                );
                let chunks: Vec<_> = parser.collect().await;
                black_box(chunks);
            });
        });
    }
    group.finish();
}

fn bench_server_info_metrics_map(c: &mut Criterion) {
    let info = serde_json::json!({
        "internal_states": [{
            "num_running_reqs": 17,
            "num_queue_reqs": 4,
            "num_used_tokens": 245_760,
            "max_total_num_tokens": 2_097_152,
            "token_usage": 0.117,
            "gen_throughput": 812.4
        }]
    });

    c.bench_function("server_info_metrics_map", |b| {
        b.iter(|| {
            let m = map_server_info_metrics(black_box(&info), black_box(16), black_box(0));
            black_box(m);
        });
    });
}

fn bench_circuit_breaker(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_breaker");
    group.sample_size(200);

    let registry = CircuitBreakerRegistry::new(&ResilienceConfig::default());
    let healthy = BackendId::new("r1", "healthy");
    let sick = BackendId::new("r1", "sick");

    // Prime both breakers.
    let _ = registry.allow(&healthy);
    let _ = registry.allow(&sick);

    // Closed breaker (healthy backend) — the common hot path.
    group.bench_function("allow_closed", |b| {
        b.iter(|| black_box(registry.allow(black_box(&healthy))));
    });

    // Open breaker (backend in failover): skipped without any I/O.
    for _ in 0..ResilienceConfig::default().circuit_breaker_failure_threshold {
        registry.on_failure(&sick);
    }
    group.bench_function("allow_open", |b| {
        b.iter(|| black_box(registry.allow(black_box(&sick))));
    });

    // Bookkeeping after an attempt.
    group.bench_function("on_success", |b| {
        b.iter(|| registry.on_success(black_box(&healthy)));
    });

    let other = BackendId::new("r1", "flapping");
    let _ = registry.allow(&other);
    group.bench_function("on_failure", |b| {
        b.iter(|| registry.on_failure(black_box(&other)));
    });

    group.finish();
}

fn bench_retry_backoff(c: &mut Criterion) {
    let policy = RetryPolicy::new(Duration::from_millis(50), Duration::from_millis(1_000));

    let mut group = c.benchmark_group("retry_backoff");
    group.sample_size(200);
    for attempt in [0u32, 2, 5, 10] {
        group.bench_with_input(BenchmarkId::new("attempt", attempt), &attempt, |b, &a| {
            b.iter(|| black_box(policy.backoff(black_box(a))));
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_generate_stream_parse,
        bench_server_info_metrics_map,
        bench_circuit_breaker,
        bench_retry_backoff,
);
criterion_main!(benches);
