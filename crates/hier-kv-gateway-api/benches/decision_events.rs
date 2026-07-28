//! Benchmarks for the decision-event telemetry pipeline.
//!
//! What we measure
//! ----------------
//! * `event_serialize` — `serde_json::to_string` cost for a
//!   [`DecisionEvent`], varying the number of candidate scores / forward
//!   attempts. This is the per-request serialization cost paid by the
//!   tracing and NDJSON-file sinks.
//! * `ring_buffer_push` — pushing into the in-memory
//!   [`DecisionEventBuffer`] at capacity (evict + push) vs into a
//!   not-yet-full buffer; this is what `GET /admin/decision_events`
//!   storage costs per request.
//! * `ring_buffer_snapshot` — admin-endpoint read cost at several buffer
//!   occupancies.
//! * `sink_emit` — full `emit` through the sink chain: [`NoopSink`]
//!   baseline, [`RingBufferSink`], and a [`MultiSink`] fan-out
//!   (ring buffer + tracing with the subscriber disabled, i.e. the
//!   serialize-only cost of the tracing path).
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-api --bench decision_events
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use hier_kv_gateway_api::telemetry::{DecisionEventBuffer, RingBufferSink, TracingSink};
use hier_kv_gateway_core::decision_event::{
    CandidateScore, DecisionEvent, DecisionEventSink, DecisionOutcome, ForwardAttempt, MultiSink,
    NoopSink, WeightSnapshot,
};

/// Build a representative event with `n_candidates` scored candidates and
/// `n_attempts` forward attempts.
fn build_event(n_candidates: usize, n_attempts: usize) -> DecisionEvent {
    DecisionEvent {
        event_id: "3f6f1b2c-0000-4000-8000-abcdefabcdef".to_string(),
        timestamp_unix_ms: 1_752_000_000_000,
        gateway_instance: "gateway-1".to_string(),
        gateway_region: "cloud-cn-beijing".to_string(),
        request_id: "req-42".to_string(),
        model: "qwen2.5-7b".to_string(),
        session_id: Some("sess-7".to_string()),
        strategy: "hybrid".to_string(),
        weights: Some(WeightSnapshot {
            kv: 0.41,
            load: 0.33,
            topology: 0.26,
        }),
        candidates: (0..n_candidates)
            .map(|i| CandidateScore {
                backend: format!("cloud-cn-beijing/inst-{i}"),
                score: 1.0 / (i as f64 + 1.0),
                kv_overlap: (16 - i) as u32,
            })
            .collect(),
        attempts: (0..n_attempts)
            .map(|i| ForwardAttempt {
                backend: format!("cloud-cn-beijing/inst-{i}"),
                success: i + 1 == n_attempts,
                skipped_open_circuit: i % 2 == 0,
                error: if i + 1 == n_attempts {
                    None
                } else {
                    Some("connection refused".to_string())
                },
            })
            .collect(),
        selected_backend: Some("cloud-cn-beijing/inst-3".to_string()),
        kv_overlap: 12,
        prompt_blocks: 16,
        routing_latency_us: 240,
        total_latency_us: 5_100,
        outcome: DecisionOutcome::Success,
    }
}

fn bench_event_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_serialize");
    group.sample_size(200);

    // (candidates, attempts): 1/1 is the single-backend happy path,
    // 10/4 a mid-size fleet with failover, 50/8 a large fleet.
    for (nc, na) in [(1usize, 1usize), (10, 4), (50, 8)] {
        let ev = build_event(nc, na);
        group.bench_with_input(
            BenchmarkId::new("candidates_attempts", format!("{nc}_{na}")),
            &ev,
            |b, ev| {
                b.iter(|| {
                    let s = serde_json::to_string(black_box(ev)).unwrap();
                    black_box(s);
                });
            },
        );
    }
    group.finish();
}

fn bench_ring_buffer_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_push");
    group.sample_size(200);

    let ev = build_event(10, 4);

    // Buffer with headroom: push only.
    let buf_growing = DecisionEventBuffer::new(1024);
    group.bench_function("growing", |b| {
        b.iter(|| buf_growing.push(black_box(ev.clone())));
    });

    // Full buffer: every push evicts the oldest entry first.
    let buf_full = DecisionEventBuffer::new(256);
    for _ in 0..256 {
        buf_full.push(ev.clone());
    }
    group.bench_function("at_capacity_evicting", |b| {
        b.iter(|| buf_full.push(black_box(ev.clone())));
    });

    group.finish();
}

fn bench_ring_buffer_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_snapshot");
    group.sample_size(100);

    let ev = build_event(10, 4);
    for size in [64usize, 256, 1024] {
        let buf = DecisionEventBuffer::new(size);
        for _ in 0..size {
            buf.push(ev.clone());
        }
        group.bench_with_input(BenchmarkId::new("occupancy", size), &buf, |b, buf| {
            b.iter(|| {
                let snap = buf.snapshot(black_box(0));
                black_box(snap);
            });
        });
    }
    group.finish();
}

fn bench_sink_emit(c: &mut Criterion) {
    let mut group = c.benchmark_group("sink_emit");
    group.sample_size(200);

    let ev = build_event(10, 4);

    group.bench_function("noop", |b| {
        let sink = NoopSink;
        b.iter(|| sink.emit(black_box(&ev)));
    });

    group.bench_function("ring_buffer", |b| {
        let buf = DecisionEventBuffer::new(1024);
        let sink = RingBufferSink::new(buf);
        b.iter(|| sink.emit(black_box(&ev)));
    });

    // Tracing sink with no subscriber installed: measures the serialize +
    // callsite cost of the `tracing` path (a subscriber would add its own).
    group.bench_function("tracing_no_subscriber", |b| {
        let sink = TracingSink;
        b.iter(|| sink.emit(black_box(&ev)));
    });

    group.bench_function("multi_ring_plus_tracing", |b| {
        let buf = DecisionEventBuffer::new(1024);
        let sink = MultiSink::new(vec![
            Box::new(RingBufferSink::new(buf)),
            Box::new(TracingSink),
        ]);
        b.iter(|| sink.emit(black_box(&ev)));
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_event_serialize,
        bench_ring_buffer_push,
        bench_ring_buffer_snapshot,
        bench_sink_emit,
);
criterion_main!(benches);
