//! Benchmarks for load metrics encoding: JSON vs postcard (binary) vs LoadPayload.
//!
//! Measures three dimensions:
//! 1. **Serialization size** — bytes per backend for each encoding
//! 2. **Encode speed** — serialize + (optional) base64 wrap throughput
//! 3. **Decode speed** — deserialize throughput
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-cluster --bench load_encoding
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hier_kv_gateway_core::ids::{BackendId, BackendInstanceId, RegionId};
use hier_kv_gateway_core::metrics::{BackendMetrics, LatencyStats};

use hier_kv_gateway_cluster::messages::LoadPayload;

// --------------------------------------------------------------------------
// Test-data helpers
// --------------------------------------------------------------------------

fn sample_metrics(i: usize) -> BackendMetrics {
    BackendMetrics {
        active_requests: (i % 10) as u64,
        queue_depth: (i % 5) as u64,
        active_decode_blocks: (i % 8) as u64 * 2,
        active_prefill_tokens: (i % 100) as u64 * 16,
        kv_used_blocks: (i % 50) as u64 + 10,
        kv_total_blocks: 100,
        gpu_utilization: (i as f64 * 0.07) % 1.0,
        gpu_memory_used_mb: (i as u64 % 30) * 1000 + 5000,
        gpu_memory_total_mb: 40_000,
        latency: LatencyStats {
            p50_ms: 10.0 + (i as f64 * 0.3),
            p99_ms: 50.0 + (i as f64 * 1.2),
            p999_ms: 80.0 + (i as f64 * 2.1),
            sample_count: 1000 + i as u64,
        },
        timestamp: 1_700_000_000 + i as i64,
    }
}

fn build_backends(n: usize) -> Vec<(String, BackendMetrics)> {
    let region = RegionId::new("r1");
    (0..n)
        .map(|i| {
            let backend = BackendId::new(region.clone(), BackendInstanceId::new(format!("inst-{i}")));
            (backend.to_string(), sample_metrics(i))
        })
        .collect()
}

// --------------------------------------------------------------------------
// Size measurement (not a Criterion bench, just printed once)
// --------------------------------------------------------------------------

fn print_sizes() {
    println!("\n=== Load Metrics Encoding Size Comparison ===\n");
    println!(
        "{:<8} {:<12} {:<14} {:<16} {:<16} {:<12}",
        "N", "Backends", "JSON (bytes)", "Postcard (raw)", "LoadPayload (b64)", "Compression"
    );
    println!("{}", "-".repeat(80));

    for n in [1usize, 5, 10, 20, 50] {
        let backends = build_backends(n);

        // JSON serialization
        let json = serde_json::to_vec(&backends).unwrap();
        let json_size = json.len();

        // Postcard raw binary
        let postcard_bytes = postcard::to_allocvec(&backends).unwrap();
        let postcard_size = postcard_bytes.len();

        // LoadPayload (postcard + base64)
        let payload = LoadPayload::encode_full(&backends);
        let payload_size = serde_json::to_vec(&payload).unwrap().len();

        let ratio = json_size as f64 / payload_size as f64;

        println!(
            "{:<8} {:<12} {:<14} {:<16} {:<16} {:<12.2}x",
            n, n, json_size, postcard_size, payload_size, ratio
        );
    }
    println!();
}

// --------------------------------------------------------------------------
// Benchmarks
// --------------------------------------------------------------------------

fn bench_encode(c: &mut Criterion) {
    print_sizes();

    let mut group = c.benchmark_group("load_encode");
    group.sample_size(200);

    for n in [1usize, 5, 10, 20, 50] {
        let backends = build_backends(n);

        // JSON serialize
        group.bench_with_input(
            BenchmarkId::new("json_serialize", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    let json = serde_json::to_vec(black_box(&backends)).unwrap();
                    black_box(json);
                });
            },
        );

        // Postcard serialize (raw binary)
        group.bench_with_input(
            BenchmarkId::new("postcard_serialize", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    let bytes = postcard::to_allocvec(black_box(&backends)).unwrap();
                    black_box(bytes);
                });
            },
        );

        // LoadPayload encode_full (postcard + base64 + wrap)
        group.bench_with_input(
            BenchmarkId::new("loadpayload_encode", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    let payload = LoadPayload::encode_full(black_box(&backends));
                    black_box(payload);
                });
            },
        );
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_decode");
    group.sample_size(200);

    for n in [1usize, 5, 10, 20, 50] {
        let backends = build_backends(n);

        // Pre-encode for decode benchmarks
        let json_bytes = serde_json::to_vec(&backends).unwrap();
        let postcard_bytes = postcard::to_allocvec(&backends).unwrap();
        let payload = LoadPayload::encode_full(&backends);

        // JSON deserialize
        group.bench_with_input(
            BenchmarkId::new("json_deserialize", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    let decoded: Vec<(String, BackendMetrics)> =
                        serde_json::from_slice(black_box(&json_bytes)).unwrap();
                    black_box(decoded);
                });
            },
        );

        // Postcard deserialize (raw binary)
        group.bench_with_input(
            BenchmarkId::new("postcard_deserialize", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    let decoded: Vec<(String, BackendMetrics)> =
                        postcard::from_bytes(black_box(&postcard_bytes)).unwrap();
                    black_box(decoded);
                });
            },
        );

        // LoadPayload decode (base64 + postcard)
        group.bench_with_input(
            BenchmarkId::new("loadpayload_decode", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    let decoded = black_box(&payload).decode().unwrap();
                    black_box(decoded);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
