//! Benchmarks for the metadata-store hot path.
//!
//! What we measure
//! ----------------
//! * `radix_find_matches_vs_find_all_matches` — the central question behind
//!   optimization #③: does calling `find_matches` once per candidate (current
//!   `KvAwareStrategy` behavior) lose to a single `find_all_matches` round-trip?
//! * `ckf_lane_of_scan` — optimization #⑤: `CkfConsumer::lane_of` is an O(N)
//!   linear scan over a `HashMap`. How bad is it really at lane counts we care
//!   about (1, 8, 16)?
//! * `ckf_estimate_overlap` — the cross-Region query path; varies the hash
//!   sequence length to expose the prefix-break behavior.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-metadata
//! ```
//!
//! The resulting HTML report lands under `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hier_kv_gateway_core::ids::{BackendId, RegionId, WorkerWithRank};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_metadata::ckf_consumer::{CkfConsumer, LANE_COUNT};
use hier_kv_gateway_metadata::ckf_producer::CkfProducer;
use hier_kv_gateway_metadata::cuckoo_filter::{try_insert, CkfSnapshot, PackedBucket, BUCKETS_PER_LANE};
use hier_kv_gateway_metadata::local_ckf::LocalCkf;
use hier_kv_gateway_metadata::radix_tree::RadixTree;

// --------------------------------------------------------------------------
// Test-data helpers
// --------------------------------------------------------------------------

/// Build a RadixTree populated with `n_backends` backends, each owning the
/// same `prefix_len`-long block-hash prefix (so every candidate has overlap).
///
/// Note: we keep a single tokio Runtime for all `apply_event` calls and keep
/// an extra `RadixTree` clone alive for the whole duration, so that the
/// `Drop` impl on `RadixTree` (which sends a best-effort `Shutdown` command
/// via `try_send`) does not actually cause the background worker thread to
/// terminate before subsequent setup calls finish.
fn build_radix_tree(n_backends: usize, prefix_len: usize) -> (RadixTree, Vec<BackendId>, Vec<u64>) {
    let tree = RadixTree::new();
    // Keep an extra clone alive so that dropping the closure's clone does not
    // become the last sender (the Drop impl's try_send(Shutdown) would otherwise
    // get buffered and processed by the worker thread, terminating it early).
    let _keep_alive = tree.clone();
    let region = RegionId::new("r1");
    let mut backends = Vec::with_capacity(n_backends);
    let prefix: Vec<u64> = (1..=prefix_len as u64).collect();

    let rt = tokio::runtime::Runtime::new().expect("failed to build tokio runtime for setup");
    for i in 0..n_backends {
        let b = BackendId::new(region.clone(), format!("inst-{i}"));
        let event = KvCacheEvent::Stored {
            worker: WorkerWithRank::from_worker_id(i as u64),
            block_hashes: prefix.clone(),
            parent_hash: None,
            num_block_tokens: Vec::new(),
        };
        // Clone before moving into the closure so `b` remains usable after.
        let b_for_event = b.clone();
        rt.block_on(async {
            tree.apply_event(b_for_event, event).await.unwrap();
        });
        backends.push(b);
    }
    (tree, backends, prefix)
}

/// Build a CkfConsumer with `lane_count` lanes bound to RegionIds
/// `"r0".."r{lane_count-1}"`. Each lane is activated with a snapshot that
/// contains a single fingerprint for hash `100`, so `estimate_overlap` on
/// `[100, 101, ...]` returns 1.
fn build_ckf_consumer(lane_count: usize) -> (CkfConsumer, Vec<RegionId>) {
    assert!(lane_count <= LANE_COUNT);
    let consumer = CkfConsumer::new();

    // Snapshot: only bucket for hash 100 has a fingerprint.
    let (fp, bucket_idx) = hier_kv_gateway_metadata::cuckoo_filter::probe(100);
    let mut buckets = vec![0u64; BUCKETS_PER_LANE];
    let mut packed: PackedBucket = 0;
    try_insert(&mut packed, fp);
    buckets[bucket_idx] = packed;
    let snapshot = CkfSnapshot { sequence: 1, buckets };

    let mut regions = Vec::with_capacity(lane_count);
    for lane in 0..lane_count {
        let region = RegionId::new(format!("r{lane}"));
        consumer.assign_lane(lane, region.clone());
        consumer.install_snapshot(lane, &snapshot);
        regions.push(region);
    }
    (consumer, regions)
}

// We need a CkfProducer too for the lane_of scan test to have realistic data,
// but the snapshot we build above is enough. Keep the import used.
fn _unused_producer_ref() -> CkfProducer {
    CkfProducer::with_seed(0)
}

// --------------------------------------------------------------------------
// Benchmarks
// --------------------------------------------------------------------------

fn bench_radix_find_matches(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_find_matches");
    group.sample_size(50);

    for n_backends in [1usize, 5, 10, 20, 50] {
        let prefix_len = 16;
        let (tree, backends, prefix) = build_radix_tree(n_backends, prefix_len);

        // Path A: current KvAwareStrategy behavior — N independent find_matches calls.
        group.bench_with_input(
            BenchmarkId::new("per_candidate_n_calls", n_backends),
            &n_backends,
            |b, &_n| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(&rt).iter(|| async {
                    let mut total = 0u32;
                    for cand in &backends {
                        let v = tree
                            .find_matches(prefix.clone(), cand.clone())
                            .await;
                        total = total.wrapping_add(v);
                    }
                    black_box(total);
                });
            },
        );

        // Path B: optimized — single find_all_matches call.
        group.bench_with_input(
            BenchmarkId::new("single_find_all_matches", n_backends),
            &n_backends,
            |b, &_n| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(&rt).iter(|| async {
                    let all = tree.find_all_matches(prefix.clone()).await;
                    black_box(all);
                });
            },
        );
    }
    group.finish();
}

fn bench_ckf_lane_of(c: &mut Criterion) {
    let mut group = c.benchmark_group("ckf_lane_of");
    group.sample_size(100);

    for lane_count in [1usize, 4, 8, 16] {
        let (consumer, regions) = build_ckf_consumer(lane_count);
        // Query the last region (worst-case for linear scan).
        let target = regions.last().unwrap().clone();

        group.bench_with_input(
            BenchmarkId::new("scan", lane_count),
            &lane_count,
            |b, &_n| {
                b.iter(|| {
                    let lane = consumer.lane_of(black_box(&target));
                    black_box(lane);
                });
            },
        );
    }
    group.finish();
}

fn bench_ckf_estimate_overlap(c: &mut Criterion) {
    let mut group = c.benchmark_group("ckf_estimate_overlap");
    group.sample_size(100);

    let (consumer, regions) = build_ckf_consumer(16);
    let region = regions[0].clone();

    for hash_len in [8usize, 32, 128, 512] {
        // The first hash (100) hits; the rest miss → prefix-break at index 1.
        let hashes: Vec<u64> = std::iter::once(100u64)
            .chain((0..hash_len as u64).map(|i| 1000 + i))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("prefix_break", hash_len),
            &hash_len,
            |b, &_n| {
                b.iter(|| {
                    let overlap = consumer.estimate_overlap(black_box(&hashes), &region);
                    black_box(overlap);
                });
            },
        );
    }
    group.finish();
}

// --------------------------------------------------------------------------
// LocalCkf benchmark: compare sync LocalCkf scan vs async RadixTree round-trip.
// --------------------------------------------------------------------------

/// Build a LocalCkf populated with `n_backends` backends, each owning the
/// same `prefix_len`-long block-hash prefix.
fn build_local_ckf(n_backends: usize, prefix_len: usize) -> (LocalCkf, Vec<BackendId>, Vec<u64>) {
    let ckf = LocalCkf::new();
    let region = RegionId::new("r1");
    let prefix: Vec<u64> = (1..=prefix_len as u64).collect();

    for i in 0..n_backends {
        let b = BackendId::new(region.clone(), format!("inst-{i}"));
        let lane = ckf.assign_lane(&b).expect("lane should be assigned");
        for &h in &prefix {
            ckf.insert(h, lane);
        }
    }
    (ckf, Vec::new(), prefix)
}

fn bench_local_ckf_vs_radix(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_ckf_vs_radix");
    group.sample_size(100);

    for n_backends in [1usize, 5, 10, 16] {
        let prefix_len = 16;

        // RadixTree path (async, channel round-trip)
        let (tree, _backends, prefix) = build_radix_tree(n_backends, prefix_len);
        group.bench_with_input(
            BenchmarkId::new("radix_find_all_matches", n_backends),
            &n_backends,
            |b, &_n| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(&rt).iter(|| async {
                    let all = tree.find_all_matches(prefix.clone()).await;
                    black_box(all);
                });
            },
        );

        // LocalCkf path (sync, cache-friendly transposed scan)
        let (ckf, _backends, prefix) = build_local_ckf(n_backends, prefix_len);
        group.bench_with_input(
            BenchmarkId::new("local_ckf_estimate_all", n_backends),
            &n_backends,
            |b, &_n| {
                b.iter(|| {
                    let all = ckf.estimate_all_overlaps(black_box(&prefix));
                    black_box(all);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_radix_find_matches, bench_ckf_lane_of, bench_ckf_estimate_overlap, bench_local_ckf_vs_radix,
);
criterion_main!(benches);
