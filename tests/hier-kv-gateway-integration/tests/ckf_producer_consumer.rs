//! Real CKF Producer / Consumer cross-Region KV projection integration test.
//!
//! This test drives `CkfProducer` to apply multiple real KV events, generates a full
//! `CkfSnapshot`, and installs it via `CkfConsumer::install_snapshot`; then it verifies
//! the hit behavior of `estimate_overlap` for stored hashes, the interruption behavior
//! for unstored hashes, and the real data flow that makes new hashes queryable on the
//! consumer side immediately after they are incrementally published via `CkfDelta`.

use hier_kv_gateway_core::ids::{BackendId, RegionId, WorkerWithRank};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_metadata::ckf_consumer::CkfConsumer;
use hier_kv_gateway_metadata::ckf_producer::CkfProducer;

/// Construct a `Stored` event.
fn stored(hashes: Vec<u64>) -> KvCacheEvent {
    KvCacheEvent::Stored {
        worker: WorkerWithRank::from_worker_id(1),
        block_hashes: hashes,
        parent_hash: None,
        num_block_tokens: Vec::new(),
    }
}

/// Construct a backend identifier for tests.
fn backend(name: &str) -> BackendId {
    BackendId::new("r1", name)
}

#[tokio::test]
async fn ckf_snapshot_install_then_estimate_overlap() {
    // 1) Producer side: apply multiple KV events to produce fingerprints
    let mut producer = CkfProducer::with_seed(42);
    let b1 = backend("worker-1");
    let b2 = backend("worker-2");

    // b1 stores 5 block hashes
    producer.apply_event(&stored(vec![101, 102, 103, 104, 105]), &b1);
    // b2 shares 101 with b1 and adds 106; 101 should be de-duplicated (not repeated within the same backend)
    producer.apply_event(&stored(vec![101, 106]), &b2);

    // The number of tracked distinct hashes should be 6: {101, 102, 103, 104, 105, 106}
    assert_eq!(
        producer.tracked_hashes(),
        6,
        "should track 6 distinct hashes"
    );
    // The number of fingerprints should be 6 (one insertion per distinct hash)
    assert_eq!(
        producer.num_items(),
        6,
        "should insert 6 fingerprints"
    );

    // 2) Generate a full snapshot
    let snapshot = producer.snapshot();
    assert_eq!(snapshot.sequence, 1, "first snapshot sequence should be 1");
    assert_eq!(
        snapshot.buckets.len(),
        hier_kv_gateway_metadata::cuckoo_filter::BUCKETS_PER_LANE,
        "snapshot should contain the full number of buckets for the lane"
    );

    // 3) Consumer side: allocate lane + install snapshot
    let consumer = CkfConsumer::new();
    let region = RegionId::new("r1");
    consumer.assign_lane(0, region.clone());
    consumer.install_snapshot(0, &snapshot);
    assert!(
        consumer.is_lane_active(0),
        "after installing the snapshot, lane 0 should be in Active state"
    );

    // 4) estimate_overlap should fully hit the stored hash sequence
    let overlap_known = consumer.estimate_overlap(&[101, 102, 103, 104, 105, 106], &region);
    assert_eq!(
        overlap_known, 6,
        "all 6 stored hashes should hit, actual: {}",
        overlap_known
    );

    // 5) Unstored hashes should stop at the first miss; hash 999 does not exist, so the first one interrupts
    let overlap_unknown = consumer.estimate_overlap(&[999], &region);
    assert_eq!(
        overlap_unknown, 0,
        "unstored hashes should not hit, actual: {}",
        overlap_unknown
    );

    // 6) A mixed sequence of stored + unstored should interrupt at the prefix
    let overlap_mixed = consumer.estimate_overlap(&[101, 102, 999], &region);
    assert_eq!(
        overlap_mixed, 2,
        "after prefix [101, 102] hits, the 3rd hash 999 interrupts -> 2"
    );

    // 7) Querying a Region with no allocated lane returns 0
    let other_region = RegionId::new("other-region");
    let overlap_no_lane = consumer.estimate_overlap(&[101], &other_region);
    assert_eq!(
        overlap_no_lane, 0,
        "querying a Region with no allocated lane should return 0"
    );
}

#[tokio::test]
async fn ckf_delta_propagation_makes_new_hash_queryable() {
    // This case verifies delta incremental publishing: first install a snapshot, then synchronize new hashes via delta.
    let mut producer = CkfProducer::with_seed(7);
    let b = backend("worker-1");

    // Initial: store [1, 2, 3] and publish a snapshot
    producer.apply_event(&stored(vec![1, 2, 3]), &b);
    let snapshot = producer.snapshot();

    let consumer = CkfConsumer::new();
    let region = RegionId::new("r2");
    consumer.assign_lane(2, region.clone());
    consumer.install_snapshot(2, &snapshot);

    // Verify that in the initial state [1, 2, 3] are queryable and [4] is not
    assert_eq!(
        consumer.estimate_overlap(&[1, 2, 3], &region),
        3,
        "initial snapshot should make [1, 2, 3] all hit"
    );
    assert_eq!(
        consumer.estimate_overlap(&[4], &region),
        0,
        "hash 4 has not been published yet and should not hit"
    );

    // Apply a new event: store [4], then generate a delta
    producer.apply_event(&stored(vec![4]), &b);
    let delta = producer
        .delta()
        .expect("after applying a new event, a non-empty delta should be generated");
    assert_eq!(
        delta.prev_sequence, 1,
        "delta's prev_sequence should be the snapshot's sequence=1"
    );
    assert_eq!(delta.sequence, 2, "delta's sequence should increment to 2");
    assert!(
        !delta.buckets.is_empty(),
        "delta should contain at least one changed bucket"
    );

    // consumer applies the delta
    consumer.apply_delta(2, &delta);

    // Verify that the new hash 4 is now queryable
    let overlap_after_delta = consumer.estimate_overlap(&[1, 2, 3, 4], &region);
    assert_eq!(
        overlap_after_delta, 4,
        "after applying the delta, [1, 2, 3, 4] should all hit, actual: {}",
        overlap_after_delta
    );
}

#[tokio::test]
async fn ckf_false_positive_rate_is_low_for_unstored_hashes() {
    // This case verifies that the CKF keeps a low false-positive rate for unstored hashes.
    // We insert a small number of real hashes, then randomly probe many unstored hashes; the false-positive count should be close to 0.
    let mut producer = CkfProducer::with_seed(123);
    let b = backend("worker-1");

    // Insert 50 real hashes
    let stored_hashes: Vec<u64> = (0..50u64).map(|i| 10_000 + i * 7).collect();
    producer.apply_event(&stored(stored_hashes.clone()), &b);
    let snapshot = producer.snapshot();

    let consumer = CkfConsumer::new();
    let region = RegionId::new("r3");
    consumer.assign_lane(5, region.clone());
    consumer.install_snapshot(5, &snapshot);

    // All real hashes should hit
    let overlap_real = consumer.estimate_overlap(&stored_hashes, &region);
    assert_eq!(
        overlap_real, 50,
        "all 50 real hashes should hit, actual: {}",
        overlap_real
    );

    // Probe 200 unstored hashes; the total false-positive count should be well below 1%
    // Note: the theoretical FPR of CKF is about 8.5e-5; for 200 hashes, the expected false-positive count is << 1.
    // Since estimate_overlap interrupts at the first miss, if the false-positive happens to occur at position 1,
    // overlap will be 1; otherwise it is 0.
    let mut false_positive_count = 0usize;
    for i in 0..200u64 {
        let probe = 1_000_000 + i * 13; // Does not overlap with stored_hashes
        let overlap = consumer.estimate_overlap(&[probe], &region);
        if overlap > 0 {
            false_positive_count += 1;
        }
    }
    assert!(
        false_positive_count <= 2,
        "false-positive count should be <= 2, actual: {}",
        false_positive_count
    );
}
