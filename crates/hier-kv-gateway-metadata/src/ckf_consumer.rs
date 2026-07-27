//! CKF Consumer: cross-Region approximate KV index (transposed layout).
//!
//! The bucket data of 16 Region lanes is arranged in bucket-major order
//! (`Vec<[AtomicU64; 16]>`, where each element holds the same bucket index
//! across all 16 lanes), so a single query can probe multiple lanes within the
//! same cache line.
//!
//! Query semantics: for a given block hash sequence, look up the fingerprint for
//! each hash one by one; on a hit, overlap++; on a miss, stop early (prefix
//! break). The lane state (Active/Retired) is maintained via an atomic byte;
//! when installing a snapshot, the lane is first retired, then the buckets are
//! written, then re-activated, avoiding exposure to intermediate states.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use hier_kv_gateway_core::ids::RegionId;
use parking_lot::RwLock;

use crate::cuckoo_filter::{
    self, CkfDelta, CkfSnapshot, BUCKETS_PER_LANE,
};

/// Number of lanes supported by this consumer (i.e. the maximum number of Regions tracked simultaneously).
pub const LANE_COUNT: usize = 16;

/// Lane states: 0 = Active, 1 = Retired.
pub const LANE_ACTIVE: u8 = 0;
pub const LANE_RETIRED: u8 = 1;

/// Transposed CKF consumer.
pub struct CkfConsumer {
    /// Bucket-major layout: `buckets[i][lane]` is the packed value of bucket i in lane.
    buckets: Vec<[AtomicU64; LANE_COUNT]>,
    /// Lane index → the RegionId currently bound to that lane.
    lane_regions: RwLock<HashMap<usize, RegionId>>,
    /// State of each lane: Active or Retired.
    lane_status: [AtomicU8; LANE_COUNT],
}

impl std::fmt::Debug for CkfConsumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CkfConsumer")
            .field("bucket_count", &self.buckets.len())
            .field("lane_count", &LANE_COUNT)
            .finish()
    }
}

impl CkfConsumer {
    /// Create an empty consumer.
    pub fn new() -> Self {
        // Initialize the fixed-size bucket array (each element is [AtomicU64; 16]).
        let buckets = (0..BUCKETS_PER_LANE)
            .map(|_| {
                let mut arr: [AtomicU64; LANE_COUNT] = [const { AtomicU64::new(0) }; LANE_COUNT];
                // The const syntax above requires nightly compatibility; use a loop as a fallback for stable Rust.
                for slot in arr.iter_mut() {
                    slot.store(0, Ordering::Relaxed);
                }
                arr
            })
            .collect::<Vec<_>>();
        Self {
            buckets,
            lane_regions: RwLock::new(HashMap::new()),
            lane_status: core::array::from_fn(|_| AtomicU8::new(LANE_RETIRED)),
        }
    }

    /// Estimate the prefix overlap length of the given hash sequence on the lane
    /// for the specified Region.
    ///
    /// For each hash, compute its (fp, bucket_idx), load the bucket value for
    /// that lane, and if it contains fp, increment overlap; otherwise stop early.
    pub fn estimate_overlap(&self, hashes: &[u64], region: &RegionId) -> u32 {
        let lane = self.lane_of(region);
        let Some(lane) = lane else {
            return 0;
        };
        if self.lane_status[lane].load(Ordering::Acquire) != LANE_ACTIVE {
            return 0;
        }
        let mut overlap = 0u32;
        for &hash in hashes {
            let (fp, bucket_idx) = cuckoo_filter::probe(hash);
            let packed = self.buckets[bucket_idx][lane].load(Ordering::Acquire);
            if cuckoo_filter::bucket_contains(packed, fp) {
                overlap += 1;
            } else {
                break;
            }
        }
        overlap
    }

    /// Install a full snapshot for a lane.
    ///
    /// Order: retire first (block reads) → write all buckets → activate (restore reads).
    pub fn install_snapshot(&self, lane: usize, snapshot: &CkfSnapshot) {
        debug_assert!(lane < LANE_COUNT);
        debug_assert_eq!(snapshot.buckets.len(), BUCKETS_PER_LANE);
        self.lane_status[lane].store(LANE_RETIRED, Ordering::Release);
        for (i, &value) in snapshot.buckets.iter().enumerate() {
            self.buckets[i][lane].store(value, Ordering::Relaxed);
        }
        self.lane_status[lane].store(LANE_ACTIVE, Ordering::Release);
    }

    /// Apply a delta update to a lane.
    ///
    /// Deltas are weakly consistent multi-bucket writes; readers may observe a
    /// partially applied state. This is a design tradeoff of CKF and we do not
    /// introduce seqlock or retries at this layer.
    pub fn apply_delta(&self, lane: usize, delta: &CkfDelta) {
        debug_assert!(lane < LANE_COUNT);
        for &(bucket_idx, value) in &delta.buckets {
            if bucket_idx < self.buckets.len() {
                self.buckets[bucket_idx][lane].store(value, Ordering::Release);
            }
        }
    }

    /// Mark the specified lane as Retired (no longer participates in queries).
    pub fn retire_lane(&self, lane: usize) {
        debug_assert!(lane < LANE_COUNT);
        self.lane_status[lane].store(LANE_RETIRED, Ordering::Release);
    }

    /// Mark the specified lane as Active (restore queries).
    pub fn activate_lane(&self, lane: usize) {
        debug_assert!(lane < LANE_COUNT);
        self.lane_status[lane].store(LANE_ACTIVE, Ordering::Release);
    }

    /// Bind a lane to the specified Region.
    pub fn assign_lane(&self, lane: usize, region: RegionId) {
        debug_assert!(lane < LANE_COUNT);
        let mut regions = self.lane_regions.write();
        regions.insert(lane, region);
    }

    /// Unbind a lane (used when a Region migrates out).
    pub fn unassign_lane(&self, lane: usize) {
        debug_assert!(lane < LANE_COUNT);
        let mut regions = self.lane_regions.write();
        regions.remove(&lane);
    }

    /// Query the lane index currently bound to the specified Region.
    pub fn lane_of(&self, region: &RegionId) -> Option<usize> {
        let regions = self.lane_regions.read();
        regions
            .iter()
            .find_map(|(lane, r)| if r == region { Some(*lane) } else { None })
    }

    /// Get the current state of a lane (true means Active).
    pub fn is_lane_active(&self, lane: usize) -> bool {
        self.lane_status[lane].load(Ordering::Acquire) == LANE_ACTIVE
    }
}

impl Default for CkfConsumer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuckoo_filter::{try_insert, PackedBucket, BUCKETS_PER_LANE};
    use crate::ckf_producer::CkfProducer;
    use hier_kv_gateway_core::ids::{BackendId, WorkerWithRank};
    use hier_kv_gateway_core::kv_event::KvCacheEvent;

    fn backend(n: u8) -> BackendId {
        BackendId::new(format!("r{n}"), format!("i{n}"))
    }

    fn stored(hashes: Vec<u64>) -> KvCacheEvent {
        KvCacheEvent::Stored {
            worker: WorkerWithRank::from_worker_id(1),
            block_hashes: hashes,
            parent_hash: None,
            num_block_tokens: Vec::new(),
        }
    }

    #[test]
    fn install_snapshot_then_estimate_overlap() {
        let mut producer = CkfProducer::with_seed(99);
        let backend = backend(1);
        producer.apply_event(&stored(vec![1, 2, 3, 4, 5]), &backend);
        let snapshot = producer.snapshot();

        let consumer = CkfConsumer::new();
        let region = RegionId::new("r7");
        consumer.assign_lane(0, region.clone());
        consumer.install_snapshot(0, &snapshot);

        let overlap = consumer.estimate_overlap(&[1, 2, 3, 4, 5, 6], &region);
        assert!(overlap >= 5, "expected overlap of 5, got {}", overlap);
    }

    #[test]
    fn prefix_break_stops_at_first_miss() {
        // Build a snapshot containing only some of the hashes
        let mut buckets = vec![0u64; BUCKETS_PER_LANE];
        let (fp1, b1) = crate::cuckoo_filter::probe(100);
        let mut packed: PackedBucket = 0;
        try_insert(&mut packed, fp1);
        buckets[b1] = packed;
        let snapshot = CkfSnapshot { sequence: 1, buckets };

        let consumer = CkfConsumer::new();
        let region = RegionId::new("r1");
        consumer.assign_lane(0, region.clone());
        consumer.install_snapshot(0, &snapshot);

        // hash 100 hits, hash 101 misses → should break at 101
        let overlap = consumer.estimate_overlap(&[100, 101], &region);
        assert_eq!(overlap, 1);
    }
}
