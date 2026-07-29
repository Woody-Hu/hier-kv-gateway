//! Local CKF projection: per-backend cuckoo filter lanes in transposed layout.
//!
//! This is the local counterpart of [`crate::ckf_consumer::CkfConsumer`]. Where
//! the consumer receives cross-Region snapshots/deltas via Gossip, the local
//! projection is **written directly** by the [`crate::radix_tree::RadixTree`]
//! ingestion path: every `Stored`/`Removed` event that lands on the RadixTree
//! also inserts/deletes a fingerprint here, so queries can be served without a
//! channel round-trip.
//!
//! ## Layout
//!
//! Same transposed layout as `CkfConsumer`: `buckets[i][lane]` groups the same
//! bucket index across all lanes into one cache line, so a single load probes
//! all lanes for a given hash simultaneously.
//!
//! ## Lane assignment
//!
//! Each backend is assigned a lane index (0..LANE_COUNT) via [`assign_lane`].
//! Writes target the lane bound to the backend; queries return a `Vec<u32>`
//! with one overlap per active lane.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::cuckoo_filter::{self, BUCKETS_PER_LANE};

/// Number of local backend lanes supported.
pub const LOCAL_LANE_COUNT: usize = 16;

/// Lane states: 0 = Active, 1 = Retired (not yet assigned or being reset).
pub const LANE_ACTIVE: u8 = 0;
pub const LANE_RETIRED: u8 = 1;

/// Local CKF projection with per-backend lanes.
///
/// Writes are **not** lock-free: each lane's fingerprint insertion uses cuckoo
/// eviction which may touch multiple buckets. To keep writes simple we use a
/// per-lane `RwLock` — write contention is low because each backend's KV events
/// are typically driven by a single ingestion task. Reads (estimate_all_overlaps)
/// are lock-free over the atomic bucket array.
pub struct LocalCkf {
    /// Bucket-major layout: `buckets[i][lane]` is the packed value of bucket i in lane.
    buckets: Vec<[AtomicU64; LOCAL_LANE_COUNT]>,
    /// Lane index → BackendId currently bound to that lane.
    lane_backends: RwLock<HashMap<usize, hier_kv_gateway_core::ids::BackendId>>,
    /// BackendId → lane index (reverse of lane_backends for O(1) lookup).
    backend_lanes: RwLock<HashMap<hier_kv_gateway_core::ids::BackendId, usize>>,
    /// State of each lane: Active or Retired.
    lane_status: [AtomicU8; LOCAL_LANE_COUNT],
}

impl std::fmt::Debug for LocalCkf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalCkf")
            .field("bucket_count", &self.buckets.len())
            .field("lane_count", &LOCAL_LANE_COUNT)
            .finish()
    }
}

impl Default for LocalCkf {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalCkf {
    /// Create an empty local CKF projection.
    pub fn new() -> Self {
        let buckets = (0..BUCKETS_PER_LANE)
            .map(|_| {
                let mut arr: [AtomicU64; LOCAL_LANE_COUNT] =
                    [const { AtomicU64::new(0) }; LOCAL_LANE_COUNT];
                for slot in arr.iter_mut() {
                    slot.store(0, Ordering::Relaxed);
                }
                arr
            })
            .collect::<Vec<_>>();
        Self {
            buckets,
            lane_backends: RwLock::new(HashMap::new()),
            backend_lanes: RwLock::new(HashMap::new()),
            lane_status: core::array::from_fn(|_| AtomicU8::new(LANE_RETIRED)),
        }
    }

    /// Assign a lane to a backend. If the backend already has a lane, returns it.
    /// If no free lane is available, returns `None`.
    pub fn assign_lane(&self, backend: &hier_kv_gateway_core::ids::BackendId) -> Option<usize> {
        // Fast path: already assigned
        if let Some(&lane) = self.backend_lanes.read().get(backend) {
            return Some(lane);
        }
        // Slow path: find a free lane
        let mut lane_backends = self.lane_backends.write();
        let mut backend_lanes = self.backend_lanes.write();
        // Double-check after acquiring write lock
        if let Some(&lane) = backend_lanes.get(backend) {
            return Some(lane);
        }
        // Find a retired lane
        for lane in 0..LOCAL_LANE_COUNT {
            if !lane_backends.contains_key(&lane) {
                // Clear the lane's buckets
                for i in 0..BUCKETS_PER_LANE {
                    self.buckets[i][lane].store(0, Ordering::Relaxed);
                }
                lane_backends.insert(lane, backend.clone());
                backend_lanes.insert(backend.clone(), lane);
                self.lane_status[lane].store(LANE_ACTIVE, Ordering::Release);
                return Some(lane);
            }
        }
        None
    }

    /// Unassign a backend's lane (clears all its fingerprints).
    pub fn unassign_lane(&self, backend: &hier_kv_gateway_core::ids::BackendId) {
        let lane = {
            let mut backend_lanes = self.backend_lanes.write();
            backend_lanes.remove(backend)
        };
        if let Some(lane) = lane {
            self.lane_status[lane].store(LANE_RETIRED, Ordering::Release);
            // Clear buckets
            for i in 0..BUCKETS_PER_LANE {
                self.buckets[i][lane].store(0, Ordering::Relaxed);
            }
            self.lane_backends.write().remove(&lane);
        }
    }

    /// Get the lane index for a backend, if assigned.
    pub fn lane_of(&self, backend: &hier_kv_gateway_core::ids::BackendId) -> Option<usize> {
        self.backend_lanes.read().get(backend).copied()
    }

    /// Get the backend bound to a lane, if any.
    pub fn backend_of_lane(&self, lane: usize) -> Option<hier_kv_gateway_core::ids::BackendId> {
        self.lane_backends.read().get(&lane).cloned()
    }

    /// Insert a fingerprint for a hash into the specified lane.
    ///
    /// Uses a CAS loop on the two candidate buckets. Returns `true` on
    /// success, `false` if the lane is full for this hash's buckets.
    pub fn insert(&self, hash: u64, lane: usize) -> bool {
        if self.lane_status[lane].load(Ordering::Acquire) != LANE_ACTIVE {
            return false;
        }
        let (fp, bucket_a) = cuckoo_filter::probe(hash);
        let bucket_b = cuckoo_filter::alt_index(bucket_a, fp);

        // Try bucket_a then bucket_b with CAS.
        for &bucket_idx in &[bucket_a, bucket_b] {
            loop {
                let packed = self.buckets[bucket_idx][lane].load(Ordering::Acquire);
                match try_insert_atomic(packed, fp) {
                    Some(new_packed) => {
                        match self.buckets[bucket_idx][lane].compare_exchange(
                            packed,
                            new_packed,
                            Ordering::Release,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => return true,
                            Err(_) => continue, // retry CAS with fresh load
                        }
                    }
                    None => break, // bucket full, try next
                }
            }
        }
        false
    }

    /// Delete a fingerprint for a hash from the specified lane.
    pub fn delete(&self, hash: u64, lane: usize) -> bool {
        if self.lane_status[lane].load(Ordering::Acquire) != LANE_ACTIVE {
            return false;
        }
        let (fp, bucket_a) = cuckoo_filter::probe(hash);
        let bucket_b = cuckoo_filter::alt_index(bucket_a, fp);

        for &bucket_idx in &[bucket_a, bucket_b] {
            loop {
                let packed = self.buckets[bucket_idx][lane].load(Ordering::Acquire);
                match try_delete_atomic(packed, fp) {
                    Some(new_packed) => {
                        match self.buckets[bucket_idx][lane].compare_exchange(
                            packed,
                            new_packed,
                            Ordering::Release,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => return true,
                            Err(_) => continue,
                        }
                    }
                    None => break,
                }
            }
        }
        false
    }

    /// Estimate the prefix overlap length for a single backend's lane.
    ///
    /// Walks the hash sequence, probing each hash; stops at the first miss
    /// (prefix break).
    pub fn estimate_overlap(&self, hashes: &[u64], lane: usize) -> u32 {
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

    /// Batch query: estimate the prefix overlap length for **all active lanes**
    /// in a single pass.
    ///
    /// Returns a `Vec<(lane, overlap)>` for all active lanes. The transposed
    /// layout ensures that `buckets[bucket_idx]` (all lanes for one bucket
    /// index) is in the same cache line, so probing all lanes for a given hash
    /// is cache-friendly.
    ///
    /// This is the primary query path for KV-aware routing: it replaces N
    /// individual RadixTree `find_matches` calls (each with a channel
    /// round-trip) with a single lock-free scan.
    pub fn estimate_all_overlaps(&self, hashes: &[u64]) -> Vec<(usize, u32)> {
        // Collect active lanes upfront
        let active_lanes: Vec<usize> = (0..LOCAL_LANE_COUNT)
            .filter(|&l| self.lane_status[l].load(Ordering::Acquire) == LANE_ACTIVE)
            .collect();
        if active_lanes.is_empty() {
            return Vec::new();
        }

        let mut overlaps: Vec<u32> = vec![0u32; LOCAL_LANE_COUNT];
        for &hash in hashes {
            let (fp, bucket_idx) = cuckoo_filter::probe(hash);
            // Load the entire bucket row (all lanes for this bucket index) —
            // this is a single cache line in the transposed layout.
            let bucket_row = &self.buckets[bucket_idx];
            let mut any_hit = false;
            for &lane in &active_lanes {
                let packed = bucket_row[lane].load(Ordering::Acquire);
                if cuckoo_filter::bucket_contains(packed, fp) {
                    overlaps[lane] += 1;
                    any_hit = true;
                }
            }
            // Prefix break: if no lane hit this hash, all lanes stop here.
            // We continue only if at least one lane hit (prefix semantics
            // are per-lane, but if no lane has it, further hashes can't match
            // any lane's prefix either).
            if !any_hit {
                break;
            }
        }
        active_lanes
            .into_iter()
            .map(|lane| (lane, overlaps[lane]))
            .collect()
    }

    /// Clear all fingerprints for a lane (used when a backend goes offline).
    pub fn clear_lane(&self, lane: usize) {
        self.lane_status[lane].store(LANE_RETIRED, Ordering::Release);
        for i in 0..BUCKETS_PER_LANE {
            self.buckets[i][lane].store(0, Ordering::Relaxed);
        }
        self.lane_status[lane].store(LANE_ACTIVE, Ordering::Release);
    }

    /// Number of currently assigned lanes.
    pub fn assigned_count(&self) -> usize {
        self.lane_backends.read().len()
    }
}

/// Try to insert a fingerprint into a packed bucket value (non-atomic helper).
/// Returns the new packed value if the fingerprint was inserted, `None` if full.
#[inline]
fn try_insert_atomic(packed: u64, fp: cuckoo_filter::Fp) -> Option<u64> {
    let mut bucket = packed;
    if cuckoo_filter::try_insert(&mut bucket, fp) {
        Some(bucket)
    } else {
        None
    }
}

/// Try to delete a fingerprint from a packed bucket value (non-atomic helper).
#[inline]
fn try_delete_atomic(packed: u64, fp: cuckoo_filter::Fp) -> Option<u64> {
    let mut bucket = packed;
    if cuckoo_filter::try_delete(&mut bucket, fp) {
        Some(bucket)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::ids::BackendId;

    fn backend(n: u8) -> BackendId {
        BackendId::new(format!("r{n}"), format!("i{n}"))
    }

    #[test]
    fn assign_and_query_overlap() {
        let ckf = LocalCkf::new();
        let b = backend(1);
        let lane = ckf.assign_lane(&b).expect("lane should be assigned");

        // Insert hashes 1..5
        for h in 1..=5u64 {
            assert!(ckf.insert(h, lane), "insert hash {h} should succeed");
        }

        // Query: overlap should be 5 for hashes 1..5, then break at 6
        let overlap = ckf.estimate_overlap(&[1, 2, 3, 4, 5, 6], lane);
        assert_eq!(overlap, 5);

        // Batch query
        let results = ckf.estimate_all_overlaps(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, lane);
        assert_eq!(results[0].1, 5);
    }

    #[test]
    fn delete_reduces_overlap() {
        let ckf = LocalCkf::new();
        let b = backend(1);
        let lane = ckf.assign_lane(&b).unwrap();

        for h in 1..=5u64 {
            ckf.insert(h, lane);
        }
        // Delete hash 3
        assert!(ckf.delete(3, lane));

        // Overlap for [1,2,3,4,5] should be 2 (breaks at 3)
        let overlap = ckf.estimate_overlap(&[1, 2, 3, 4, 5], lane);
        assert_eq!(overlap, 2);
    }

    #[test]
    fn multiple_lanes_batch_query() {
        let ckf = LocalCkf::new();
        let b1 = backend(1);
        let b2 = backend(2);
        let l1 = ckf.assign_lane(&b1).unwrap();
        let l2 = ckf.assign_lane(&b2).unwrap();

        // b1 has hashes 1,2,3
        for h in 1..=3u64 {
            ckf.insert(h, l1);
        }
        // b2 has hashes 1,2,3,4,5
        for h in 1..=5u64 {
            ckf.insert(h, l2);
        }

        let results = ckf.estimate_all_overlaps(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(results.len(), 2);

        let mut by_lane: HashMap<usize, u32> = results.into_iter().collect();
        assert_eq!(by_lane.remove(&l1), Some(3));
        assert_eq!(by_lane.remove(&l2), Some(5));
    }

    #[test]
    fn unassign_clears_lane() {
        let ckf = LocalCkf::new();
        let b = backend(1);
        let lane = ckf.assign_lane(&b).unwrap();
        for h in 1..=5u64 {
            ckf.insert(h, lane);
        }
        ckf.unassign_lane(&b);

        let results = ckf.estimate_all_overlaps(&[1, 2, 3]);
        assert!(results.is_empty(), "no active lanes after unassign");
    }

    #[test]
    fn assign_returns_existing_lane() {
        let ckf = LocalCkf::new();
        let b = backend(1);
        let lane1 = ckf.assign_lane(&b).unwrap();
        let lane2 = ckf.assign_lane(&b).unwrap();
        assert_eq!(lane1, lane2, "assigning twice should return the same lane");
    }
}
