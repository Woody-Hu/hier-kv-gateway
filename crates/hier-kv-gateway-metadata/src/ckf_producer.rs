//! CKF Producer: one local Cuckoo Filter producer per pool.
//!
//! The producer maintains a lane of cuckoo buckets locally along with "refcount
//! per hash + set of all workers" tracking, to support precise deduplication
//! when multiple workers share the same block. After events are applied,
//! changes can be published to remote consumers as either a [`CkfSnapshot`] or
//! [`CkfDelta`].
//!
//! Ownership rules:
//! - `Stored`: first owner → insert fingerprint; subsequent owners → refcount++ only.
//! - `Removed`: one of multiple owners → refcount-- only; last owner → delete fingerprint.
//! - `Clear`: clear all hashes for that worker (only those held by this worker;
//!   other workers are unaffected).
//! - `Reset`: generation fence; clear the entire producer state.

use std::collections::{HashMap, HashSet};

use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use tracing::warn;

use crate::cuckoo_filter::{
    self, CkfDelta, CkfSnapshot, PackedBucket, BUCKETS_PER_LANE, MAX_KICKS,
};

/// Default RNG seed used by the CKF producer.
const CKF_PRODUCER_DEFAULT_SEED: u64 = 0xA7C5_4B7E_38A1_2F5D;

/// Internal hash tracking entry: holds the reference count and the set of all
/// backends that hold this hash.
#[derive(Debug, Default, Clone)]
struct HashEntry {
    refcount: u32,
    owners: HashSet<BackendId>,
}

/// CKF producer state.
///
/// All mutating methods require `&mut self`; the Rust borrow rules guarantee
/// write mutual exclusion within a single thread. For cross-task sharing the
/// upper-layer caller is responsible for serialization (the producer is
/// typically driven by a single ingestion task).
pub struct CkfProducer {
    /// Current values of all buckets in the lane.
    buckets: Vec<PackedBucket>,
    /// Number of fingerprints inserted (counting multi-owner duplicates).
    num_items: u64,
    /// Set of bucket indices changed since the last snapshot/delta.
    dirty_buckets: HashSet<usize>,
    /// Largest sequence number published.
    pub_seq: u64,
    /// hash → (refcount, owners) tracking table.
    hash_refcount: HashMap<u64, HashEntry>,
    /// backend → set of all hashes held by that backend, for fast Clear cleanup.
    worker_hashes: HashMap<BackendId, HashSet<u64>>,
    /// Deterministic RNG state used during cuckoo eviction (splitmix64).
    rng_state: u64,
}

impl std::fmt::Debug for CkfProducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CkfProducer")
            .field("num_items", &self.num_items)
            .field("pub_seq", &self.pub_seq)
            .field("dirty_buckets", &self.dirty_buckets.len())
            .field("tracked_hashes", &self.hash_refcount.len())
            .finish()
    }
}

impl CkfProducer {
    /// Create an empty producer.
    pub fn new() -> Self {
        Self::with_seed(CKF_PRODUCER_DEFAULT_SEED)
    }

    /// Create a producer with a custom RNG seed (mainly for testing).
    pub fn with_seed(seed: u64) -> Self {
        Self {
            buckets: vec![0u64; BUCKETS_PER_LANE],
            num_items: 0,
            dirty_buckets: HashSet::new(),
            pub_seq: 0,
            hash_refcount: HashMap::new(),
            worker_hashes: HashMap::new(),
            rng_state: seed,
        }
    }

    /// Apply a KV cache event, updating this producer's ownership and fingerprint state.
    ///
    /// Ownership of the event is determined by the `backend` passed in by the
    /// caller (the event's own `worker` field is used for logging/auditing; this
    /// producer deduplicates by backend).
    pub fn apply_event(&mut self, event: &KvCacheEvent, backend: &BackendId) {
        match event {
            KvCacheEvent::Stored { block_hashes, .. } => {
                for &hash in block_hashes {
                    self.apply_stored(hash, backend.clone());
                }
            }
            KvCacheEvent::Removed { block_hashes, .. } => {
                for &hash in block_hashes {
                    self.apply_removed(hash, backend.clone());
                }
            }
            KvCacheEvent::Clear { .. } => {
                self.apply_clear(backend.clone());
            }
            KvCacheEvent::Reset { .. } => {
                warn!("CkfProducer received Reset, clearing all state");
                self.reset_internal();
            }
        }
    }

    fn apply_stored(&mut self, hash: u64, worker: BackendId) {
        let entry = self.hash_refcount.entry(hash).or_default();
        let first_owner = entry.owners.is_empty();
        let inserted = entry.owners.insert(worker.clone());
        if !inserted {
            // Already held by this worker; dedup: do not increment refcount or insert another fingerprint.
            return;
        }
        entry.refcount += 1;
        self.worker_hashes
            .entry(worker)
            .or_default()
            .insert(hash);
        if first_owner {
            // First owner: insert fingerprint
            if !self.insert_fingerprint(hash) {
                warn!(hash, "CkfProducer insert_fingerprint failed (lane full?)");
            }
        }
    }

    fn apply_removed(&mut self, hash: u64, worker: BackendId) {
        let Some(entry) = self.hash_refcount.get_mut(&hash) else {
            return;
        };
        if !entry.owners.remove(&worker) {
            return;
        }
        if let Some(s) = self.worker_hashes.get_mut(&worker) {
            s.remove(&hash);
        }
        entry.refcount = entry.refcount.saturating_sub(1);
        if entry.refcount == 0 {
            // Last owner: delete fingerprint
            self.delete_fingerprint(hash);
            self.hash_refcount.remove(&hash);
        }
    }

    fn apply_clear(&mut self, worker: BackendId) {
        let hashes = self.worker_hashes.remove(&worker).unwrap_or_default();
        for hash in hashes {
            let Some(entry) = self.hash_refcount.get_mut(&hash) else {
                continue;
            };
            entry.owners.remove(&worker);
            entry.refcount = entry.refcount.saturating_sub(1);
            if entry.refcount == 0 {
                self.delete_fingerprint(hash);
                self.hash_refcount.remove(&hash);
            }
        }
    }

    fn reset_internal(&mut self) {
        for b in self.buckets.iter_mut() {
            *b = 0;
        }
        self.num_items = 0;
        self.dirty_buckets.clear();
        self.hash_refcount.clear();
        self.worker_hashes.clear();
    }

    /// Perform a cuckoo insertion (with eviction). Returns `true` on success;
    /// returns `false` and rolls back all evictions if MAX_KICKS is reached without success.
    fn insert_fingerprint(&mut self, hash: u64) -> bool {
        let (fp, bucket_a) = cuckoo_filter::probe(hash);
        let bucket_b = cuckoo_filter::alt_index(bucket_a, fp);

        // Try inserting directly into the two candidate buckets first.
        if cuckoo_filter::try_insert(&mut self.buckets[bucket_a], fp) {
            self.dirty_buckets.insert(bucket_a);
            self.num_items += 1;
            return true;
        }
        if cuckoo_filter::try_insert(&mut self.buckets[bucket_b], fp) {
            self.dirty_buckets.insert(bucket_b);
            self.num_items += 1;
            return true;
        }

        // Enter the eviction loop: pick a random starting point and evict fingerprints one by one.
        let mut touched: Vec<(usize, PackedBucket)> = Vec::with_capacity(MAX_KICKS + 1);
        let mut current_bucket = if self.next_random() & 1 == 0 {
            bucket_a
        } else {
            bucket_b
        };
        let mut current_fp = fp;

        for _ in 0..MAX_KICKS {
            let before = self.buckets[current_bucket];
            let slot_idx = (self.next_random() as usize) & 0x3;
            let evicted = cuckoo_filter::slot(before, slot_idx);
            let after = cuckoo_filter::with_slot(before, slot_idx, current_fp);
            self.buckets[current_bucket] = after;
            touched.push((current_bucket, before));
            current_fp = evicted;
            current_bucket = cuckoo_filter::alt_index(current_bucket, current_fp);
            if cuckoo_filter::try_insert(&mut self.buckets[current_bucket], current_fp) {
                self.dirty_buckets.insert(current_bucket);
                for &(idx, _) in &touched {
                    self.dirty_buckets.insert(idx);
                }
                self.num_items += 1;
                return true;
            }
        }

        // Roll back the evictions
        for &(idx, before) in touched.iter().rev() {
            self.buckets[idx] = before;
        }
        false
    }

    /// Delete the fingerprint for the specified hash.
    fn delete_fingerprint(&mut self, hash: u64) {
        let (fp, bucket_a) = cuckoo_filter::probe(hash);
        let bucket_b = cuckoo_filter::alt_index(bucket_a, fp);
        if cuckoo_filter::try_delete(&mut self.buckets[bucket_a], fp) {
            self.dirty_buckets.insert(bucket_a);
            self.num_items = self.num_items.saturating_sub(1);
        } else if cuckoo_filter::try_delete(&mut self.buckets[bucket_b], fp) {
            self.dirty_buckets.insert(bucket_b);
            self.num_items = self.num_items.saturating_sub(1);
        }
    }

    /// Generate a full snapshot (also clears the dirty set).
    pub fn snapshot(&mut self) -> CkfSnapshot {
        self.pub_seq = self.pub_seq.wrapping_add(1);
        self.dirty_buckets.clear();
        CkfSnapshot {
            sequence: self.pub_seq,
            buckets: self.buckets.clone(),
        }
    }

    /// Generate a delta (containing only dirty buckets); returns `None` if there are no changes.
    pub fn delta(&mut self) -> Option<CkfDelta> {
        if self.dirty_buckets.is_empty() {
            return None;
        }
        let prev = self.pub_seq;
        self.pub_seq = self.pub_seq.wrapping_add(1);
        let mut buckets: Vec<(usize, PackedBucket)> =
            self.dirty_buckets.iter().copied().map(|idx| (idx, self.buckets[idx])).collect();
        buckets.sort_unstable_by_key(|(idx, _)| *idx);
        self.dirty_buckets.clear();
        Some(CkfDelta {
            sequence: self.pub_seq,
            prev_sequence: prev,
            buckets,
        })
    }

    /// Current number of inserted fingerprints (counting multi-owner duplicates).
    pub fn num_items(&self) -> u64 {
        self.num_items
    }

    /// Current number of distinct tracked hashes.
    pub fn tracked_hashes(&self) -> usize {
        self.hash_refcount.len()
    }

    /// splitmix64-style deterministic PRNG, avoiding an extra dependency.
    fn next_random(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Default for CkfProducer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::ids::WorkerWithRank;

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

    fn removed(hashes: Vec<u64>) -> KvCacheEvent {
        KvCacheEvent::Removed {
            worker: WorkerWithRank::from_worker_id(1),
            block_hashes: hashes,
        }
    }

    #[test]
    fn stored_inserts_fingerprint_once_for_multiple_owners() {
        let mut producer = CkfProducer::with_seed(42);
        let b1 = backend(1);
        let b2 = backend(2);

        // Simulate first owner
        producer.apply_event(&stored(vec![100]), &b1);
        assert_eq!(producer.num_items(), 1);
        assert_eq!(producer.tracked_hashes(), 1);

        // Simulate second owner: should not insert another fingerprint
        producer.apply_event(&stored(vec![100]), &b2);
        assert_eq!(producer.num_items(), 1);
        assert_eq!(producer.tracked_hashes(), 1);
    }

    #[test]
    fn removed_deletes_only_when_last_owner() {
        let mut producer = CkfProducer::with_seed(7);
        let b1 = backend(1);
        let b2 = backend(2);
        producer.apply_event(&stored(vec![100]), &b1);
        producer.apply_event(&stored(vec![100]), &b2);
        // First owner removes: should not delete the fingerprint
        producer.apply_event(&removed(vec![100]), &b1);
        assert_eq!(producer.num_items(), 1);
        // Last owner removes: should delete the fingerprint
        producer.apply_event(&removed(vec![100]), &b2);
        assert_eq!(producer.num_items(), 0);
    }

    #[test]
    fn snapshot_clears_dirty_and_delta_returns_none() {
        let mut producer = CkfProducer::with_seed(11);
        let b = backend(1);
        producer.apply_event(&stored(vec![1, 2, 3]), &b);
        let snap = producer.snapshot();
        assert_eq!(snap.sequence, 1);
        assert!(producer.delta().is_none());
    }
}
