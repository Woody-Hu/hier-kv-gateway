//! Unified KV index interface: combines the local exact RadixTree, a local
//! per-backend CKF projection, and the cross-Region approximate CkfConsumer.
//!
//! - **Write path**: KV events are applied to both the RadixTree (exact refcount
//!   tracking) and the [`LocalCkf`] (per-backend cuckoo filter projection). The
//!   RadixTree remains the source of truth for refcount; LocalCkf serves queries.
//! - **Query path (local)**: [`LocalCkf::estimate_all_overlaps`] performs a
//!   single cache-friendly scan across all backend lanes — no channel
//!   round-trip needed (unlike `RadixTree::find_all_matches`).
//! - **Query path (cross-Region)**: [`CkfConsumer::estimate_overlap`] provides
//!   approximate overlap per Region.
//! - [`KvIndex::kv_confidence`] returns the confidence of approximate queries (1 - FPR).

use std::collections::HashMap;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::{BackendId, RegionId};
use hier_kv_gateway_core::kv_event::KvCacheEvent;

use crate::ckf_consumer::CkfConsumer;
use crate::cuckoo_filter::{FINGERPRINT_BITS, FP_PER_BUCKET};
use crate::local_ckf::LocalCkf;
use crate::radix_tree::RadixTree;

/// Unified KV index entry point.
pub struct KvIndex {
    /// Local exact index (source of truth for refcount / ownership).
    radix: RadixTree,
    /// Local per-backend CKF projection (serves cache-friendly batch queries).
    local_ckf: LocalCkf,
    /// Cross-Region approximate index.
    consumer: CkfConsumer,
}

impl KvIndex {
    /// Create a new KV index; internally starts the RadixTree background thread.
    pub fn new() -> Self {
        Self {
            radix: RadixTree::new(),
            local_ckf: LocalCkf::new(),
            consumer: CkfConsumer::new(),
        }
    }

    /// Shared RadixTree handle (for direct calls such as stats / remove_backend).
    pub fn radix(&self) -> &RadixTree {
        &self.radix
    }

    /// Shared LocalCkf handle (for lane management and direct queries).
    pub fn local_ckf(&self) -> &LocalCkf {
        &self.local_ckf
    }

    /// Shared CkfConsumer handle (for lane management).
    pub fn consumer(&self) -> &CkfConsumer {
        &self.consumer
    }

    /// Local exact query: returns the prefix overlap length of the specified
    /// backend for the hash sequence.
    ///
    /// Uses the LocalCkf projection for cache-friendly, lock-free access.
    /// Falls back to RadixTree exact query if the backend has no lane assigned.
    pub async fn kv_find_local_overlap(
        &self,
        hashes: &[u64],
        backend: BackendId,
    ) -> u32 {
        if let Some(lane) = self.local_ckf.lane_of(&backend) {
            return self.local_ckf.estimate_overlap(hashes, lane);
        }
        // Fallback: RadixTree exact query (backend not yet assigned a lane)
        self.radix.find_matches(hashes.to_vec(), backend).await
    }

    /// Local batched query: returns the prefix overlap length of *all* backends
    /// for the hash sequence in a single pass.
    ///
    /// This is the preferred query path for KV-aware routing. It uses
    /// [`LocalCkf::estimate_all_overlaps`] which scans all active lanes in one
    /// cache-friendly pass over the transposed bucket layout — no channel
    /// round-trip to the RadixTree worker thread.
    ///
    /// The returned `HashMap` maps `BackendId → overlap` for all backends
    /// that have an assigned lane.
    pub fn kv_find_all_local_overlap(
        &self,
        hashes: &[u64],
    ) -> HashMap<BackendId, u32> {
        let lane_results = self.local_ckf.estimate_all_overlaps(hashes);
        let mut out = HashMap::with_capacity(lane_results.len());
        for (lane, overlap) in lane_results {
            if let Some(backend) = self.local_ckf.backend_of_lane(lane) {
                out.insert(backend, overlap);
            }
        }
        out
    }

    /// Cross-Region approximate query: returns the prefix overlap length of the
    /// specified Region for the hash sequence.
    pub fn kv_find_global_overlap(&self, hashes: &[u64], region: &RegionId) -> u32 {
        self.consumer.estimate_overlap(hashes, region)
    }

    /// Apply a KV cache event to the local index.
    ///
    /// Writes to both the RadixTree (exact refcount) and the LocalCkf projection.
    /// The RadixTree write is async (channel-based); the LocalCkf write is
    /// synchronous (CAS on atomic buckets). The backend is auto-assigned a lane
    /// on first event.
    pub async fn kv_apply_event(
        &self,
        event: KvCacheEvent,
        backend: BackendId,
    ) -> Result<()> {
        // Ensure the backend has a lane assigned before writing to LocalCkf.
        let lane = self.local_ckf.assign_lane(&backend);

        // Write to RadixTree first (source of truth).
        let event_clone = event.clone();
        self.radix.apply_event(backend.clone(), event_clone).await?;

        // Project to LocalCkf.
        if let Some(lane) = lane {
            match &event {
                KvCacheEvent::Stored { block_hashes, .. } => {
                    for &hash in block_hashes {
                        self.local_ckf.insert(hash, lane);
                    }
                }
                KvCacheEvent::Removed { block_hashes, .. } => {
                    for &hash in block_hashes {
                        self.local_ckf.delete(hash, lane);
                    }
                }
                KvCacheEvent::Clear { .. } | KvCacheEvent::Reset { .. } => {
                    self.local_ckf.clear_lane(lane);
                }
            }
        }

        Ok(())
    }

    /// Remove a backend from the local index (RadixTree + LocalCkf).
    pub async fn kv_remove_backend(&self, backend: BackendId) {
        // Clear LocalCkf lane
        self.local_ckf.unassign_lane(&backend);
        // Clear RadixTree ownership
        self.radix.remove_backend(backend).await;
    }

    /// Confidence of the current approximate queries: `1 - FPR`.
    ///
    /// The cuckoo filter theoretical FPR ≈ `(2 * b * ln2) / 2^f`, where
    /// b = `FP_PER_BUCKET` and f = `FINGERPRINT_BITS`. With this implementation's
    /// parameters the FPR ≈ 8.5e-5.
    pub fn kv_confidence(&self) -> f64 {
        let b = FP_PER_BUCKET as f64;
        let f = FINGERPRINT_BITS as f64;
        let fpr = (2.0 * b * std::f64::consts::LN_2) / (2f64.powf(f));
        (1.0 - fpr).clamp(0.0, 1.0)
    }
}

impl Default for KvIndex {
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

    #[tokio::test]
    async fn local_overlap_via_ckf() {
        let idx = KvIndex::new();
        let b = backend(1);
        idx.kv_apply_event(stored(vec![1, 2, 3]), b.clone())
            .await
            .unwrap();
        let overlap = idx.kv_find_local_overlap(&[1, 2, 3, 4], b).await;
        assert_eq!(overlap, 3);
    }

    #[tokio::test]
    async fn batched_query_via_ckf() {
        let idx = KvIndex::new();
        let b1 = backend(1);
        let b2 = backend(2);
        idx.kv_apply_event(stored(vec![1, 2, 3]), b1.clone())
            .await
            .unwrap();
        idx.kv_apply_event(stored(vec![1, 2, 3, 4, 5]), b2.clone())
            .await
            .unwrap();

        let results = idx.kv_find_all_local_overlap(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(results.len(), 2);
        assert_eq!(results.get(&b1), Some(&3));
        assert_eq!(results.get(&b2), Some(&5));
    }

    #[tokio::test]
    async fn removed_decreases_overlap() {
        let idx = KvIndex::new();
        let b = backend(1);
        idx.kv_apply_event(stored(vec![1, 2, 3]), b.clone())
            .await
            .unwrap();
        idx.kv_apply_event(removed(vec![3]), b.clone())
            .await
            .unwrap();
        let overlap = idx.kv_find_local_overlap(&[1, 2, 3], b).await;
        assert_eq!(overlap, 2);
    }

    #[tokio::test]
    async fn remove_backend_clears_lane() {
        let idx = KvIndex::new();
        let b = backend(1);
        idx.kv_apply_event(stored(vec![1, 2, 3]), b.clone())
            .await
            .unwrap();
        idx.kv_remove_backend(b.clone()).await;
        let results = idx.kv_find_all_local_overlap(&[1, 2, 3]);
        assert!(results.is_empty());
    }

    #[test]
    fn confidence_is_close_to_one() {
        let idx = KvIndex::new();
        let c = idx.kv_confidence();
        assert!(c > 0.99, "confidence should be > 0.99, got {}", c);
    }
}
