//! Unified KV index interface: combines the local exact RadixTree with the
//! cross-Region approximate CkfConsumer.
//!
//! - Local queries go through [`RadixTree`]; results are exact but only cover
//!   backends in this Region.
//! - Cross-Region queries go through [`CkfConsumer`]; results are approximate
//!   (subject to cuckoo filter FPR) but cover multiple Regions at once.
//! - [`KvIndex::kv_confidence`] returns the confidence of approximate queries (1 - FPR).

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::{BackendId, RegionId};
use hier_kv_gateway_core::kv_event::KvCacheEvent;

use crate::ckf_consumer::CkfConsumer;
use crate::cuckoo_filter::{FINGERPRINT_BITS, FP_PER_BUCKET};
use crate::radix_tree::RadixTree;

/// Unified KV index entry point.
pub struct KvIndex {
    /// Local exact index.
    radix: RadixTree,
    /// Cross-Region approximate index.
    consumer: CkfConsumer,
}

impl KvIndex {
    /// Create a new KV index; internally starts the RadixTree background thread.
    pub fn new() -> Self {
        Self {
            radix: RadixTree::new(),
            consumer: CkfConsumer::new(),
        }
    }

    /// Shared RadixTree handle (for direct calls such as stats / remove_backend).
    pub fn radix(&self) -> &RadixTree {
        &self.radix
    }

    /// Shared CkfConsumer handle (for lane management).
    pub fn consumer(&self) -> &CkfConsumer {
        &self.consumer
    }

    /// Local exact query: returns the prefix overlap length of the specified
    /// backend for the hash sequence.
    pub async fn kv_find_local_overlap(
        &self,
        hashes: &[u64],
        backend: BackendId,
    ) -> u32 {
        self.radix.find_matches(hashes.to_vec(), backend).await
    }

    /// Cross-Region approximate query: returns the prefix overlap length of the
    /// specified Region for the hash sequence.
    pub fn kv_find_global_overlap(&self, hashes: &[u64], region: &RegionId) -> u32 {
        self.consumer.estimate_overlap(hashes, region)
    }

    /// Apply a KV cache event to the local index.
    pub async fn kv_apply_event(
        &self,
        event: KvCacheEvent,
        backend: BackendId,
    ) -> Result<()> {
        self.radix.apply_event(backend, event).await
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

    #[tokio::test]
    async fn local_overlap_via_radix() {
        let idx = KvIndex::new();
        let b = backend(1);
        idx.kv_apply_event(stored(vec![1, 2, 3]), b.clone())
            .await
            .unwrap();
        let overlap = idx.kv_find_local_overlap(&[1, 2, 3, 4], b).await;
        assert_eq!(overlap, 3);
    }

    #[test]
    fn confidence_is_close_to_one() {
        let idx = KvIndex::new();
        let c = idx.kv_confidence();
        assert!(c > 0.99, "confidence should be > 0.99, got {}", c);
    }
}
