//! KV 索引统一接口：组合本地精确 RadixTree 与跨 Region 近似 CkfConsumer。
//!
//! - 本地查询走 [`RadixTree`]，结果精确，但仅覆盖本 Region 的 backend。
//! - 跨 Region 查询走 [`CkfConsumer`]，结果近似（受 cuckoo filter FPR 影响），
//!   但可同时覆盖多个 Region。
//! - [`KvIndex::kv_confidence`] 返回近似查询的可信度（1 - FPR）。

use aether_core::error::Result;
use aether_core::ids::{BackendId, RegionId};
use aether_core::kv_event::KvCacheEvent;

use crate::ckf_consumer::CkfConsumer;
use crate::cuckoo_filter::{FINGERPRINT_BITS, FP_PER_BUCKET};
use crate::radix_tree::RadixTree;

/// KV 索引统一入口。
pub struct KvIndex {
    /// 本地精确索引。
    radix: RadixTree,
    /// 跨 Region 近似索引。
    consumer: CkfConsumer,
}

impl KvIndex {
    /// 创建一个新的 KV 索引，内部启动 RadixTree 后台线程。
    pub fn new() -> Self {
        Self {
            radix: RadixTree::new(),
            consumer: CkfConsumer::new(),
        }
    }

    /// 共享 RadixTree 句柄（用于直接调用 stats / remove_backend 等）。
    pub fn radix(&self) -> &RadixTree {
        &self.radix
    }

    /// 共享 CkfConsumer 句柄（用于 lane 管理）。
    pub fn consumer(&self) -> &CkfConsumer {
        &self.consumer
    }

    /// 本地精确查询：返回指定 backend 对 hash 序列的前缀重叠长度。
    pub async fn kv_find_local_overlap(
        &self,
        hashes: &[u64],
        backend: BackendId,
    ) -> u32 {
        self.radix.find_matches(hashes.to_vec(), backend).await
    }

    /// 跨 Region 近似查询：返回指定 Region 对 hash 序列的前缀重叠长度。
    pub fn kv_find_global_overlap(&self, hashes: &[u64], region: &RegionId) -> u32 {
        self.consumer.estimate_overlap(hashes, region)
    }

    /// 应用一个 KV cache 事件到本地索引。
    pub async fn kv_apply_event(
        &self,
        event: KvCacheEvent,
        backend: BackendId,
    ) -> Result<()> {
        self.radix.apply_event(backend, event).await
    }

    /// 当前近似查询的可信度：`1 - FPR`。
    ///
    /// Cuckoo filter 理论 FPR ≈ `(2 * b * ln2) / 2^f`，其中 b = `FP_PER_BUCKET`，
    /// f = `FINGERPRINT_BITS`。本实现参数下 FPR ≈ 8.5e-5。
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
    use aether_core::ids::WorkerWithRank;

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
