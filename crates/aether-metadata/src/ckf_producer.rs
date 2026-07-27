//! CKF Producer：每个 pool 一个的本地 Cuckoo Filter 生产者。
//!
//! 参考 Dynamo 的 ingestion pool 设计：生产者负责在本地维护一个 lane 的 cuckoo
//! 桶与“每个 hash 的引用计数 + 所有 worker 集合”，以支持多 worker 共享同一 block
//! 时的精确去重。在事件应用后，可以将变更以 [`CkfSnapshot`] 或 [`CkfDelta`] 的
//! 形式发布给远端消费者。
//!
//! 所有权规则（参考 Dynamo CKF）：
//! - `Stored`：首个 owner → 插入 fingerprint；后续 owner → 仅 refcount++。
//! - `Removed`：多个 owner 之一 → 仅 refcount--；最后一个 owner → 删除 fingerprint。
//! - `Clear`：清空该 worker 的所有 hash（仅清理该 worker 持有的，不影响其他 worker）。
//! - `Reset`：代际 fence，清空整个 producer 的状态。

use std::collections::{HashMap, HashSet};

use aether_core::ids::BackendId;
use aether_core::kv_event::KvCacheEvent;
use tracing::warn;

use crate::cuckoo_filter::{
    self, CkfDelta, CkfSnapshot, PackedBucket, BUCKETS_PER_LANE, MAX_KICKS,
};

/// CKF 生产者默认使用的 RNG 种子。
const CKF_PRODUCER_DEFAULT_SEED: u64 = 0xA7C5_4B7E_38A1_2F5D;

/// 生产者内部的 hash 跟踪条目：保存引用计数与所有持有该 hash 的 backend 集合。
#[derive(Debug, Default, Clone)]
struct HashEntry {
    refcount: u32,
    owners: HashSet<BackendId>,
}

/// CKF 生产者状态。
///
/// 所有变更方法均要求 `&mut self`，由 Rust 借用规则保证单线程内的写互斥；
/// 跨任务共享时由上层调用方自行串行化（生产者通常由单一 ingestion 任务驱动）。
pub struct CkfProducer {
    /// lane 内所有 bucket 的当前值。
    buckets: Vec<PackedBucket>,
    /// 已插入的 fingerprint 数量（含多 owner 重复计入）。
    num_items: u64,
    /// 自上次 snapshot/delta 以来变动的 bucket 索引集合。
    dirty_buckets: HashSet<usize>,
    /// 已发布的最大序列号。
    pub_seq: u64,
    /// hash → (refcount, owners) 跟踪表。
    hash_refcount: HashMap<u64, HashEntry>,
    /// backend → 该 backend 持有的所有 hash 集合，用于 Clear 时快速清理。
    worker_hashes: HashMap<BackendId, HashSet<u64>>,
    /// cuckoo 踢出过程中使用的确定性 RNG 状态（splitmix64）。
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
    /// 创建一个空的生产者。
    pub fn new() -> Self {
        Self::with_seed(CKF_PRODUCER_DEFAULT_SEED)
    }

    /// 使用自定义 RNG 种子创建生产者（主要用于测试）。
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

    /// 应用一个 KV cache 事件，更新本生产者的所有权与 fingerprint 状态。
    ///
    /// 事件的归属权以调用方传入的 `backend` 为准（事件本身的 `worker` 字段
    /// 用于日志/审计，本生产者按 backend 维度去重）。
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
            // 已被该 worker 持有，去重：不增 refcount 也不再插 fingerprint。
            return;
        }
        entry.refcount += 1;
        self.worker_hashes
            .entry(worker)
            .or_default()
            .insert(hash);
        if first_owner {
            // 首个 owner：插入 fingerprint
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
            // 最后一个 owner：删除 fingerprint
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

    /// 执行 cuckoo 插入（含踢出）。成功返回 `true`，达到 MAX_KICKS 仍未成功
    /// 则回滚所有踢出并返回 `false`。
    fn insert_fingerprint(&mut self, hash: u64) -> bool {
        let (fp, bucket_a) = cuckoo_filter::probe(hash);
        let bucket_b = cuckoo_filter::alt_index(bucket_a, fp);

        // 先尝试直接插入两个候选 bucket。
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

        // 进入踢出循环：随机选一个起点，逐次踢出已有指纹。
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

        // 回滚踢出
        for &(idx, before) in touched.iter().rev() {
            self.buckets[idx] = before;
        }
        false
    }

    /// 删除指定 hash 的 fingerprint。
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

    /// 生成全量快照（同时清空 dirty 集合）。
    pub fn snapshot(&mut self) -> CkfSnapshot {
        self.pub_seq = self.pub_seq.wrapping_add(1);
        self.dirty_buckets.clear();
        CkfSnapshot {
            sequence: self.pub_seq,
            buckets: self.buckets.clone(),
        }
    }

    /// 生成增量（仅含 dirty bucket），如果没有变动返回 `None`。
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

    /// 当前已插入的 fingerprint 数量（含多 owner 重复）。
    pub fn num_items(&self) -> u64 {
        self.num_items
    }

    /// 当前跟踪的 distinct hash 数量。
    pub fn tracked_hashes(&self) -> usize {
        self.hash_refcount.len()
    }

    /// splitmix64 风格的确定性 PRNG，避免引入额外依赖。
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

        // 模拟 first owner
        producer.apply_event(&stored(vec![100]), &b1);
        assert_eq!(producer.num_items(), 1);
        assert_eq!(producer.tracked_hashes(), 1);

        // 模拟 second owner：不应再插 fingerprint
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
        // 第一个 owner 移除：不应删除 fingerprint
        producer.apply_event(&removed(vec![100]), &b1);
        assert_eq!(producer.num_items(), 1);
        // 最后一个 owner 移除：应删除 fingerprint
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
