//! CKF Consumer：跨 Region 近似 KV 索引（transposed 布局）。
//!
//! 参考 Dynamo 的 transposed CKF replica：将 16 个 Region lane 的 bucket 数据
//! 按 bucket-major 方式排列（`Vec<[AtomicU64; 16]>`，每个元素是 16 个 lane 的
//! 同一 bucket），使得一次查询可以在同一缓存行内 probe 多个 lane。
//!
//! 查询语义：对给定的 block hash 序列，逐个 hash 查找 fingerprint，命中则
//! overlap++，未命中则提前停止（前缀中断）。lane 的状态（Active/Retired）通过
//! 原子字节维护，安装快照时先 retired 再写桶再 active，避免读到中间状态。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use aether_core::ids::RegionId;
use parking_lot::RwLock;

use crate::cuckoo_filter::{
    self, CkfDelta, CkfSnapshot, BUCKETS_PER_LANE,
};

/// 该 consumer 支持的 lane 数量（即同时跟踪的 Region 数上限）。
pub const LANE_COUNT: usize = 16;

/// lane 状态：0 = Active，1 = Retired。
pub const LANE_ACTIVE: u8 = 0;
pub const LANE_RETIRED: u8 = 1;

/// Transposed CKF 消费者。
pub struct CkfConsumer {
    /// bucket-major 布局：`buckets[i][lane]` 是 bucket i 在 lane 上的 packed 值。
    buckets: Vec<[AtomicU64; LANE_COUNT]>,
    /// lane 索引 → 该 lane 当前对应的 RegionId。
    lane_regions: RwLock<HashMap<usize, RegionId>>,
    /// 每个 lane 的状态：Active 或 Retired。
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
    /// 创建一个空的 consumer。
    pub fn new() -> Self {
        // 初始化固定大小的 bucket 数组（每个数组 [AtomicU64; 16]）。
        let buckets = (0..BUCKETS_PER_LANE)
            .map(|_| {
                let mut arr: [AtomicU64; LANE_COUNT] = [const { AtomicU64::new(0) }; LANE_COUNT];
                // 上面 const 语法要求 nightly 兼容；用循环兜底以兼容稳定 Rust。
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

    /// 估算给定 hash 序列在指定 Region 的 lane 上的前缀重叠长度。
    ///
    /// 遍历每个 hash，计算其 (fp, bucket_idx)，加载该 lane 上的 bucket 值，
    /// 若包含 fp 则 overlap++，否则提前停止。
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

    /// 安装一个 lane 的全量快照。
    ///
    /// 顺序：先 retired（屏蔽读）→ 写所有 bucket → active（恢复读）。
    pub fn install_snapshot(&self, lane: usize, snapshot: &CkfSnapshot) {
        debug_assert!(lane < LANE_COUNT);
        debug_assert_eq!(snapshot.buckets.len(), BUCKETS_PER_LANE);
        self.lane_status[lane].store(LANE_RETIRED, Ordering::Release);
        for (i, &value) in snapshot.buckets.iter().enumerate() {
            self.buckets[i][lane].store(value, Ordering::Relaxed);
        }
        self.lane_status[lane].store(LANE_ACTIVE, Ordering::Release);
    }

    /// 应用一个 lane 的增量更新。
    ///
    /// 增量是弱一致的多 bucket 写入，读端可能观察到部分应用的状态；
    /// 这是 Dynamo CKF 的设计取舍，不在本层引入 seqlock 或重试。
    pub fn apply_delta(&self, lane: usize, delta: &CkfDelta) {
        debug_assert!(lane < LANE_COUNT);
        for &(bucket_idx, value) in &delta.buckets {
            if bucket_idx < self.buckets.len() {
                self.buckets[bucket_idx][lane].store(value, Ordering::Release);
            }
        }
    }

    /// 将指定 lane 标记为 Retired（不再参与查询）。
    pub fn retire_lane(&self, lane: usize) {
        debug_assert!(lane < LANE_COUNT);
        self.lane_status[lane].store(LANE_RETIRED, Ordering::Release);
    }

    /// 将指定 lane 标记为 Active（恢复查询）。
    pub fn activate_lane(&self, lane: usize) {
        debug_assert!(lane < LANE_COUNT);
        self.lane_status[lane].store(LANE_ACTIVE, Ordering::Release);
    }

    /// 将 lane 绑定到指定 Region。
    pub fn assign_lane(&self, lane: usize, region: RegionId) {
        debug_assert!(lane < LANE_COUNT);
        let mut regions = self.lane_regions.write();
        regions.insert(lane, region);
    }

    /// 解绑 lane（用于 Region 迁出）。
    pub fn unassign_lane(&self, lane: usize) {
        debug_assert!(lane < LANE_COUNT);
        let mut regions = self.lane_regions.write();
        regions.remove(&lane);
    }

    /// 查询指定 Region 当前所在的 lane 索引。
    pub fn lane_of(&self, region: &RegionId) -> Option<usize> {
        let regions = self.lane_regions.read();
        regions
            .iter()
            .find_map(|(lane, r)| if r == region { Some(*lane) } else { None })
    }

    /// 获取 lane 当前状态（true 表示 Active）。
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
    use aether_core::ids::{BackendId, WorkerWithRank};
    use aether_core::kv_event::KvCacheEvent;

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
        // 构造一个只含部分 hash 的快照
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

        // hash 100 命中，hash 101 未命中 → 应在 101 处中断
        let overlap = consumer.estimate_overlap(&[100, 101], &region);
        assert_eq!(overlap, 1);
    }
}
