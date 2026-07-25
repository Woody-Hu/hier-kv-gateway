//! Cuckoo Filter 基础原语（参考 Dynamo CKF）。
//!
//! 该模块只提供无状态的桶操作与寻址函数，以及快照/增量数据结构。
//! 上层的 [`crate::ckf_producer::CkfProducer`] 与 [`crate::ckf_consumer::CkfConsumer`]
//! 负责组合这些原语实现具体的写入与查询逻辑。
//!
//! 设计要点：
//! - 指纹位数 = 16，每个 u64 桶装 4 个指纹（共 64 bit）。
//! - 使用 partial-key cuckoo 寻址：`idx2 = idx1 ^ hash(fp)`，避免存储完整键。
//! - 每个 lane 容量固定为 `BUCKETS_PER_LANE`（65536），是 2 的幂便于位掩码寻址。

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

/// 指纹位数。
pub const FINGERPRINT_BITS: usize = 16;
/// 每个 bucket 中的指纹数量。
pub const FP_PER_BUCKET: usize = 4;
/// 单次插入最大踢出次数。
pub const MAX_KICKS: usize = 500;
/// 单个 lane 的 bucket 数量（必须为 2 的幂）。
pub const BUCKETS_PER_LANE: usize = 65536;

/// bucket 数量对应的位掩码。
pub const BUCKET_MASK: usize = BUCKETS_PER_LANE - 1;

/// 指纹类型。0 保留为“空槽”哨兵。
pub type Fp = u16;
/// 打包后的 bucket：4 个 16-bit 指纹存于一个 u64 中（slot 0 在低位）。
pub type PackedBucket = u64;

/// 用于 alt_index 计算的混合常量，避免与主索引哈希撞设计。
const ALT_MIX_DOMAIN: u64 = 0x9E37_79B9_7F4A_7C15;

/// 计算指定指纹在 lane 内的备选 bucket 索引。
///
/// 使用 partial-key cuckoo hashing：`idx2 = idx1 ^ hash(fp)`。
/// 当 `delta == 0` 时强制返回 `idx ^ 1`，避免两个候选 bucket 重合。
#[inline]
pub fn alt_index(idx: usize, fp: Fp) -> usize {
    let mixed = xxh3::xxh3_64_with_seed(&fp.to_le_bytes(), ALT_MIX_DOMAIN);
    let delta = (mixed as usize) & BUCKET_MASK;
    let delta = if delta == 0 { 1 } else { delta };
    (idx ^ delta) & BUCKET_MASK
}

/// 计算指定 block hash 的指纹与主 bucket 索引。
#[inline]
pub fn probe(hash: u64) -> (Fp, usize) {
    let mixed = xxh3::xxh3_64_with_seed(&hash.to_le_bytes(), 0);
    let fp = (mixed as u16) | 1; // 0 保留为空槽，最低位置 1 避免生成 0
    let bucket = ((mixed >> 16) as usize) & BUCKET_MASK;
    (fp, bucket)
}

/// 读取 bucket 中指定槽位的指纹。
#[inline]
pub fn slot(bucket: PackedBucket, slot_idx: usize) -> Fp {
    debug_assert!(slot_idx < FP_PER_BUCKET);
    (bucket >> (slot_idx * FINGERPRINT_BITS)) as Fp
}

/// 将指定槽位写入指纹，返回新的 bucket 值。
#[inline]
pub fn with_slot(bucket: PackedBucket, slot_idx: usize, fp: Fp) -> PackedBucket {
    debug_assert!(slot_idx < FP_PER_BUCKET);
    let shift = slot_idx * FINGERPRINT_BITS;
    let mask = (u64::from(u16::MAX)) << shift;
    (bucket & !mask) | (u64::from(fp) << shift)
}

/// 判断 bucket 中是否包含指定指纹。
#[inline]
pub fn bucket_contains(bucket: PackedBucket, fp: Fp) -> bool {
    debug_assert_ne!(fp, 0);
    let repeated = u64::from(fp) * 0x0001_0001_0001_0001;
    let different = bucket ^ repeated;
    let high_bits = 0x8000_8000_8000_8000;
    different.wrapping_sub(0x0001_0001_0001_0001) & !different & high_bits != 0
}

/// 查找 bucket 中第一个等于 `fp` 的槽位。
#[inline]
pub fn first_match(bucket: PackedBucket, fp: Fp) -> Option<usize> {
    (0..FP_PER_BUCKET).find(|&i| slot(bucket, i) == fp)
}

/// 查找 bucket 中第一个空槽（指纹为 0）。
#[inline]
pub fn first_empty(bucket: PackedBucket) -> Option<usize> {
    (0..FP_PER_BUCKET).find(|&i| slot(bucket, i) == 0)
}

/// 尝试将指纹插入 bucket（找空槽写入）。成功返回 `true`，bucket 满返回 `false`。
#[inline]
pub fn try_insert(bucket: &mut PackedBucket, fp: Fp) -> bool {
    debug_assert_ne!(fp, 0);
    if let Some(i) = first_empty(*bucket) {
        *bucket = with_slot(*bucket, i, fp);
        return true;
    }
    false
}

/// 尝试从 bucket 删除指定指纹（找匹配槽位置 0）。成功返回 `true`，未找到返回 `false`。
#[inline]
pub fn try_delete(bucket: &mut PackedBucket, fp: Fp) -> bool {
    debug_assert_ne!(fp, 0);
    if let Some(i) = first_match(*bucket, fp) {
        *bucket = with_slot(*bucket, i, 0);
        return true;
    }
    false
}

/// CKF 全量快照：包含 lane 内所有 bucket 的当前值与单调递增的序列号。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CkfSnapshot {
    /// 单调递增的快照序列号。
    pub sequence: u64,
    /// lane 内所有 bucket 的值，按 bucket 索引排列。
    pub buckets: Vec<PackedBucket>,
}

/// CKF 增量：仅包含自上一个序列号以来变动的 bucket。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CkfDelta {
    /// 本增量对应的序列号。
    pub sequence: u64,
    /// 本增量基于的前一个序列号。
    pub prev_sequence: u64,
    /// 变动的 bucket 列表：(bucket_idx, new_value)。
    pub buckets: Vec<(usize, PackedBucket)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alt_index_is_involution() {
        let fp = 0x1234u16;
        for idx in [0usize, 1, 100, 65535] {
            let alt = alt_index(idx, fp);
            assert_eq!(alt_index(alt, fp), idx, "alt_index must be an involution");
            assert_ne!(alt, idx, "alt_index must differ from idx");
        }
    }

    #[test]
    fn try_insert_and_contains() {
        let mut bucket = 0u64;
        assert!(try_insert(&mut bucket, 0x1111));
        assert!(try_insert(&mut bucket, 0x2222));
        assert!(bucket_contains(bucket, 0x1111));
        assert!(bucket_contains(bucket, 0x2222));
        assert!(!bucket_contains(bucket, 0x3333));
    }

    #[test]
    fn try_insert_fails_when_full() {
        let mut bucket = 0u64;
        assert!(try_insert(&mut bucket, 1));
        assert!(try_insert(&mut bucket, 2));
        assert!(try_insert(&mut bucket, 3));
        assert!(try_insert(&mut bucket, 4));
        assert!(!try_insert(&mut bucket, 5));
    }

    #[test]
    fn try_delete_removes_fingerprint() {
        let mut bucket = 0u64;
        assert!(try_insert(&mut bucket, 0xabcd));
        assert!(try_delete(&mut bucket, 0xabcd));
        assert!(!bucket_contains(bucket, 0xabcd));
        assert!(!try_delete(&mut bucket, 0xabcd));
    }
}
