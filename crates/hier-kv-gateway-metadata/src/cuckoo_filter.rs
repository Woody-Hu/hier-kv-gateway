//! Cuckoo Filter primitives.
//!
//! This module only provides stateless bucket operations and addressing
//! functions, as well as snapshot/delta data structures. The upper-layer
//! [`crate::ckf_producer::CkfProducer`] and [`crate::ckf_consumer::CkfConsumer`]
//! combine these primitives to implement the concrete write and query logic.
//!
//! Design points:
//! - Fingerprint width = 16 bits; each u64 bucket holds 4 fingerprints (64 bits total).
//! - Uses partial-key cuckoo addressing: `idx2 = idx1 ^ hash(fp)`, avoiding storing the full key.
//! - Each lane has a fixed capacity of `BUCKETS_PER_LANE` (65536), a power of two for bitmask addressing.

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

/// Fingerprint width in bits.
pub const FINGERPRINT_BITS: usize = 16;
/// Number of fingerprints per bucket.
pub const FP_PER_BUCKET: usize = 4;
/// Maximum number of evictions per insertion.
pub const MAX_KICKS: usize = 500;
/// Number of buckets per lane (must be a power of two).
pub const BUCKETS_PER_LANE: usize = 65536;

/// Bitmask corresponding to the bucket count.
pub const BUCKET_MASK: usize = BUCKETS_PER_LANE - 1;

/// Fingerprint type. 0 is reserved as the "empty slot" sentinel.
pub type Fp = u16;
/// Packed bucket: 4 16-bit fingerprints stored in one u64 (slot 0 in the low bits).
pub type PackedBucket = u64;

/// Mixing constant used for alt_index computation, to avoid design collisions with the main index hash.
const ALT_MIX_DOMAIN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Compute the alternate bucket index for a given fingerprint within a lane.
///
/// Uses partial-key cuckoo hashing: `idx2 = idx1 ^ hash(fp)`.
/// When `delta == 0` it forces a return of `idx ^ 1` so the two candidate buckets do not collide.
#[inline]
pub fn alt_index(idx: usize, fp: Fp) -> usize {
    let mixed = xxh3::xxh3_64_with_seed(&fp.to_le_bytes(), ALT_MIX_DOMAIN);
    let delta = (mixed as usize) & BUCKET_MASK;
    let delta = if delta == 0 { 1 } else { delta };
    (idx ^ delta) & BUCKET_MASK
}

/// Compute the fingerprint and primary bucket index for a given block hash.
#[inline]
pub fn probe(hash: u64) -> (Fp, usize) {
    let mixed = xxh3::xxh3_64_with_seed(&hash.to_le_bytes(), 0);
    let fp = (mixed as u16) | 1; // 0 is reserved for empty slots; set the lowest bit to avoid generating 0
    let bucket = ((mixed >> 16) as usize) & BUCKET_MASK;
    (fp, bucket)
}

/// Read the fingerprint in the specified slot of the bucket.
#[inline]
pub fn slot(bucket: PackedBucket, slot_idx: usize) -> Fp {
    debug_assert!(slot_idx < FP_PER_BUCKET);
    (bucket >> (slot_idx * FINGERPRINT_BITS)) as Fp
}

/// Write the fingerprint into the specified slot and return the new bucket value.
#[inline]
pub fn with_slot(bucket: PackedBucket, slot_idx: usize, fp: Fp) -> PackedBucket {
    debug_assert!(slot_idx < FP_PER_BUCKET);
    let shift = slot_idx * FINGERPRINT_BITS;
    let mask = (u64::from(u16::MAX)) << shift;
    (bucket & !mask) | (u64::from(fp) << shift)
}

/// Determine whether the bucket contains the specified fingerprint.
#[inline]
pub fn bucket_contains(bucket: PackedBucket, fp: Fp) -> bool {
    debug_assert_ne!(fp, 0);
    let repeated = u64::from(fp) * 0x0001_0001_0001_0001;
    let different = bucket ^ repeated;
    let high_bits = 0x8000_8000_8000_8000;
    different.wrapping_sub(0x0001_0001_0001_0001) & !different & high_bits != 0
}

/// Find the first slot in the bucket equal to `fp`.
#[inline]
pub fn first_match(bucket: PackedBucket, fp: Fp) -> Option<usize> {
    (0..FP_PER_BUCKET).find(|&i| slot(bucket, i) == fp)
}

/// Find the first empty slot in the bucket (fingerprint is 0).
#[inline]
pub fn first_empty(bucket: PackedBucket) -> Option<usize> {
    (0..FP_PER_BUCKET).find(|&i| slot(bucket, i) == 0)
}

/// Try to insert a fingerprint into the bucket (find an empty slot to write).
/// Returns `true` on success, `false` if the bucket is full.
#[inline]
pub fn try_insert(bucket: &mut PackedBucket, fp: Fp) -> bool {
    debug_assert_ne!(fp, 0);
    if let Some(i) = first_empty(*bucket) {
        *bucket = with_slot(*bucket, i, fp);
        return true;
    }
    false
}

/// Try to delete the specified fingerprint from the bucket (find a matching slot
/// and set it to 0). Returns `true` on success, `false` if not found.
#[inline]
pub fn try_delete(bucket: &mut PackedBucket, fp: Fp) -> bool {
    debug_assert_ne!(fp, 0);
    if let Some(i) = first_match(*bucket, fp) {
        *bucket = with_slot(*bucket, i, 0);
        return true;
    }
    false
}

/// Full CKF snapshot: contains the current value of all buckets in a lane and a
/// monotonically increasing sequence number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CkfSnapshot {
    /// Monotonically increasing snapshot sequence number.
    pub sequence: u64,
    /// Values of all buckets in the lane, ordered by bucket index.
    pub buckets: Vec<PackedBucket>,
}

/// CKF delta: contains only the buckets changed since the previous sequence number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CkfDelta {
    /// Sequence number this delta corresponds to.
    pub sequence: u64,
    /// Previous sequence number this delta is based on.
    pub prev_sequence: u64,
    /// List of changed buckets: (bucket_idx, new_value).
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
