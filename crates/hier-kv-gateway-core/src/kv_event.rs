//! KV Cache event types and block hash computation.
//!
//! When a KV cache block is written or removed, the backend publishes a
//! [`KvCacheEvent`] to notify the indexer; events are serialized with internal
//! tagging via the `type` field so consumers can dispatch by type.
//!
//! Block hashes are computed using the XXH3 algorithm from [`xxhash-rust`], with
//! the seed derived from `cache_namespace` and `lora_name`, ensuring that
//! same-named blocks under different tenants/adapters never collide.

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

use crate::ids::WorkerWithRank;

/// A store / remove / clear / reset event for KV cache blocks.
///
/// Serialized using internal tagging via the `type` field with `snake_case` naming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KvCacheEvent {
    /// Event for storing one or more blocks.
    Stored {
        /// Worker and dp_rank that stored the blocks.
        worker: WorkerWithRank,
        /// List of block hashes.
        block_hashes: Vec<u64>,
        /// Parent block hash (`None` for the first block).
        parent_hash: Option<u64>,
        /// Actual number of tokens in each block.
        num_block_tokens: Vec<u64>,
    },
    /// Event for removing one or more blocks.
    Removed {
        /// Worker that removed the blocks.
        worker: WorkerWithRank,
        /// List of block hashes to remove.
        block_hashes: Vec<u64>,
    },
    /// Event for clearing all blocks on a given worker.
    Clear {
        /// Worker being cleared.
        worker: WorkerWithRank,
    },
    /// Event for resetting the generation of a worker, invalidating its historical events.
    Reset {
        /// Worker being reset.
        worker: WorkerWithRank,
        /// Generation identifier after reset; the indexer should ignore events with a smaller generation.
        generation: u64,
    },
}

/// Inputs required to compute block hashes.
///
/// `cache_namespace` and `lora_name` are mixed into the XXH3 seed, ensuring that
/// identical token sequences under different tenants or LoRA adapters produce
/// different hashes. Empty strings are treated as not provided.
#[derive(Debug, Clone, Default)]
pub struct BlockHashInput<'a> {
    /// Token sequence; each token is written as little-endian `u32` bytes into the hash.
    pub tokens: &'a [u32],
    /// Number of tokens per KV block.
    pub kv_block_size: u32,
    /// Optional cache namespace.
    pub cache_namespace: Option<&'a str>,
    /// Optional LoRA adapter name.
    pub lora_name: Option<&'a str>,
}

/// Fixed base seed for XXH3 computation, agreed upon with the indexer.
const XXH3_SEED: u64 = 1337;

/// Salt mixed with the namespace hash, isolating namespace influence from the default seed.
const NS_SALT: u64 = 0x4e53_5f4c_4f5f_4c4f;
/// Salt mixed with the LoRA hash, independent of the namespace salt so the two influences do not overlap.
const LORA_SALT: u64 = 0x4c52_4f5f_4c4f_5f4c;

/// Derive an XXH3 seed from `cache_namespace` and `lora_name`.
///
/// - Empty strings are treated as not provided;
/// - The namespace and LoRA are each hashed with their own salt, then combined with the
///   base seed via `wrapping_add` and XOR, ensuring their influences are independent
///   and cannot cancel each other out.
fn compute_seed(cache_namespace: Option<&str>, lora_name: Option<&str>) -> u64 {
    let mut seed = XXH3_SEED;
    if let Some(ns) = cache_namespace.filter(|s| !s.is_empty()) {
        let ns_hash = xxh3::xxh3_64_with_seed(ns.as_bytes(), NS_SALT);
        seed = seed.wrapping_add(ns_hash);
        seed ^= NS_SALT;
    }
    if let Some(lora) = lora_name.filter(|s| !s.is_empty()) {
        let lora_hash = xxh3::xxh3_64_with_seed(lora.as_bytes(), LORA_SALT);
        seed = seed.wrapping_add(lora_hash);
        seed ^= LORA_SALT;
    }
    seed
}

/// Compute the list of block hashes for the given input.
///
/// - Splits `tokens` into non-overlapping windows of `kv_block_size`; the final
///   incomplete block is discarded;
/// - Tokens within each window are written as little-endian bytes into a buffer,
///   then [`xxh3::xxh3_64_with_seed`] is called to produce the block hash;
/// - `cache_namespace` and `lora_name` are mixed in via the derived seed; empty
///   strings are treated as not provided.
///
/// Returns an empty vector when `kv_block_size == 0`.
pub fn compute_block_hashes(input: &BlockHashInput<'_>) -> Vec<u64> {
    if input.kv_block_size == 0 {
        return Vec::new();
    }
    let seed = compute_seed(input.cache_namespace, input.lora_name);
    let stride = input.kv_block_size as usize;
    let mut hashes = Vec::with_capacity(input.tokens.len() / stride);
    let mut bytes: Vec<u8> = Vec::with_capacity(stride * std::mem::size_of::<u32>());
    let mut start = 0;
    while start + stride <= input.tokens.len() {
        bytes.clear();
        for &token in &input.tokens[start..start + stride] {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        hashes.push(xxh3::xxh3_64_with_seed(&bytes, seed));
        start += stride;
    }
    hashes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_count_matches_stride() {
        let tokens: Vec<u32> = (0..11).collect();
        let hashes = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: None,
            lora_name: None,
        });
        // 11 / 4 = 2 complete blocks
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn block_count_exact_multiple() {
        let tokens: Vec<u32> = (0..8).collect();
        let hashes = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: None,
            lora_name: None,
        });
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn zero_block_size_returns_empty() {
        let tokens: Vec<u32> = (0..4).collect();
        let hashes = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 0,
            cache_namespace: None,
            lora_name: None,
        });
        assert!(hashes.is_empty());
    }

    #[test]
    fn identical_inputs_produce_identical_hashes() {
        let tokens: Vec<u32> = (0..4).collect();
        let a = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: Some("tenant-a"),
            lora_name: Some("adapter-a"),
        });
        let b = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: Some("tenant-a"),
            lora_name: Some("adapter-a"),
        });
        assert_eq!(a, b);
    }

    #[test]
    fn different_lora_produces_different_hashes() {
        let tokens: Vec<u32> = (0..4).collect();
        let a = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: None,
            lora_name: Some("adapter-a"),
        });
        let b = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: None,
            lora_name: Some("adapter-b"),
        });
        assert_ne!(a, b);
    }

    #[test]
    fn different_namespace_produces_different_hashes() {
        let tokens: Vec<u32> = (0..4).collect();
        let a = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: Some("tenant-a"),
            lora_name: None,
        });
        let b = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: Some("tenant-b"),
            lora_name: None,
        });
        assert_ne!(a, b);
    }

    #[test]
    fn empty_namespace_normalized_to_none() {
        let tokens: Vec<u32> = (0..4).collect();
        let base = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: None,
            lora_name: None,
        });
        let empty_ns = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: Some(""),
            lora_name: None,
        });
        let empty_lora = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: None,
            lora_name: Some(""),
        });
        assert_eq!(base, empty_ns, "an empty cache_namespace should be treated as not provided");
        assert_eq!(base, empty_lora, "an empty lora_name should be treated as not provided");
    }

    #[test]
    fn namespace_and_lora_independent() {
        let tokens: Vec<u32> = (0..4).collect();
        let ns = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: Some("foo"),
            lora_name: None,
        });
        let lora = compute_block_hashes(&BlockHashInput {
            tokens: &tokens,
            kv_block_size: 4,
            cache_namespace: None,
            lora_name: Some("foo"),
        });
        assert_ne!(
            ns, lora,
            "namespace and lora_name with the same value should still produce independent hashes"
        );
    }

    #[test]
    fn stored_event_round_trip_json() {
        let event = KvCacheEvent::Stored {
            worker: WorkerWithRank::new(7, 0),
            block_hashes: vec![1, 2, 3],
            parent_hash: Some(0),
            num_block_tokens: vec![4, 4, 3],
        };
        let s = serde_json::to_string(&event).unwrap();
        let back: KvCacheEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(event, back);
        // Internal tag field
        assert!(s.contains(r#""type":"stored""#));
    }

    #[test]
    fn clear_and_reset_event_tags() {
        let clear = KvCacheEvent::Clear {
            worker: WorkerWithRank::new(1, 0),
        };
        let s = serde_json::to_string(&clear).unwrap();
        assert!(s.contains(r#""type":"clear""#));

        let reset = KvCacheEvent::Reset {
            worker: WorkerWithRank::new(1, 0),
            generation: 42,
        };
        let s = serde_json::to_string(&reset).unwrap();
        assert!(s.contains(r#""type":"reset""#));
    }

    #[test]
    fn removed_event_round_trip() {
        let event = KvCacheEvent::Removed {
            worker: WorkerWithRank::new(7, 1),
            block_hashes: vec![10, 20],
        };
        let s = serde_json::to_string(&event).unwrap();
        assert!(s.contains(r#""type":"removed""#));
        let back: KvCacheEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(event, back);
    }
}
