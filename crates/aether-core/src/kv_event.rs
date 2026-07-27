//! KV Cache 事件类型与块哈希计算。
//!
//! 后端在 KV 缓存块写入或移除时，通过发布 [`KvCacheEvent`] 通知索引器；
//! 事件以 `type` 字段进行内部标签（internally tagged）序列化，便于消费端按类型分发。
//!
//! 块哈希基于 [`xxhash-rust`] 的 XXH3 算法计算，并通过 `cache_namespace` 与 `lora_name`
//! 派生种子，保证不同租户/适配器的同名块互不冲突。

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

use crate::ids::WorkerWithRank;

/// 一个 KV 缓存块的存储 / 移除 / 清除 / 重置事件。
///
/// 序列化时使用 `type` 字段做内部标签，并采用 `snake_case` 命名。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KvCacheEvent {
    /// 写入一个或多个块的事件。
    Stored {
        /// 写入块的 worker 与 dp_rank。
        worker: WorkerWithRank,
        /// 块哈希列表。
        block_hashes: Vec<u64>,
        /// 父块哈希（首块为 `None`）。
        parent_hash: Option<u64>,
        /// 每个块的实际 token 数量。
        num_block_tokens: Vec<u64>,
    },
    /// 移除一个或多个块的事件。
    Removed {
        /// 移除块的 worker。
        worker: WorkerWithRank,
        /// 待移除的块哈希列表。
        block_hashes: Vec<u64>,
    },
    /// 清除指定 worker 上所有块的事件。
    Clear {
        /// 被清除的 worker。
        worker: WorkerWithRank,
    },
    /// 重置 worker 的事件代次，使其历史事件失效。
    Reset {
        /// 被重置的 worker。
        worker: WorkerWithRank,
        /// 重置后的代次标识，索引器应忽略小于此代次的事件。
        generation: u64,
    },
}

/// 计算块哈希所需的输入。
///
/// `cache_namespace` 与 `lora_name` 会混合进 XXH3 的种子中，
/// 确保不同租户或不同 LoRA 适配器下的相同 token 序列产生不同的哈希。
/// 空字符串视为未提供。
#[derive(Debug, Clone, Default)]
pub struct BlockHashInput<'a> {
    /// token 序列，每个 token 以 `u32` 小端序字节写入哈希。
    pub tokens: &'a [u32],
    /// 单个 KV 块的 token 数量。
    pub kv_block_size: u32,
    /// 可选缓存命名空间。
    pub cache_namespace: Option<&'a str>,
    /// 可选 LoRA 适配器名。
    pub lora_name: Option<&'a str>,
}

/// XXH3 计算用的固定基础种子，与索引器约定一致。
const XXH3_SEED: u64 = 1337;

/// 与命名空间哈希混合的 salt，确保命名空间影响与默认种子隔离。
const NS_SALT: u64 = 0x4e53_5f4c_4f5f_4c4f;
/// 与 LoRA 哈希混合的 salt，独立于命名空间 salt，保证两者影响互不重叠。
const LORA_SALT: u64 = 0x4c52_4f5f_4c4f_5f4c;

/// 根据 `cache_namespace` 与 `lora_name` 派生 XXH3 种子。
///
/// - 空字符串视为未提供；
/// - 命名空间与 LoRA 各自用独立 salt 哈希，再与基础种子做 `wrapping_add` 与异或，
///   保证两者影响相互独立且不可抵消。
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

/// 计算给定输入的块哈希列表。
///
/// - 按 `kv_block_size` 把 `tokens` 切分为多个不重叠窗口，最后一个不完整的块被丢弃；
/// - 每个窗口内 token 以小端字节写入缓冲，调用 [`xxh3::xxh3_64_with_seed`] 得到块哈希；
/// - `cache_namespace` 与 `lora_name` 通过派生种子混入，空字符串视为未提供。
///
/// 当 `kv_block_size == 0` 时返回空向量。
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
        // 11 / 4 = 2 个完整块
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
        assert_eq!(base, empty_ns, "空 cache_namespace 应视为未提供");
        assert_eq!(base, empty_lora, "空 lora_name 应视为未提供");
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
            "namespace 与 lora_name 同名时也应产生独立哈希"
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
        // 内部标签字段
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
