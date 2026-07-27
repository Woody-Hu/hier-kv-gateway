//! Local prefix reuse history for degradation routing.
//!
//! Records dispatch decisions keyed by incremental prefix hashes of request
//! block_hashes. During degradation (when KV/model metadata is stale or
//! unavailable), the degradation router queries this history to find the
//! longest matching prefix and returns the previously-dispatched backend.
//!
//! Prefix hashing is incremental:
//!   p[0] = hash(block_hashes[0])
//!   p[i] = hash(p[i-1] || block_hashes[i])
//! This allows O(k) insert and O(k) lookup where k = number of blocks.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use hier_kv_gateway_core::ids::BackendId;
use parking_lot::RwLock;
use xxhash_rust::xxh3;

/// Seed used for incremental prefix hashing. The bytes spell "HIERKVGG".
pub const PREFIX_SEED: u64 = 0x4849_4552_4B56_4757;

/// Default maximum number of dispatch records retained.
pub const DEFAULT_MAX_ENTRIES: usize = 10_000;

/// A dispatch history entry: which backend handled a prefix, how many times.
#[derive(Debug, Clone)]
pub struct DispatchRecord {
    /// The backend that was dispatched to for this prefix.
    pub backend: BackendId,
    /// How many times this prefix was dispatched to this backend.
    pub dispatch_count: u64,
    /// Last dispatch Unix timestamp (seconds).
    pub last_dispatched_unix: u64,
}

/// Prefix reuse history with a bounded number of entries (LRU-ish eviction).
///
/// When the entry count exceeds `max_entries`, the oldest entries (by
/// last_dispatched_unix) are evicted. This keeps memory bounded while
/// retaining the most useful recent dispatch patterns.
pub struct PrefixReuseHistory {
    /// prefix_hash -> DispatchRecord
    entries: RwLock<HashMap<u64, DispatchRecord>>,
    /// Maximum number of entries before eviction.
    max_entries: usize,
}

impl PrefixReuseHistory {
    /// Create a new history with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(max_entries.min(1024))),
            max_entries,
        }
    }

    /// Create a new history with the default capacity.
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }

    /// Record a dispatch: for the given block_hashes, the request was routed
    /// to `backend`. Records ALL prefix lengths (1..=len) as incremental hashes.
    ///
    /// If `block_hashes` is empty, this is a no-op (nothing to learn).
    pub fn record_dispatch(&self, block_hashes: &[u64], backend: &BackendId) {
        if block_hashes.is_empty() {
            return;
        }
        let prefix_hashes = Self::compute_prefix_hashes(block_hashes);
        let now = now_unix();

        let mut entries = self.entries.write();
        for ph in &prefix_hashes {
            match entries.get_mut(ph) {
                Some(rec) => {
                    rec.dispatch_count = rec.dispatch_count.saturating_add(1);
                    rec.last_dispatched_unix = now;
                    // Backend should not change for the same prefix in normal
                    // operation; if it does, prefer the latest routing.
                    rec.backend = backend.clone();
                }
                None => {
                    entries.insert(
                        *ph,
                        DispatchRecord {
                            backend: backend.clone(),
                            dispatch_count: 1,
                            last_dispatched_unix: now,
                        },
                    );
                }
            }
        }
        drop(entries);

        self.evict_if_needed();
    }

    /// Find the longest matching prefix in history for the given block_hashes.
    /// Returns (prefix_length, DispatchRecord) for the longest match, or None.
    /// Iterates from the longest prefix down to length 1.
    pub fn find_longest_match(&self, block_hashes: &[u64]) -> Option<(usize, DispatchRecord)> {
        if block_hashes.is_empty() {
            return None;
        }
        let prefix_hashes = Self::compute_prefix_hashes(block_hashes);
        let entries = self.entries.read();
        // Iterate from longest (last index) to shortest (index 0).
        for (idx, ph) in prefix_hashes.iter().enumerate().rev() {
            if let Some(rec) = entries.get(ph) {
                // prefix length = idx + 1 (since idx is 0-based and prefix
                // includes the (idx+1)-th block).
                return Some((idx + 1, rec.clone()));
            }
        }
        None
    }

    /// Get a specific prefix hash's record.
    pub fn get(&self, prefix_hash: u64) -> Option<DispatchRecord> {
        self.entries.read().get(&prefix_hash).cloned()
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Maximum number of entries before eviction.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Evict oldest entries if over capacity. Called internally on record_dispatch.
    fn evict_if_needed(&self) {
        let mut entries = self.entries.write();
        if entries.len() <= self.max_entries {
            return;
        }
        let surplus = entries.len() - self.max_entries;
        // Collect prefix_hash -> last_dispatched_unix for all entries, then
        // sort ascending by timestamp and remove the oldest `surplus` entries.
        let mut stamped: Vec<(u64, u64)> = entries
            .iter()
            .map(|(k, v)| (*k, v.last_dispatched_unix))
            .collect();
        stamped.sort_unstable_by_key(|&(_, ts)| ts);
        let to_remove = stamped.into_iter().take(surplus).map(|(k, _)| k);
        for k in to_remove {
            entries.remove(&k);
        }
    }

    /// Compute the incremental prefix hashes for a block_hashes sequence.
    /// Returns a Vec of len block_hashes.len(), where result[i] is the hash
    /// of the prefix block_hashes[0..=i].
    fn compute_prefix_hashes(block_hashes: &[u64]) -> Vec<u64> {
        let mut out = Vec::with_capacity(block_hashes.len());
        let mut prev: u64 = 0;
        let mut first = true;
        for &h in block_hashes {
            let next = if first {
                first = false;
                xxh3::xxh3_64_with_seed(&h.to_le_bytes(), PREFIX_SEED)
            } else {
                let mut buf = [0u8; 16];
                buf[..8].copy_from_slice(&prev.to_le_bytes());
                buf[8..].copy_from_slice(&h.to_le_bytes());
                xxh3::xxh3_64_with_seed(&buf, PREFIX_SEED)
            };
            out.push(next);
            prev = next;
        }
        out
    }
}

impl Default for PrefixReuseHistory {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(name: &str) -> BackendId {
        BackendId::new("r1", name)
    }

    #[test]
    fn test_record_and_find_exact_match() {
        let hist = PrefixReuseHistory::new(100);
        let a = backend("a");
        // Record dispatch for prefix [1, 2, 3].
        hist.record_dispatch(&[1, 2, 3], &a);
        // Look up a sequence that extends the recorded prefix.
        let m = hist.find_longest_match(&[1, 2, 3, 4]).expect("should match");
        assert_eq!(m.0, 3);
        assert_eq!(m.1.backend, a);
    }

    #[test]
    fn test_find_partial_match() {
        let hist = PrefixReuseHistory::new(100);
        let a = backend("a");
        hist.record_dispatch(&[1, 2, 3], &a);
        // [1, 2, 5] shares prefix [1, 2] of length 2 with the recorded entry.
        let m = hist.find_longest_match(&[1, 2, 5]).expect("should match");
        assert_eq!(m.0, 2);
        assert_eq!(m.1.backend, a);
    }

    #[test]
    fn test_no_match() {
        let hist = PrefixReuseHistory::new(100);
        let a = backend("a");
        hist.record_dispatch(&[1, 2, 3], &a);
        // Completely disjoint hashes should not match.
        assert!(hist.find_longest_match(&[4, 5, 6]).is_none());
    }

    #[test]
    fn test_multiple_dispatches_increment_count() {
        let hist = PrefixReuseHistory::new(100);
        let a = backend("a");
        hist.record_dispatch(&[1, 2, 3], &a);
        hist.record_dispatch(&[1, 2, 3], &a);
        // The longest prefix [1,2,3] should have dispatch_count == 2.
        let m = hist.find_longest_match(&[1, 2, 3]).expect("should match");
        assert_eq!(m.0, 3);
        assert_eq!(m.1.dispatch_count, 2);
    }

    #[test]
    fn test_eviction() {
        // Set capacity to 3 entries; record 5 different single-block prefixes.
        // Each prefix writes exactly 1 entry, so we expect 3 to remain after eviction.
        let hist = PrefixReuseHistory::new(3);
        let a = backend("a");
        let hashes = [10u64, 20, 30, 40, 50];
        for h in hashes {
            hist.record_dispatch(&[h], &a);
        }
        assert_eq!(hist.len(), 3);
        // The eviction policy is LRU-ish by `last_dispatched_unix`. When
        // multiple entries share the same second-granularity timestamp
        // (as in this tight test loop), the specific victims are
        // non-deterministic. We therefore only verify the count invariant
        // and that every remaining entry corresponds to one of the recorded
        // prefixes.
        let all_prefix_hashes: std::collections::HashSet<u64> = hashes
            .iter()
            .map(|h| PrefixReuseHistory::compute_prefix_hashes(&[*h])[0])
            .collect();
        for h in &hashes {
            let ph = PrefixReuseHistory::compute_prefix_hashes(&[*h])[0];
            if let Some(rec) = hist.get(ph) {
                assert!(all_prefix_hashes.contains(&ph));
                assert_eq!(rec.backend, a);
            }
        }
    }

    #[test]
    fn test_empty_hashes() {
        let hist = PrefixReuseHistory::new(100);
        let a = backend("a");
        // Recording empty hashes is a no-op.
        hist.record_dispatch(&[], &a);
        assert!(hist.is_empty());
        // Looking up empty hashes returns None gracefully.
        assert!(hist.find_longest_match(&[]).is_none());
    }

    #[test]
    fn test_prefix_hashes_are_incremental_and_distinct() {
        // The incremental prefix hash property: hashes for distinct prefix
        // lengths should differ; identical prefixes should produce identical hashes.
        let h1 = PrefixReuseHistory::compute_prefix_hashes(&[1, 2, 3]);
        let h2 = PrefixReuseHistory::compute_prefix_hashes(&[1, 2, 3, 4]);
        assert_eq!(h1.len(), 3);
        assert_eq!(h2.len(), 4);
        // Prefix of length 3 should match for both.
        assert_eq!(h1[2], h2[2]);
        // But the 4-block hash should be different from the 3-block hash.
        assert_ne!(h2[3], h2[2]);
        // All hashes for distinct prefix lengths should differ.
        assert_ne!(h1[0], h1[1]);
        assert_ne!(h1[1], h1[2]);
    }

    #[test]
    fn test_longest_match_wins_over_shorter() {
        let hist = PrefixReuseHistory::new(100);
        let a = backend("a");
        let b = backend("b");
        // Record two prefixes that share a common shorter prefix.
        // [1, 2] -> a, [1, 2, 3] -> b
        hist.record_dispatch(&[1, 2], &a);
        hist.record_dispatch(&[1, 2, 3], &b);
        // Looking up [1, 2, 3, 4] should match the longest available (3) -> b.
        let m = hist.find_longest_match(&[1, 2, 3, 4]).expect("should match");
        assert_eq!(m.0, 3);
        assert_eq!(m.1.backend, b);
    }
}
