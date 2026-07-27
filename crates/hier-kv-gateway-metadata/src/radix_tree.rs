//! Local exact KV block index (Radix Tree).
//!
//! All write operations are serialized through a dedicated background thread; read
//! operations synchronously return results via an mpsc channel + oneshot. This
//! keeps the internal implementation lock-free while remaining safe to call from
//! async contexts.
//!
//! Each non-root node represents a block hash in the sequence prefix. The node's
//! `owners` set records all `(BackendId, rank)` pairs that hold that block.
//! `find_matches` walks the prefix path matching block hashes one by one; on a
//! hit where the backend is an owner, overlap is incremented; on a miss it stops early.

use std::collections::{HashMap, HashSet};

use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

/// Node ownership granularity: a `(backend, rank)` pair, matching `WorkerWithDpRank` semantics.
pub type Owner = (BackendId, u32);

/// Internal Radix Tree node.
#[derive(Debug)]
pub struct Node {
    /// Block hash corresponding to this node. The root node has hash 0.
    pub hash: u64,
    /// All `(backend, rank)` pairs that hold this block.
    pub owners: HashSet<Owner>,
    /// Child nodes, indexed by block hash.
    pub children: HashMap<u64, Node>,
    /// Reference count (a cache of the owners count, for quickly checking whether the node is reclaimable).
    pub ref_count: u32,
}

impl Node {
    fn new(hash: u64) -> Self {
        Self {
            hash,
            owners: HashSet::new(),
            children: HashMap::new(),
            ref_count: 0,
        }
    }

    fn add_owner(&mut self, owner: Owner) {
        if self.owners.insert(owner) {
            self.ref_count = self.ref_count.saturating_add(1);
        }
    }

    fn remove_owner(&mut self, owner: &Owner) {
        if self.owners.remove(owner) {
            self.ref_count = self.ref_count.saturating_sub(1);
        }
    }

    fn is_owned_by(&self, backend: &BackendId) -> bool {
        self.owners.iter().any(|(b, _)| b == backend)
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Background thread statistics.
#[derive(Debug, Clone, Default)]
pub struct RadixStats {
    /// Total number of non-root nodes in the tree.
    pub total_nodes: usize,
    /// Sum of owners across all nodes (including duplicates).
    pub total_blocks: usize,
    /// Number of currently active backends.
    pub backends: usize,
    /// Maximum depth of the tree.
    pub depth: usize,
}

/// Commands handled by the background thread.
enum RadixCommand {
    /// Apply a KV cache event.
    ApplyEvent {
        backend: BackendId,
        event: KvCacheEvent,
        done: oneshot::Sender<Result<()>>,
    },
    /// Query the prefix overlap length of a single backend for a given hash sequence.
    FindMatches {
        hashes: Vec<u64>,
        backend: BackendId,
        done: oneshot::Sender<u32>,
    },
    /// Query the prefix overlap length of all backends for a given hash sequence.
    FindAllMatches {
        hashes: Vec<u64>,
        done: oneshot::Sender<HashMap<BackendId, u32>>,
    },
    /// Remove all ownership for a backend.
    RemoveBackend {
        backend: BackendId,
        done: oneshot::Sender<()>,
    },
    /// Get statistics.
    Stats {
        done: oneshot::Sender<RadixStats>,
    },
    /// Shut down the background thread.
    Shutdown,
}

/// Thread-safe Radix Tree handle.
///
/// The handle itself only holds an mpsc sender, so cloning is cheap; all calls
/// are forwarded to the background thread via the channel. The background thread
/// is hinted to exit via `try_send(Shutdown)` when the last handle is dropped
/// (if the thread is currently waiting for a command it will receive it
/// immediately; if the channel is already closed the error is ignored).
#[derive(Clone)]
pub struct RadixTree {
    tx: mpsc::Sender<RadixCommand>,
}

impl RadixTree {
    /// Create a new Radix Tree and start the background processing thread.
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel::<RadixCommand>(4096);
        std::thread::Builder::new()
            .name("hier-kv-gateway-radix-tree".to_string())
            .spawn(move || {
                let mut state = TreeState::default();
                debug!("hier-kv-gateway-radix-tree worker started");
                while let Some(cmd) = rx.blocking_recv() {
                    match cmd {
                        RadixCommand::ApplyEvent {
                            backend,
                            event,
                            done,
                        } => {
                            let result = state.apply_event(backend, event);
                            let _ = done.send(result);
                        }
                        RadixCommand::FindMatches {
                            hashes,
                            backend,
                            done,
                        } => {
                            let overlap = state.find_matches(&hashes, &backend);
                            let _ = done.send(overlap);
                        }
                        RadixCommand::FindAllMatches { hashes, done } => {
                            let all = state.find_all_matches(&hashes);
                            let _ = done.send(all);
                        }
                        RadixCommand::RemoveBackend { backend, done } => {
                            state.remove_backend(&backend);
                            let _ = done.send(());
                        }
                        RadixCommand::Stats { done } => {
                            let stats = state.stats();
                            let _ = done.send(stats);
                        }
                        RadixCommand::Shutdown => {
                            debug!("hier-kv-gateway-radix-tree worker shutting down");
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn hier-kv-gateway-radix-tree worker thread");
        Self { tx }
    }

    /// Apply a KV cache event to the specified backend.
    pub async fn apply_event(
        &self,
        backend: BackendId,
        event: KvCacheEvent,
    ) -> Result<()> {
        let (done, rx) = oneshot::channel();
        self.tx
            .send(RadixCommand::ApplyEvent {
                backend,
                event,
                done,
            })
            .await
            .map_err(|_| {
                HierKvGatewayError::Internal(
                    "radix tree worker thread terminated".to_string(),
                )
            })?;
        rx.await.map_err(|_| {
            HierKvGatewayError::Internal(
                "radix tree worker dropped response".to_string(),
            )
        })?
    }

    /// Query the prefix overlap length of the specified backend for the given hash sequence.
    pub async fn find_matches(&self, hashes: Vec<u64>, backend: BackendId) -> u32 {
        let (done, rx) = oneshot::channel();
        if self
            .tx
            .send(RadixCommand::FindMatches {
                hashes,
                backend,
                done,
            })
            .await
            .is_err()
        {
            return 0;
        }
        rx.await.unwrap_or(0)
    }

    /// Query the prefix overlap length of all backends for the given hash sequence.
    pub async fn find_all_matches(
        &self,
        hashes: Vec<u64>,
    ) -> HashMap<BackendId, u32> {
        let (done, rx) = oneshot::channel();
        if self
            .tx
            .send(RadixCommand::FindAllMatches { hashes, done })
            .await
            .is_err()
        {
            return HashMap::new();
        }
        rx.await.unwrap_or_default()
    }

    /// Remove all ownership for the specified backend (used when a backend goes offline).
    pub async fn remove_backend(&self, backend: BackendId) {
        let (done, rx) = oneshot::channel();
        if self
            .tx
            .send(RadixCommand::RemoveBackend { backend, done })
            .await
            .is_err()
        {
            return;
        }
        let _ = rx.await;
    }

    /// Get current statistics.
    pub async fn stats(&self) -> RadixStats {
        let (done, rx) = oneshot::channel();
        if self
            .tx
            .send(RadixCommand::Stats { done })
            .await
            .is_err()
        {
            return RadixStats::default();
        }
        rx.await.unwrap_or_default()
    }

    /// Shut down the background thread. Drop attempts to shut down automatically;
    /// calling this explicitly forces termination even when unreachable clones still exist.
    pub fn shutdown(&self) {
        let _ = self.tx.try_send(RadixCommand::Shutdown);
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RadixTree {
    fn drop(&mut self) {
        // Best-effort shutdown only: if clones still hold a sender, the shutdown is ignored.
        let _ = self.tx.try_send(RadixCommand::Shutdown);
    }
}

/// Tree state held by the background thread.
#[derive(Default)]
struct TreeState {
    root: Node,
    /// All backends ever seen, used for stats and find_all_matches.
    backends: HashSet<BackendId>,
}

impl TreeState {
    fn apply_event(&mut self, backend: BackendId, event: KvCacheEvent) -> Result<()> {
        match event {
            KvCacheEvent::Stored { block_hashes, .. } => {
                self.backends.insert(backend.clone());
                self.apply_stored(backend, &block_hashes);
                Ok(())
            }
            KvCacheEvent::Removed { block_hashes, .. } => {
                self.apply_removed(backend, &block_hashes);
                Ok(())
            }
            KvCacheEvent::Clear { .. } => {
                self.clear_backend(&backend);
                Ok(())
            }
            KvCacheEvent::Reset { .. } => {
                // Reset acts as a generation fence: clear all ownership for this backend.
                // Semantically equivalent to Clear, but triggered for a different reason (worker restart / generation switch).
                warn!(
                    ?backend,
                    "radix tree received Reset event, clearing all ownership"
                );
                self.clear_backend(&backend);
                Ok(())
            }
        }
    }

    fn apply_stored(&mut self, backend: BackendId, hashes: &[u64]) {
        // Each rank defaults to 0; if hier-kv-gateway-core exposes finer-grained ranks, extend here.
        let owner = (backend, 0u32);
        let mut current = &mut self.root;
        for &hash in hashes {
            let child = current
                .children
                .entry(hash)
                .or_insert_with(|| Node::new(hash));
            child.add_owner(owner.clone());
            current = child;
        }
    }

    fn apply_removed(&mut self, backend: BackendId, hashes: &[u64]) {
        let owner = (backend, 0u32);
        // The `block_hashes` in a `Removed` event are a "list of block hashes to remove"
        // (independent blocks, not a prefix path from the root), so we must search the
        // whole tree for nodes whose hash matches and remove that backend's ownership.
        // This matches CkfProducer's semantics for Removed: each hash is removed
        // independently. The root node (hash=0) is not reclaimed, keeping the tree
        // structure always present.
        let hash_set: HashSet<u64> = hashes.iter().copied().collect();
        let mut empty_keys = Vec::new();
        for (key, child) in self.root.children.iter_mut() {
            if remove_owners_by_hash(child, &owner, &hash_set) {
                empty_keys.push(*key);
            }
        }
        for key in empty_keys {
            self.root.children.remove(&key);
        }
    }

    fn clear_backend(&mut self, backend: &BackendId) {
        self.backends.remove(backend);
        clear_backend_recursive(&mut self.root, backend);
    }

    fn find_matches(&self, hashes: &[u64], backend: &BackendId) -> u32 {
        let mut current = &self.root;
        let mut overlap = 0u32;
        for &hash in hashes {
            let Some(child) = current.children.get(&hash) else {
                break;
            };
            if !child.is_owned_by(backend) {
                break;
            }
            overlap += 1;
            current = child;
        }
        overlap
    }

    fn find_all_matches(&self, hashes: &[u64]) -> HashMap<BackendId, u32> {
        // Walk the prefix path collecting the maximum overlap length for each backend.
        let mut scores: HashMap<BackendId, u32> = HashMap::new();
        let mut current = &self.root;
        for &hash in hashes {
            let Some(child) = current.children.get(&hash) else {
                break;
            };
            if child.owners.is_empty() {
                break;
            }
            for (backend, _) in &child.owners {
                *scores.entry(backend.clone()).or_insert(0) += 1;
            }
            current = child;
        }
        scores
    }

    fn remove_backend(&mut self, backend: &BackendId) {
        self.clear_backend(backend);
    }

    fn stats(&self) -> RadixStats {
        let mut total_nodes = 0usize;
        let mut total_blocks = 0usize;
        let mut depth = 0usize;
        for child in self.root.children.values() {
            let (nodes, blocks, d) = stats_recursive(child, 1);
            total_nodes += nodes;
            total_blocks += blocks;
            depth = depth.max(d);
        }
        RadixStats {
            total_nodes,
            total_blocks,
            backends: self.backends.len(),
            depth,
        }
    }
}

/// Recursively search the subtree, removing the specified owner's ownership from
/// all nodes whose hash is in the `hashes` set, and reclaim empty nodes bottom-up.
/// Returns `true` if the current node has no owner and no children and should be
/// removed by the parent.
///
/// This matches the semantics of `Removed { block_hashes }`: `block_hashes` is a
/// set of independent block hashes; any node in the tree whose hash matches will
/// have that owner removed (content-addressed blocks are shared across prefixes
/// in the cache, so global removal is correct).
fn remove_owners_by_hash(node: &mut Node, owner: &Owner, hashes: &HashSet<u64>) -> bool {
    if hashes.contains(&node.hash) {
        node.remove_owner(owner);
    }
    let mut empty_keys = Vec::new();
    for (key, child) in node.children.iter_mut() {
        if remove_owners_by_hash(child, owner, hashes) {
            empty_keys.push(*key);
        }
    }
    for key in empty_keys {
        node.children.remove(&key);
    }
    node.ref_count == 0 && node.children.is_empty()
}

/// Recursively remove all ownership of the specified backend within this subtree,
/// and reclaim empty nodes. Returns `true` if the current node should be removed
/// by the parent.
fn clear_backend_recursive(node: &mut Node, backend: &BackendId) -> bool {
    let before = node.owners.len();
    node.owners.retain(|(b, _)| b != backend);
    let removed = before - node.owners.len();
    node.ref_count = node.ref_count.saturating_sub(removed as u32);

    let mut empty_keys = Vec::new();
    for (key, child) in node.children.iter_mut() {
        if clear_backend_recursive(child, backend) {
            empty_keys.push(*key);
        }
    }
    for key in empty_keys {
        node.children.remove(&key);
    }
    node.ref_count == 0 && node.children.is_empty()
}

fn stats_recursive(node: &Node, depth: usize) -> (usize, usize, usize) {
    let mut nodes = 1usize;
    let mut blocks = node.owners.len();
    let mut max_depth = depth;
    for child in node.children.values() {
        let (n, b, d) = stats_recursive(child, depth + 1);
        nodes += n;
        blocks += b;
        max_depth = max_depth.max(d);
    }
    (nodes, blocks, max_depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::ids::WorkerWithRank;

    fn backend(n: u8) -> BackendId {
        BackendId::new(
            format!("r{n}"),
            format!("i{n}"),
        )
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

    fn clear() -> KvCacheEvent {
        KvCacheEvent::Clear {
            worker: WorkerWithRank::from_worker_id(1),
        }
    }

    #[tokio::test]
    async fn stored_and_find_matches() {
        let tree = RadixTree::new();
        let backend = backend(1);
        tree.apply_event(backend.clone(), stored(vec![10, 20, 30]))
            .await
            .unwrap();
        let overlap = tree.find_matches(vec![10, 20, 30, 40], backend).await;
        assert_eq!(overlap, 3);
    }

    #[tokio::test]
    async fn removed_decrements_overlap() {
        let tree = RadixTree::new();
        let backend = backend(1);
        tree.apply_event(backend.clone(), stored(vec![1, 2, 3]))
            .await
            .unwrap();
        tree.apply_event(backend.clone(), removed(vec![3]))
            .await
            .unwrap();
        let overlap = tree.find_matches(vec![1, 2, 3], backend).await;
        assert_eq!(overlap, 2);
    }

    #[tokio::test]
    async fn clear_removes_all_ownership() {
        let tree = RadixTree::new();
        let backend = backend(1);
        tree.apply_event(backend.clone(), stored(vec![1, 2, 3]))
            .await
            .unwrap();
        tree.apply_event(backend.clone(), clear()).await.unwrap();
        let overlap = tree.find_matches(vec![1, 2, 3], backend).await;
        assert_eq!(overlap, 0);
    }
}
