//! 本地精确 KV block 索引（Radix Tree）。
//!
//! 参考 Dynamo 的 RadixTree binding 思路：所有写操作通过专用后台线程串行化执行，
//! 读操作通过 mpsc channel + oneshot 同步返回结果。这样既保证了内部无锁的简单实现，
//! 又能从异步上下文安全调用。
//!
//! 每个非根节点表示序列前缀中的一个 block hash，节点的 `owners` 集合记录所有
//! 持有该 block 的 `(BackendId, rank)` 对。`find_matches` 沿前缀路径逐个匹配
//! block hash，命中且 backend 为 owner 则 overlap 自增，未命中则提前停止。

use std::collections::{HashMap, HashSet};

use aether_core::error::{AetherError, Result};
use aether_core::ids::BackendId;
use aether_core::kv_event::KvCacheEvent;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

/// 节点所有权粒度：`(backend, rank)` 对，对应 Dynamo 中 `WorkerWithDpRank` 的语义。
pub type Owner = (BackendId, u32);

/// Radix Tree 内部节点。
#[derive(Debug)]
pub struct Node {
    /// 该节点对应的 block hash。根节点为 0。
    pub hash: u64,
    /// 持有该 block 的所有 `(backend, rank)` 对。
    pub owners: HashSet<Owner>,
    /// 子节点，按 block hash 索引。
    pub children: HashMap<u64, Node>,
    /// 引用计数（即 owners 数量的缓存，便于快速判断节点是否可回收）。
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

/// 后台线程统计信息。
#[derive(Debug, Clone, Default)]
pub struct RadixStats {
    /// 树中非根节点总数。
    pub total_nodes: usize,
    /// 所有节点的 owners 总和（含重复）。
    pub total_blocks: usize,
    /// 当前活跃 backend 数量。
    pub backends: usize,
    /// 树的最大深度。
    pub depth: usize,
}

/// 后台线程处理的命令枚举。
enum RadixCommand {
    /// 应用一个 KV cache 事件。
    ApplyEvent {
        backend: BackendId,
        event: KvCacheEvent,
        done: oneshot::Sender<Result<()>>,
    },
    /// 查询单个 backend 对给定 hash 序列的前缀重叠长度。
    FindMatches {
        hashes: Vec<u64>,
        backend: BackendId,
        done: oneshot::Sender<u32>,
    },
    /// 查询所有 backend 对给定 hash 序列的前缀重叠长度。
    FindAllMatches {
        hashes: Vec<u64>,
        done: oneshot::Sender<HashMap<BackendId, u32>>,
    },
    /// 移除某个 backend 的全部所有权。
    RemoveBackend {
        backend: BackendId,
        done: oneshot::Sender<()>,
    },
    /// 获取统计信息。
    Stats {
        done: oneshot::Sender<RadixStats>,
    },
    /// 关闭后台线程。
    Shutdown,
}

/// 线程安全的 Radix Tree 句柄。
///
/// 句柄本身仅持有一个 mpsc sender，clone 廉价；所有调用通过 channel 转发到
/// 后台线程。后台线程在最后一个句柄 drop 时通过 `try_send(Shutdown)` 提示退出
/// （若线程恰好在等待命令则会立即收到；若 channel 已关闭则忽略错误）。
#[derive(Clone)]
pub struct RadixTree {
    tx: mpsc::Sender<RadixCommand>,
}

impl RadixTree {
    /// 创建一棵新的 Radix Tree，并启动后台处理线程。
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel::<RadixCommand>(4096);
        std::thread::Builder::new()
            .name("aether-radix-tree".to_string())
            .spawn(move || {
                let mut state = TreeState::default();
                debug!("aether-radix-tree worker started");
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
                            debug!("aether-radix-tree worker shutting down");
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn aether-radix-tree worker thread");
        Self { tx }
    }

    /// 应用一个 KV cache 事件到指定 backend。
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
                AetherError::Internal(
                    "radix tree worker thread terminated".to_string(),
                )
            })?;
        rx.await.map_err(|_| {
            AetherError::Internal(
                "radix tree worker dropped response".to_string(),
            )
        })?
    }

    /// 查询指定 backend 对给定 hash 序列的前缀重叠长度。
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

    /// 查询所有 backend 对给定 hash 序列的前缀重叠长度。
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

    /// 移除指定 backend 的全部所有权（用于 backend 下线）。
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

    /// 获取当前统计信息。
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

    /// 关闭后台线程。Drop 时会自动尝试关闭，显式调用可在不可达 clone 仍存在时
    /// 强制终止后台线程。
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
        // 仅作尽力关闭：若仍有 clone 持有 sender，shutdown 会被忽略。
        let _ = self.tx.try_send(RadixCommand::Shutdown);
    }
}

/// 后台线程持有的树状态。
#[derive(Default)]
struct TreeState {
    root: Node,
    /// 所有出现过的 backend，用于 stats 与 find_all_matches。
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
                // Reset 视为代际 fence：清空该 backend 全部所有权。
                // 语义上等价于 Clear，但触发原因不同（worker 重启 / 代际切换）。
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
        // 每个 rank 默认为 0；若 aether-core 暴露更细粒度的 rank，可在此扩展。
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
        // `Removed` 事件中的 `block_hashes` 是"待移除的块哈希列表"（独立块，
        // 而非从 root 起的前缀路径），因此需要在整棵树中搜索匹配 hash 的节点并
        // 移除该 backend 的 ownership。这与 CkfProducer 对 Removed 的处理语义一致：
        // 每个 hash 独立删除。根节点（hash=0）不参与回收，保证树结构始终保留。
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
        // 沿前缀路径收集每个 backend 的最大重叠长度。
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

/// 递归搜索子树，移除所有 hash 命中 `hashes` 集合的节点上指定 owner 的 ownership，
/// 并自底向上回收空节点。返回 `true` 表示当前节点已无 owner 且无子节点，应被父节点删除。
///
/// 与 `Removed { block_hashes }` 的语义对应：`block_hashes` 是一组独立块哈希，
/// 树中任何 hash 匹配的节点都会移除该 owner（content-addressed 块在缓存中
/// 跨前缀共享，故全局移除是正确的）。
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

/// 递归移除指定 backend 在该子树下的所有 ownership，并回收空节点。
/// 返回 `true` 表示当前节点应被父节点删除。
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
    use aether_core::ids::WorkerWithRank;

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
