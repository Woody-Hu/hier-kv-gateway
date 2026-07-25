//! 真实 RadixTree 的 KV event 处理集成测试。
//!
//! 这些测试直接驱动 `aether_metadata::radix_tree::RadixTree` 的后台线程，
//! 应用真实 `KvCacheEvent` 并验证 `find_matches` 的前缀匹配语义。不使用任何
//! mock 对象，所有断言均基于后台线程状态机的真实演化结果。

use aether_core::ids::{BackendId, WorkerWithRank};
use aether_core::kv_event::KvCacheEvent;
use aether_metadata::radix_tree::RadixTree;

/// 构造一个 `Stored` 事件，承载给定的块哈希列表。
fn stored(hashes: Vec<u64>) -> KvCacheEvent {
    KvCacheEvent::Stored {
        worker: WorkerWithRank::from_worker_id(1),
        block_hashes: hashes,
        parent_hash: None,
        num_block_tokens: Vec::new(),
    }
}

/// 构造一个 `Removed` 事件，承载给定的块哈希列表。
fn removed(hashes: Vec<u64>) -> KvCacheEvent {
    KvCacheEvent::Removed {
        worker: WorkerWithRank::from_worker_id(1),
        block_hashes: hashes,
    }
}

/// 构造一个 `Clear` 事件。
fn clear() -> KvCacheEvent {
    KvCacheEvent::Clear {
        worker: WorkerWithRank::from_worker_id(1),
    }
}

/// 构造测试用 backend 标识。
fn backend(name: &str) -> BackendId {
    BackendId::new("test-region", name)
}

#[tokio::test]
async fn radix_tree_kv_event_lifecycle() {
    let tree = RadixTree::new();

    let backend_a = backend("a");
    let backend_b = backend("b");

    // 1) backend A 存入 [1, 2, 3]，应能在 [1, 2, 3] 上完全命中
    tree.apply_event(backend_a.clone(), stored(vec![1, 2, 3]))
        .await
        .expect("应用 Stored([1,2,3]) 到 backend A 应成功");
    let overlap_a_full = tree.find_matches(vec![1, 2, 3], backend_a.clone()).await;
    assert_eq!(
        overlap_a_full, 3,
        "backend A 应匹配全部 3 个块哈希"
    );

    // 2) backend B 存入 [1, 2, 4]，与 A 共享前缀 [1, 2]
    tree.apply_event(backend_b.clone(), stored(vec![1, 2, 4]))
        .await
        .expect("应用 Stored([1,2,4]) 到 backend B 应成功");

    // 2a) 查询 [1, 2, 3] 对 B：前缀 [1, 2] 命中，第 3 个块不属于 B → 2
    let overlap_b_partial = tree
        .find_matches(vec![1, 2, 3], backend_b.clone())
        .await;
    assert_eq!(
        overlap_b_partial, 2,
        "backend B 仅应匹配前缀 [1, 2] 共 2 个块"
    );

    // 2b) 查询 [1, 2, 4] 对 A：前缀 [1, 2] 命中，第 3 个块不属于 A → 2
    let overlap_a_partial = tree
        .find_matches(vec![1, 2, 4], backend_a.clone())
        .await;
    assert_eq!(
        overlap_a_partial, 2,
        "backend A 仅应匹配前缀 [1, 2] 共 2 个块"
    );

    // 3) 从 backend A 移除块 3：tree 中持有 hash=3 的节点应清除 A 的所有权
    tree.apply_event(backend_a.clone(), removed(vec![3]))
        .await
        .expect("应用 Removed([3]) 到 backend A 应成功");
    let overlap_a_after_removed = tree
        .find_matches(vec![1, 2, 3], backend_a.clone())
        .await;
    assert_eq!(
        overlap_a_after_removed, 2,
        "移除块 3 后，backend A 在 [1, 2, 3] 上的命中应降至 2"
    );

    // 4) 清空 backend B 的全部所有权
    tree.apply_event(backend_b.clone(), clear())
        .await
        .expect("应用 Clear 到 backend B 应成功");
    let overlap_b_after_clear = tree
        .find_matches(vec![1, 2, 4], backend_b.clone())
        .await;
    assert_eq!(
        overlap_b_after_clear, 0,
        "Clear 后 backend B 不应再命中任何块"
    );

    // 5) 额外验证：清空 B 不应影响 A 的所有权
    let overlap_a_still = tree
        .find_matches(vec![1, 2], backend_a.clone())
        .await;
    assert_eq!(
        overlap_a_still, 2,
        "backend A 在前缀 [1, 2] 上的命中不应受 B 的 Clear 影响"
    );

    // 关闭后台线程，避免在测试结束时泄漏
    tree.shutdown();
}

#[tokio::test]
async fn radix_tree_find_all_matches_aggregates_owners() {
    // 该用例验证 `find_all_matches` 能沿前缀路径聚合每个 backend 的命中数。
    let tree = RadixTree::new();
    let backend_a = backend("a");
    let backend_b = backend("b");
    let backend_c = backend("c");

    // A 存入 [10, 20, 30]，B 存入 [10, 20, 40]，C 存入 [10, 50]
    tree.apply_event(backend_a.clone(), stored(vec![10, 20, 30]))
        .await
        .unwrap();
    tree.apply_event(backend_b.clone(), stored(vec![10, 20, 40]))
        .await
        .unwrap();
    tree.apply_event(backend_c.clone(), stored(vec![10, 50]))
        .await
        .unwrap();

    // 查询 [10, 20, 30, 40] 应得到：
    // - A: 3 (10, 20, 30)
    // - B: 2 (10, 20，第 3 个块 30 不属于 B 中断)
    // - C: 1 (10，第 2 个块 20 不属于 C 中断)
    let all = tree.find_all_matches(vec![10, 20, 30, 40]).await;
    assert_eq!(
        all.get(&backend_a).copied().unwrap_or(0),
        3,
        "A 应命中 3 个块"
    );
    assert_eq!(
        all.get(&backend_b).copied().unwrap_or(0),
        2,
        "B 应命中前缀 [10, 20] 共 2 个块"
    );
    assert_eq!(
        all.get(&backend_c).copied().unwrap_or(0),
        1,
        "C 应仅命中 [10] 共 1 个块"
    );

    tree.shutdown();
}
