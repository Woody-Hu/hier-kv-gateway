//! Real RadixTree KV event processing integration test.
//!
//! These tests directly drive the background thread of
//! `hier_kv_gateway_metadata::radix_tree::RadixTree`, apply real `KvCacheEvent`s, and
//! verify the prefix-matching semantics of `find_matches`. No mock objects are used; all
//! assertions are based on the real evolution of the background-thread state machine.

use hier_kv_gateway_core::ids::{BackendId, WorkerWithRank};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_metadata::radix_tree::RadixTree;

/// Construct a `Stored` event carrying the given block hash list.
fn stored(hashes: Vec<u64>) -> KvCacheEvent {
    KvCacheEvent::Stored {
        worker: WorkerWithRank::from_worker_id(1),
        block_hashes: hashes,
        parent_hash: None,
        num_block_tokens: Vec::new(),
    }
}

/// Construct a `Removed` event carrying the given block hash list.
fn removed(hashes: Vec<u64>) -> KvCacheEvent {
    KvCacheEvent::Removed {
        worker: WorkerWithRank::from_worker_id(1),
        block_hashes: hashes,
    }
}

/// Construct a `Clear` event.
fn clear() -> KvCacheEvent {
    KvCacheEvent::Clear {
        worker: WorkerWithRank::from_worker_id(1),
    }
}

/// Construct a backend identifier for tests.
fn backend(name: &str) -> BackendId {
    BackendId::new("test-region", name)
}

#[tokio::test]
async fn radix_tree_kv_event_lifecycle() {
    let tree = RadixTree::new();

    let backend_a = backend("a");
    let backend_b = backend("b");

    // 1) backend A stores [1, 2, 3], should fully hit on [1, 2, 3]
    tree.apply_event(backend_a.clone(), stored(vec![1, 2, 3]))
        .await
        .expect("applying Stored([1,2,3]) to backend A should succeed");
    let overlap_a_full = tree.find_matches(vec![1, 2, 3], backend_a.clone()).await;
    assert_eq!(
        overlap_a_full, 3,
        "backend A should match all 3 block hashes"
    );

    // 2) backend B stores [1, 2, 4], sharing the prefix [1, 2] with A
    tree.apply_event(backend_b.clone(), stored(vec![1, 2, 4]))
        .await
        .expect("applying Stored([1,2,4]) to backend B should succeed");

    // 2a) Query [1, 2, 3] against B: prefix [1, 2] hits, the 3rd block does not belong to B -> 2
    let overlap_b_partial = tree
        .find_matches(vec![1, 2, 3], backend_b.clone())
        .await;
    assert_eq!(
        overlap_b_partial, 2,
        "backend B should only match the prefix [1, 2] (2 blocks)"
    );

    // 2b) Query [1, 2, 4] against A: prefix [1, 2] hits, the 3rd block does not belong to A -> 2
    let overlap_a_partial = tree
        .find_matches(vec![1, 2, 4], backend_a.clone())
        .await;
    assert_eq!(
        overlap_a_partial, 2,
        "backend A should only match the prefix [1, 2] (2 blocks)"
    );

    // 3) Remove block 3 from backend A: nodes holding hash=3 in the tree should clear A's ownership
    tree.apply_event(backend_a.clone(), removed(vec![3]))
        .await
        .expect("applying Removed([3]) to backend A should succeed");
    let overlap_a_after_removed = tree
        .find_matches(vec![1, 2, 3], backend_a.clone())
        .await;
    assert_eq!(
        overlap_a_after_removed, 2,
        "after removing block 3, backend A's hit count on [1, 2, 3] should drop to 2"
    );

    // 4) Clear all ownership of backend B
    tree.apply_event(backend_b.clone(), clear())
        .await
        .expect("applying Clear to backend B should succeed");
    let overlap_b_after_clear = tree
        .find_matches(vec![1, 2, 4], backend_b.clone())
        .await;
    assert_eq!(
        overlap_b_after_clear, 0,
        "after Clear, backend B should no longer hit any blocks"
    );

    // 5) Additional verification: clearing B should not affect A's ownership
    let overlap_a_still = tree
        .find_matches(vec![1, 2], backend_a.clone())
        .await;
    assert_eq!(
        overlap_a_still, 2,
        "backend A's hit count on prefix [1, 2] should not be affected by B's Clear"
    );

    // Shut down the background thread to avoid leaking it at the end of the test
    tree.shutdown();
}

#[tokio::test]
async fn radix_tree_find_all_matches_aggregates_owners() {
    // This case verifies that `find_all_matches` can aggregate the hit count of each
    // backend along the prefix path.
    let tree = RadixTree::new();
    let backend_a = backend("a");
    let backend_b = backend("b");
    let backend_c = backend("c");

    // A stores [10, 20, 30], B stores [10, 20, 40], C stores [10, 50]
    tree.apply_event(backend_a.clone(), stored(vec![10, 20, 30]))
        .await
        .unwrap();
    tree.apply_event(backend_b.clone(), stored(vec![10, 20, 40]))
        .await
        .unwrap();
    tree.apply_event(backend_c.clone(), stored(vec![10, 50]))
        .await
        .unwrap();

    // Querying [10, 20, 30, 40] should yield:
    // - A: 3 (10, 20, 30)
    // - B: 2 (10, 20, the 3rd block 30 does not belong to B and interrupts)
    // - C: 1 (10, the 2nd block 20 does not belong to C and interrupts)
    let all = tree.find_all_matches(vec![10, 20, 30, 40]).await;
    assert_eq!(
        all.get(&backend_a).copied().unwrap_or(0),
        3,
        "A should hit 3 blocks"
    );
    assert_eq!(
        all.get(&backend_b).copied().unwrap_or(0),
        2,
        "B should hit the prefix [10, 20] (2 blocks)"
    );
    assert_eq!(
        all.get(&backend_c).copied().unwrap_or(0),
        1,
        "C should only hit [10] (1 block)"
    );

    tree.shutdown();
}
