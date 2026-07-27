//! Shared state store integration test.
//!
//! Verifies the `LocalSharedState` implementation of `SharedStateStore`:
//! - Instance registration, heartbeat, listing, deregistration
//! - Session affinity set/get with TTL
//! - Prefix dispatch history set/get
//! - Distributed lock acquire/release with holder checking
//! - Backend metrics set/get/get_all
//!
//! This test uses the in-memory `LocalSharedState` (no external dependencies).
//! The Redis-backed implementation is tested separately behind `#[ignore]`.

use std::time::Duration;

use hier_kv_gateway_cluster::shared_state::{
    InstanceEntry, LocalSharedState, PrefixDispatchEntry, SharedStateStore,
};
use hier_kv_gateway_core::ids::{BackendId, InstanceId, RegionId, SessionId};
use hier_kv_gateway_core::metrics::{BackendMetrics, LatencyStats};

fn instance_entry(id: &str, region: &str, addr: &str) -> InstanceEntry {
    InstanceEntry {
        instance_id: InstanceId::new(id),
        region: RegionId::new(region),
        addr: addr.to_string(),
        last_heartbeat_unix: 0,
    }
}

fn backend(region: &str, instance: &str) -> BackendId {
    BackendId::new(region, instance)
}

fn sample_metrics(active: u64) -> BackendMetrics {
    BackendMetrics {
        active_requests: active,
        queue_depth: active / 2,
        active_decode_blocks: 8,
        active_prefill_tokens: 128,
        kv_used_blocks: 40,
        kv_total_blocks: 100,
        gpu_utilization: 0.5,
        gpu_memory_used_mb: 20_000,
        gpu_memory_total_mb: 40_000,
        latency: LatencyStats {
            p50_ms: 100.0,
            p99_ms: 500.0,
            p999_ms: 800.0,
            sample_count: 100,
        },
        timestamp: 1_700_000_000,
    }
}

#[tokio::test]
async fn instance_register_list_and_deregister() {
    let store = LocalSharedState::new();
    assert_eq!(store.backend_name(), "local");

    let e1 = instance_entry("g1", "cloud-beijing", "10.0.0.1:8080");
    let e2 = instance_entry("g2", "edge-shanghai", "10.0.0.2:8080");

    store
        .register_instance(&e1, Duration::from_secs(60))
        .await
        .unwrap();
    store
        .register_instance(&e2, Duration::from_secs(60))
        .await
        .unwrap();

    let instances = store.list_instances().await.unwrap();
    assert_eq!(instances.len(), 2, "should list 2 registered instances");

    store.deregister_instance(&InstanceId::new("g1")).await.unwrap();
    let instances = store.list_instances().await.unwrap();
    assert_eq!(instances.len(), 1, "should list 1 after deregister");
    assert_eq!(instances[0].instance_id, InstanceId::new("g2"));
}

#[tokio::test]
async fn heartbeat_refreshes_ttl() {
    let store = LocalSharedState::new();
    let entry = instance_entry("g1", "cloud-beijing", "10.0.0.1:8080");

    // Register with a very short TTL
    store
        .register_instance(&entry, Duration::from_secs(1))
        .await
        .unwrap();

    // Heartbeat to refresh
    store
        .heartbeat(&InstanceId::new("g1"), Duration::from_secs(60))
        .await
        .unwrap();

    // Sleep past the original 1s TTL
    tokio::time::sleep(Duration::from_millis(1100)).await;

    // Instance should still be alive because heartbeat refreshed the TTL
    let instances = store.list_instances().await.unwrap();
    assert_eq!(instances.len(), 1, "heartbeat should have refreshed the TTL");
    assert!(instances[0].last_heartbeat_unix > 0);
}

#[tokio::test]
async fn session_affinity_set_and_get() {
    let store = LocalSharedState::new();
    let session = SessionId::new("sess-123");
    let backend_id = backend("cloud-beijing", "worker-0");

    // Initially no session affinity
    let result = store.session_get(&session).await.unwrap();
    assert!(result.is_none(), "session should not exist initially");

    // Set session affinity
    store
        .session_set(&session, &backend_id, Duration::from_secs(60))
        .await
        .unwrap();

    // Retrieve it
    let result = store.session_get(&session).await.unwrap();
    assert_eq!(
        result,
        Some(backend_id),
        "session should map to the correct backend"
    );
}

#[tokio::test]
async fn session_ttl_expiry() {
    let store = LocalSharedState::new();
    let session = SessionId::new("sess-expire");
    let backend_id = backend("cloud-beijing", "worker-0");

    store
        .session_set(&session, &backend_id, Duration::from_secs(1))
        .await
        .unwrap();

    // Should exist immediately
    assert!(store.session_get(&session).await.unwrap().is_some());

    // Wait for TTL to expire
    tokio::time::sleep(Duration::from_millis(1200)).await;

    // Should be gone
    assert!(
        store.session_get(&session).await.unwrap().is_none(),
        "session should expire after TTL"
    );
}

#[tokio::test]
async fn prefix_dispatch_history_set_and_get() {
    let store = LocalSharedState::new();
    let prefix_hash: u64 = 0x1234_5678;
    let entry = PrefixDispatchEntry {
        backend: backend("edge-shanghai", "worker-1"),
        dispatch_count: 5,
        last_dispatched_unix: 1700000000,
    };

    // Initially no entry
    let result = store.prefix_history_get(prefix_hash).await.unwrap();
    assert!(result.is_none());

    // Set the entry
    store
        .prefix_history_set(prefix_hash, &entry, Duration::from_secs(60))
        .await
        .unwrap();

    // Retrieve it
    let result = store.prefix_history_get(prefix_hash).await.unwrap();
    assert_eq!(result, Some(entry), "prefix history entry should match");
}

#[tokio::test]
async fn distributed_lock_acquire_and_release() {
    let store = LocalSharedState::new();
    let key = "leader-election";
    let holder1 = "gateway-1";
    let holder2 = "gateway-2";

    // holder1 acquires the lock
    let acquired1 = store
        .try_lock(key, holder1, Duration::from_secs(60))
        .await
        .unwrap();
    assert!(acquired1, "first acquire should succeed");

    // holder2 tries to acquire — should fail
    let acquired2 = store
        .try_lock(key, holder2, Duration::from_secs(60))
        .await
        .unwrap();
    assert!(!acquired2, "second acquire by different holder should fail");

    // holder2 tries to release — should not affect the lock
    store.unlock(key, holder2).await.unwrap();
    let still_locked = store
        .try_lock(key, holder2, Duration::from_secs(60))
        .await
        .unwrap();
    assert!(
        !still_locked,
        "unlock by non-holder should not release the lock"
    );

    // holder1 releases the lock
    store.unlock(key, holder1).await.unwrap();

    // holder2 can now acquire
    let acquired3 = store
        .try_lock(key, holder2, Duration::from_secs(60))
        .await
        .unwrap();
    assert!(acquired3, "acquire after release should succeed");
}

#[tokio::test]
async fn distributed_lock_ttl_expiry() {
    let store = LocalSharedState::new();
    let key = "ttl-lock";
    let holder = "gateway-1";

    // Acquire with 1-second TTL
    let acquired = store
        .try_lock(key, holder, Duration::from_secs(1))
        .await
        .unwrap();
    assert!(acquired);

    // Wait for TTL to expire
    tokio::time::sleep(Duration::from_millis(1200)).await;

    // Another holder should be able to acquire (stale lock reclamation)
    let acquired2 = store
        .try_lock(key, "gateway-2", Duration::from_secs(60))
        .await
        .unwrap();
    assert!(
        acquired2,
        "should acquire after the original lock's TTL expired"
    );
}

#[tokio::test]
async fn metrics_set_get_and_get_all() {
    let store = LocalSharedState::new();
    let b1 = backend("cloud-beijing", "worker-0");
    let b2 = backend("edge-shanghai", "worker-1");

    // Initially no metrics
    assert!(store.metrics_get(&b1).await.unwrap().is_none());

    // Set metrics for two backends
    let m1 = sample_metrics(10);
    let m2 = sample_metrics(20);
    store
        .metrics_set(&b1, &m1, Duration::from_secs(60))
        .await
        .unwrap();
    store
        .metrics_set(&b2, &m2, Duration::from_secs(60))
        .await
        .unwrap();

    // Get individual
    let result1 = store.metrics_get(&b1).await.unwrap();
    assert_eq!(result1.as_ref().unwrap().active_requests, 10);

    // Get all
    let all = store.metrics_get_all().await.unwrap();
    assert_eq!(all.len(), 2, "should have metrics for 2 backends");
    assert!(all.contains_key(&b1));
    assert!(all.contains_key(&b2));
}

#[tokio::test]
async fn metrics_ttl_expiry() {
    let store = LocalSharedState::new();
    let b1 = backend("cloud-beijing", "worker-0");
    let m1 = sample_metrics(5);

    store
        .metrics_set(&b1, &m1, Duration::from_secs(1))
        .await
        .unwrap();
    assert!(store.metrics_get(&b1).await.unwrap().is_some());

    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert!(
        store.metrics_get(&b1).await.unwrap().is_none(),
        "metrics should expire after TTL"
    );
}

#[tokio::test]
async fn multi_instance_session_sharing() {
    // Simulate two gateway instances sharing state via the same LocalSharedState.
    // (In production, this would be RedisSharedState backed by a Redis server.)
    let store = std::sync::Arc::new(LocalSharedState::new());

    // Instance 1 registers itself
    let e1 = instance_entry("g1", "cloud-beijing", "10.0.0.1:8080");
    store
        .register_instance(&e1, Duration::from_secs(60))
        .await
        .unwrap();

    // Instance 2 registers itself
    let e2 = instance_entry("g2", "edge-shanghai", "10.0.0.2:8080");
    store
        .register_instance(&e2, Duration::from_secs(60))
        .await
        .unwrap();

    // Both instances should see each other in the member list
    let members = store.list_instances().await.unwrap();
    assert_eq!(members.len(), 2);

    // Instance 1 sets a session affinity
    let session = SessionId::new("shared-session");
    let backend_id = backend("cloud-beijing", "worker-0");
    store
        .session_set(&session, &backend_id, Duration::from_secs(60))
        .await
        .unwrap();

    // Instance 2 reads the session affinity (shared state)
    let result = store.session_get(&session).await.unwrap();
    assert_eq!(
        result,
        Some(backend_id.clone()),
        "session affinity should be visible across instances via shared state"
    );

    // Instance 1 records a prefix dispatch
    let prefix_hash: u64 = 0xABCD_1234;
    let entry = PrefixDispatchEntry {
        backend: backend_id.clone(),
        dispatch_count: 1,
        last_dispatched_unix: 1700000000,
    };
    store
        .prefix_history_set(prefix_hash, &entry, Duration::from_secs(60))
        .await
        .unwrap();

    // Instance 2 reads the prefix dispatch history (shared state)
    let result = store.prefix_history_get(prefix_hash).await.unwrap();
    assert_eq!(
        result,
        Some(entry),
        "prefix dispatch history should be visible across instances via shared state"
    );
}
