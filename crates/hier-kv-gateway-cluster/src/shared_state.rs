//! Shared state coordination abstraction for multi-instance gateway clusters.
//!
//! Provides a trait [`SharedStateStore`] that abstracts cross-instance state
//! sharing (session affinity, routing history, member registry, distributed locks).
//! Two implementations are provided:
//! - [`LocalSharedState`]: in-memory, no network (default, single-instance or testing).
//! - [`RedisSharedState`]: backed by Redis (feature-gated, for multi-instance coordination).

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::{BackendId, InstanceId, RegionId, SessionId};
use hier_kv_gateway_core::metrics::BackendMetrics;

/// A gateway instance's registration info stored in shared state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstanceEntry {
    pub instance_id: InstanceId,
    pub region: RegionId,
    pub addr: String,
    pub last_heartbeat_unix: u64,
}

/// A prefix dispatch history entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PrefixDispatchEntry {
    pub backend: BackendId,
    pub dispatch_count: u64,
    pub last_dispatched_unix: u64,
}

/// Shared state store abstraction.
///
/// Implementations provide cross-instance coordination for:
/// - Member registry (instance discovery)
/// - Session affinity (shared across instances)
/// - Prefix dispatch history (for degradation routing)
/// - Distributed locks (leader election, coordination)
/// - Backend metrics sharing (cross-instance visibility)
#[async_trait]
pub trait SharedStateStore: Send + Sync {
    /// Register this instance in the shared store with a TTL.
    async fn register_instance(&self, entry: &InstanceEntry, ttl: Duration) -> Result<()>;

    /// Send a heartbeat, refreshing the instance's TTL.
    async fn heartbeat(&self, instance_id: &InstanceId, ttl: Duration) -> Result<()>;

    /// List all known alive instances.
    async fn list_instances(&self) -> Result<Vec<InstanceEntry>>;

    /// Deregister an instance.
    async fn deregister_instance(&self, instance_id: &InstanceId) -> Result<()>;

    /// Store a session affinity mapping with TTL.
    async fn session_set(
        &self,
        session: &SessionId,
        backend: &BackendId,
        ttl: Duration,
    ) -> Result<()>;

    /// Retrieve the backend for a session.
    async fn session_get(&self, session: &SessionId) -> Result<Option<BackendId>>;

    /// Store a prefix dispatch entry (keyed by prefix hash).
    async fn prefix_history_set(
        &self,
        prefix_hash: u64,
        entry: &PrefixDispatchEntry,
        ttl: Duration,
    ) -> Result<()>;

    /// Retrieve a prefix dispatch entry.
    async fn prefix_history_get(&self, prefix_hash: u64) -> Result<Option<PrefixDispatchEntry>>;

    /// Try to acquire a distributed lock. Returns true if acquired.
    async fn try_lock(&self, key: &str, holder: &str, ttl: Duration) -> Result<bool>;

    /// Release a distributed lock (only if held by `holder`).
    async fn unlock(&self, key: &str, holder: &str) -> Result<()>;

    /// Get backend metrics for a specific backend (shared across instances).
    async fn metrics_get(&self, backend: &BackendId) -> Result<Option<BackendMetrics>>;

    /// Set backend metrics (shared across instances).
    async fn metrics_set(
        &self,
        backend: &BackendId,
        metrics: &BackendMetrics,
        ttl: Duration,
    ) -> Result<()>;

    /// Get all backends' metrics from all instances.
    async fn metrics_get_all(&self) -> Result<HashMap<BackendId, BackendMetrics>>;

    /// Human-readable name for the backend (e.g. "local", "redis").
    fn backend_name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// LocalSharedState: in-memory implementation
// ---------------------------------------------------------------------------

/// In-memory shared state store for single-instance deployments and testing.
///
/// All entries are stored with an expiry unix timestamp; reads check the TTL
/// and treat expired entries as missing. Locks are stored without TTL expiry
/// in the map itself (they must be explicitly released), but the TTL is
/// tracked so that a stale lock can be reclaimed after expiry.
pub struct LocalSharedState {
    /// entry + expiry unix timestamp (seconds).
    instances: DashMap<InstanceId, (InstanceEntry, u64)>,
    /// backend + expiry unix timestamp (seconds).
    sessions: DashMap<SessionId, (BackendId, u64)>,
    /// entry + expiry unix timestamp (seconds).
    prefix_history: DashMap<u64, (PrefixDispatchEntry, u64)>,
    /// key -> (holder, expiry unix timestamp seconds).
    locks: DashMap<String, (String, u64)>,
    /// metrics + expiry unix timestamp (seconds).
    metrics: DashMap<BackendId, (BackendMetrics, u64)>,
}

impl LocalSharedState {
    /// Create a new empty in-memory shared state store.
    pub fn new() -> Self {
        Self {
            instances: DashMap::new(),
            sessions: DashMap::new(),
            prefix_history: DashMap::new(),
            locks: DashMap::new(),
            metrics: DashMap::new(),
        }
    }

    /// Current unix timestamp in seconds.
    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

impl Default for LocalSharedState {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SharedStateStore for LocalSharedState {
    async fn register_instance(&self, entry: &InstanceEntry, ttl: Duration) -> Result<()> {
        let expiry = Self::now_unix().saturating_add(ttl.as_secs());
        let stored = InstanceEntry {
            last_heartbeat_unix: Self::now_unix(),
            ..entry.clone()
        };
        self.instances.insert(entry.instance_id.clone(), (stored, expiry));
        Ok(())
    }

    async fn heartbeat(&self, instance_id: &InstanceId, ttl: Duration) -> Result<()> {
        let now = Self::now_unix();
        let expiry = now.saturating_add(ttl.as_secs());
        if let Some(mut e) = self.instances.get_mut(instance_id) {
            e.0.last_heartbeat_unix = now;
            e.1 = expiry;
        }
        Ok(())
    }

    async fn list_instances(&self) -> Result<Vec<InstanceEntry>> {
        let now = Self::now_unix();
        let mut out = Vec::new();
        for entry in self.instances.iter() {
            if entry.1 > now {
                out.push(entry.0.clone());
            }
        }
        Ok(out)
    }

    async fn deregister_instance(&self, instance_id: &InstanceId) -> Result<()> {
        self.instances.remove(instance_id);
        Ok(())
    }

    async fn session_set(
        &self,
        session: &SessionId,
        backend: &BackendId,
        ttl: Duration,
    ) -> Result<()> {
        let expiry = Self::now_unix().saturating_add(ttl.as_secs());
        self.sessions.insert(session.clone(), (backend.clone(), expiry));
        Ok(())
    }

    async fn session_get(&self, session: &SessionId) -> Result<Option<BackendId>> {
        let now = Self::now_unix();
        if let Some(entry) = self.sessions.get(session) {
            if entry.1 > now {
                return Ok(Some(entry.0.clone()));
            }
        }
        // Expired or missing: remove if present.
        self.sessions.remove(session);
        Ok(None)
    }

    async fn prefix_history_set(
        &self,
        prefix_hash: u64,
        entry: &PrefixDispatchEntry,
        ttl: Duration,
    ) -> Result<()> {
        let expiry = Self::now_unix().saturating_add(ttl.as_secs());
        self.prefix_history
            .insert(prefix_hash, (entry.clone(), expiry));
        Ok(())
    }

    async fn prefix_history_get(&self, prefix_hash: u64) -> Result<Option<PrefixDispatchEntry>> {
        let now = Self::now_unix();
        if let Some(entry) = self.prefix_history.get(&prefix_hash) {
            if entry.1 > now {
                return Ok(Some(entry.0.clone()));
            }
        }
        self.prefix_history.remove(&prefix_hash);
        Ok(None)
    }

    async fn try_lock(&self, key: &str, holder: &str, ttl: Duration) -> Result<bool> {
        let now = Self::now_unix();
        let expiry = now.saturating_add(ttl.as_secs());
        // Atomically acquire: entry API gives us single-key CAS semantics.
        match self.locks.entry(key.to_string()) {
            dashmap::mapref::entry::Entry::Occupied(mut o) => {
                if o.get().1 <= now {
                    // Stale lock — reclaim.
                    o.insert((holder.to_string(), expiry));
                    Ok(true)
                } else if o.get().0 == holder {
                    // Re-entrant refresh by same holder.
                    o.insert((holder.to_string(), expiry));
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            dashmap::mapref::entry::Entry::Vacant(v) => {
                v.insert((holder.to_string(), expiry));
                Ok(true)
            }
        }
    }

    async fn unlock(&self, key: &str, holder: &str) -> Result<()> {
        // Remove only if the current holder matches (atomic check-and-remove).
        self.locks.remove_if(key, |_k, (h, _)| h == holder);
        Ok(())
    }

    async fn metrics_get(&self, backend: &BackendId) -> Result<Option<BackendMetrics>> {
        let now = Self::now_unix();
        if let Some(entry) = self.metrics.get(backend) {
            if entry.1 > now {
                return Ok(Some(entry.0.clone()));
            }
        }
        self.metrics.remove(backend);
        Ok(None)
    }

    async fn metrics_set(
        &self,
        backend: &BackendId,
        metrics: &BackendMetrics,
        ttl: Duration,
    ) -> Result<()> {
        let expiry = Self::now_unix().saturating_add(ttl.as_secs());
        self.metrics.insert(backend.clone(), (metrics.clone(), expiry));
        Ok(())
    }

    async fn metrics_get_all(&self) -> Result<HashMap<BackendId, BackendMetrics>> {
        let now = Self::now_unix();
        let mut out = HashMap::new();
        for entry in self.metrics.iter() {
            if entry.1 > now {
                out.insert(entry.key().clone(), entry.0.clone());
            }
        }
        Ok(out)
    }

    fn backend_name(&self) -> &'static str {
        "local"
    }
}

// ---------------------------------------------------------------------------
// RedisSharedState: Redis-backed implementation (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "redis")]
mod redis_impl {
    use super::*;
    use hier_kv_gateway_core::error::HierKvGatewayError;

    /// Key prefixes used in the Redis keyspace.
    const INSTANCE_PREFIX: &str = "hier_kv_gateway:instance:";
    const SESSION_PREFIX: &str = "hier_kv_gateway:session:";
    const PREFIX_HISTORY_PREFIX: &str = "hier_kv_gateway:prefix:";
    const LOCK_PREFIX: &str = "hier_kv_gateway:lock:";
    const METRICS_PREFIX: &str = "hier_kv_gateway:metrics:";

    /// Lua script for safe lock release: delete only if the current holder matches.
    const UNLOCK_SCRIPT: &str = r#"
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"#;

    /// Wrapper that bundles a [`BackendId`] with its [`BackendMetrics`] so that
    /// `metrics_get_all` can recover the backend identity from the stored value
    /// rather than parsing it out of the Redis key.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct StoredMetrics {
        backend: BackendId,
        metrics: BackendMetrics,
    }

    /// Redis-backed shared state store for multi-instance coordination.
    ///
    /// Uses a [`redis::aio::MultiplexedConnection`] which is cheap to clone and
    /// safe for concurrent use across tasks. Each method clones the connection
    /// to obtain a `&mut` handle as required by the redis async command API.
    pub struct RedisSharedState {
        conn: redis::aio::MultiplexedConnection,
    }

    impl RedisSharedState {
        /// Connect to a Redis server at the given URL (e.g. `redis://127.0.0.1:6379`).
        pub async fn connect(url: &str) -> Result<Self> {
            let client = redis::Client::open(url)
                .map_err(|e| HierKvGatewayError::Internal(format!("Redis connect failed: {e}")))?;
            let conn = client
                .get_multiplexed_async_connection()
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!(
                    "Redis multiplexed connection failed: {e}"
                )))?;
            Ok(Self { conn })
        }

        /// Build the Redis key for an instance entry.
        fn instance_key(instance_id: &InstanceId) -> String {
            format!("{INSTANCE_PREFIX}{}", instance_id.as_str())
        }

        /// Build the Redis key for a session affinity mapping.
        fn session_key(session: &SessionId) -> String {
            format!("{SESSION_PREFIX}{}", session.as_str())
        }

        /// Build the Redis key for a prefix dispatch entry.
        fn prefix_key(prefix_hash: u64) -> String {
            format!("{PREFIX_HISTORY_PREFIX}{prefix_hash}")
        }

        /// Build the Redis key for a distributed lock.
        fn lock_key(key: &str) -> String {
            format!("{LOCK_PREFIX}{key}")
        }

        /// Build the Redis key for backend metrics.
        fn metrics_key(backend: &BackendId) -> String {
            format!("{METRICS_PREFIX}{backend}")
        }

        /// SCAN a keyspace pattern and return all matching keys.
        async fn scan_keys(
            &self,
            pattern: &str,
        ) -> Result<Vec<String>> {
            let mut conn = self.conn.clone();
            let mut cursor: u64 = 0;
            let mut keys = Vec::new();
            loop {
                let (next_cursor, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg(pattern)
                    .arg("COUNT")
                    .arg(100)
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| {
                        HierKvGatewayError::Internal(format!("redis SCAN failed: {e}"))
                    })?;
                keys.extend(batch);
                cursor = next_cursor;
                if cursor == 0 {
                    break;
                }
            }
            Ok(keys)
        }
    }

    #[async_trait]
    impl SharedStateStore for RedisSharedState {
        async fn register_instance(&self, entry: &InstanceEntry, ttl: Duration) -> Result<()> {
            let mut conn = self.conn.clone();
            let key = Self::instance_key(&entry.instance_id);
            let value = serde_json::to_string(entry)
                .map_err(|e| HierKvGatewayError::Internal(format!("serialize instance: {e}")))?;
            let _: () = redis::AsyncCommands::set_ex(
                &mut conn,
                key,
                value,
                ttl.as_secs().max(1),
            )
            .await
            .map_err(|e| HierKvGatewayError::Internal(format!("redis set_ex instance: {e}")))?;
            Ok(())
        }

        async fn heartbeat(&self, instance_id: &InstanceId, ttl: Duration) -> Result<()> {
            let mut conn = self.conn.clone();
            let key = Self::instance_key(instance_id);
            // Refresh TTL; if the key is gone, re-registration is required by the caller.
            let refreshed: bool = redis::AsyncCommands::expire(
                &mut conn,
                &key,
                ttl.as_secs().max(1) as i64,
            )
            .await
            .map_err(|e| HierKvGatewayError::Internal(format!("redis expire instance: {e}")))?;
            if !refreshed {
                return Err(HierKvGatewayError::NotFound(format!(
                    "instance not found in shared state: {instance_id}"
                )));
            }
            // Also bump the heartbeat timestamp inside the stored JSON.
            let raw: Option<String> = redis::AsyncCommands::get(&mut conn, &key)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis get instance: {e}")))?;
            if let Some(raw) = raw {
                if let Ok(mut entry) = serde_json::from_str::<InstanceEntry>(&raw) {
                    entry.last_heartbeat_unix = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    if let Ok(json) = serde_json::to_string(&entry) {
                        let _: std::result::Result<(), _> =
                            redis::AsyncCommands::set_ex(&mut conn, &key, json, ttl.as_secs().max(1))
                                .await;
                    }
                }
            }
            Ok(())
        }

        async fn list_instances(&self) -> Result<Vec<InstanceEntry>> {
            let keys = self.scan_keys(&format!("{INSTANCE_PREFIX}*")).await?;
            if keys.is_empty() {
                return Ok(Vec::new());
            }
            let mut conn = self.conn.clone();
            let values: Vec<Option<String>> = redis::AsyncCommands::mget(&mut conn, keys)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis mget instances: {e}")))?;
            let mut out = Vec::new();
            for v in values.into_iter().flatten() {
                if let Ok(entry) = serde_json::from_str::<InstanceEntry>(&v) {
                    out.push(entry);
                }
            }
            Ok(out)
        }

        async fn deregister_instance(&self, instance_id: &InstanceId) -> Result<()> {
            let mut conn = self.conn.clone();
            let key = Self::instance_key(instance_id);
            let _: () = redis::AsyncCommands::del(&mut conn, key)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis del instance: {e}")))?;
            Ok(())
        }

        async fn session_set(
            &self,
            session: &SessionId,
            backend: &BackendId,
            ttl: Duration,
        ) -> Result<()> {
            let mut conn = self.conn.clone();
            let key = Self::session_key(session);
            let value = serde_json::to_string(backend)
                .map_err(|e| HierKvGatewayError::Internal(format!("serialize backend: {e}")))?;
            let _: () = redis::AsyncCommands::set_ex(
                &mut conn,
                key,
                value,
                ttl.as_secs().max(1),
            )
            .await
            .map_err(|e| HierKvGatewayError::Internal(format!("redis set_ex session: {e}")))?;
            Ok(())
        }

        async fn session_get(&self, session: &SessionId) -> Result<Option<BackendId>> {
            let mut conn = self.conn.clone();
            let key = Self::session_key(session);
            let raw: Option<String> = redis::AsyncCommands::get(&mut conn, key)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis get session: {e}")))?;
            match raw {
                Some(s) => {
                    let backend: BackendId = serde_json::from_str(&s).map_err(|e| {
                        HierKvGatewayError::Internal(format!("deserialize backend: {e}"))
                    })?;
                    Ok(Some(backend))
                }
                None => Ok(None),
            }
        }

        async fn prefix_history_set(
            &self,
            prefix_hash: u64,
            entry: &PrefixDispatchEntry,
            ttl: Duration,
        ) -> Result<()> {
            let mut conn = self.conn.clone();
            let key = Self::prefix_key(prefix_hash);
            let value = serde_json::to_string(entry).map_err(|e| {
                HierKvGatewayError::Internal(format!("serialize prefix entry: {e}"))
            })?;
            let _: () = redis::AsyncCommands::set_ex(
                &mut conn,
                key,
                value,
                ttl.as_secs().max(1),
            )
            .await
            .map_err(|e| HierKvGatewayError::Internal(format!("redis set_ex prefix: {e}")))?;
            Ok(())
        }

        async fn prefix_history_get(&self, prefix_hash: u64) -> Result<Option<PrefixDispatchEntry>> {
            let mut conn = self.conn.clone();
            let key = Self::prefix_key(prefix_hash);
            let raw: Option<String> = redis::AsyncCommands::get(&mut conn, key)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis get prefix: {e}")))?;
            match raw {
                Some(s) => {
                    let entry: PrefixDispatchEntry = serde_json::from_str(&s).map_err(|e| {
                        HierKvGatewayError::Internal(format!("deserialize prefix entry: {e}"))
                    })?;
                    Ok(Some(entry))
                }
                None => Ok(None),
            }
        }

        async fn try_lock(&self, key: &str, holder: &str, ttl: Duration) -> Result<bool> {
            let mut conn = self.conn.clone();
            let redis_key = Self::lock_key(key);
            // SET key holder NX EX ttl -> returns "OK" if acquired, nil otherwise.
            let result: Option<String> = redis::cmd("SET")
                .arg(redis_key)
                .arg(holder)
                .arg("NX")
                .arg("EX")
                .arg(ttl.as_secs().max(1))
                .query_async(&mut conn)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis try_lock: {e}")))?;
            Ok(result.is_some())
        }

        async fn unlock(&self, key: &str, holder: &str) -> Result<()> {
            let mut conn = self.conn.clone();
            let redis_key = Self::lock_key(key);
            let _: i64 = redis::cmd("EVAL")
                .arg(UNLOCK_SCRIPT)
                .arg(1)
                .arg(redis_key)
                .arg(holder)
                .query_async(&mut conn)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis unlock: {e}")))?;
            Ok(())
        }

        async fn metrics_get(&self, backend: &BackendId) -> Result<Option<BackendMetrics>> {
            let mut conn = self.conn.clone();
            let key = Self::metrics_key(backend);
            let raw: Option<String> = redis::AsyncCommands::get(&mut conn, key)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis get metrics: {e}")))?;
            match raw {
                Some(s) => {
                    let stored: StoredMetrics = serde_json::from_str(&s).map_err(|e| {
                        HierKvGatewayError::Internal(format!("deserialize metrics: {e}"))
                    })?;
                    Ok(Some(stored.metrics))
                }
                None => Ok(None),
            }
        }

        async fn metrics_set(
            &self,
            backend: &BackendId,
            metrics: &BackendMetrics,
            ttl: Duration,
        ) -> Result<()> {
            let mut conn = self.conn.clone();
            let key = Self::metrics_key(backend);
            let stored = StoredMetrics {
                backend: backend.clone(),
                metrics: metrics.clone(),
            };
            let value = serde_json::to_string(&stored)
                .map_err(|e| HierKvGatewayError::Internal(format!("serialize metrics: {e}")))?;
            let _: () = redis::AsyncCommands::set_ex(
                &mut conn,
                key,
                value,
                ttl.as_secs().max(1),
            )
            .await
            .map_err(|e| HierKvGatewayError::Internal(format!("redis set_ex metrics: {e}")))?;
            Ok(())
        }

        async fn metrics_get_all(&self) -> Result<HashMap<BackendId, BackendMetrics>> {
            let keys = self.scan_keys(&format!("{METRICS_PREFIX}*")).await?;
            if keys.is_empty() {
                return Ok(HashMap::new());
            }
            let mut conn = self.conn.clone();
            let values: Vec<Option<String>> = redis::AsyncCommands::mget(&mut conn, keys)
                .await
                .map_err(|e| HierKvGatewayError::Internal(format!("redis mget metrics: {e}")))?;
            let mut out = HashMap::new();
            for v in values.into_iter().flatten() {
                if let Ok(stored) = serde_json::from_str::<StoredMetrics>(&v) {
                    out.insert(stored.backend, stored.metrics);
                }
            }
            Ok(out)
        }

        fn backend_name(&self) -> &'static str {
            "redis"
        }
    }
}

#[cfg(feature = "redis")]
pub use redis_impl::RedisSharedState;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::metrics::BackendMetrics;
    use std::time::Duration;

    fn sample_instance(id: &str) -> InstanceEntry {
        InstanceEntry {
            instance_id: InstanceId::new(id),
            region: RegionId::new("us-east-1"),
            addr: format!("127.0.0.1:7000"),
            last_heartbeat_unix: 0,
        }
    }

    fn sample_backend() -> BackendId {
        BackendId::new("us-east-1", "worker-0")
    }

    fn sample_metrics() -> BackendMetrics {
        BackendMetrics {
            active_requests: 4,
            queue_depth: 2,
            active_decode_blocks: 8,
            active_prefill_tokens: 128,
            kv_used_blocks: 40,
            kv_total_blocks: 100,
            gpu_utilization: 0.5,
            gpu_memory_used_mb: 20_000,
            gpu_memory_total_mb: 40_000,
            latency: hier_kv_gateway_core::metrics::LatencyStats {
                p50_ms: 10.0,
                p99_ms: 50.0,
                p999_ms: 80.0,
                sample_count: 1_000,
            },
            timestamp: 1_700_000_000,
        }
    }

    #[tokio::test]
    async fn test_instance_register_and_list() {
        let store = LocalSharedState::new();
        let entry = sample_instance("inst-1");
        store
            .register_instance(&entry, Duration::from_secs(60))
            .await
            .unwrap();
        let list = store.list_instances().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].instance_id, entry.instance_id);
        // Heartbeat timestamp should have been set on register.
        assert!(list[0].last_heartbeat_unix > 0);
    }

    #[tokio::test]
    async fn test_instance_ttl_expiry() {
        let store = LocalSharedState::new();
        let entry = sample_instance("inst-ttl");
        store
            .register_instance(&entry, Duration::from_millis(100))
            .await
            .unwrap();
        // The local store tracks TTL in whole seconds; 100ms rounds down to 0,
        // so the entry is immediately expired. Verify with a 1-second TTL instead.
        let entry2 = sample_instance("inst-ttl-2");
        store
            .register_instance(&entry2, Duration::from_secs(1))
            .await
            .unwrap();
        // Sleep long enough for the entry to expire.
        tokio::time::sleep(Duration::from_millis(1100)).await;
        let list = store.list_instances().await.unwrap();
        assert!(
            list.iter().all(|e| e.instance_id != entry2.instance_id),
            "instance should have expired"
        );
    }

    #[tokio::test]
    async fn test_session_set_get() {
        let store = LocalSharedState::new();
        let session = SessionId::new("sess-1");
        let backend = sample_backend();
        store
            .session_set(&session, &backend, Duration::from_secs(60))
            .await
            .unwrap();
        let got = store.session_get(&session).await.unwrap();
        assert_eq!(got, Some(backend.clone()));
        // Unknown session returns None.
        let got = store.session_get(&SessionId::new("nope")).await.unwrap();
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn test_prefix_history() {
        let store = LocalSharedState::new();
        let entry = PrefixDispatchEntry {
            backend: sample_backend(),
            dispatch_count: 7,
            last_dispatched_unix: 1234,
        };
        store
            .prefix_history_set(42, &entry, Duration::from_secs(60))
            .await
            .unwrap();
        let got = store.prefix_history_get(42).await.unwrap();
        assert_eq!(got.as_ref().unwrap().dispatch_count, 7);
        assert_eq!(got.as_ref().unwrap().backend, sample_backend());
        // Unknown prefix returns None.
        let got = store.prefix_history_get(999).await.unwrap();
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn test_lock_acquire_and_release() {
        let store = LocalSharedState::new();
        // Acquire.
        let acquired = store
            .try_lock("leader", "node-a", Duration::from_secs(60))
            .await
            .unwrap();
        assert!(acquired);
        // Second acquire by a different holder fails.
        let acquired2 = store
            .try_lock("leader", "node-b", Duration::from_secs(60))
            .await
            .unwrap();
        assert!(!acquired2);
        // Release by the wrong holder does nothing.
        store.unlock("leader", "node-b").await.unwrap();
        let acquired3 = store
            .try_lock("leader", "node-c", Duration::from_secs(60))
            .await
            .unwrap();
        assert!(!acquired3);
        // Release by the correct holder.
        store.unlock("leader", "node-a").await.unwrap();
        // Now another holder can acquire.
        let acquired4 = store
            .try_lock("leader", "node-c", Duration::from_secs(60))
            .await
            .unwrap();
        assert!(acquired4);
    }

    #[tokio::test]
    async fn test_metrics_set_get() {
        let store = LocalSharedState::new();
        let backend = sample_backend();
        let metrics = sample_metrics();
        store
            .metrics_set(&backend, &metrics, Duration::from_secs(60))
            .await
            .unwrap();
        let got = store.metrics_get(&backend).await.unwrap();
        assert_eq!(got.as_ref().unwrap(), &metrics);
        // metrics_get_all should include the entry.
        let all = store.metrics_get_all().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all.get(&backend), Some(&metrics));
    }

    #[tokio::test]
    async fn test_deregister_instance() {
        let store = LocalSharedState::new();
        let entry = sample_instance("inst-del");
        store
            .register_instance(&entry, Duration::from_secs(60))
            .await
            .unwrap();
        assert_eq!(store.list_instances().await.unwrap().len(), 1);
        store
            .deregister_instance(&entry.instance_id)
            .await
            .unwrap();
        assert_eq!(store.list_instances().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_heartbeat_refreshes_ttl() {
        let store = LocalSharedState::new();
        let entry = sample_instance("inst-hb");
        store
            .register_instance(&entry, Duration::from_secs(1))
            .await
            .unwrap();
        // Sleep half the TTL, then heartbeat to refresh.
        tokio::time::sleep(Duration::from_millis(500)).await;
        store
            .heartbeat(&entry.instance_id, Duration::from_secs(5))
            .await
            .unwrap();
        // Sleep past the original TTL — the entry should still be alive.
        tokio::time::sleep(Duration::from_millis(1100)).await;
        let list = store.list_instances().await.unwrap();
        assert!(list.iter().any(|e| e.instance_id == entry.instance_id));
    }
}

// Redis-backed tests require a running Redis server and the `redis` feature.
// Run with: cargo test -p hier-kv-gateway-cluster --features redis -- --ignored
#[cfg(all(test, feature = "redis"))]
mod redis_tests {
    use super::*;
    use std::time::Duration;

    fn redis_url() -> String {
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
    }

    fn sample_instance(id: &str) -> InstanceEntry {
        InstanceEntry {
            instance_id: InstanceId::new(id),
            region: RegionId::new("us-east-1"),
            addr: format!("127.0.0.1:7000"),
            last_heartbeat_unix: 0,
        }
    }

    fn sample_backend() -> BackendId {
        BackendId::new("us-east-1", "worker-0")
    }

    #[tokio::test]
    #[ignore]
    async fn redis_instance_register_list_deregister() {
        let store = RedisSharedState::connect(&redis_url()).await.unwrap();
        let entry = sample_instance("redis-inst-1");
        store
            .register_instance(&entry, Duration::from_secs(60))
            .await
            .unwrap();
        let list = store.list_instances().await.unwrap();
        assert!(list.iter().any(|e| e.instance_id == entry.instance_id));
        store
            .deregister_instance(&entry.instance_id)
            .await
            .unwrap();
        let list = store.list_instances().await.unwrap();
        assert!(!list.iter().any(|e| e.instance_id == entry.instance_id));
    }

    #[tokio::test]
    #[ignore]
    async fn redis_session_set_get() {
        let store = RedisSharedState::connect(&redis_url()).await.unwrap();
        let session = SessionId::new("redis-sess-1");
        let backend = sample_backend();
        store
            .session_set(&session, &backend, Duration::from_secs(60))
            .await
            .unwrap();
        let got = store.session_get(&session).await.unwrap();
        assert_eq!(got, Some(backend));
    }

    #[tokio::test]
    #[ignore]
    async fn redis_lock_acquire_release() {
        let store = RedisSharedState::connect(&redis_url()).await.unwrap();
        let acquired = store
            .try_lock("redis-leader", "node-a", Duration::from_secs(30))
            .await
            .unwrap();
        assert!(acquired);
        let acquired2 = store
            .try_lock("redis-leader", "node-b", Duration::from_secs(30))
            .await
            .unwrap();
        assert!(!acquired2);
        store.unlock("redis-leader", "node-a").await.unwrap();
        let acquired3 = store
            .try_lock("redis-leader", "node-c", Duration::from_secs(30))
            .await
            .unwrap();
        assert!(acquired3);
        store.unlock("redis-leader", "node-c").await.unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn redis_metrics_set_get() {
        let store = RedisSharedState::connect(&redis_url()).await.unwrap();
        let backend = sample_backend();
        let metrics = BackendMetrics {
            active_requests: 1,
            ..Default::default()
        };
        store
            .metrics_set(&backend, &metrics, Duration::from_secs(60))
            .await
            .unwrap();
        let got = store.metrics_get(&backend).await.unwrap();
        assert_eq!(got.as_ref().unwrap(), &metrics);
    }
}
