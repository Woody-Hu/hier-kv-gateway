//! Bridge between [`MetadataStore`] and the cluster gossip engine.
//!
//! This module implements [`GossipHandler`] on top of [`MetadataStore`] so that
//! metadata received via gossip messages (`MetricsBroadcast`, `CkfDelta`,
//! `TopologyUpdate`, `SessionAffinityBroadcast`, `SyncRequest`/`SyncResponse`)
//! is applied to the local store instead of being dropped.
//!
//! It is deliberately kept in the main binary crate (rather than in
//! `hier-kv-gateway-cluster` or `hier-kv-gateway-metadata`) so neither of those
//! crates needs to take a dependency on the other; the bridge is the single
//! place that sees both sides.
//!
//! ## Versioning model
//!
//! Each metadata category (kv / model / load / topology / members) carries an
//! atomic version counter inside [`MetadataGossipHandler`]. The counter is
//! bumped whenever a gossip-driven write lands on the local store, so the
//! digest we advertise in Ping/Pong reflects "how much gossip has converged".
//!
//! Writes performed directly on `MetadataStore` by the local routing path
//! (e.g. `load_update` called by the connector metrics loop) do *not* bump
//! these counters automatically — that would require intercepting every write
//! API on `MetadataStore`. The trade-off is that purely-local changes do not
//! trigger SyncRequest from peers via the digest path; they still propagate
//! when the local instance explicitly broadcasts them. Callers that want the
//! digest to reflect a local change can call [`MetadataGossipHandler::bump_*`]
//! after the write.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

use hier_kv_gateway_cluster::gossip::GossipHandler;
use hier_kv_gateway_cluster::member::MemberList;
use hier_kv_gateway_cluster::messages::{MetaDigest, MetaEntry, MetaKey};
use hier_kv_gateway_core::ids::{BackendId, PoolId, RegionId, SessionId};
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::topology::LatencyEstimate;
use hier_kv_gateway_metadata::cuckoo_filter::{CkfDelta, CkfSnapshot};
use hier_kv_gateway_metadata::store::MetadataStore;

/// Predefined string keys used inside [`MetaEntry`].
///
/// These mirror [`MetaKey::as_str`] but are repeated here as `&'static str`
/// constants so the match arms below don't need to allocate.
mod keys {
    pub const KV_STATE: &str = "kv_state";
    pub const MODEL_STATE: &str = "model_state";
    pub const LOAD_STATE: &str = "load_state";
    pub const TOPOLOGY_STATE: &str = "topology_state";
    pub const MEMBERS: &str = "members";
}

/// [`GossipHandler`] implementation backed by a shared [`MetadataStore`].
///
/// See the module docs for the versioning model and the trade-offs around
/// local vs gossip-driven writes.
pub struct MetadataGossipHandler {
    store: Arc<MetadataStore>,
    /// Shared member list; currently unused by the handler itself but kept so
    /// future handlers can validate broadcast sources against the membership
    /// table without requiring callers to thread a separate handle through.
    #[allow(dead_code)]
    members: Arc<MemberList>,
    versions: VersionCounter,
}

struct VersionCounter {
    kv: AtomicU64,
    model: AtomicU64,
    load: AtomicU64,
    topology: AtomicU64,
    members: AtomicU64,
}

impl VersionCounter {
    fn new() -> Self {
        Self {
            kv: AtomicU64::new(0),
            model: AtomicU64::new(0),
            load: AtomicU64::new(0),
            topology: AtomicU64::new(0),
            members: AtomicU64::new(0),
        }
    }
}

impl MetadataGossipHandler {
    /// Wrap a `MetadataStore` + `MemberList` pair with version tracking.
    ///
    /// Both handles are held by `Arc` so the handler can be cheaply cloned
    /// into the gossip engine's background tasks.
    pub fn new(store: Arc<MetadataStore>, members: Arc<MemberList>) -> Self {
        Self {
            store,
            members,
            versions: VersionCounter::new(),
        }
    }

    // ----- Manual version bumps for local writes -----
    //
    // These are part of the public API so that the main binary (or future
    // connectors) can signal "I just mutated the store locally; please
    // advertise a new digest so peers pull a sync". They are currently
    // unused — `#[allow(dead_code)]` silences the warning until the main
    // binary wires them into its write paths.

    /// Bump the KV index version. Call after local `kv_apply_event` calls if
    /// the digest should reflect them.
    #[allow(dead_code)]
    pub fn bump_kv_version(&self) {
        self.versions.kv.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the model registry version.
    #[allow(dead_code)]
    pub fn bump_model_version(&self) {
        self.versions.model.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the load statistics version.
    #[allow(dead_code)]
    pub fn bump_load_version(&self) {
        self.versions.load.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the topology version.
    #[allow(dead_code)]
    pub fn bump_topology_version(&self) {
        self.versions.topology.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the member-list version.
    #[allow(dead_code)]
    pub fn bump_members_version(&self) {
        self.versions.members.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot the current backends + metrics into a JSON value for
    /// `load_state` sync responses.
    fn serialize_load_state(&self) -> serde_json::Value {
        let entries: Vec<serde_json::Value> = self
            .store
            .backends_all()
            .into_iter()
            .filter_map(|info| {
                let metrics = self.store.load_get_metrics(&info.id)?;
                Some(serde_json::json!({
                    "backend_id": info.id.to_string(),
                    "metrics": metrics,
                }))
            })
            .collect();
        serde_json::Value::Array(entries)
    }

    /// Snapshot all known regions into a JSON value for `topology_state`
    /// sync responses.
    fn serialize_topology_state(&self) -> serde_json::Value {
        let topology = self.store.topology();
        let entries: Vec<serde_json::Value> = topology
            .all_regions()
            .into_iter()
            .filter_map(|rid| {
                let info = topology.get_region(&rid)?;
                Some(serde_json::json!({
                    "region": info,
                }))
            })
            .collect();
        serde_json::Value::Array(entries)
    }

    /// Apply a `load_state` JSON payload produced by [`serialize_load_state`].
    fn apply_load_state(&self, value: &serde_json::Value) {
        let Some(arr) = value.as_array() else {
            debug!("load_state payload is not an array; ignoring");
            return;
        };
        for entry in arr {
            let Some(id_str) = entry.get("backend_id").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(backend_id) = BackendId::parse(id_str) else {
                debug!(id = %id_str, "load_state: malformed backend_id");
                continue;
            };
            let Some(metrics) = entry
                .get("metrics")
                .and_then(|v| serde_json::from_value::<BackendMetrics>(v.clone()).ok())
            else {
                debug!(id = %id_str, "load_state: malformed metrics");
                continue;
            };
            self.store.load_update(backend_id, metrics);
        }
        self.versions.load.fetch_add(1, Ordering::Relaxed);
    }

    /// Apply a `topology_state` JSON payload produced by
    /// [`serialize_topology_state`].
    fn apply_topology_state(&self, value: &serde_json::Value) {
        let Some(arr) = value.as_array() else {
            debug!("topology_state payload is not an array; ignoring");
            return;
        };
        for entry in arr {
            let Some(info) = entry
                .get("region")
                .and_then(|v| serde_json::from_value::<hier_kv_gateway_core::topology::RegionInfo>(v.clone()).ok())
            else {
                continue;
            };
            self.store.topo_add_region(info);
        }
        self.versions.topology.fetch_add(1, Ordering::Relaxed);
    }
}

#[async_trait]
impl GossipHandler for MetadataGossipHandler {
    fn current_meta_digest(&self) -> MetaDigest {
        MetaDigest {
            kv_version: self.versions.kv.load(Ordering::Relaxed),
            model_version: self.versions.model.load(Ordering::Relaxed),
            load_version: self.versions.load.load(Ordering::Relaxed),
            topology_version: self.versions.topology.load(Ordering::Relaxed),
            members_version: self.versions.members.load(Ordering::Relaxed),
        }
    }

    fn handle_sync_request(&self, keys: &[String]) -> Vec<MetaEntry> {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let (value, version) = match key.as_str() {
                keys::LOAD_STATE => {
                    (self.serialize_load_state(), self.versions.load.load(Ordering::Relaxed))
                }
                keys::TOPOLOGY_STATE => (
                    self.serialize_topology_state(),
                    self.versions.topology.load(Ordering::Relaxed),
                ),
                // KV/model/members state is structurally complex (RadixTree,
                // model registry, member table). Full serialization is left
                // as a follow-up; for now we report an empty payload so peers
                // at least learn the version.
                keys::KV_STATE => (
                    serde_json::Value::Null,
                    self.versions.kv.load(Ordering::Relaxed),
                ),
                keys::MODEL_STATE => (
                    serde_json::Value::Null,
                    self.versions.model.load(Ordering::Relaxed),
                ),
                keys::MEMBERS => (
                    serde_json::Value::Null,
                    self.versions.members.load(Ordering::Relaxed),
                ),
                _ => {
                    debug!(key = %key, "sync_request: unknown key");
                    continue;
                }
            };
            out.push(MetaEntry {
                key: key.clone(),
                value,
                version,
            });
        }
        out
    }

    fn apply_sync_response(&self, entries: &[MetaEntry]) {
        for entry in entries {
            match entry.key.as_str() {
                keys::LOAD_STATE => self.apply_load_state(&entry.value),
                keys::TOPOLOGY_STATE => self.apply_topology_state(&entry.value),
                keys::KV_STATE | keys::MODEL_STATE | keys::MEMBERS => {
                    // Received a version-only marker; no payload to apply yet.
                    debug!(key = %entry.key, version = entry.version, "sync_response: skipping payload-less entry");
                }
                _ => {
                    debug!(key = %entry.key, "sync_response: unknown key");
                }
            }
        }
    }

    fn handle_metrics_broadcast(
        &self,
        region: &RegionId,
        backends: &[(String, BackendMetrics)],
    ) {
        let mut updated = 0u64;
        for (id_str, metrics) in backends {
            let Some(backend_id) = BackendId::parse(id_str) else {
                debug!(id = %id_str, "metrics_broadcast: malformed backend_id");
                continue;
            };
            // Only accept metrics for backends that belong to the announced
            // region — guards against accidental cross-region crosstalk.
            if backend_id.region != *region {
                debug!(
                    backend = %id_str,
                    announced_region = %region,
                    backend_region = %backend_id.region,
                    "metrics_broadcast: region mismatch, skipping"
                );
                continue;
            }
            self.store.load_update(backend_id, metrics.clone());
            updated += 1;
        }
        if updated > 0 {
            self.versions.load.fetch_add(updated, Ordering::Relaxed);
        }
    }

    fn handle_ckf_barrier_snapshot(
        &self,
        pool: &PoolId,
        sequence: u64,
        buckets: &[u64],
        num_buckets: usize,
    ) {
        let consumer = self.store.ckf_consumer();
        // Lane assignment is the responsibility of the upper layer (e.g. the
        // KV relay). If we don't have a lane for this Region yet, the
        // snapshot is dropped — it will be re-sent after lane assignment.
        let Some(lane) = consumer.lane_of(&pool.region) else {
            debug!(
                region = %pool.region,
                sequence,
                "ckf_barrier_snapshot: no lane assigned for region, dropping"
            );
            return;
        };
        if buckets.len() != num_buckets {
            warn!(
                region = %pool.region,
                sequence,
                buckets_len = buckets.len(),
                num_buckets,
                "ckf_barrier_snapshot: bucket count mismatch, dropping"
            );
            return;
        }
        let snapshot = CkfSnapshot {
            sequence,
            buckets: buckets.to_vec(),
        };
        consumer.install_snapshot(lane, &snapshot);
        self.versions.kv.fetch_add(1, Ordering::Relaxed);
    }

    fn handle_ckf_delta(
        &self,
        pool: &PoolId,
        sequence: u64,
        prev_sequence: u64,
        dirty_buckets: &[(usize, u64)],
    ) {
        let consumer = self.store.ckf_consumer();
        let Some(lane) = consumer.lane_of(&pool.region) else {
            debug!(
                region = %pool.region,
                sequence,
                "ckf_delta: no lane assigned for region, dropping"
            );
            return;
        };
        let delta = CkfDelta {
            sequence,
            prev_sequence,
            buckets: dirty_buckets.to_vec(),
        };
        consumer.apply_delta(lane, &delta);
        self.versions.kv.fetch_add(1, Ordering::Relaxed);
    }

    fn handle_topology_update(
        &self,
        from: &RegionId,
        to: &RegionId,
        estimate: &LatencyEstimate,
    ) {
        self.store
            .topo_update_latency(from, to, estimate.clone());
        self.versions.topology.fetch_add(1, Ordering::Relaxed);
    }

    fn handle_session_affinity(
        &self,
        session: &SessionId,
        backend_region: &RegionId,
        backend_instance: &str,
    ) {
        let backend_id = BackendId::new(backend_region.clone(), backend_instance);
        // kv_overlap_at_route is not carried in the broadcast; we record 0
        // so the session is at least pinned. The next local routing decision
        // will refresh the overlap.
        self.store.session_set(session.clone(), backend_id, 0);
    }
}

/// Convenience helper: convert a [`MetaKey`] to its string form without
/// requiring the caller to import `MetaKey::as_str`.
#[allow(dead_code)]
fn meta_key_str(key: MetaKey) -> &'static str {
    key.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::backend::{
        BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
        ModelInstance, Protocol, Quantization,
    };
    use hier_kv_gateway_core::ids::IndexerDomainId;
    use hier_kv_gateway_core::metrics::LatencyStats;
    use hier_kv_gateway_core::topology::RegionInfo;

    fn sample_metrics(n: u64) -> BackendMetrics {
        BackendMetrics {
            active_requests: n,
            queue_depth: 0,
            active_decode_blocks: 0,
            active_prefill_tokens: 0,
            kv_used_blocks: n,
            kv_total_blocks: 100,
            gpu_utilization: 0.1 * n as f64,
            gpu_memory_used_mb: 0,
            gpu_memory_total_mb: 0,
            latency: LatencyStats {
                p50_ms: 1.0,
                p99_ms: 2.0,
                p999_ms: 3.0,
                sample_count: 1,
            },
            timestamp: 1,
        }
    }

    /// Register a minimal backend so `backends_all()` can enumerate it.
    /// `load_update` alone does not populate the backend registry.
    fn register_minimal_backend(store: &MetadataStore, id: &BackendId) {
        let info = BackendInfo {
            id: id.clone(),
            backend_type: BackendType::VllmEngine,
            endpoint: Endpoint {
                url: format!("http://{}", id),
                protocol: Protocol::Http,
            },
            models: vec![ModelInstance {
                model_name: "test-model".to_string(),
                model_architecture: "test".to_string(),
                quantization: Quantization::Fp16,
                max_context_len: 4096,
                supports_tool_calling: false,
                supports_streaming: true,
            }],
            region: id.region.clone(),
            indexer_domain: IndexerDomainId::new(0),
            capabilities: BackendCapabilities {
                supports_kv_events: true,
                supports_batching: true,
                max_batch_size: 32,
                gpu_count: 1,
                gpu_memory_gb: 24,
            },
            kv_config: KvConfig {
                block_size: 16,
                cache_namespace: "default".to_string(),
                max_kv_blocks: 1024,
            },
            status: BackendStatus::Healthy,
        };
        store.register_backend(info);
    }

    #[test]
    fn backend_id_parse_round_trip() {
        let b = BackendId::parse("r1/i1").unwrap();
        assert_eq!(b.region.as_str(), "r1");
        assert_eq!(b.instance.as_str(), "i1");
        assert!(BackendId::parse("no-slash").is_none());
        assert!(BackendId::parse("/empty-region").is_none());
        assert!(BackendId::parse("empty-instance/").is_none());
    }

    #[test]
    fn digest_starts_at_zero_and_bumps_on_metrics() {
        let store = Arc::new(MetadataStore::new());
        let members = Arc::new(MemberList::new());
        let h = MetadataGossipHandler::new(store, members);

        let d = h.current_meta_digest();
        assert_eq!(d.load_version, 0);

        h.handle_metrics_broadcast(
            &RegionId::new("r1"),
            &[("r1/i1".to_string(), sample_metrics(1))],
        );
        let d = h.current_meta_digest();
        assert_eq!(d.load_version, 1);
    }

    #[test]
    fn metrics_broadcast_skips_region_mismatch() {
        let store = Arc::new(MetadataStore::new());
        let members = Arc::new(MemberList::new());
        let h = MetadataGossipHandler::new(store.clone(), members);

        h.handle_metrics_broadcast(
            &RegionId::new("r1"),
            &[("r2/i1".to_string(), sample_metrics(1))],
        );
        // Should not bump load_version because of region mismatch.
        let d = h.current_meta_digest();
        assert_eq!(d.load_version, 0);
        // Backend should not have been registered.
        let b = BackendId::new("r2", "i1");
        assert!(store.load_get_metrics(&b).is_none());
    }

    #[test]
    fn load_state_sync_round_trip() {
        let store_a = Arc::new(MetadataStore::new());
        let store_b = Arc::new(MetadataStore::new());
        let members = Arc::new(MemberList::new());

        // Seed store_a with one registered backend + metrics.
        let b = BackendId::new("r1", "i1");
        register_minimal_backend(&store_a, &b);
        store_a.load_update(b.clone(), sample_metrics(7));

        let h_a = MetadataGossipHandler::new(store_a, members.clone());
        let h_b = MetadataGossipHandler::new(store_b.clone(), members);

        // A serializes load_state; B applies it.
        let entries = h_a.handle_sync_request(&[keys::LOAD_STATE.to_string()]);
        assert_eq!(entries.len(), 1);
        h_b.apply_sync_response(&entries);

        // B should now have the metrics.
        let m = store_b.load_get_metrics(&b).expect("metrics should be synced");
        assert_eq!(m.active_requests, 7);
    }

    #[test]
    fn topology_state_sync_round_trip() {
        let store_a = Arc::new(MetadataStore::new());
        let store_b = Arc::new(MetadataStore::new());
        let members = Arc::new(MemberList::new());

        let info = RegionInfo {
            id: RegionId::new("r-remote"),
            tier: hier_kv_gateway_core::ids::RegionTier::Cloud,
            geo: None,
            network_zone: "z1".to_string(),
            endpoints: vec![],
        };
        store_a.topo_add_region(info.clone());

        let h_a = MetadataGossipHandler::new(store_a, members.clone());
        let h_b = MetadataGossipHandler::new(store_b.clone(), members);

        let entries = h_a.handle_sync_request(&[keys::TOPOLOGY_STATE.to_string()]);
        assert_eq!(entries.len(), 1);
        h_b.apply_sync_response(&entries);

        let got = store_b.topo_get_region(&RegionId::new("r-remote"));
        assert!(got.is_some(), "region should be synced to store_b");
    }

    #[test]
    fn topology_update_bumps_version_and_applies() {
        let store = Arc::new(MetadataStore::new());
        let members = Arc::new(MemberList::new());
        let h = MetadataGossipHandler::new(store.clone(), members);

        let estimate = LatencyEstimate {
            rtt_p50_ms: 10.0,
            rtt_p99_ms: 20.0,
            bandwidth_mbps: 1000.0,
            last_updated_unix: 1,
        };
        h.handle_topology_update(&RegionId::new("r1"), &RegionId::new("r2"), &estimate);

        let rtt = store.topo_rtt_ms(&RegionId::new("r1"), &RegionId::new("r2"));
        assert_eq!(rtt, 10.0);
        let d = h.current_meta_digest();
        assert_eq!(d.topology_version, 1);
    }

    #[test]
    fn ckf_snapshot_without_lane_is_dropped() {
        let store = Arc::new(MetadataStore::new());
        let members = Arc::new(MemberList::new());
        let h = MetadataGossipHandler::new(store, members);

        let pool = PoolId {
            domain: hier_kv_gateway_core::ids::IndexerDomainId::new(0),
            region: RegionId::new("r-no-lane"),
        };
        h.handle_ckf_barrier_snapshot(&pool, 1, &[0u64; 4], 4);

        // Version should not bump since the snapshot was dropped.
        let d = h.current_meta_digest();
        assert_eq!(d.kv_version, 0);
    }
}
