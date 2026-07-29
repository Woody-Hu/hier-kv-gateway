//! Cluster message types and metadata digest / entry definitions.
//!
//! Refers to the Gossip message type design in architecture doc §5.1:
//! - `Ping/Pong` carry the sender's metadata digest, used for heartbeat and version synchronization.
//! - `Meet` is used for a new instance to join the cluster; the receiver adds it to the member list and broadcasts propagation.
//! - `SyncRequest/SyncResponse` are used to pull metadata entries by key when the version lags behind.
//! - `CkfBarrierSnapshot/CkfDelta` are used for cross-Region KV projection publishing (two-phase publishing).
//! - `MetricsBroadcast` is used to broadcast backend load metrics.
//! - `TopologyUpdate` is used to synchronize cross-Region latency estimates.
//! - `SessionAffinityBroadcast` is used to share session affinity routing across instances.
//!
//! All messages use the `type` field as an internal tag, with `snake_case` naming,
//! to maintain stable compatibility across multiple transport serialization formats
//! such as JSON / msgpack.

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use hier_kv_gateway_core::ids::{InstanceId, PoolId, RegionId, SessionId};
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::topology::LatencyEstimate;

/// Compact payload for backend load metrics.
///
/// Encodes `Vec<(backend_id_string, BackendMetrics)>` using postcard (binary)
/// and wraps it in a versioned envelope so future delta-encoded variants can
/// be added without breaking older peers.
///
/// Wire format in JSON: `{"v": 1, "data": "<base64-postcard>"}`
/// Future delta variant: `{"v": 2, "data": "<base64-postcard-delta>"}`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoadPayload {
    /// Encoding version. 1 = full postcard payload.
    pub v: u8,
    /// Base64-encoded postcard bytes of `Vec<(String, BackendMetrics)>`.
    pub data: String,
}

impl LoadPayload {
    /// Encoding version for the current full-payload format.
    pub const VERSION_FULL: u8 = 1;

    /// Encode a list of (backend_id, metrics) into a compact `LoadPayload`.
    pub fn encode_full(backends: &[(String, BackendMetrics)]) -> Self {
        let bytes = postcard::to_allocvec(backends)
            .expect("postcard serialization of BackendMetrics cannot fail");
        Self {
            v: Self::VERSION_FULL,
            data: base64::engine::general_purpose::STANDARD.encode(&bytes),
        }
    }

    /// Decode the payload back into `Vec<(String, BackendMetrics)>`.
    ///
    /// Returns `None` if the version is unknown or decoding fails, allowing
    /// the receiver to gracefully skip unsupported payloads.
    pub fn decode(&self) -> Option<Vec<(String, BackendMetrics)>> {
        match self.v {
            Self::VERSION_FULL => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(&self.data)
                    .ok()?;
                postcard::from_bytes::<Vec<(String, BackendMetrics)>>(&bytes).ok()
            }
            // Future delta variant would go here.
            _ => None,
        }
    }
}

/// Unified cluster message envelope.
///
/// Serialized with the `type` field as an internal tag (internally tagged),
/// and uses `snake_case` naming to match the architecture doc conventions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClusterMessage {
    /// Heartbeat Ping, carrying the sender's metadata digest.
    Ping {
        /// Sender instance identifier.
        sender: InstanceId,
        /// Region where the sender resides.
        region: RegionId,
        /// Sender metadata digest (used for version comparison).
        meta_digest: MetaDigest,
        /// Send timestamp (Unix milliseconds).
        timestamp: u64,
    },
    /// Heartbeat Pong, replying to the peer of a Ping, and carries the latest digest of itself.
    Pong {
        /// Sender instance identifier.
        sender: InstanceId,
        /// Region where the sender resides.
        region: RegionId,
        /// Sender metadata digest.
        meta_digest: MetaDigest,
        /// Send timestamp (Unix milliseconds).
        timestamp: u64,
    },
    /// New instance join request.
    ///
    /// Sent by a newly joining instance to a known seed node; the receiver should add the sender
    /// to the member list and propagate it in subsequent Gossip rounds.
    Meet {
        /// Sender instance identifier.
        sender: InstanceId,
        /// Region where the sender resides.
        region: RegionId,
        /// Externally reachable address of the sender (host:port).
        addr: String,
    },
    /// Metadata sync request.
    ///
    /// The receiver returns the corresponding [`MetaEntry`] for each [`MetaKey`] string name
    /// listed in `keys`.
    SyncRequest {
        /// Sender instance identifier.
        sender: InstanceId,
        /// List of metadata key names to sync.
        keys: Vec<String>,
    },
    /// Metadata sync response, containing all entries requested by the requester.
    SyncResponse {
        /// List of metadata entries returned.
        entries: Vec<MetaEntry>,
    },
    /// CKF full snapshot publish (barrier snapshot).
    ///
    /// Published by the KV Relay on first publish or after reconnection; the receiver should
    /// install it into the corresponding Region's lane.
    CkfBarrierSnapshot {
        /// The pool this CKF belongs to.
        pool: PoolId,
        /// Snapshot sequence number.
        sequence: u64,
        /// Current values of all buckets in the lane.
        buckets: Vec<u64>,
        /// Number of buckets in the lane.
        num_buckets: usize,
    },
    /// CKF delta publish (sequenced delta).
    ///
    /// Contains only the buckets changed since the previous sequence number; the receiver writes
    /// them to the corresponding lane by (idx, value).
    CkfDelta {
        /// The pool this CKF belongs to.
        pool: PoolId,
        /// Sequence number this delta corresponds to.
        sequence: u64,
        /// Previous sequence number this delta is based on.
        prev_sequence: u64,
        /// List of changed buckets: (bucket_idx, new_value).
        dirty_buckets: Vec<(usize, u64)>,
    },
    /// Backend load metrics broadcast.
    ///
    /// `payload` is a compact binary-encoded [`LoadPayload`] (versioned envelope +
    /// base64-wrapped postcard bytes) so peers that don't understand the version
    /// can skip it gracefully. The [`GossipHandler::handle_metrics_broadcast`]
    /// implementation decodes it back into `Vec<(String, BackendMetrics)>`.
    MetricsBroadcast {
        /// Region the metrics belong to.
        region: RegionId,
        /// Compact payload of `(backend_id, metrics)` pairs.
        payload: LoadPayload,
    },
    /// Cross-Region topology latency update.
    ///
    /// Broadcast by an instance that perceives an RTT change; the receiver updates its local
    /// [`LatencyMatrix`].
    TopologyUpdate {
        /// Source Region.
        from: RegionId,
        /// Target Region.
        to: RegionId,
        /// Latency estimate between the two Regions.
        estimate: LatencyEstimate,
    },
    /// Session affinity routing broadcast.
    ///
    /// Broadcast to other instances in the cluster after an instance routes a session to a
    /// specified backend, so that subsequent requests hit the same backend.
    SessionAffinityBroadcast {
        /// Session identifier.
        session: SessionId,
        /// Region of the hit backend.
        backend_region: RegionId,
        /// Instance identifier of the hit backend.
        backend_instance: String,
    },
}

/// Metadata digest.
///
/// Used in PING/PONG to carry the version of each part of the sender's metadata;
/// the receiver decides whether to initiate a [`ClusterMessage::SyncRequest`] based on this.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetaDigest {
    /// KV cache index version (generation of the local exact index).
    pub kv_version: u64,
    /// Model registry version.
    pub model_version: u64,
    /// Load statistics version.
    pub load_version: u64,
    /// Topology graph version.
    pub topology_version: u64,
    /// Member list version.
    pub members_version: u64,
}

/// Metadata entry.
///
/// Used in [`ClusterMessage::SyncResponse`] to return a single metadata item; `value` is
/// carried as a generic JSON value, interpreted by the receiver according to `key`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaEntry {
    /// Metadata key name (corresponds to the string form of [`MetaKey`] or a custom key).
    pub key: String,
    /// Metadata value (JSON value).
    pub value: serde_json::Value,
    /// Version of this entry.
    pub version: u64,
}

/// Predefined categories of metadata keys.
///
/// Corresponds to each version field in [`MetaDigest`]; the caller can convert it to a string
/// via `as_str` for use in [`ClusterMessage::SyncRequest`] and [`MetaEntry::key`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MetaKey {
    /// KV cache index state.
    KvState,
    /// Model registry state.
    ModelState,
    /// Load statistics state.
    LoadState,
    /// Topology graph state.
    TopologyState,
    /// Member list.
    Members,
}

impl MetaKey {
    /// Returns the string name of this key.
    pub fn as_str(&self) -> &'static str {
        match self {
            MetaKey::KvState => "kv_state",
            MetaKey::ModelState => "model_state",
            MetaKey::LoadState => "load_state",
            MetaKey::TopologyState => "topology_state",
            MetaKey::Members => "members",
        }
    }
}

impl From<MetaKey> for String {
    fn from(key: MetaKey) -> Self {
        key.as_str().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::ids::{BackendInstanceId, IndexerDomainId};

    #[test]
    fn ping_message_round_trip() {
        let msg = ClusterMessage::Ping {
            sender: InstanceId::new("g1"),
            region: RegionId::new("r1"),
            meta_digest: MetaDigest {
                kv_version: 1,
                model_version: 2,
                load_version: 3,
                topology_version: 4,
                members_version: 5,
            },
            timestamp: 1_700_000_000_000,
        };
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.contains(r#""type":"ping""#), "serialized: {s}");
        let back: ClusterMessage = serde_json::from_str(&s).unwrap();
        match back {
            ClusterMessage::Ping {
                sender,
                region,
                meta_digest,
                timestamp,
            } => {
                assert_eq!(sender.as_str(), "g1");
                assert_eq!(region.as_str(), "r1");
                assert_eq!(meta_digest.kv_version, 1);
                assert_eq!(timestamp, 1_700_000_000_000);
            }
            _ => panic!("Deserialization should yield Ping"),
        }
    }

    #[test]
    fn meet_message_round_trip() {
        let msg = ClusterMessage::Meet {
            sender: InstanceId::new("g2"),
            region: RegionId::new("r2"),
            addr: "10.0.0.2:7946".to_string(),
        };
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.contains(r#""type":"meet""#));
        let back: ClusterMessage = serde_json::from_str(&s).unwrap();
        match back {
            ClusterMessage::Meet { sender, region, addr } => {
                assert_eq!(sender.as_str(), "g2");
                assert_eq!(region.as_str(), "r2");
                assert_eq!(addr, "10.0.0.2:7946");
            }
            _ => panic!("Deserialization should yield Meet"),
        }
    }

    #[test]
    fn ckf_barrier_snapshot_round_trip() {
        let pool = PoolId {
            domain: IndexerDomainId::new(7),
            region: RegionId::new("r1"),
        };
        let msg = ClusterMessage::CkfBarrierSnapshot {
            pool,
            sequence: 42,
            buckets: vec![1, 2, 3],
            num_buckets: 3,
        };
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.contains(r#""type":"ckf_barrier_snapshot""#));
        let back: ClusterMessage = serde_json::from_str(&s).unwrap();
        match back {
            ClusterMessage::CkfBarrierSnapshot {
                pool,
                sequence,
                buckets,
                num_buckets,
            } => {
                assert_eq!(pool.domain.0, 7);
                assert_eq!(sequence, 42);
                assert_eq!(buckets, vec![1, 2, 3]);
                assert_eq!(num_buckets, 3);
            }
            _ => panic!("Deserialization should yield CkfBarrierSnapshot"),
        }
    }

    #[test]
    fn ckf_delta_round_trip() {
        let pool = PoolId {
            domain: IndexerDomainId::new(7),
            region: RegionId::new("r1"),
        };
        let msg = ClusterMessage::CkfDelta {
            pool,
            sequence: 10,
            prev_sequence: 9,
            dirty_buckets: vec![(0, 100), (3, 200)],
        };
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.contains(r#""type":"ckf_delta""#));
        let back: ClusterMessage = serde_json::from_str(&s).unwrap();
        match back {
            ClusterMessage::CkfDelta {
                pool,
                sequence,
                prev_sequence,
                dirty_buckets,
            } => {
                assert_eq!(sequence, 10);
                assert_eq!(prev_sequence, 9);
                assert_eq!(dirty_buckets, vec![(0, 100), (3, 200)]);
                assert_eq!(pool.domain.0, 7);
            }
            _ => panic!("Deserialization should yield CkfDelta"),
        }
    }

    #[test]
    fn meta_key_as_str_matches_serde_name() {
        assert_eq!(MetaKey::KvState.as_str(), "kv_state");
        assert_eq!(MetaKey::ModelState.as_str(), "model_state");
        assert_eq!(MetaKey::LoadState.as_str(), "load_state");
        assert_eq!(MetaKey::TopologyState.as_str(), "topology_state");
        assert_eq!(MetaKey::Members.as_str(), "members");

        // After serde serialization it should match as_str
        let s = serde_json::to_string(&MetaKey::KvState).unwrap();
        assert_eq!(s, r#""kv_state""#);
    }

    #[test]
    fn meta_digest_default() {
        let d = MetaDigest::default();
        assert_eq!(d.kv_version, 0);
        assert_eq!(d.model_version, 0);
        assert_eq!(d.load_version, 0);
        assert_eq!(d.topology_version, 0);
        assert_eq!(d.members_version, 0);
    }

    #[test]
    fn session_affinity_round_trip() {
        let msg = ClusterMessage::SessionAffinityBroadcast {
            session: SessionId::new("sess-1"),
            backend_region: RegionId::new("r1"),
            backend_instance: BackendInstanceId::new("i1").to_string(),
        };
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.contains(r#""type":"session_affinity_broadcast""#));
        let back: ClusterMessage = serde_json::from_str(&s).unwrap();
        match back {
            ClusterMessage::SessionAffinityBroadcast {
                session,
                backend_region,
                backend_instance,
            } => {
                assert_eq!(session.as_str(), "sess-1");
                assert_eq!(backend_region.as_str(), "r1");
                assert_eq!(backend_instance, "i1");
            }
            _ => panic!("Deserialization should yield SessionAffinityBroadcast"),
        }
    }
}
