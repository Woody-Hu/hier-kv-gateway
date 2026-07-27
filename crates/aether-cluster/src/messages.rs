//! 集群消息类型与元数据摘要 / 条目定义。
//!
//! 参考架构文档 §5.1 的 Gossip 消息类型设计：
//! - `Ping/Pong` 携带发送方元数据摘要，用于心跳与版本同步。
//! - `Meet` 用于新实例加入集群，由接收方加入成员列表并广播传播。
//! - `SyncRequest/SyncResponse` 用于在版本落后时按 key 拉取元数据条目。
//! - `CkfBarrierSnapshot/CkfDelta` 用于跨 Region 的 KV 投影发布（参考 Dynamo
//!   Multi-DC Relay 的两阶段发布）。
//! - `MetricsBroadcast` 用于广播后端负载指标。
//! - `TopologyUpdate` 用于同步跨 Region 延迟估计。
//! - `SessionAffinityBroadcast` 用于跨实例共享会话亲和路由。
//!
//! 所有消息以 `type` 字段做内部标签，采用 `snake_case` 命名，便于在 JSON / msgpack
//! 等多种传输序列化格式下保持稳定兼容。

use serde::{Deserialize, Serialize};

use aether_core::ids::{InstanceId, PoolId, RegionId, SessionId};
use aether_core::metrics::BackendMetrics;
use aether_core::topology::LatencyEstimate;

/// 集群消息统一封装。
///
/// 序列化时使用 `type` 字段做内部标签（internally tagged），
/// 并采用 `snake_case` 命名以匹配架构文档约定。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClusterMessage {
    /// 心跳 Ping，携带发送方的元数据摘要。
    Ping {
        /// 发送方实例标识。
        sender: InstanceId,
        /// 发送方所在区域。
        region: RegionId,
        /// 发送方元数据摘要（用于版本对比）。
        meta_digest: MetaDigest,
        /// 发送时戳（Unix 毫秒）。
        timestamp: u64,
    },
    /// 心跳 Pong，回应 Ping 的对端，并携带自身最新摘要。
    Pong {
        /// 发送方实例标识。
        sender: InstanceId,
        /// 发送方所在区域。
        region: RegionId,
        /// 发送方元数据摘要。
        meta_digest: MetaDigest,
        /// 发送时戳（Unix 毫秒）。
        timestamp: u64,
    },
    /// 新实例加入请求。
    ///
    /// 由新加入的实例发送给已知种子节点，接收方应将发送方加入成员列表，
    /// 并在后续 Gossip 中传播。
    Meet {
        /// 发送方实例标识。
        sender: InstanceId,
        /// 发送方所在区域。
        region: RegionId,
        /// 发送方对外可达地址（host:port）。
        addr: String,
    },
    /// 元数据同步请求。
    ///
    /// 接收方按 `keys` 中列出的 [`MetaKey`] 字符串名返回对应的 [`MetaEntry`]。
    SyncRequest {
        /// 发送方实例标识。
        sender: InstanceId,
        /// 需要同步的元数据键名列表。
        keys: Vec<String>,
    },
    /// 元数据同步响应，包含请求方所请求的全部条目。
    SyncResponse {
        /// 返回的元数据条目列表。
        entries: Vec<MetaEntry>,
    },
    /// CKF 全量快照发布（barrier snapshot）。
    ///
    /// 由 KV Relay 在首次发布或重连后发布，接收方应安装到对应 Region 的 lane。
    CkfBarrierSnapshot {
        /// 该 CKF 所属连接池。
        pool: PoolId,
        /// 快照序列号。
        sequence: u64,
        /// lane 内所有 bucket 的当前值。
        buckets: Vec<u64>,
        /// lane 中的 bucket 数量。
        num_buckets: usize,
    },
    /// CKF 增量发布（sequenced delta）。
    ///
    /// 仅包含自上一个序列号以来变动的 bucket，接收方按 (idx, value) 写入对应 lane。
    CkfDelta {
        /// 该 CKF 所属连接池。
        pool: PoolId,
        /// 本增量对应的序列号。
        sequence: u64,
        /// 本增量基于的前一个序列号。
        prev_sequence: u64,
        /// 变动的 bucket 列表：(bucket_idx, new_value)。
        dirty_buckets: Vec<(usize, u64)>,
    },
    /// 后端负载指标广播。
    ///
    /// `backends` 中每项为 `(backend 标识字符串, 指标快照)`。
    MetricsBroadcast {
        /// 指标所属 Region。
        region: RegionId,
        /// 该 Region 内多个后端的最新指标。
        backends: Vec<(String, BackendMetrics)>,
    },
    /// 跨 Region 拓扑延迟更新。
    ///
    /// 由感知到 RTT 变化的实例广播，接收方更新本地 [`LatencyMatrix`]。
    TopologyUpdate {
        /// 源 Region。
        from: RegionId,
        /// 目标 Region。
        to: RegionId,
        /// 两区域间的延迟估计。
        estimate: LatencyEstimate,
    },
    /// 会话亲和路由广播。
    ///
    /// 当某实例将会话路由到指定后端后，广播给集群内其他实例以便后续请求命中同一后端。
    SessionAffinityBroadcast {
        /// 会话标识。
        session: SessionId,
        /// 命中的后端所在 Region。
        backend_region: RegionId,
        /// 命中的后端实例标识。
        backend_instance: String,
    },
}

/// 元数据摘要。
///
/// 用于 PING/PONG 中携带发送方各部分元数据的版本号，接收方据此决定是否需要发起
/// [`ClusterMessage::SyncRequest`]。
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetaDigest {
    /// KV cache 索引版本（本地精确索引的代次）。
    pub kv_version: u64,
    /// 模型注册表版本。
    pub model_version: u64,
    /// 负载统计版本。
    pub load_version: u64,
    /// 拓扑图版本。
    pub topology_version: u64,
    /// 成员列表版本。
    pub members_version: u64,
}

/// 元数据条目。
///
/// 用于 [`ClusterMessage::SyncResponse`] 中返回单项元数据，`value` 以通用 JSON 值
/// 形式承载，由接收方按 `key` 解释。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaEntry {
    /// 元数据键名（对应 [`MetaKey`] 的字符串形式或自定义键）。
    pub key: String,
    /// 元数据值（JSON 值）。
    pub value: serde_json::Value,
    /// 该条目的版本号。
    pub version: u64,
}

/// 元数据键的预定义分类。
///
/// 对应 [`MetaDigest`] 中的各个版本字段；调用方可通过 `as_str` 转为字符串用于
/// [`ClusterMessage::SyncRequest`] 与 [`MetaEntry::key`]。
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MetaKey {
    /// KV cache 索引状态。
    KvState,
    /// 模型注册表状态。
    ModelState,
    /// 负载统计状态。
    LoadState,
    /// 拓扑图状态。
    TopologyState,
    /// 成员列表。
    Members,
}

impl MetaKey {
    /// 返回该键对应的字符串名。
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
    use aether_core::ids::{BackendInstanceId, IndexerDomainId};

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
            _ => panic!("反序列化应为 Ping"),
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
            _ => panic!("反序列化应为 Meet"),
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
            _ => panic!("反序列化应为 CkfBarrierSnapshot"),
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
            _ => panic!("反序列化应为 CkfDelta"),
        }
    }

    #[test]
    fn meta_key_as_str_matches_serde_name() {
        assert_eq!(MetaKey::KvState.as_str(), "kv_state");
        assert_eq!(MetaKey::ModelState.as_str(), "model_state");
        assert_eq!(MetaKey::LoadState.as_str(), "load_state");
        assert_eq!(MetaKey::TopologyState.as_str(), "topology_state");
        assert_eq!(MetaKey::Members.as_str(), "members");

        // serde 序列化后应与 as_str 一致
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
            _ => panic!("反序列化应为 SessionAffinityBroadcast"),
        }
    }
}
