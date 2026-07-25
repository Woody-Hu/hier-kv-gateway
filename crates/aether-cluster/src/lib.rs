//! Aether LLM Gateway 的集群通信层。
//!
//! 该 crate 实现了 Aether 网关实例组之间的集群通信能力，主要包括：
//!
//! - [`transport`]：定义集群传输抽象 [`ClusterTransport`]，便于上层替换不同的
//!   传输实现（默认 Gossip Bus，可换为 NATS、gRPC mesh 等）。
//! - [`gossip`]：参考 Redis Cluster 的 Gossip 协议实现 [`GossipEngine`]，
//!   维护成员列表、心跳探活与跨实例的元数据摘要同步。
//! - [`member`]：集群成员列表 [`MemberList`] 与成员状态机 [`MemberStatus`]。
//! - [`ckf_relay`]：参考 Dynamo Multi-DC Relay 的 [`KvRelay`]，将本地
//!   Cuckoo Filter 投影以 barrier snapshot + sequenced delta 的方式发布到
//!   跨 Region 的 Gossip Bus。
//! - [`messages`]：集群消息类型 [`ClusterMessage`] 与元数据摘要 / 条目定义。

pub mod transport;
pub mod gossip;
pub mod member;
pub mod ckf_relay;
pub mod messages;
