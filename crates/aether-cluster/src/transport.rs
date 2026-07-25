//! 集群传输抽象。
//!
//! [`ClusterTransport`] 是 aether-cluster 的传输层 trait：将消息收发、广播、
//! 成员视图等能力抽象出来，便于上层替换不同的传输实现（默认 Gossip Bus，
//! 可换为 NATS、gRPC mesh、QUIC mesh 等）。
//!
//! 设计要点：
//! - `start` 接收自身身份（`InstanceId` / `RegionId` / 绑定地址），完成后开始收发消息。
//! - `messages` 返回一个 [`tokio::sync::mpsc::Receiver`]，由实现向其中推送
//!   所有从对端收到的 [`ClusterMessage`]；上层（如 [`crate::gossip::GossipEngine`])
//!   持有 receiver 并在事件循环中处理。
//! - `members` 返回当前传输层已知的成员列表（通常由实现内部维护一份与
//!   [`crate::member::MemberList`] 一致的视图，或委托给上游成员管理）。

use async_trait::async_trait;

use aether_core::error::Result;
use aether_core::ids::{InstanceId, RegionId};

use crate::member::ClusterMember;
use crate::messages::ClusterMessage;

/// 集群传输层抽象。
///
/// 所有方法都设计为非阻塞（async），允许实现内部使用任意异步运行时与 IO 模型。
#[async_trait]
pub trait ClusterTransport: Send + Sync {
    /// 启动传输层。
    ///
    /// `self_id` 为本实例标识，`region` 为所在区域，`addr` 为绑定地址
    /// （如 `0.0.0.0:7946`）。启动成功后，传输层应开始监听入站消息，并将收到的
    /// 消息通过 [`messages`](ClusterTransport::messages) 返回的 channel 推送出去。
    async fn start(&self, self_id: &InstanceId, region: &RegionId, addr: &str) -> Result<()>;

    /// 停止传输层，关闭监听与所有连接。
    async fn stop(&self) -> Result<()>;

    /// 向指定目标地址发送一条消息。
    ///
    /// `target` 形如 `host:port`。实现需保证单条消息原子送达（或失败上报）。
    async fn send(&self, target: &str, msg: &ClusterMessage) -> Result<()>;

    /// 向当前已知所有 alive 成员广播一条消息。
    async fn broadcast(&self, msg: &ClusterMessage) -> Result<()>;

    /// 返回入站消息的接收端。
    ///
    /// 通常实现会在 [`start`](ClusterTransport::start) 后于内部 spawn 一个任务
    /// 负责从底层 IO 读取消息并推送到该 channel。
    ///
    /// 注意：调用此方法会获取 receiver 所有权，多次调用通常返回同一个 receiver
    /// 或新创建的 receiver（取决于实现）。
    fn messages(&self) -> tokio::sync::mpsc::Receiver<ClusterMessage>;

    /// 返回当前已知的成员列表快照。
    ///
    /// 通常由实现委托给一个共享的 [`crate::member::MemberList`]。
    fn members(&self) -> Vec<ClusterMember>;
}
