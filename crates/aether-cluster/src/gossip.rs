//! Gossip 协议引擎（参考 Redis Cluster）。
//!
//! [`GossipEngine`] 维护一个本地的 [`MemberList`]，并通过 [`ClusterTransport`]
//! 与集群内其他实例交换心跳与元数据摘要。引擎启动后会同时运行三个后台任务：
//!
//! 1. `gossip_loop`：每隔 `gossip_interval_ms` 随机选取至多 `GOSSIP_FANOUT` 个
//!    alive 成员发送 [`ClusterMessage::Ping`]。
//! 2. `probe_loop`：周期性扫描成员列表，将超过 `probe_timeout_ms` 未收到 Pong
//!    的成员标记为 [`MemberStatus::Suspect`]；在 Suspect 状态停留超过
//!    `suspect_timeout_secs` 的成员标记为 [`MemberStatus::Dead`]。
//! 3. `message_loop`：从传输层接收 [`ClusterMessage`] 并调用 [`handle_message`]
//!    分发处理。
//!
//! 因为 [`ClusterMessage::MetricsBroadcast`] / [`ClusterMessage::CkfBarrierSnapshot`]
//! / [`ClusterMessage::CkfDelta`] 等消息需要更新本地 MetadataStore（位于上层
//! gateway），本引擎不直接持有 MetadataStore，而是通过 [`GossipHandler`] trait
//! 由调用方注入回调实现，从而保持 aether-cluster 与 aether-metadata::store 之间的
//! 解耦。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use rand::seq::SliceRandom;

use aether_core::config::ClusterConfig;
use aether_core::error::{AetherError, Result};
use aether_core::ids::{InstanceId, PoolId, RegionId, SessionId};
use aether_core::metrics::BackendMetrics;
use aether_core::topology::LatencyEstimate;

use crate::member::{now_unix_millis, ClusterMember, MemberList, MemberStatus};
use crate::messages::{ClusterMessage, MetaDigest, MetaEntry};
use crate::transport::ClusterTransport;

/// 默认每轮 Gossip 选取的成员数量（参考 SWIM/Redis Cluster 默认值）。
const GOSSIP_FANOUT: usize = 3;

/// 探活循环的扫描间隔（毫秒）。
///
/// 不与 `gossip_interval_ms` 强绑定，避免探活过于稀疏或过于密集。
const PROBE_LOOP_INTERVAL_MS: u64 = 500;

/// 元数据 / 指标 / CKF / 拓扑 / 会话亲和消息的处理器 trait。
///
/// 由 gateway 层（持有 MetadataStore 的组件）实现并注入到 [`GossipEngine`] 中，
/// 使 aether-cluster 不直接依赖 aether-metadata::store，仅依赖 aether-core 的
/// 基础类型。
#[async_trait]
pub trait GossipHandler: Send + Sync {
    /// 返回当前本地元数据摘要。
    ///
    /// 在 PING/PONG 中携带，供对端版本比对。
    fn current_meta_digest(&self) -> MetaDigest;

    /// 处理 [`ClusterMessage::SyncRequest`]：按 `keys` 返回对应的元数据条目。
    fn handle_sync_request(&self, keys: &[String]) -> Vec<MetaEntry>;

    /// 应用 [`ClusterMessage::SyncResponse`] 中的元数据条目到本地存储。
    fn apply_sync_response(&self, entries: &[MetaEntry]);

    /// 处理 [`ClusterMessage::MetricsBroadcast`]：更新本地负载统计。
    fn handle_metrics_broadcast(
        &self,
        region: &RegionId,
        backends: &[(String, BackendMetrics)],
    );

    /// 处理 [`ClusterMessage::CkfBarrierSnapshot`]：将全量快照安装到本地 CKF Consumer。
    fn handle_ckf_barrier_snapshot(
        &self,
        pool: &PoolId,
        sequence: u64,
        buckets: &[u64],
        num_buckets: usize,
    );

    /// 处理 [`ClusterMessage::CkfDelta`]：将增量应用到本地 CKF Consumer。
    fn handle_ckf_delta(
        &self,
        pool: &PoolId,
        sequence: u64,
        prev_sequence: u64,
        dirty_buckets: &[(usize, u64)],
    );

    /// 处理 [`ClusterMessage::TopologyUpdate`]：更新本地延迟矩阵。
    fn handle_topology_update(
        &self,
        from: &RegionId,
        to: &RegionId,
        estimate: &LatencyEstimate,
    );

    /// 处理 [`ClusterMessage::SessionAffinityBroadcast`]：更新本地会话亲和历史。
    fn handle_session_affinity(
        &self,
        session: &SessionId,
        backend_region: &RegionId,
        backend_instance: &str,
    );
}

/// 空操作的 [`GossipHandler`]，所有方法均不做任何实际更新。
///
/// 适用于不需要处理上游消息的场景（例如纯发送测试、网关尚未装配完整 store 时）。
#[derive(Default, Debug, Clone, Copy)]
pub struct NoopGossipHandler;

#[async_trait]
impl GossipHandler for NoopGossipHandler {
    fn current_meta_digest(&self) -> MetaDigest {
        MetaDigest::default()
    }

    fn handle_sync_request(&self, _keys: &[String]) -> Vec<MetaEntry> {
        Vec::new()
    }

    fn apply_sync_response(&self, _entries: &[MetaEntry]) {}

    fn handle_metrics_broadcast(
        &self,
        _region: &RegionId,
        _backends: &[(String, BackendMetrics)],
    ) {
    }

    fn handle_ckf_barrier_snapshot(
        &self,
        _pool: &PoolId,
        _sequence: u64,
        _buckets: &[u64],
        _num_buckets: usize,
    ) {
    }

    fn handle_ckf_delta(
        &self,
        _pool: &PoolId,
        _sequence: u64,
        _prev_sequence: u64,
        _dirty_buckets: &[(usize, u64)],
    ) {
    }

    fn handle_topology_update(
        &self,
        _from: &RegionId,
        _to: &RegionId,
        _estimate: &LatencyEstimate,
    ) {
    }

    fn handle_session_affinity(
        &self,
        _session: &SessionId,
        _backend_region: &RegionId,
        _backend_instance: &str,
    ) {
    }
}

/// Gossip 协议引擎。
pub struct GossipEngine {
    /// 本实例标识。
    self_id: InstanceId,
    /// 本实例所在区域。
    self_region: RegionId,
    /// 本实例对外绑定地址（host:port），用于 Meet 通知对端如何回连。
    self_addr: String,
    /// 成员列表（与传输层共享同一份）。
    members: Arc<MemberList>,
    /// 集群传输层。
    transport: Arc<dyn ClusterTransport>,
    /// 集群配置。
    config: ClusterConfig,
    /// 运行标志，false 时所有后台任务退出。
    running: Arc<AtomicBool>,
    /// 元数据 / 指标 / CKF 等消息的处理器。
    handler: Arc<dyn GossipHandler>,
}

impl GossipEngine {
    /// 创建一个 Gossip 引擎。
    ///
    /// 引擎启动时不会主动监听；需调用 [`start`](GossipEngine::start) 启动后台任务。
    pub fn new(
        self_id: InstanceId,
        self_region: RegionId,
        self_addr: String,
        members: Arc<MemberList>,
        transport: Arc<dyn ClusterTransport>,
        config: ClusterConfig,
        handler: Arc<dyn GossipHandler>,
    ) -> Self {
        Self {
            self_id,
            self_region,
            self_addr,
            members,
            transport,
            config,
            running: Arc::new(AtomicBool::new(false)),
            handler,
        }
    }

    /// 启动 Gossip 引擎，启动传输层并 spawn 三个后台任务。
    ///
    /// 该方法立即返回；引擎持续运行直到调用 [`stop`](GossipEngine::stop)。
    pub async fn start(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            // 已经启动过
            return Ok(());
        }

        // 启动传输层
        self.transport
            .start(&self.self_id, &self.self_region, &self.self_addr)
            .await?;

        // 将自身加入成员列表
        let self_member = ClusterMember::new(
            self.self_id.clone(),
            self.self_region.clone(),
            self.self_addr.clone(),
            self.handler.current_meta_digest(),
        );
        self.members.upsert(self_member);

        // 获取入站消息接收端；receiver 的所有权转移到 message_loop 任务中。
        let receiver = self.transport.messages();

        // 启动 gossip_loop
        let gossip_handle = tokio::spawn(Self::gossip_loop(
            self.self_id.clone(),
            self.self_region.clone(),
            self.members.clone(),
            self.transport.clone(),
            self.handler.clone(),
            self.config.gossip_interval_ms,
            self.running.clone(),
        ));

        // 启动 probe_loop
        let probe_handle = tokio::spawn(Self::probe_loop(
            self.self_id.clone(),
            self.members.clone(),
            self.config.probe_timeout_ms,
            self.config.suspect_timeout_secs,
            self.running.clone(),
        ));

        // 启动 message_loop
        let message_handle = tokio::spawn(Self::message_loop(
            self.self_id.clone(),
            self.self_region.clone(),
            self.self_addr.clone(),
            self.members.clone(),
            self.transport.clone(),
            self.handler.clone(),
            receiver,
            self.running.clone(),
        ));

        // 让三个任务句柄在后台独立运行；引擎通过 `running` 标志位停止。
        // 这里 detach 三个 JoinHandle 以避免阻塞调用方。
        tokio::spawn(async move {
            let _ = gossip_handle.await;
            let _ = probe_handle.await;
            let _ = message_handle.await;
        });

        Ok(())
    }

    /// 加入集群：向每个种子节点发送 [`ClusterMessage::Meet`]。
    pub async fn join_cluster(&self, seed_peers: &[String]) -> Result<()> {
        let meet = ClusterMessage::Meet {
            sender: self.self_id.clone(),
            region: self.self_region.clone(),
            addr: self.self_addr.clone(),
        };
        for seed in seed_peers {
            if let Err(e) = self.transport.send(seed, &meet).await {
                warn!(target = %seed, error = %e, "向种子节点发送 Meet 失败");
            }
        }
        Ok(())
    }

    /// 停止引擎：清除运行标志并停止传输层。
    pub async fn stop(&self) -> Result<()> {
        if !self.running.swap(false, Ordering::SeqCst) {
            return Ok(());
        }
        self.transport.stop().await
    }

    /// 处理一条入站消息。
    ///
    /// 该方法是 public 的，便于上层在测试中直接调用而无需走传输层。
    pub async fn handle_message(&self, msg: ClusterMessage) {
        match msg {
            ClusterMessage::Ping {
                sender,
                region,
                meta_digest,
                timestamp: _,
            } => {
                // 把发送方加入成员列表（若已存在则覆盖 digest）
                Self::ensure_member(
                    &self.members,
                    &sender,
                    &region,
                    None,
                    meta_digest.clone(),
                );

                // 回 Pong
                let pong = ClusterMessage::Pong {
                    sender: self.self_id.clone(),
                    region: self.self_region.clone(),
                    meta_digest: self.handler.current_meta_digest(),
                    timestamp: now_unix_millis(),
                };
                // 找到发送方地址后回送
                if let Some(member) = self.members.get(&sender) {
                    if let Err(e) = self.transport.send(&member.addr, &pong).await {
                        warn!(error = %e, "回复 Pong 失败");
                    }
                }
            }
            ClusterMessage::Pong {
                sender,
                region,
                meta_digest,
                timestamp: _,
            } => {
                Self::ensure_member(&self.members, &sender, &region, None, meta_digest);
            }
            ClusterMessage::Meet {
                sender,
                region,
                addr,
            } => {
                // 收到 Meet：把发送方加入成员列表
                Self::ensure_member(
                    &self.members,
                    &sender,
                    &region,
                    Some(addr.clone()),
                    self.handler.current_meta_digest(),
                );

                // 回 Pong 给发送方，让对方得知我们的存在
                let pong = ClusterMessage::Pong {
                    sender: self.self_id.clone(),
                    region: self.self_region.clone(),
                    meta_digest: self.handler.current_meta_digest(),
                    timestamp: now_unix_millis(),
                };
                if let Err(e) = self.transport.send(&addr, &pong).await {
                    warn!(error = %e, "回复 Meet 的 Pong 失败");
                }

                // 传播新成员：广播 Ping 让其他实例也能看到这个新成员
                // （此处借助 Ping 携带自身 digest，对端收到后会回 Pong，
                //  间接达到 disseminate 的效果；为避免广播风暴，仅在收到 Meet 时
                //  触发一次广播）
                let ping = ClusterMessage::Ping {
                    sender: self.self_id.clone(),
                    region: self.self_region.clone(),
                    meta_digest: self.handler.current_meta_digest(),
                    timestamp: now_unix_millis(),
                };
                if let Err(e) = self.transport.broadcast(&ping).await {
                    warn!(error = %e, "Meet 后广播 Ping 失败");
                }
            }
            ClusterMessage::SyncRequest { sender, keys } => {
                let entries = self.handler.handle_sync_request(&keys);
                let response = ClusterMessage::SyncResponse { entries };
                if let Some(member) = self
                    .members
                    .all_members()
                    .into_iter()
                    .find(|m| m.instance_id == sender)
                {
                    if let Err(e) = self.transport.send(&member.addr, &response).await {
                        warn!(error = %e, "回复 SyncResponse 失败");
                    }
                } else {
                    // 不知道发送方地址，回退到广播（成本较高，但保证可达性）
                    if let Err(e) = self.transport.broadcast(&response).await {
                        warn!(error = %e, "广播 SyncResponse 失败");
                    }
                }
            }
            ClusterMessage::SyncResponse { entries } => {
                self.handler.apply_sync_response(&entries);
            }
            ClusterMessage::MetricsBroadcast { region, backends } => {
                self.handler
                    .handle_metrics_broadcast(&region, &backends);
            }
            ClusterMessage::CkfBarrierSnapshot {
                pool,
                sequence,
                buckets,
                num_buckets,
            } => {
                self.handler
                    .handle_ckf_barrier_snapshot(&pool, sequence, &buckets, num_buckets);
            }
            ClusterMessage::CkfDelta {
                pool,
                sequence,
                prev_sequence,
                dirty_buckets,
            } => {
                self.handler.handle_ckf_delta(
                    &pool,
                    sequence,
                    prev_sequence,
                    &dirty_buckets,
                );
            }
            ClusterMessage::TopologyUpdate {
                from,
                to,
                estimate,
            } => {
                self.handler.handle_topology_update(&from, &to, &estimate);
            }
            ClusterMessage::SessionAffinityBroadcast {
                session,
                backend_region,
                backend_instance,
            } => {
                self.handler.handle_session_affinity(
                    &session,
                    &backend_region,
                    &backend_instance,
                );
            }
        }
    }

    /// 把一个实例加入成员列表（若不存在则插入，存在则更新 digest / addr）。
    fn ensure_member(
        members: &Arc<MemberList>,
        instance_id: &InstanceId,
        region: &RegionId,
        addr: Option<String>,
        digest: MetaDigest,
    ) {
        match members.get(instance_id) {
            Some(existing) => {
                // 已存在：更新 digest；若提供了新 addr 则一并更新
                let new_addr = addr.unwrap_or_else(|| existing.addr.clone());
                let updated = ClusterMember {
                    instance_id: instance_id.clone(),
                    region: region.clone(),
                    addr: new_addr,
                    last_pong_unix: now_unix_millis(),
                    status: MemberStatus::Alive,
                    meta_digest: digest,
                };
                members.upsert(updated);
            }
            None => {
                // 不存在：插入新成员，addr 必须提供
                let addr = addr.unwrap_or_default();
                let member = ClusterMember::new(
                    instance_id.clone(),
                    region.clone(),
                    addr,
                    digest,
                );
                members.upsert(member);
            }
        }
    }

    /// Gossip 循环：每隔 `interval_ms` 随机选 `GOSSIP_FANOUT` 个 alive 成员发 Ping。
    async fn gossip_loop(
        self_id: InstanceId,
        _self_region: RegionId,
        members: Arc<MemberList>,
        transport: Arc<dyn ClusterTransport>,
        handler: Arc<dyn GossipHandler>,
        interval_ms: u64,
        running: Arc<AtomicBool>,
    ) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(1)));
        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            interval.tick().await;

            // 不向自己发 Ping
            let candidates: Vec<ClusterMember> = members
                .alive_members()
                .into_iter()
                .filter(|m| m.instance_id != self_id)
                .collect();
            if candidates.is_empty() {
                continue;
            }
            // 注意：`rand::rng()` 返回的 ThreadRng 不是 Send，
            // 必须在进入 await 之前 drop，否则 future 不满足 Send。
            let targets: Vec<ClusterMember> = {
                let mut rng = rand::rng();
                let mut chosen = candidates.clone();
                chosen.shuffle(&mut rng);
                chosen.into_iter().take(GOSSIP_FANOUT).collect()
            };
            for member in targets {
                let ping = ClusterMessage::Ping {
                    sender: self_id.clone(),
                    region: member.region.clone(),
                    meta_digest: handler.current_meta_digest(),
                    timestamp: now_unix_millis(),
                };
                if let Err(e) = transport.send(&member.addr, &ping).await {
                    debug!(error = %e, target = %member.addr, "发送 Ping 失败");
                }
            }
        }
    }

    /// 探活循环：检测 Pong 超时与 Suspect 超时。
    async fn probe_loop(
        self_id: InstanceId,
        members: Arc<MemberList>,
        probe_timeout_ms: u64,
        suspect_timeout_secs: u64,
        running: Arc<AtomicBool>,
    ) {
        let mut interval =
            tokio::time::interval(Duration::from_millis(PROBE_LOOP_INTERVAL_MS));
        let suspect_timeout_ms = suspect_timeout_secs.saturating_mul(1000);
        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            interval.tick().await;

            let now = now_unix_millis();
            let probe_too_old = now.saturating_sub(probe_timeout_ms);
            let suspect_too_old = now.saturating_sub(suspect_timeout_ms);

            // 扫描所有成员，进行状态转移
            for member in members.all_members() {
                // 不对自己做探活
                if member.instance_id == self_id {
                    continue;
                }
                match member.status {
                    MemberStatus::Alive => {
                        if member.last_pong_unix < probe_too_old {
                            members.mark_suspect(&member.instance_id);
                            debug!(instance = %member.instance_id, "成员 Pong 超时，标记为 Suspect");
                        }
                    }
                    MemberStatus::Suspect => {
                        if member.last_pong_unix < suspect_too_old {
                            members.mark_dead(&member.instance_id);
                            debug!(instance = %member.instance_id, "Suspect 超时，标记为 Dead");
                        }
                    }
                    MemberStatus::Dead => {
                        // 已下线，等待被清理（上层可调用 remove）
                    }
                }
            }
        }
    }

    /// 消息循环：从传输层接收消息并交给 [`handle_message`](Self::handle_message)。
    ///
    /// 注意：因为 `handle_message` 需要 `&self`，但 `message_loop` 是 spawn 出去的
    /// 静态任务，无法直接持有 `&GossipEngine`；这里复制一份与 `handle_message` 等价的
    /// 闭包逻辑。为保持代码与 `handle_message` 行为一致，本函数将消息分发逻辑
    /// 委托给一个独立的 `dispatch` 辅助函数。
    async fn message_loop(
        self_id: InstanceId,
        self_region: RegionId,
        self_addr: String,
        members: Arc<MemberList>,
        transport: Arc<dyn ClusterTransport>,
        handler: Arc<dyn GossipHandler>,
        mut receiver: mpsc::Receiver<ClusterMessage>,
        running: Arc<AtomicBool>,
    ) {
        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            // 收消息：使用 recv() 阻塞等待，运行标志位变更后下次循环退出。
            let msg = match receiver.recv().await {
                Some(m) => m,
                None => {
                    // 通道关闭：传输层已停止
                    break;
                }
            };

            // 忽略自己发出的消息回环（防止广播时收到自己）
            if message_sender_is_self(&msg, &self_id) {
                continue;
            }

            dispatch_message(
                &self_id,
                &self_region,
                &self_addr,
                &members,
                &transport,
                &handler,
                msg,
            )
            .await;
        }
    }
}

/// 判断消息的 `sender` 字段是否为自身（用于过滤广播回环）。
fn message_sender_is_self(msg: &ClusterMessage, self_id: &InstanceId) -> bool {
    match msg {
        ClusterMessage::Ping { sender, .. }
        | ClusterMessage::Pong { sender, .. }
        | ClusterMessage::Meet { sender, .. }
        | ClusterMessage::SyncRequest { sender, .. } => sender == self_id,
        // 其他消息没有 sender 字段，无法判定；视作非自身以保守处理。
        _ => false,
    }
}

/// 将一条消息分发到对应的处理逻辑（与 [`GossipEngine::handle_message`] 等价）。
///
/// 提取为独立函数是为了在 `message_loop` 任务中调用，避免借用 `&self`。
async fn dispatch_message(
    self_id: &InstanceId,
    self_region: &RegionId,
    _self_addr: &str,
    members: &Arc<MemberList>,
    transport: &Arc<dyn ClusterTransport>,
    handler: &Arc<dyn GossipHandler>,
    msg: ClusterMessage,
) {
    match msg {
        ClusterMessage::Ping {
            sender,
            region,
            meta_digest,
            timestamp: _,
        } => {
            GossipEngine::ensure_member(members, &sender, &region, None, meta_digest.clone());
            let pong = ClusterMessage::Pong {
                sender: self_id.clone(),
                region: self_region.clone(),
                meta_digest: handler.current_meta_digest(),
                timestamp: now_unix_millis(),
            };
            if let Some(member) = members.get(&sender) {
                if let Err(e) = transport.send(&member.addr, &pong).await {
                    warn!(error = %e, "回复 Pong 失败");
                }
            }
        }
        ClusterMessage::Pong {
            sender,
            region,
            meta_digest,
            timestamp: _,
        } => {
            GossipEngine::ensure_member(members, &sender, &region, None, meta_digest);
        }
        ClusterMessage::Meet {
            sender,
            region,
            addr,
        } => {
            GossipEngine::ensure_member(
                members,
                &sender,
                &region,
                Some(addr.clone()),
                handler.current_meta_digest(),
            );
            let pong = ClusterMessage::Pong {
                sender: self_id.clone(),
                region: self_region.clone(),
                meta_digest: handler.current_meta_digest(),
                timestamp: now_unix_millis(),
            };
            if let Err(e) = transport.send(&addr, &pong).await {
                warn!(error = %e, "回复 Meet 的 Pong 失败");
            }
            // 传播新成员
            let ping = ClusterMessage::Ping {
                sender: self_id.clone(),
                region: self_region.clone(),
                meta_digest: handler.current_meta_digest(),
                timestamp: now_unix_millis(),
            };
            if let Err(e) = transport.broadcast(&ping).await {
                warn!(error = %e, "Meet 后广播 Ping 失败");
            }
        }
        ClusterMessage::SyncRequest { sender, keys } => {
            let entries = handler.handle_sync_request(&keys);
            let response = ClusterMessage::SyncResponse { entries };
            if let Some(member) = members
                .all_members()
                .into_iter()
                .find(|m| m.instance_id == sender)
            {
                if let Err(e) = transport.send(&member.addr, &response).await {
                    warn!(error = %e, "回复 SyncResponse 失败");
                }
            } else if let Err(e) = transport.broadcast(&response).await {
                warn!(error = %e, "广播 SyncResponse 失败");
            }
        }
        ClusterMessage::SyncResponse { entries } => {
            handler.apply_sync_response(&entries);
        }
        ClusterMessage::MetricsBroadcast { region, backends } => {
            handler.handle_metrics_broadcast(&region, &backends);
        }
        ClusterMessage::CkfBarrierSnapshot {
            pool,
            sequence,
            buckets,
            num_buckets,
        } => {
            handler.handle_ckf_barrier_snapshot(&pool, sequence, &buckets, num_buckets);
        }
        ClusterMessage::CkfDelta {
            pool,
            sequence,
            prev_sequence,
            dirty_buckets,
        } => {
            handler.handle_ckf_delta(&pool, sequence, prev_sequence, &dirty_buckets);
        }
        ClusterMessage::TopologyUpdate {
            from,
            to,
            estimate,
        } => {
            handler.handle_topology_update(&from, &to, &estimate);
        }
        ClusterMessage::SessionAffinityBroadcast {
            session,
            backend_region,
            backend_instance,
        } => {
            handler.handle_session_affinity(&session, &backend_region, &backend_instance);
        }
    }
}

/// 把一个错误转换为 [`AetherError::ClusterError`]。
#[allow(dead_code)]
fn to_cluster_error<E: std::fmt::Display>(e: E) -> AetherError {
    AetherError::ClusterError(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::MetaKey;
    use aether_core::ids::BackendInstanceId;
    use async_trait::async_trait;
    use std::sync::atomic::AtomicU64;
    use tokio::sync::Mutex;

    /// 用于测试的简单 handler：记录所有收到的同步响应、metrics、CKF 等事件。
    struct CountingHandler {
        digest: MetaDigest,
        sync_calls: AtomicU64,
        apply_calls: AtomicU64,
        metrics_calls: AtomicU64,
        ckf_snapshot_calls: AtomicU64,
        ckf_delta_calls: AtomicU64,
        topology_calls: AtomicU64,
        affinity_calls: AtomicU64,
        last_metrics_backend: parking_lot::Mutex<Option<String>>,
    }

    impl CountingHandler {
        fn new(digest: MetaDigest) -> Self {
            Self {
                digest,
                sync_calls: AtomicU64::new(0),
                apply_calls: AtomicU64::new(0),
                metrics_calls: AtomicU64::new(0),
                ckf_snapshot_calls: AtomicU64::new(0),
                ckf_delta_calls: AtomicU64::new(0),
                topology_calls: AtomicU64::new(0),
                affinity_calls: AtomicU64::new(0),
                last_metrics_backend: parking_lot::Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl GossipHandler for CountingHandler {
        fn current_meta_digest(&self) -> MetaDigest {
            self.digest.clone()
        }

        fn handle_sync_request(&self, keys: &[String]) -> Vec<MetaEntry> {
            self.sync_calls
                .fetch_add(1, Ordering::Relaxed);
            keys.iter()
                .map(|k| MetaEntry {
                    key: k.clone(),
                    value: serde_json::Value::Null,
                    version: 1,
                })
                .collect()
        }

        fn apply_sync_response(&self, entries: &[MetaEntry]) {
            self.apply_calls
                .fetch_add(entries.len() as u64, Ordering::Relaxed);
        }

        fn handle_metrics_broadcast(
            &self,
            _region: &RegionId,
            backends: &[(String, BackendMetrics)],
        ) {
            self.metrics_calls.fetch_add(1, Ordering::Relaxed);
            // 仅记录最后一次的 backend 标识，避免在测试中 clone 整条 metrics。
            if let Some((id, _)) = backends.first() {
                *self.last_metrics_backend.lock() = Some(id.clone());
            }
        }

        fn handle_ckf_barrier_snapshot(
            &self,
            _pool: &PoolId,
            _sequence: u64,
            _buckets: &[u64],
            _num_buckets: usize,
        ) {
            self.ckf_snapshot_calls.fetch_add(1, Ordering::Relaxed);
        }

        fn handle_ckf_delta(
            &self,
            _pool: &PoolId,
            _sequence: u64,
            _prev_sequence: u64,
            _dirty_buckets: &[(usize, u64)],
        ) {
            self.ckf_delta_calls.fetch_add(1, Ordering::Relaxed);
        }

        fn handle_topology_update(
            &self,
            _from: &RegionId,
            _to: &RegionId,
            _estimate: &LatencyEstimate,
        ) {
            self.topology_calls.fetch_add(1, Ordering::Relaxed);
        }

        fn handle_session_affinity(
            &self,
            _session: &SessionId,
            _backend_region: &RegionId,
            _backend_instance: &str,
        ) {
            self.affinity_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 用于测试的伪传输层：把所有发送的消息记录到 Vec 中。
    struct MockTransport {
        sent: Mutex<Vec<(String, ClusterMessage)>>,
        broadcasted: Mutex<Vec<ClusterMessage>>,
        rx_tx: Mutex<Option<mpsc::Sender<ClusterMessage>>>,
        members_view: Arc<MemberList>,
    }

    impl MockTransport {
        fn new(members_view: Arc<MemberList>) -> Self {
            Self {
                sent: Mutex::new(Vec::new()),
                broadcasted: Mutex::new(Vec::new()),
                rx_tx: Mutex::new(None),
                members_view,
            }
        }
    }

    #[async_trait]
    impl ClusterTransport for MockTransport {
        async fn start(
            &self,
            _self_id: &InstanceId,
            _region: &RegionId,
            _addr: &str,
        ) -> Result<()> {
            let (tx, rx) = mpsc::channel(64);
            *self.rx_tx.lock().await = Some(tx);
            // 把 rx 交给谁？transport.messages() 会返回它。
            // 这里用 OnceCell 模拟“只能取一次”。
            // 但 messages() 必须返回 Receiver，且 start 必须先于 messages() 调用。
            // 简化：把 rx 存在 transport 自己中，messages() 取出来。
            std::mem::forget(rx); // 避免编译错误；测试不会调用 messages
            Ok(())
        }

        async fn stop(&self) -> Result<()> {
            Ok(())
        }

        async fn send(&self, target: &str, msg: &ClusterMessage) -> Result<()> {
            self.sent.lock().await.push((target.to_string(), msg.clone()));
            Ok(())
        }

        async fn broadcast(&self, msg: &ClusterMessage) -> Result<()> {
            self.broadcasted.lock().await.push(msg.clone());
            Ok(())
        }

        fn messages(&self) -> mpsc::Receiver<ClusterMessage> {
            // 测试中不会通过 messages() 驱动；返回一个永远收不到消息的 receiver。
            mpsc::channel(1).1
        }

        fn members(&self) -> Vec<ClusterMember> {
            self.members_view.all_members()
        }
    }

    #[tokio::test]
    async fn ensure_member_inserts_and_updates() {
        let members = Arc::new(MemberList::new());
        let id = InstanceId::new("g1");
        let region = RegionId::new("r1");
        let digest = MetaDigest {
            kv_version: 5,
            ..MetaDigest::default()
        };
        GossipEngine::ensure_member(
            &members,
            &id,
            &region,
            Some("1.1.1.1:1".to_string()),
            digest.clone(),
        );
        assert_eq!(members.len(), 1);
        let m = members.get(&id).unwrap();
        assert_eq!(m.meta_digest.kv_version, 5);

        // 更新 digest
        let digest2 = MetaDigest {
            kv_version: 9,
            ..MetaDigest::default()
        };
        GossipEngine::ensure_member(&members, &id, &region, None, digest2.clone());
        let m = members.get(&id).unwrap();
        assert_eq!(m.meta_digest.kv_version, 9);
        // addr 不应被清空
        assert_eq!(m.addr, "1.1.1.1:1");
    }

    #[tokio::test]
    async fn handle_pong_updates_member_digest() {
        let members = Arc::new(MemberList::new());
        let transport: Arc<dyn ClusterTransport> = Arc::new(MockTransport::new(members.clone()));
        let handler: Arc<dyn GossipHandler> = Arc::new(CountingHandler::new(MetaDigest::default()));
        let engine = GossipEngine::new(
            InstanceId::new("self"),
            RegionId::new("r-self"),
            "0.0.0.0:1".to_string(),
            members.clone(),
            transport,
            ClusterConfig {
                bind_addr: "0.0.0.0:7946".to_string(),
                seed_peers: vec![],
                gossip_interval_ms: 1000,
                probe_timeout_ms: 5000,
                suspect_timeout_secs: 5,
            },
            handler,
        );

        let pong = ClusterMessage::Pong {
            sender: InstanceId::new("g2"),
            region: RegionId::new("r2"),
            meta_digest: MetaDigest {
                kv_version: 7,
                ..MetaDigest::default()
            },
            timestamp: 0,
        };
        engine.handle_message(pong).await;
        let m = members.get(&InstanceId::new("g2")).unwrap();
        assert_eq!(m.meta_digest.kv_version, 7);
        assert_eq!(m.status, MemberStatus::Alive);
    }

    #[tokio::test]
    async fn handle_metrics_broadcast_invokes_handler() {
        let members = Arc::new(MemberList::new());
        let transport: Arc<dyn ClusterTransport> = Arc::new(MockTransport::new(members.clone()));
        let handler = Arc::new(CountingHandler::new(MetaDigest::default()));
        let engine = GossipEngine::new(
            InstanceId::new("self"),
            RegionId::new("r-self"),
            "0.0.0.0:1".to_string(),
            members.clone(),
            transport,
            ClusterConfig {
                bind_addr: "0.0.0.0:7946".to_string(),
                seed_peers: vec![],
                gossip_interval_ms: 1000,
                probe_timeout_ms: 5000,
                suspect_timeout_secs: 5,
            },
            handler.clone(),
        );

        let metrics = BackendMetrics {
            active_requests: 1,
            queue_depth: 0,
            active_decode_blocks: 0,
            active_prefill_tokens: 0,
            kv_used_blocks: 0,
            kv_total_blocks: 10,
            gpu_utilization: 0.0,
            gpu_memory_used_mb: 0,
            gpu_memory_total_mb: 0,
            latency: aether_core::metrics::LatencyStats {
                p50_ms: 1.0,
                p99_ms: 2.0,
                p999_ms: 3.0,
                sample_count: 1,
            },
            timestamp: 1,
        };
        let msg = ClusterMessage::MetricsBroadcast {
            region: RegionId::new("r1"),
            backends: vec![("b1".to_string(), metrics)],
        };
        engine.handle_message(msg).await;
        assert_eq!(
            handler.metrics_calls.load(Ordering::Relaxed),
            1,
            "应调用一次 handle_metrics_broadcast"
        );
    }

    #[tokio::test]
    async fn handle_sync_request_returns_entries() {
        let members = Arc::new(MemberList::new());
        let transport: Arc<dyn ClusterTransport> = Arc::new(MockTransport::new(members.clone()));
        let handler = Arc::new(CountingHandler::new(MetaDigest::default()));
        let engine = GossipEngine::new(
            InstanceId::new("self"),
            RegionId::new("r-self"),
            "0.0.0.0:1".to_string(),
            members.clone(),
            transport.clone(),
            ClusterConfig {
                bind_addr: "0.0.0.0:7946".to_string(),
                seed_peers: vec![],
                gossip_interval_ms: 1000,
                probe_timeout_ms: 5000,
                suspect_timeout_secs: 5,
            },
            handler.clone(),
        );

        // 直接调用 handle_sync_request（不经传输）以验证返回值
        let keys: Vec<String> = vec![MetaKey::KvState.into(), MetaKey::Members.into()];
        let entries = handler.handle_sync_request(&keys);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key, "kv_state");
        assert_eq!(entries[1].key, "members");

        // 调用 handle_message(SyncRequest) 应通过 transport.send 回 SyncResponse
        // 由于对端不在成员列表中，会走 broadcast
        let req = ClusterMessage::SyncRequest {
            sender: InstanceId::new("g-other"),
            keys: keys.clone(),
        };
        engine.handle_message(req).await;
        // 至少一次 sync_request 被记录
        assert!(handler.sync_calls.load(Ordering::Relaxed) >= 1);
    }

    #[tokio::test]
    async fn noop_handler_does_not_panic() {
        let members = Arc::new(MemberList::new());
        let transport: Arc<dyn ClusterTransport> = Arc::new(MockTransport::new(members.clone()));
        let handler: Arc<dyn GossipHandler> = Arc::new(NoopGossipHandler);
        let engine = GossipEngine::new(
            InstanceId::new("self"),
            RegionId::new("r-self"),
            "0.0.0.0:1".to_string(),
            members.clone(),
            transport,
            ClusterConfig {
                bind_addr: "0.0.0.0:7946".to_string(),
                seed_peers: vec![],
                gossip_interval_ms: 1000,
                probe_timeout_ms: 5000,
                suspect_timeout_secs: 5,
            },
            handler,
        );

        let metrics = BackendMetrics {
            active_requests: 0,
            queue_depth: 0,
            active_decode_blocks: 0,
            active_prefill_tokens: 0,
            kv_used_blocks: 0,
            kv_total_blocks: 0,
            gpu_utilization: 0.0,
            gpu_memory_used_mb: 0,
            gpu_memory_total_mb: 0,
            latency: aether_core::metrics::LatencyStats {
                p50_ms: 0.0,
                p99_ms: 0.0,
                p999_ms: 0.0,
                sample_count: 0,
            },
            timestamp: 0,
        };

        // 各种消息都应能正常分发，不 panic
        engine
            .handle_message(ClusterMessage::MetricsBroadcast {
                region: RegionId::new("r"),
                backends: vec![("b".to_string(), metrics.clone())],
            })
            .await;

        engine
            .handle_message(ClusterMessage::SyncResponse { entries: vec![] })
            .await;

        engine
            .handle_message(ClusterMessage::SessionAffinityBroadcast {
                session: SessionId::new("s"),
                backend_region: RegionId::new("r"),
                backend_instance: BackendInstanceId::new("i").to_string(),
            })
            .await;
    }

    #[tokio::test]
    async fn handle_meet_adds_member_and_replies_pong() {
        let members = Arc::new(MemberList::new());
        let transport = Arc::new(MockTransport::new(members.clone()));
        let handler = Arc::new(CountingHandler::new(MetaDigest::default()));
        let engine = GossipEngine::new(
            InstanceId::new("self"),
            RegionId::new("r-self"),
            "0.0.0.0:1".to_string(),
            members.clone(),
            transport.clone(),
            ClusterConfig {
                bind_addr: "0.0.0.0:7946".to_string(),
                seed_peers: vec![],
                gossip_interval_ms: 1000,
                probe_timeout_ms: 5000,
                suspect_timeout_secs: 5,
            },
            handler,
        );

        let meet = ClusterMessage::Meet {
            sender: InstanceId::new("g-new"),
            region: RegionId::new("r-new"),
            addr: "10.0.0.2:7946".to_string(),
        };
        engine.handle_message(meet).await;

        // 新成员应被加入
        assert!(members.get(&InstanceId::new("g-new")).is_some());

        // 应该有一次 send（Pong）+ 一次 broadcast（Ping 传播）
        let sent = transport.sent.lock().await;
        assert!(!sent.is_empty());
        let broadcasted = transport.broadcasted.lock().await;
        assert!(!broadcasted.is_empty());
    }
}
