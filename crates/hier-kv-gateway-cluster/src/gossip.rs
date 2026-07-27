//! Gossip protocol engine.
//!
//! [`GossipEngine`] maintains a local [`MemberList`] and exchanges heartbeats and
//! metadata digests with other instances in the cluster via [`ClusterTransport`]. After
//! the engine starts, it runs three background tasks concurrently:
//!
//! 1. `gossip_loop`: every `gossip_interval_ms`, randomly selects at most `GOSSIP_FANOUT`
//!    alive members and sends them a [`ClusterMessage::Ping`].
//! 2. `probe_loop`: periodically scans the member list, marking members that have not
//!    received a Pong for more than `probe_timeout_ms` as [`MemberStatus::Suspect`];
//!    members that stay in Suspect for more than `suspect_timeout_secs` are marked as
//!    [`MemberStatus::Dead`].
//! 3. `message_loop`: receives [`ClusterMessage`] from the transport layer and dispatches
//!    it via [`handle_message`].
//!
//! Because messages like [`ClusterMessage::MetricsBroadcast`] /
//! [`ClusterMessage::CkfBarrierSnapshot`] / [`ClusterMessage::CkfDelta`] need to update
//! the local MetadataStore (in the upper-layer gateway), this engine does not directly
//! hold the MetadataStore; instead, the caller injects callback implementations via the
//! [`GossipHandler`] trait, keeping hier-kv-gateway-cluster decoupled from
//! hier-kv-gateway-metadata::store.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use rand::seq::SliceRandom;

use hier_kv_gateway_core::config::ClusterConfig;
use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::{InstanceId, PoolId, RegionId, SessionId};
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::topology::LatencyEstimate;

use crate::member::{now_unix_millis, ClusterMember, MemberList, MemberStatus};
use crate::messages::{ClusterMessage, MetaDigest, MetaEntry};
use crate::transport::ClusterTransport;

/// Default number of members selected per Gossip round (refer to the SWIM default).
const GOSSIP_FANOUT: usize = 3;

/// Scan interval of the probe loop (milliseconds).
///
/// Not tightly bound to `gossip_interval_ms`, to avoid probing being too sparse or too dense.
const PROBE_LOOP_INTERVAL_MS: u64 = 500;

/// Handler trait for metadata / metrics / CKF / topology / session affinity messages.
///
/// Implemented by the gateway layer (the component holding the MetadataStore) and injected
/// into the [`GossipEngine`], so that hier-kv-gateway-cluster does not directly depend on
/// hier-kv-gateway-metadata::store, only on the basic types of hier-kv-gateway-core.
#[async_trait]
pub trait GossipHandler: Send + Sync {
    /// Returns the current local metadata digest.
    ///
    /// Carried in PING/PONG for peer version comparison.
    fn current_meta_digest(&self) -> MetaDigest;

    /// Handles [`ClusterMessage::SyncRequest`]: returns the corresponding metadata entries for `keys`.
    fn handle_sync_request(&self, keys: &[String]) -> Vec<MetaEntry>;

    /// Applies the metadata entries in [`ClusterMessage::SyncResponse`] to the local store.
    fn apply_sync_response(&self, entries: &[MetaEntry]);

    /// Handles [`ClusterMessage::MetricsBroadcast`]: updates the local load statistics.
    fn handle_metrics_broadcast(
        &self,
        region: &RegionId,
        backends: &[(String, BackendMetrics)],
    );

    /// Handles [`ClusterMessage::CkfBarrierSnapshot`]: installs the full snapshot into the local CKF Consumer.
    fn handle_ckf_barrier_snapshot(
        &self,
        pool: &PoolId,
        sequence: u64,
        buckets: &[u64],
        num_buckets: usize,
    );

    /// Handles [`ClusterMessage::CkfDelta`]: applies the delta to the local CKF Consumer.
    fn handle_ckf_delta(
        &self,
        pool: &PoolId,
        sequence: u64,
        prev_sequence: u64,
        dirty_buckets: &[(usize, u64)],
    );

    /// Handles [`ClusterMessage::TopologyUpdate`]: updates the local latency matrix.
    fn handle_topology_update(
        &self,
        from: &RegionId,
        to: &RegionId,
        estimate: &LatencyEstimate,
    );

    /// Handles [`ClusterMessage::SessionAffinityBroadcast`]: updates the local session affinity history.
    fn handle_session_affinity(
        &self,
        session: &SessionId,
        backend_region: &RegionId,
        backend_instance: &str,
    );
}

/// No-op [`GossipHandler`]; all methods perform no actual updates.
///
/// Suitable for scenarios that do not need to handle upstream messages (e.g. pure send
/// tests, or when the gateway has not yet assembled a complete store).
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

/// Gossip protocol engine.
pub struct GossipEngine {
    /// This instance's identifier.
    self_id: InstanceId,
    /// The Region where this instance resides.
    self_region: RegionId,
    /// This instance's externally bound address (host:port), used to tell peers how to dial back via Meet.
    self_addr: String,
    /// Member list (shared with the transport layer).
    members: Arc<MemberList>,
    /// Cluster transport layer.
    transport: Arc<dyn ClusterTransport>,
    /// Cluster configuration.
    config: ClusterConfig,
    /// Running flag; all background tasks exit when false.
    running: Arc<AtomicBool>,
    /// Handler for metadata / metrics / CKF and other messages.
    handler: Arc<dyn GossipHandler>,
}

impl GossipEngine {
    /// Create a Gossip engine.
    ///
    /// The engine does not actively listen on creation; you must call
    /// [`start`](GossipEngine::start) to launch background tasks.
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

    /// Start the Gossip engine, starting the transport layer and spawning three background tasks.
    ///
    /// This method returns immediately; the engine keeps running until [`stop`](GossipEngine::stop) is called.
    pub async fn start(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            // Already started
            return Ok(());
        }

        // Start the transport layer
        self.transport
            .start(&self.self_id, &self.self_region, &self.self_addr)
            .await?;

        // Add self to the member list
        let self_member = ClusterMember::new(
            self.self_id.clone(),
            self.self_region.clone(),
            self.self_addr.clone(),
            self.handler.current_meta_digest(),
        );
        self.members.upsert(self_member);

        // Get the inbound message receiver; ownership of the receiver is moved into the message_loop task.
        let receiver = self.transport.messages();

        // Start gossip_loop
        let gossip_handle = tokio::spawn(Self::gossip_loop(
            self.self_id.clone(),
            self.self_region.clone(),
            self.members.clone(),
            self.transport.clone(),
            self.handler.clone(),
            self.config.gossip_interval_ms,
            self.running.clone(),
        ));

        // Start probe_loop
        let probe_handle = tokio::spawn(Self::probe_loop(
            self.self_id.clone(),
            self.members.clone(),
            self.config.probe_timeout_ms,
            self.config.suspect_timeout_secs,
            self.running.clone(),
        ));

        // Start message_loop
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

        // Let the three task handles run independently in the background; the engine stops via the `running` flag.
        // Here we detach the three JoinHandles to avoid blocking the caller.
        tokio::spawn(async move {
            let _ = gossip_handle.await;
            let _ = probe_handle.await;
            let _ = message_handle.await;
        });

        Ok(())
    }

    /// Join the cluster: send [`ClusterMessage::Meet`] to each seed node.
    pub async fn join_cluster(&self, seed_peers: &[String]) -> Result<()> {
        let meet = ClusterMessage::Meet {
            sender: self.self_id.clone(),
            region: self.self_region.clone(),
            addr: self.self_addr.clone(),
        };
        for seed in seed_peers {
            if let Err(e) = self.transport.send(seed, &meet).await {
                warn!(target = %seed, error = %e, "Failed to send Meet to seed node");
            }
        }
        Ok(())
    }

    /// Stop the engine: clear the running flag and stop the transport layer.
    pub async fn stop(&self) -> Result<()> {
        if !self.running.swap(false, Ordering::SeqCst) {
            return Ok(());
        }
        self.transport.stop().await
    }

    /// Handle an inbound message.
    ///
    /// This method is public so the upper layer can call it directly in tests without going through the transport layer.
    pub async fn handle_message(&self, msg: ClusterMessage) {
        match msg {
            ClusterMessage::Ping {
                sender,
                region,
                meta_digest,
                timestamp: _,
            } => {
                // Add the sender to the member list (overwrites the digest if it already exists)
                Self::ensure_member(
                    &self.members,
                    &sender,
                    &region,
                    None,
                    meta_digest.clone(),
                );

                // Reply with Pong
                let pong = ClusterMessage::Pong {
                    sender: self.self_id.clone(),
                    region: self.self_region.clone(),
                    meta_digest: self.handler.current_meta_digest(),
                    timestamp: now_unix_millis(),
                };
                // Look up the sender's address and send it back
                if let Some(member) = self.members.get(&sender) {
                    if let Err(e) = self.transport.send(&member.addr, &pong).await {
                        warn!(error = %e, "Failed to reply Pong");
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
                // Meet received: add the sender to the member list
                Self::ensure_member(
                    &self.members,
                    &sender,
                    &region,
                    Some(addr.clone()),
                    self.handler.current_meta_digest(),
                );

                // Reply with Pong to the sender, so the peer learns about us
                let pong = ClusterMessage::Pong {
                    sender: self.self_id.clone(),
                    region: self.self_region.clone(),
                    meta_digest: self.handler.current_meta_digest(),
                    timestamp: now_unix_millis(),
                };
                if let Err(e) = self.transport.send(&addr, &pong).await {
                    warn!(error = %e, "Failed to reply Pong for Meet");
                }

                // Propagate the new member: broadcast a Ping so other instances can also see this new member
                // (Here we leverage Ping to carry our own digest; on receiving it, peers reply with Pong,
                //  indirectly achieving dissemination; to avoid broadcast storms, we only trigger one
                //  broadcast on receiving a Meet)
                let ping = ClusterMessage::Ping {
                    sender: self.self_id.clone(),
                    region: self.self_region.clone(),
                    meta_digest: self.handler.current_meta_digest(),
                    timestamp: now_unix_millis(),
                };
                if let Err(e) = self.transport.broadcast(&ping).await {
                    warn!(error = %e, "Failed to broadcast Ping after Meet");
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
                        warn!(error = %e, "Failed to reply SyncResponse");
                    }
                } else {
                    // Sender address unknown: fall back to broadcast (higher cost, but ensures reachability)
                    if let Err(e) = self.transport.broadcast(&response).await {
                        warn!(error = %e, "Failed to broadcast SyncResponse");
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

    /// Add an instance to the member list (insert if absent, otherwise update digest / addr).
    fn ensure_member(
        members: &Arc<MemberList>,
        instance_id: &InstanceId,
        region: &RegionId,
        addr: Option<String>,
        digest: MetaDigest,
    ) {
        match members.get(instance_id) {
            Some(existing) => {
                // Exists: update the digest; also update the addr if a new one is provided
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
                // Does not exist: insert a new member; addr must be provided
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

    /// Gossip loop: every `interval_ms`, randomly select `GOSSIP_FANOUT` alive members and send them a Ping.
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

            // Do not Ping ourselves
            let candidates: Vec<ClusterMember> = members
                .alive_members()
                .into_iter()
                .filter(|m| m.instance_id != self_id)
                .collect();
            if candidates.is_empty() {
                continue;
            }
            // Note: the ThreadRng returned by `rand::rng()` is not Send,
            // and must be dropped before entering an await, otherwise the future is not Send.
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
                    debug!(error = %e, target = %member.addr, "Failed to send Ping");
                }
            }
        }
    }

    /// Probe loop: detects Pong timeout and Suspect timeout.
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

            // Scan all members and perform state transitions
            for member in members.all_members() {
                // Do not probe ourselves
                if member.instance_id == self_id {
                    continue;
                }
                match member.status {
                    MemberStatus::Alive => {
                        if member.last_pong_unix < probe_too_old {
                            members.mark_suspect(&member.instance_id);
                            debug!(instance = %member.instance_id, "Member Pong timed out, marking as Suspect");
                        }
                    }
                    MemberStatus::Suspect => {
                        if member.last_pong_unix < suspect_too_old {
                            members.mark_dead(&member.instance_id);
                            debug!(instance = %member.instance_id, "Suspect timed out, marking as Dead");
                        }
                    }
                    MemberStatus::Dead => {
                        // Already offline, awaiting cleanup (the upper layer can call remove)
                    }
                }
            }
        }
    }

    /// Message loop: receives messages from the transport layer and hands them to [`handle_message`](Self::handle_message).
    ///
    /// Note: because `handle_message` requires `&self`, but `message_loop` is a spawned
    /// static task and cannot directly hold `&GossipEngine`; here we replicate a closure
    /// logic equivalent to `handle_message`. To keep the code consistent with
    /// `handle_message`, this function delegates the message dispatch logic to a
    /// standalone `dispatch` helper function.
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
            // Receive a message: use recv() to block waiting; the loop exits on the next iteration after the running flag changes.
            let msg = match receiver.recv().await {
                Some(m) => m,
                None => {
                    // Channel closed: the transport layer has stopped
                    break;
                }
            };

            // Ignore loopback of messages sent by ourselves (to avoid receiving our own broadcasts)
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

/// Determines whether the `sender` field of a message is self (used to filter broadcast loopback).
fn message_sender_is_self(msg: &ClusterMessage, self_id: &InstanceId) -> bool {
    match msg {
        ClusterMessage::Ping { sender, .. }
        | ClusterMessage::Pong { sender, .. }
        | ClusterMessage::Meet { sender, .. }
        | ClusterMessage::SyncRequest { sender, .. } => sender == self_id,
        // Other messages have no sender field and cannot be determined; treat as non-self conservatively.
        _ => false,
    }
}

/// Dispatches a message to the corresponding handling logic (equivalent to [`GossipEngine::handle_message`]).
///
/// Extracted as a standalone function to be called from the `message_loop` task, avoiding borrowing `&self`.
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
                    warn!(error = %e, "Failed to reply Pong");
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
                warn!(error = %e, "Failed to reply Pong for Meet");
            }
            // Propagate the new member
            let ping = ClusterMessage::Ping {
                sender: self_id.clone(),
                region: self_region.clone(),
                meta_digest: handler.current_meta_digest(),
                timestamp: now_unix_millis(),
            };
            if let Err(e) = transport.broadcast(&ping).await {
                warn!(error = %e, "Failed to broadcast Ping after Meet");
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
                    warn!(error = %e, "Failed to reply SyncResponse");
                }
            } else if let Err(e) = transport.broadcast(&response).await {
                warn!(error = %e, "Failed to broadcast SyncResponse");
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

/// Converts an error into a [`HierKvGatewayError::ClusterError`].
#[allow(dead_code)]
fn to_cluster_error<E: std::fmt::Display>(e: E) -> HierKvGatewayError {
    HierKvGatewayError::ClusterError(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::MetaKey;
    use hier_kv_gateway_core::ids::BackendInstanceId;
    use async_trait::async_trait;
    use std::sync::atomic::AtomicU64;
    use tokio::sync::Mutex;

    /// A simple handler for tests: records all received sync responses, metrics, CKF and other events.
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
            // Only record the last backend identifier, to avoid cloning the entire metrics in tests.
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

    /// A mock transport for tests: records all sent messages into a Vec.
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
            // Who do we hand rx to? transport.messages() will return it.
            // Here we use a OnceCell pattern to simulate "take-once".
            // But messages() must return a Receiver, and start must be called before messages().
            // Simplification: store rx in transport itself; messages() takes it out.
            std::mem::forget(rx); // Avoid compile errors; tests will not call messages
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
            // Not driven via messages() in tests; return a receiver that never receives messages.
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

        // Update the digest
        let digest2 = MetaDigest {
            kv_version: 9,
            ..MetaDigest::default()
        };
        GossipEngine::ensure_member(&members, &id, &region, None, digest2.clone());
        let m = members.get(&id).unwrap();
        assert_eq!(m.meta_digest.kv_version, 9);
        // addr should not be cleared
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
            latency: hier_kv_gateway_core::metrics::LatencyStats {
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
            "handle_metrics_broadcast should be called once"
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

        // Call handle_sync_request directly (without going through the transport) to verify the return value
        let keys: Vec<String> = vec![MetaKey::KvState.into(), MetaKey::Members.into()];
        let entries = handler.handle_sync_request(&keys);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key, "kv_state");
        assert_eq!(entries[1].key, "members");

        // Calling handle_message(SyncRequest) should reply with a SyncResponse via transport.send
        // Since the peer is not in the member list, it will go through broadcast
        let req = ClusterMessage::SyncRequest {
            sender: InstanceId::new("g-other"),
            keys: keys.clone(),
        };
        engine.handle_message(req).await;
        // At least one sync_request is recorded
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
            latency: hier_kv_gateway_core::metrics::LatencyStats {
                p50_ms: 0.0,
                p99_ms: 0.0,
                p999_ms: 0.0,
                sample_count: 0,
            },
            timestamp: 0,
        };

        // All kinds of messages should be dispatchable without panicking
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

        // The new member should be added
        assert!(members.get(&InstanceId::new("g-new")).is_some());

        // There should be one send (Pong) + one broadcast (Ping propagation)
        let sent = transport.sent.lock().await;
        assert!(!sent.is_empty());
        let broadcasted = transport.broadcasted.lock().await;
        assert!(!broadcasted.is_empty());
    }
}
