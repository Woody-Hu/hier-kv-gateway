//! KV Relay: cross-Region CKF projection publishing.
//!
//! Each Region's Gateway instance maintains a [`KvRelay`], which internally maintains an
//! independent [`CkfProducer`] for each [`PoolId`]. When a backend KV Cache event arrives,
//! call [`apply_event`](KvRelay::apply_event) to feed the event to the corresponding pool's
//! producer; when the accumulated number of pending events reaches `publication_threshold`
//! or the background periodic timer fires, call [`flush`](KvRelay::flush) to broadcast the
//! changes as [`ClusterMessage::CkfDelta`] (preferred) or [`ClusterMessage::CkfBarrierSnapshot`]
//! (fallback when there is no delta) to the cluster.
//!
//! This "delta-first + barrier-fallback" design adopts a two-phase publishing model: delta
//! is used for low-overhead incremental synchronization in steady state, and barrier
//! snapshot is used to realign lanes after reconnection recovery.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use tracing::warn;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::{BackendId, PoolId, RegionId};
use hier_kv_gateway_core::kv_event::KvCacheEvent;

use hier_kv_gateway_metadata::ckf_producer::CkfProducer;
use hier_kv_gateway_metadata::ckf_consumer::CkfConsumer;
use hier_kv_gateway_metadata::cuckoo_filter::BUCKETS_PER_LANE;

use crate::messages::ClusterMessage;
use crate::transport::ClusterTransport;

/// Default pending event threshold: triggers a flush when 16 events have accumulated.
pub const DEFAULT_PUBLICATION_THRESHOLD: usize = 16;

/// Default publish period: checks for pending every 1ms.
///
/// It does not actually publish every millisecond; instead, it triggers a flush when there is pending.
pub const DEFAULT_PUBLICATION_DELAY_MS: u64 = 1;

/// KV Relay: one per Region, maintains the local CKF producer per pool and broadcasts changes.
pub struct KvRelay {
    /// The Region this instance resides in.
    self_region: RegionId,
    /// One producer per pool.
    producers: DashMap<PoolId, CkfProducer>,
    /// Cross-Region CKF consumer (shared by the gateway layer).
    consumer: Arc<CkfConsumer>,
    /// Cluster transport layer.
    transport: Arc<dyn ClusterTransport>,
    /// Pending event accumulation threshold.
    publication_threshold: usize,
    /// Background flush task check interval (milliseconds).
    publication_delay_ms: u64,
    /// Current number of unpublished pending events.
    pending_count: AtomicUsize,
    /// Background task running flag.
    running: Arc<AtomicBool>,
}

impl KvRelay {
    /// Create a KV Relay.
    pub fn new(
        self_region: RegionId,
        consumer: Arc<CkfConsumer>,
        transport: Arc<dyn ClusterTransport>,
    ) -> Self {
        Self::with_params(
            self_region,
            consumer,
            transport,
            DEFAULT_PUBLICATION_THRESHOLD,
            DEFAULT_PUBLICATION_DELAY_MS,
        )
    }

    /// Create a KV Relay with custom publish threshold and delay.
    pub fn with_params(
        self_region: RegionId,
        consumer: Arc<CkfConsumer>,
        transport: Arc<dyn ClusterTransport>,
        publication_threshold: usize,
        publication_delay_ms: u64,
    ) -> Self {
        Self {
            self_region,
            producers: DashMap::new(),
            consumer,
            transport,
            publication_threshold,
            publication_delay_ms,
            pending_count: AtomicUsize::new(0),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Apply a KV cache event.
    ///
    /// 1. Get or create the [`CkfProducer`] for the pool;
    /// 2. Call `producer.apply_event(event, backend)` to update the internal ownership and fingerprint;
    /// 3. pending_count += 1;
    /// 4. If pending_count >= `publication_threshold`, immediately trigger [`flush`](Self::flush).
    pub fn apply_event(&self, pool: &PoolId, event: &KvCacheEvent, backend: &BackendId) {
        // The DashMap entry API calls apply_event while holding the lock; here producer is &mut,
        // and the update is done within a short lock, which is acceptable overhead.
        let mut entry = self.producers.entry(pool.clone()).or_default();
        entry.apply_event(event, backend);
        drop(entry);

        let new_pending = self.pending_count.fetch_add(1, Ordering::Relaxed) + 1;
        if new_pending >= self.publication_threshold {
            // Trigger flush immediately; ignore errors (errors are already logged)
            let _ = self.flush();
        }
    }

    /// Broadcast all producers' pending changes.
    ///
    /// For each producer:
    /// - Prefer calling `delta()` to generate a delta; if there is one, broadcast [`ClusterMessage::CkfDelta`];
    /// - If there is no delta (dirty is empty), skip this producer (do not publish a meaningless barrier snapshot).
    /// - The caller can actively call [`publish_barrier`](Self::publish_barrier) after reconnection
    ///   recovery to force-publish a full snapshot to realign the lane.
    pub fn flush(&self) -> Result<()> {
        // Reset pending_count; there may be races between concurrent calls, but both are "reset or decrement",
        // which is semantically acceptable.
        self.pending_count.store(0, Ordering::Relaxed);

        // Note: we must first collect the keys of all pools, then get_mut separately.
        // DashMap's iter() holds the shard read lock during iteration,
        // and calling get_mut directly inside the loop body would deadlock with the iter read lock.
        let pools: Vec<PoolId> = self
            .producers
            .iter()
            .map(|r| r.key().clone())
            .collect();

        for pool in pools {
            let Some(mut producer_ref) = self.producers.get_mut(&pool) else {
                continue;
            };
            // Prefer delta
            if let Some(delta) = producer_ref.delta() {
                let dirty_count = delta.buckets.len();
                let msg = ClusterMessage::CkfDelta {
                    pool: pool.clone(),
                    sequence: delta.sequence,
                    prev_sequence: delta.prev_sequence,
                    dirty_buckets: delta.buckets.clone(),
                };
                // Note: transport.broadcast is async, but this function is sync.
                // Here we send asynchronously via tokio::spawn to avoid blocking the ingestion thread.
                // Errors are logged.
                let transport = self.transport.clone();
                tokio::spawn(async move {
                    if let Err(e) = transport.broadcast(&msg).await {
                        warn!(error = %e, dirty = dirty_count, "Failed to broadcast CkfDelta");
                    }
                });
            }
            // No dirty bucket: send nothing, to avoid meaningless barrier snapshots.
        }
        Ok(())
    }

    /// Force-publish a full snapshot (barrier snapshot) of the specified pool.
    ///
    /// Used for:
    /// - Making all peers install the latest lane on startup;
    /// - Realigning when an inconsistent lane is detected.
    pub fn publish_barrier(&self, pool: &PoolId) -> Result<()> {
        let Some(mut producer_ref) = self.producers.get_mut(pool) else {
            return Ok(());
        };
        let snapshot = producer_ref.snapshot();
        drop(producer_ref);

        let msg = ClusterMessage::CkfBarrierSnapshot {
            pool: pool.clone(),
            sequence: snapshot.sequence,
            buckets: snapshot.buckets.clone(),
            num_buckets: BUCKETS_PER_LANE,
        };
        let transport = self.transport.clone();
        tokio::spawn(async move {
            if let Err(e) = transport.broadcast(&msg).await {
                warn!(error = %e, "Failed to broadcast CkfBarrierSnapshot");
            }
        });
        Ok(())
    }

    /// Start the background flush task: every `publication_delay_ms`, check pending_count and flush when > 0.
    pub fn start(self: &Arc<Self>) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let this = self.clone();
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(this.publication_delay_ms.max(1)));
            loop {
                if !this.running.load(Ordering::Relaxed) {
                    break;
                }
                interval.tick().await;
                if this.pending_count.load(Ordering::Relaxed) > 0 {
                    if let Err(e) = this.flush() {
                        warn!(error = %e, "KvRelay background flush failed");
                    }
                }
            }
        });
        Ok(())
    }

    /// Stop the background flush task.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Returns the current number of pending events.
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(Ordering::Relaxed)
    }

    /// Current number of tracked pools.
    pub fn pool_count(&self) -> usize {
        self.producers.len()
    }

    /// Take a snapshot of the producer reference for a specified pool (mainly for testing).
    pub fn producer_snapshot(&self, pool: &PoolId) -> Option<u64> {
        self.producers.get(pool).map(|r| r.num_items())
    }

    /// Returns the Region this Relay resides in.
    pub fn self_region(&self) -> &RegionId {
        &self.self_region
    }

    /// Returns a reference to the internal consumer (for upper-layer queries / lane management).
    pub fn consumer(&self) -> &Arc<CkfConsumer> {
        &self.consumer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::ids::{BackendInstanceId, IndexerDomainId, InstanceId, WorkerWithRank};
    use async_trait::async_trait;
    use tokio::sync::Mutex;

    /// A mock transport for tests: records all broadcast messages.
    struct RecordingTransport {
        broadcasted: Mutex<Vec<ClusterMessage>>,
    }

    impl RecordingTransport {
        fn new() -> Self {
            Self {
                broadcasted: Mutex::new(Vec::new()),
            }
        }

        async fn broadcasted_messages(&self) -> Vec<ClusterMessage> {
            self.broadcasted.lock().await.clone()
        }
    }

    #[async_trait]
    impl ClusterTransport for RecordingTransport {
        async fn start(
            &self,
            _self_id: &InstanceId,
            _region: &RegionId,
            _addr: &str,
        ) -> Result<()> {
            Ok(())
        }

        async fn stop(&self) -> Result<()> {
            Ok(())
        }

        async fn send(&self, _target: &str, _msg: &ClusterMessage) -> Result<()> {
            Ok(())
        }

        async fn broadcast(&self, msg: &ClusterMessage) -> Result<()> {
            self.broadcasted.lock().await.push(msg.clone());
            Ok(())
        }

        fn messages(&self) -> tokio::sync::mpsc::Receiver<ClusterMessage> {
            tokio::sync::mpsc::channel(1).1
        }

        fn members(&self) -> Vec<crate::member::ClusterMember> {
            Vec::new()
        }
    }

    fn pool(domain: u64, region: &str) -> PoolId {
        PoolId {
            domain: IndexerDomainId::new(domain),
            region: RegionId::new(region),
        }
    }

    fn stored_event(hashes: Vec<u64>) -> KvCacheEvent {
        KvCacheEvent::Stored {
            worker: WorkerWithRank::from_worker_id(1),
            block_hashes: hashes,
            parent_hash: None,
            num_block_tokens: Vec::new(),
        }
    }

    fn backend(name: &str) -> BackendId {
        BackendId::new("r1", BackendInstanceId::new(name))
    }

    #[tokio::test]
    async fn apply_event_creates_producer_and_tracks_pending() {
        let consumer = Arc::new(CkfConsumer::new());
        let transport = Arc::new(RecordingTransport::new());
        let relay = KvRelay::new(RegionId::new("r1"), consumer, transport.clone());

        let pool = pool(1, "r1");
        relay.apply_event(&pool, &stored_event(vec![1, 2, 3]), &backend("b1"));
        assert_eq!(relay.pending_count(), 1);
        assert_eq!(relay.pool_count(), 1);
        assert_eq!(relay.producer_snapshot(&pool).unwrap(), 3);
    }

    #[tokio::test]
    async fn flush_clears_pending_and_broadcasts_delta() {
        let consumer = Arc::new(CkfConsumer::new());
        let transport = Arc::new(RecordingTransport::new());
        let relay = KvRelay::new(RegionId::new("r1"), consumer, transport.clone());

        let pool = pool(1, "r1");
        relay.apply_event(&pool, &stored_event(vec![10, 20]), &backend("b1"));
        relay.apply_event(&pool, &stored_event(vec![30]), &backend("b2"));
        assert_eq!(relay.pending_count(), 2);

        relay.flush().unwrap();
        assert_eq!(relay.pending_count(), 0);

        // Async broadcast is initiated via tokio::spawn; wait a bit for the task to complete
        tokio::time::sleep(Duration::from_millis(50)).await;
        let msgs = transport.broadcasted_messages().await;
        assert!(!msgs.is_empty(), "There should be at least one CkfDelta broadcast");
        // The first one should be CkfDelta
        match &msgs[0] {
            ClusterMessage::CkfDelta {
                pool: p,
                dirty_buckets,
                ..
            } => {
                assert_eq!(p.domain.0, 1);
                assert!(!dirty_buckets.is_empty());
            }
            other => panic!("Expected CkfDelta, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn threshold_triggers_auto_flush() {
        let consumer = Arc::new(CkfConsumer::new());
        let transport = Arc::new(RecordingTransport::new());
        let relay = KvRelay::with_params(
            RegionId::new("r1"),
            consumer,
            transport.clone(),
            3,
            1000,
        );

        let pool = pool(1, "r1");
        // The 1st and 2nd events do not trigger a flush
        relay.apply_event(&pool, &stored_event(vec![1]), &backend("b1"));
        relay.apply_event(&pool, &stored_event(vec![2]), &backend("b1"));
        assert_eq!(relay.pending_count(), 2);

        // The 3rd event triggers an auto flush, pending_count goes to zero
        relay.apply_event(&pool, &stored_event(vec![3]), &backend("b1"));
        assert_eq!(relay.pending_count(), 0);

        tokio::time::sleep(Duration::from_millis(50)).await;
        let msgs = transport.broadcasted_messages().await;
        assert!(!msgs.is_empty());
    }

    #[tokio::test]
    async fn publish_barrier_broadcasts_snapshot() {
        let consumer = Arc::new(CkfConsumer::new());
        let transport = Arc::new(RecordingTransport::new());
        let relay = KvRelay::new(RegionId::new("r1"), consumer, transport.clone());

        let pool = pool(7, "r1");
        relay.apply_event(&pool, &stored_event(vec![1, 2]), &backend("b1"));

        relay.publish_barrier(&pool).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let msgs = transport.broadcasted_messages().await;
        assert!(
            msgs.iter().any(|m| matches!(
                m,
                ClusterMessage::CkfBarrierSnapshot { pool: p, .. } if p.domain.0 == 7
            )),
            "Should broadcast CkfBarrierSnapshot"
        );
    }

    #[tokio::test]
    async fn flush_without_changes_does_not_broadcast() {
        let consumer = Arc::new(CkfConsumer::new());
        let transport = Arc::new(RecordingTransport::new());
        let relay = KvRelay::new(RegionId::new("r1"), consumer, transport.clone());

        // With no producer, flush should not broadcast any messages
        relay.flush().unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(transport.broadcasted_messages().await.is_empty());

        // With a producer but dirty empty: first snapshot clears dirty, then flush should also not broadcast
        let pool = pool(1, "r1");
        relay.apply_event(&pool, &stored_event(vec![1]), &backend("b1"));
        // The first flush should broadcast a delta
        relay.flush().unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let n_after_first = transport.broadcasted_messages().await.len();
        assert!(n_after_first >= 1, "The first flush should broadcast a delta");

        // Flush again; dirty is already cleared, so nothing should be broadcast
        relay.flush().unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let n_after_second = transport.broadcasted_messages().await.len();
        assert_eq!(
            n_after_second,
            n_after_first,
            "Nothing should be broadcast when there is no dirty"
        );
    }

    #[tokio::test]
    async fn start_and_stop_background_loop() {
        let consumer = Arc::new(CkfConsumer::new());
        let transport = Arc::new(RecordingTransport::new());
        let relay = Arc::new(KvRelay::with_params(
            RegionId::new("r1"),
            consumer,
            transport.clone(),
            100,
            5,
        ));

        relay.start().unwrap();
        let pool = pool(1, "r1");
        relay.apply_event(&pool, &stored_event(vec![1, 2]), &backend("b1"));
        // Do not actively flush; wait for the background 5ms timer to trigger
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(relay.pending_count(), 0, "The background loop should have flushed");

        let msgs = transport.broadcasted_messages().await;
        assert!(!msgs.is_empty());

        relay.stop();
        // Let stop take effect
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
