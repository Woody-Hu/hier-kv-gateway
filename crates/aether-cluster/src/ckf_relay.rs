//! KV Relay：跨 Region 的 CKF 投影发布（参考 Dynamo Multi-DC Relay）。
//!
//! 每个 Region 的 Gateway 实例维护一个 [`KvRelay`]，内部为每个 [`PoolId`] 维护
//! 一个独立的 [`CkfProducer`]。当后端 KV Cache 事件到达时，调用
//! [`apply_event`](KvRelay::apply_event) 把事件喂给对应 pool 的 producer；当累计
//! pending 事件数达到 `publication_threshold` 或后台周期性 timer 到点时，调用
//! [`flush`](KvRelay::flush) 把变更以 [`ClusterMessage::CkfDelta`]（优先）或
//! [`ClusterMessage::CkfBarrierSnapshot`]（无 delta 时回退）的形式广播到集群。
//!
//! 这种“delta 优先 + barrier 兜底”的设计参考了 Dynamo Multi-DC Relay 的两阶段
//! 发布模型：delta 用于稳态的低开销增量同步，barrier snapshot 用于断连恢复后
//! 重新对齐 lane。

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use tracing::warn;

use aether_core::error::Result;
use aether_core::ids::{BackendId, PoolId, RegionId};
use aether_core::kv_event::KvCacheEvent;

use aether_metadata::ckf_producer::CkfProducer;
use aether_metadata::ckf_consumer::CkfConsumer;
use aether_metadata::cuckoo_filter::BUCKETS_PER_LANE;

use crate::messages::ClusterMessage;
use crate::transport::ClusterTransport;

/// 默认 pending 事件阈值：累计到 16 个事件即触发 flush。
pub const DEFAULT_PUBLICATION_THRESHOLD: usize = 16;

/// 默认发布周期：1ms 检查一次是否有 pending。
///
/// 实际并非每毫秒都发布，而是在有 pending 时触发一次 flush。
pub const DEFAULT_PUBLICATION_DELAY_MS: u64 = 1;

/// KV Relay：每个 Region 一个，按 pool 维度维护本地 CKF producer 并广播变更。
pub struct KvRelay {
    /// 本实例所在区域。
    self_region: RegionId,
    /// 每个 pool 一个 producer。
    producers: DashMap<PoolId, CkfProducer>,
    /// 跨 Region 的 CKF 消费者（由 gateway 层共享）。
    consumer: Arc<CkfConsumer>,
    /// 集群传输层。
    transport: Arc<dyn ClusterTransport>,
    /// pending 事件累计阈值。
    publication_threshold: usize,
    /// 后台 flush 任务检查间隔（毫秒）。
    publication_delay_ms: u64,
    /// 当前未发布的 pending 事件数。
    pending_count: AtomicUsize,
    /// 后台任务运行标志。
    running: Arc<AtomicBool>,
}

impl KvRelay {
    /// 创建一个 KV Relay。
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

    /// 创建一个 KV Relay，自定义发布阈值与延迟。
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

    /// 应用一个 KV cache 事件。
    ///
    /// 1. 获取或创建该 pool 对应的 [`CkfProducer`]；
    /// 2. 调用 `producer.apply_event(event, backend)` 更新内部所有权与 fingerprint；
    /// 3. pending_count += 1；
    /// 4. 若 pending_count >= `publication_threshold`，立即触发 [`flush`](Self::flush)。
    pub fn apply_event(&self, pool: &PoolId, event: &KvCacheEvent, backend: &BackendId) {
        // DashMap 的 entry API 在持锁期间调用 apply_event；这里 producer 是 &mut，
        // 短锁内完成更新，开销可接受。
        let mut entry = self.producers.entry(pool.clone()).or_default();
        entry.apply_event(event, backend);
        drop(entry);

        let new_pending = self.pending_count.fetch_add(1, Ordering::Relaxed) + 1;
        if new_pending >= self.publication_threshold {
            // 立即触发 flush；忽略错误（错误已通过日志记录）
            let _ = self.flush();
        }
    }

    /// 把所有 producer 的待发布变更广播出去。
    ///
    /// 对每个 producer：
    /// - 优先调用 `delta()` 生成增量；若有则广播 [`ClusterMessage::CkfDelta`]；
    /// - 若无增量（dirty 为空），跳过该 producer（不发布无意义的 barrier snapshot）。
    /// - 调用方可在断连恢复后主动调用 [`publish_barrier`](Self::publish_barrier)
    ///   强制发布全量快照以重新对齐 lane。
    pub fn flush(&self) -> Result<()> {
        // 重置 pending_count；并发调用之间可能有竞争，但都是“重置或扣减”，语义可接受。
        self.pending_count.store(0, Ordering::Relaxed);

        // 注意：必须先收集所有 pool 的 key，再单独 get_mut。
        // DashMap 的 iter() 在遍历期间会持有 shard 读锁，
        // 若在迭代体内直接 get_mut 会与自身 iter 的读锁形成死锁。
        let pools: Vec<PoolId> = self
            .producers
            .iter()
            .map(|r| r.key().clone())
            .collect();

        for pool in pools {
            let Some(mut producer_ref) = self.producers.get_mut(&pool) else {
                continue;
            };
            // 优先尝试 delta
            if let Some(delta) = producer_ref.delta() {
                let dirty_count = delta.buckets.len();
                let msg = ClusterMessage::CkfDelta {
                    pool: pool.clone(),
                    sequence: delta.sequence,
                    prev_sequence: delta.prev_sequence,
                    dirty_buckets: delta.buckets.clone(),
                };
                // 注意：transport.broadcast 是 async，但本函数是 sync。
                // 这里通过 tokio::spawn 异步发送，避免阻塞 ingestion 线程。
                // 错误通过日志记录。
                let transport = self.transport.clone();
                tokio::spawn(async move {
                    if let Err(e) = transport.broadcast(&msg).await {
                        warn!(error = %e, dirty = dirty_count, "广播 CkfDelta 失败");
                    }
                });
            }
            // 没有 dirty bucket：什么都不发，避免无意义的 barrier snapshot。
        }
        Ok(())
    }

    /// 强制发布指定 pool 的全量快照（barrier snapshot）。
    ///
    /// 用于：
    /// - 启动时让所有对端安装最新 lane；
    /// - 检测到 lane 不一致时重新对齐。
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
                warn!(error = %e, "广播 CkfBarrierSnapshot 失败");
            }
        });
        Ok(())
    }

    /// 启动后台 flush 任务：每 `publication_delay_ms` 检查 pending_count，>0 时 flush。
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
                        warn!(error = %e, "KvRelay 后台 flush 失败");
                    }
                }
            }
        });
        Ok(())
    }

    /// 停止后台 flush 任务。
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// 返回当前 pending 事件数。
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(Ordering::Relaxed)
    }

    /// 当前跟踪的 pool 数量。
    pub fn pool_count(&self) -> usize {
        self.producers.len()
    }

    /// 取指定 pool 的 producer 引用快照（主要用于测试）。
    pub fn producer_snapshot(&self, pool: &PoolId) -> Option<u64> {
        self.producers.get(pool).map(|r| r.num_items())
    }

    /// 返回本 Relay 所在 Region。
    pub fn self_region(&self) -> &RegionId {
        &self.self_region
    }

    /// 返回内部 consumer 引用（用于上层查询 / lane 管理）。
    pub fn consumer(&self) -> &Arc<CkfConsumer> {
        &self.consumer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_core::ids::{BackendInstanceId, IndexerDomainId, InstanceId, WorkerWithRank};
    use async_trait::async_trait;
    use tokio::sync::Mutex;

    /// 用于测试的伪传输层：记录所有 broadcast 的消息。
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

        // 异步 broadcast 通过 tokio::spawn 发起，等待一下让任务执行完
        tokio::time::sleep(Duration::from_millis(50)).await;
        let msgs = transport.broadcasted_messages().await;
        assert!(!msgs.is_empty(), "应有至少一条 CkfDelta 广播");
        // 第一条应为 CkfDelta
        match &msgs[0] {
            ClusterMessage::CkfDelta {
                pool: p,
                dirty_buckets,
                ..
            } => {
                assert_eq!(p.domain.0, 1);
                assert!(!dirty_buckets.is_empty());
            }
            other => panic!("期望 CkfDelta，实际为 {:?}", other),
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
        // 第 1、2 个事件不触发 flush
        relay.apply_event(&pool, &stored_event(vec![1]), &backend("b1"));
        relay.apply_event(&pool, &stored_event(vec![2]), &backend("b1"));
        assert_eq!(relay.pending_count(), 2);

        // 第 3 个事件触发自动 flush，pending_count 归零
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
            "应广播 CkfBarrierSnapshot"
        );
    }

    #[tokio::test]
    async fn flush_without_changes_does_not_broadcast() {
        let consumer = Arc::new(CkfConsumer::new());
        let transport = Arc::new(RecordingTransport::new());
        let relay = KvRelay::new(RegionId::new("r1"), consumer, transport.clone());

        // 没有 producer 时 flush 不应广播任何消息
        relay.flush().unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(transport.broadcasted_messages().await.is_empty());

        // 有 producer 但 dirty 为空：先 snapshot 清掉 dirty，再 flush 也不应广播
        let pool = pool(1, "r1");
        relay.apply_event(&pool, &stored_event(vec![1]), &backend("b1"));
        // 第一次 flush 应广播 delta
        relay.flush().unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let n_after_first = transport.broadcasted_messages().await.len();
        assert!(n_after_first >= 1, "首次 flush 应广播 delta");

        // 再次 flush，dirty 已清空，不应广播
        relay.flush().unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let n_after_second = transport.broadcasted_messages().await.len();
        assert_eq!(
            n_after_second,
            n_after_first,
            "无 dirty 时不应再广播"
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
        // 不主动 flush，等后台 5ms timer 触发
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(relay.pending_count(), 0, "后台 loop 应已 flush");

        let msgs = transport.broadcasted_messages().await;
        assert!(!msgs.is_empty());

        relay.stop();
        // 让 stop 生效
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
