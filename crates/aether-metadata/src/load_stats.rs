//! 后端负载统计：维护每个 backend 的最新指标，以及最近 N 个采样的滑动窗口。
//!
//! 读路径通过 [`arc_swap::ArcSwap`] 完全无锁；写路径通过 CAS 替换整个 `Arc<BackendMetrics>`。
//! 滑动窗口使用 ring buffer，便于计算 P50/P99 等分位数（具体聚合由 routing 层负责）。

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aether_core::ids::BackendId;
use aether_core::metrics::BackendMetrics;
use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::Mutex;

/// 滑动窗口保留的采样数量。
pub const RING_CAPACITY: usize = 60;

/// 后端负载统计集合。
pub struct LoadStats {
    /// backend → 最新指标（无锁读）。
    latest: DashMap<BackendId, Arc<ArcSwap<BackendMetrics>>>,
    /// backend → 滑动窗口历史采样。
    history: DashMap<BackendId, Mutex<RingBuffer<BackendMetrics>>>,
}

impl std::fmt::Debug for LoadStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadStats")
            .field("backends", &self.latest.len())
            .finish()
    }
}

impl Default for LoadStats {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadStats {
    /// 创建一个空的负载统计集合。
    pub fn new() -> Self {
        Self {
            latest: DashMap::new(),
            history: DashMap::new(),
        }
    }

    /// 读取 backend 的最新指标（无锁）。
    pub fn get(&self, backend: &BackendId) -> Option<BackendMetrics> {
        let entry = self.latest.get(backend)?;
        let guard = entry.load_full();
        Some((*guard).clone())
    }

    /// 写入 backend 的最新指标，并追加到滑动窗口。
    pub fn update(&self, backend: BackendId, metrics: BackendMetrics) {
        // 确保两个 map 都有条目
        let arc = self
            .latest
            .entry(backend.clone())
            .or_insert_with(|| Arc::new(ArcSwap::from_pointee(metrics.clone())))
            .clone();
        arc.store(Arc::new(metrics.clone()));

        let hist = self
            .history
            .entry(backend)
            .or_insert_with(|| Mutex::new(RingBuffer::new(RING_CAPACITY)));
        hist.lock().push(metrics);
    }

    /// 返回最新指标相对于 `now` 的过期时间。
    ///
    /// 依赖 `BackendMetrics::timestamp`（Unix 秒，i64）字段；若 aether-core 使用其他
    /// 时间表示，需在此适配。
    pub fn freshness(&self, backend: &BackendId) -> Option<Duration> {
        let metrics = self.get(backend)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?;
        let ts = metrics_timestamp(&metrics);
        Some(Duration::from_secs(now.as_secs().saturating_sub(ts)))
    }

    /// 返回最近 N 个采样（按写入顺序，最老在最前）。
    pub fn history(&self, backend: &BackendId) -> Vec<BackendMetrics> {
        let Some(hist) = self.history.get(backend) else {
            return Vec::new();
        };
        let snapshot = hist.lock().snapshot();
        snapshot
    }

    /// 移除某个 backend 的全部统计（用于下线）。
    pub fn remove(&self, backend: &BackendId) {
        self.latest.remove(backend);
        self.history.remove(backend);
    }
}

/// 简单的 ring buffer：固定容量，写满后覆盖最老元素。
pub struct RingBuffer<T> {
    buf: Vec<T>,
    head: usize,
    len: usize,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            head: 0,
            len: 0,
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.buf.len() < self.capacity {
            self.buf.push(item);
            self.len = self.buf.len();
            return;
        }
        self.buf[self.head] = item;
        self.head = (self.head + 1) % self.capacity;
        self.len = self.capacity;
    }

    /// 返回按写入顺序排列的快照（最老元素在前）。
    pub fn snapshot(&self) -> Vec<T> {
        if self.buf.len() < self.capacity {
            return self.buf.clone();
        }
        let mut out = Vec::with_capacity(self.capacity);
        for i in 0..self.capacity {
            out.push(self.buf[(self.head + i) % self.capacity].clone());
        }
        out
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// 从 `BackendMetrics` 中读取 Unix 秒时间戳并转为 `u64`。
///
/// `BackendMetrics::timestamp` 为 `i64`；负值视为 0，非负值直接转 `u64`。
fn metrics_timestamp(m: &BackendMetrics) -> u64 {
    if m.timestamp < 0 {
        0
    } else {
        m.timestamp as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_core::metrics::LatencyStats;

    fn metrics(ts: i64) -> BackendMetrics {
        BackendMetrics {
            timestamp: ts,
            active_requests: 0,
            queue_depth: 0,
            active_decode_blocks: 0,
            active_prefill_tokens: 0,
            kv_used_blocks: 0,
            kv_total_blocks: 0,
            gpu_utilization: 0.0,
            gpu_memory_used_mb: 0,
            gpu_memory_total_mb: 0,
            latency: LatencyStats {
                p50_ms: 0.0,
                p99_ms: 0.0,
                p999_ms: 0.0,
                sample_count: 0,
            },
        }
    }

    fn backend(n: u8) -> BackendId {
        BackendId::new(format!("r{n}"), format!("i{n}"))
    }

    #[test]
    fn ring_buffer_overwrites_oldest() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.push(4);
        assert_eq!(rb.snapshot(), vec![2, 3, 4]);
    }

    #[test]
    fn update_and_get() {
        let stats = LoadStats::new();
        let b = backend(1);
        stats.update(b.clone(), metrics(100));
        let m = stats.get(&b).expect("metrics should exist");
        assert_eq!(m.timestamp, 100);
    }

    #[test]
    fn history_keeps_last_n() {
        let stats = LoadStats::new();
        let b = backend(1);
        for i in 0..(RING_CAPACITY as i64 + 5) {
            stats.update(b.clone(), metrics(i));
        }
        let h = stats.history(&b);
        assert_eq!(h.len(), RING_CAPACITY);
        // 最老的一个应该是 5
        assert_eq!(h[0].timestamp, 5);
    }
}
