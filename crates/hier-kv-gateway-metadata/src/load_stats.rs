//! Backend load statistics: maintains the latest metrics for each backend along
//! with a sliding window of the most recent N samples.
//!
//! The read path is fully lock-free via [`arc_swap::ArcSwap`]; the write path
//! replaces the whole `Arc<BackendMetrics>` via CAS. The sliding window uses a
//! ring buffer to facilitate computing percentiles such as P50/P99 (the actual
//! aggregation is the responsibility of the routing layer).

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::metrics::BackendMetrics;
use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::Mutex;

/// Number of samples retained in the sliding window.
pub const RING_CAPACITY: usize = 60;

/// Backend load statistics collection.
pub struct LoadStats {
    /// backend → latest metrics (lock-free read).
    latest: DashMap<BackendId, Arc<ArcSwap<BackendMetrics>>>,
    /// backend → sliding window of historical samples.
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
    /// Create an empty load statistics collection.
    pub fn new() -> Self {
        Self {
            latest: DashMap::new(),
            history: DashMap::new(),
        }
    }

    /// Read the latest metrics for a backend (lock-free).
    pub fn get(&self, backend: &BackendId) -> Option<BackendMetrics> {
        let entry = self.latest.get(backend)?;
        let guard = entry.load_full();
        Some((*guard).clone())
    }

    /// Write the latest metrics for a backend, appending to the sliding window as well.
    pub fn update(&self, backend: BackendId, metrics: BackendMetrics) {
        // Ensure both maps have an entry
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

    /// Returns how stale the latest metrics are relative to `now`.
    ///
    /// Relies on the `BackendMetrics::timestamp` field (Unix seconds, i64); if
    /// hier-kv-gateway-core uses a different time representation, adapt here.
    pub fn freshness(&self, backend: &BackendId) -> Option<Duration> {
        let metrics = self.get(backend)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?;
        let ts = metrics_timestamp(&metrics);
        Some(Duration::from_secs(now.as_secs().saturating_sub(ts)))
    }

    /// Returns the most recent N samples (in write order, oldest first).
    pub fn history(&self, backend: &BackendId) -> Vec<BackendMetrics> {
        let Some(hist) = self.history.get(backend) else {
            return Vec::new();
        };
        let snapshot = hist.lock().snapshot();
        snapshot
    }

    /// Remove all statistics for a backend (used on offline).
    pub fn remove(&self, backend: &BackendId) {
        self.latest.remove(backend);
        self.history.remove(backend);
    }
}

/// Simple ring buffer: fixed capacity; overwrites the oldest element when full.
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

    /// Returns a snapshot in write order (oldest first).
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

/// Read the Unix-second timestamp from `BackendMetrics` and convert to `u64`.
///
/// `BackendMetrics::timestamp` is `i64`; negative values are treated as 0, and
/// non-negative values are cast directly to `u64`.
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
    use hier_kv_gateway_core::metrics::LatencyStats;

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
        // The oldest entry should be 5
        assert_eq!(h[0].timestamp, 5);
    }
}
