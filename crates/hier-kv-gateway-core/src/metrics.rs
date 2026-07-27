//! Backend load metrics.
//!
//! Reported periodically by backends and consumed by the routing layer for
//! load-aware routing decisions. Provides derived computation methods so the
//! router can quickly estimate KV cache and GPU memory usage and remaining capacity.

use serde::{Deserialize, Serialize};

/// Latency percentile snapshot.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LatencyStats {
    /// P50 latency (milliseconds).
    pub p50_ms: f64,
    /// P99 latency (milliseconds).
    pub p99_ms: f64,
    /// P999 latency (milliseconds).
    pub p999_ms: f64,
    /// Number of samples in the statistics.
    pub sample_count: u64,
}

/// Real-time backend load metrics snapshot.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BackendMetrics {
    /// Number of currently active requests.
    pub active_requests: u64,
    /// Number of requests waiting in the queue.
    pub queue_depth: u64,
    /// Number of active decode blocks.
    pub active_decode_blocks: u64,
    /// Number of active prefill tokens.
    pub active_prefill_tokens: u64,
    /// Number of KV blocks in use.
    pub kv_used_blocks: u64,
    /// Total number of KV blocks.
    pub kv_total_blocks: u64,
    /// GPU utilization, in the range [0.0, 1.0].
    pub gpu_utilization: f64,
    /// GPU memory used (MB).
    pub gpu_memory_used_mb: u64,
    /// Total GPU memory (MB).
    pub gpu_memory_total_mb: u64,
    /// Latency percentile statistics.
    pub latency: LatencyStats,
    /// Metrics collection timestamp (Unix seconds).
    pub timestamp: i64,
}

impl BackendMetrics {
    /// KV cache usage ratio (used blocks / total blocks).
    ///
    /// Returns 0.0 when the total number of blocks is 0 to avoid division by zero.
    pub fn kv_cache_usage(&self) -> f64 {
        if self.kv_total_blocks == 0 {
            0.0
        } else {
            self.kv_used_blocks as f64 / self.kv_total_blocks as f64
        }
    }

    /// GPU memory usage ratio (used memory / total memory).
    ///
    /// Returns 0.0 when total memory is 0 to avoid division by zero.
    pub fn gpu_memory_usage(&self) -> f64 {
        if self.gpu_memory_total_mb == 0 {
            0.0
        } else {
            self.gpu_memory_used_mb as f64 / self.gpu_memory_total_mb as f64
        }
    }

    /// Remaining available capacity, in KV blocks.
    ///
    /// Computed as: total blocks minus (used blocks + active decode blocks + queue wait conversion).
    /// A negative return value indicates the backend is overloaded.
    pub fn available_capacity(&self) -> i64 {
        let used = self.kv_used_blocks as i64 + self.active_decode_blocks as i64;
        self.kv_total_blocks as i64 - used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics() -> BackendMetrics {
        BackendMetrics {
            active_requests: 4,
            queue_depth: 2,
            active_decode_blocks: 8,
            active_prefill_tokens: 128,
            kv_used_blocks: 40,
            kv_total_blocks: 100,
            gpu_utilization: 0.5,
            gpu_memory_used_mb: 20_000,
            gpu_memory_total_mb: 40_000,
            latency: LatencyStats {
                p50_ms: 10.0,
                p99_ms: 50.0,
                p999_ms: 80.0,
                sample_count: 1_000,
            },
            timestamp: 1_700_000_000,
        }
    }

    #[test]
    fn kv_cache_usage_basic() {
        let m = sample_metrics();
        assert!((m.kv_cache_usage() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn kv_cache_usage_zero_total_returns_zero() {
        let mut m = sample_metrics();
        m.kv_total_blocks = 0;
        assert_eq!(m.kv_cache_usage(), 0.0);
    }

    #[test]
    fn gpu_memory_usage_basic() {
        let m = sample_metrics();
        assert!((m.gpu_memory_usage() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn gpu_memory_usage_zero_total_returns_zero() {
        let mut m = sample_metrics();
        m.gpu_memory_total_mb = 0;
        assert_eq!(m.gpu_memory_usage(), 0.0);
    }

    #[test]
    fn available_capacity_subtracts_decode() {
        let m = sample_metrics();
        // 100 - 40 - 8 = 52
        assert_eq!(m.available_capacity(), 52);
    }

    #[test]
    fn available_capacity_negative_when_overloaded() {
        let mut m = sample_metrics();
        m.kv_used_blocks = 200;
        assert_eq!(m.available_capacity(), -108);
    }
}
