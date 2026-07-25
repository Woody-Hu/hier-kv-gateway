//! 后端负载指标。
//!
//! 由后端周期性上报，被路由层用于负载感知路由决策。
//! 提供派生计算方法以便路由器快速估算 KV 缓存与 GPU 显存占用、剩余容量。

use serde::{Deserialize, Serialize};

/// 延迟统计分位数快照。
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LatencyStats {
    /// P50 延迟（毫秒）。
    pub p50_ms: f64,
    /// P99 延迟（毫秒）。
    pub p99_ms: f64,
    /// P999 延迟（毫秒）。
    pub p999_ms: f64,
    /// 统计样本数量。
    pub sample_count: u64,
}

/// 后端实时负载指标快照。
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BackendMetrics {
    /// 当前活跃请求数。
    pub active_requests: u64,
    /// 队列中等待处理的请求数。
    pub queue_depth: u64,
    /// 活跃的 decode 块数。
    pub active_decode_blocks: u64,
    /// 活跃的 prefill token 数。
    pub active_prefill_tokens: u64,
    /// 已使用的 KV 块数。
    pub kv_used_blocks: u64,
    /// KV 块总数。
    pub kv_total_blocks: u64,
    /// GPU 利用率，取值范围 [0.0, 1.0]。
    pub gpu_utilization: f64,
    /// 已用 GPU 显存（MB）。
    pub gpu_memory_used_mb: u64,
    /// GPU 总显存（MB）。
    pub gpu_memory_total_mb: u64,
    /// 延迟分位数统计。
    pub latency: LatencyStats,
    /// 指标采集时间戳（Unix 秒）。
    pub timestamp: i64,
}

impl BackendMetrics {
    /// KV 缓存占用率（已用块数 / 总块数）。
    ///
    /// 当总块数为 0 时返回 0.0 以避免除零。
    pub fn kv_cache_usage(&self) -> f64 {
        if self.kv_total_blocks == 0 {
            0.0
        } else {
            self.kv_used_blocks as f64 / self.kv_total_blocks as f64
        }
    }

    /// GPU 显存占用率（已用显存 / 总显存）。
    ///
    /// 当总显存为 0 时返回 0.0 以避免除零。
    pub fn gpu_memory_usage(&self) -> f64 {
        if self.gpu_memory_total_mb == 0 {
            0.0
        } else {
            self.gpu_memory_used_mb as f64 / self.gpu_memory_total_mb as f64
        }
    }

    /// 剩余可用容量，以 KV 块数为单位。
    ///
    /// 计算方式：总块数减去（已使用块数 + 活跃 decode 块数 + 队列等待折算）。
    /// 返回值小于 0 表示后端已超载。
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
