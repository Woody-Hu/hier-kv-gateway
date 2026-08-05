//! Multi-tenant admission scheduler with token-bucket rate limiting and
//! priority-based admission control.
//!
//! # Architecture
//!
//! Each tenant is assigned a [`TokenBucket`] that limits their request rate.
//! When the system is saturated (backend capacity usage exceeds the configured
//! threshold), the scheduler activates priority-based admission:
//!
//! 1. **Premium** tenants get their reserved capacity fraction first.
//! 2. **Normal** tenants share the remaining capacity via weighted fair queuing.
//! 3. **Background** tenants are shed first under load.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use parking_lot::Mutex;

use hier_kv_gateway_core::config::TenantConfig;
use hier_kv_gateway_core::ids::TenantId;
use hier_kv_gateway_core::tenant::{
    AdmissionDecision, SaturationState, TenantPriority, TenantQuota,
};

// ---------------------------------------------------------------------------
// Token bucket
// ---------------------------------------------------------------------------

/// A token bucket rate limiter for a single tenant.
struct TokenBucket {
    rate: f64,
    max_burst: f64,
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(rate: f64, max_burst: Option<f64>) -> Self {
        let burst = max_burst.unwrap_or(rate.max(1.0));
        Self {
            rate,
            max_burst: burst,
            tokens: burst,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            self.tokens = (self.tokens + elapsed * self.rate).min(self.max_burst);
            self.last_refill = now;
        }
    }

    fn try_consume(&mut self, n: f64, now: Instant) -> bool {
        self.refill(now);
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    fn time_until_available(&self, n: f64) -> f64 {
        if n <= self.tokens {
            return 0.0;
        }
        (n - self.tokens) / self.rate.max(1e-9)
    }
}

// ---------------------------------------------------------------------------
// Per-tenant runtime bucket
// ---------------------------------------------------------------------------

struct TenantBucket {
    quota: TenantQuota,
    bucket: Mutex<TokenBucket>,
    in_flight: AtomicU64,
}

impl TenantBucket {
    fn new(quota: TenantQuota) -> Self {
        let rate = quota.max_rps.unwrap_or(f64::MAX);
        let burst = rate.max(1.0);
        Self {
            quota,
            bucket: Mutex::new(TokenBucket::new(rate, Some(burst))),
            in_flight: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Multi-tenant admission scheduler.
///
/// Thread-safe: all methods use interior mutability and can be called from
/// multiple async tasks concurrently.
pub struct TenantScheduler {
    config: TenantConfig,
    buckets: DashMap<TenantId, Arc<TenantBucket>>,
    total_capacity: AtomicU64,
    used_capacity: AtomicU64,
}

impl TenantScheduler {
    /// Create a new scheduler from the gateway tenant configuration.
    pub fn new(config: TenantConfig) -> Self {
        Self {
            config,
            buckets: DashMap::new(),
            total_capacity: AtomicU64::new(100),
            used_capacity: AtomicU64::new(0),
        }
    }

    /// Update the global capacity view.
    pub fn update_capacity(&self, total: u64, used: u64) {
        self.total_capacity.store(total.max(1), Ordering::Relaxed);
        self.used_capacity.store(used, Ordering::Relaxed);
    }

    /// Get the current saturation state.
    pub fn saturation_state(&self) -> SaturationState {
        let total = self.total_capacity.load(Ordering::Relaxed);
        let used = self.used_capacity.load(Ordering::Relaxed);
        SaturationState::new(total, used)
    }

    /// Check whether a request from the given tenant should be admitted.
    pub fn check_admission(&self, tenant_id: &TenantId) -> AdmissionDecision {
        if !self.config.enabled {
            return AdmissionDecision::Admitted;
        }

        let bucket = self.get_or_create_bucket(tenant_id);
        let now = Instant::now();

        // 1. Token-bucket rate limit check.
        {
            let mut tb = bucket.bucket.lock();
            if !tb.try_consume(1.0, now) {
                let retry_after = tb.time_until_available(1.0).ceil() as u64;
                return AdmissionDecision::RateLimited {
                    retry_after_secs: retry_after.max(1),
                    reason: format!(
                        "tenant {} exceeded RPS limit of {:.0}",
                        tenant_id,
                        bucket.quota.max_rps.unwrap_or(f64::MAX)
                    ),
                };
            }
        }

        // 2. Concurrent request limit check.
        if let Some(max_concurrent) = bucket.quota.max_concurrent {
            let in_flight = bucket.in_flight.load(Ordering::Relaxed);
            if in_flight >= max_concurrent as u64 {
                return AdmissionDecision::RateLimited {
                    retry_after_secs: 1,
                    reason: format!(
                        "tenant {} at max concurrent requests ({})",
                        tenant_id, max_concurrent
                    ),
                };
            }
        }

        // 3. Saturation / priority check.
        let sat = self.saturation_state();
        let is_saturated = sat.capacity_used >= self.config.saturation_threshold;
        if is_saturated {
            match bucket.quota.priority {
                TenantPriority::Premium => AdmissionDecision::Admitted,
                TenantPriority::Normal => {
                    let reserved = self.total_reserved_fraction();
                    let available = 1.0 - reserved;
                    if sat.capacity_used < available {
                        AdmissionDecision::Admitted
                    } else {
                        AdmissionDecision::Queued
                    }
                }
                TenantPriority::Background => AdmissionDecision::Queued,
            }
        } else {
            AdmissionDecision::Admitted
        }
    }

    /// Record that a request is starting (increment in-flight).
    pub fn record_request_start(&self, tenant_id: &TenantId) {
        let bucket = self.get_or_create_bucket(tenant_id);
        bucket.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a request has completed (decrement in-flight).
    pub fn record_request_end(&self, tenant_id: &TenantId) {
        if let Some(bucket) = self.buckets.get(tenant_id) {
            bucket.in_flight.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn get_or_create_bucket(&self, tenant_id: &TenantId) -> Arc<TenantBucket> {
        self.buckets
            .entry(tenant_id.clone())
            .or_insert_with(|| {
                let quota = self.resolve_quota(tenant_id);
                Arc::new(TenantBucket::new(quota))
            })
            .clone()
    }

    fn resolve_quota(&self, tenant_id: &TenantId) -> TenantQuota {
        for t in &self.config.tenants {
            if t.id == tenant_id.as_str() {
                return TenantQuota {
                    tenant_id: tenant_id.clone(),
                    priority: t.priority,
                    max_rps: t.max_rps.or(self.config.default_max_rps),
                    max_input_tpm: None,
                    max_output_tpm: None,
                    max_concurrent: t.max_concurrent.or(self.config.default_max_concurrent),
                    reserved_capacity_fraction: if t.priority == TenantPriority::Premium {
                        t.reserved_capacity_fraction
                    } else {
                        0.0
                    },
                };
            }
        }
        TenantQuota {
            tenant_id: tenant_id.clone(),
            priority: TenantPriority::Normal,
            max_rps: self.config.default_max_rps,
            max_input_tpm: None,
            max_output_tpm: None,
            max_concurrent: self.config.default_max_concurrent,
            reserved_capacity_fraction: 0.0,
        }
    }

    fn total_reserved_fraction(&self) -> f64 {
        let mut total = 0.0;
        self.buckets.iter().for_each(|entry| {
            let b = entry.value();
            if b.quota.priority == TenantPriority::Premium {
                total += b.quota.reserved_capacity_fraction;
            }
        });
        total.min(0.9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::config::TenantQuotaConfig;

    fn tenant_id(s: &str) -> TenantId {
        TenantId::from(s)
    }

    fn basic_config() -> TenantConfig {
        TenantConfig {
            enabled: true,
            default_max_rps: Some(100.0),
            default_max_concurrent: Some(10),
            saturation_threshold: 0.8,
            tenants: vec![
                TenantQuotaConfig {
                    id: "premium".to_string(),
                    priority: TenantPriority::Premium,
                    max_rps: Some(500.0),
                    max_concurrent: Some(50),
                    reserved_capacity_fraction: 0.3,
                },
                TenantQuotaConfig {
                    id: "background".to_string(),
                    priority: TenantPriority::Background,
                    max_rps: Some(10.0),
                    max_concurrent: Some(2),
                    reserved_capacity_fraction: 0.0,
                },
            ],
        }
    }

    #[test]
    fn tenant_admitted_when_disabled() {
        let cfg = TenantConfig {
            enabled: false,
            ..Default::default()
        };
        let s = TenantScheduler::new(cfg);
        assert_eq!(s.check_admission(&tenant_id("anyone")), AdmissionDecision::Admitted);
    }

    #[test]
    fn tenant_admitted_within_quota() {
        let s = TenantScheduler::new(basic_config());
        assert_eq!(s.check_admission(&tenant_id("premium")), AdmissionDecision::Admitted);
    }

    #[test]
    fn tenant_rate_limited_when_exceeding_rps() {
        let s = TenantScheduler::new(basic_config());
        let tid = tenant_id("background");
        for _ in 0..11 {
            let _ = s.check_admission(&tid);
        }
        assert!(matches!(s.check_admission(&tid), AdmissionDecision::RateLimited { .. }));
    }

    #[test]
    fn premium_tenant_gets_higher_quota() {
        let s = TenantScheduler::new(basic_config());
        let tid = tenant_id("premium");
        for _ in 0..11 {
            assert_eq!(s.check_admission(&tid), AdmissionDecision::Admitted);
        }
    }

    #[test]
    fn unknown_tenant_gets_default_quota() {
        let s = TenantScheduler::new(basic_config());
        assert_eq!(s.check_admission(&tenant_id("unknown")), AdmissionDecision::Admitted);
    }

    #[test]
    fn background_tenant_queued_when_saturated() {
        let cfg = TenantConfig {
            saturation_threshold: 0.5,
            ..basic_config()
        };
        let s = TenantScheduler::new(cfg);
        s.update_capacity(100, 60);
        assert_eq!(s.check_admission(&tenant_id("background")), AdmissionDecision::Queued);
    }

    #[test]
    fn premium_tenant_admitted_even_when_saturated() {
        let cfg = TenantConfig {
            saturation_threshold: 0.5,
            ..basic_config()
        };
        let s = TenantScheduler::new(cfg);
        s.update_capacity(100, 90);
        assert_eq!(s.check_admission(&tenant_id("premium")), AdmissionDecision::Admitted);
    }

    #[test]
    fn token_bucket_refill_and_consume() {
        let mut tb = TokenBucket::new(10.0, Some(10.0));
        let now = Instant::now();
        for _ in 0..10 {
            assert!(tb.try_consume(1.0, now));
        }
        assert!(!tb.try_consume(1.0, now));
        assert!((tb.time_until_available(1.0) - 0.1).abs() < 0.01);
    }

    #[test]
    fn token_bucket_refills_over_time() {
        let mut tb = TokenBucket::new(100.0, Some(1.0));
        let t0 = Instant::now();
        assert!(tb.try_consume(1.0, t0));
        assert!(!tb.try_consume(1.0, t0));
        let t1 = t0 + std::time::Duration::from_millis(20);
        assert!(tb.try_consume(1.0, t1));
    }

    #[test]
    fn record_request_start_end_tracks_in_flight() {
        let s = TenantScheduler::new(basic_config());
        let tid = tenant_id("premium");
        s.record_request_start(&tid);
        s.record_request_start(&tid);
        {
            let bucket = s.buckets.get(&tid).unwrap();
            assert_eq!(bucket.in_flight.load(Ordering::Relaxed), 2);
        }
        s.record_request_end(&tid);
        {
            let bucket = s.buckets.get(&tid).unwrap();
            assert_eq!(bucket.in_flight.load(Ordering::Relaxed), 1);
        }
    }
}
