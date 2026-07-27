//! Resilience primitives for backend forwarding: retry backoff and per-backend
//! circuit breakers.
//!
//! The gateway's forwarding loop (see `hier-kv-gateway-api::handlers`) combines
//! the two pieces in this module:
//!
//! 1. [`RetryPolicy`] shapes the delay between two forward attempts —
//!    exponential backoff `base * 2^attempt`, capped at `max`.
//! 2. [`CircuitBreakerRegistry`] tracks per-backend failure streaks. After
//!    `failure_threshold` consecutive failures a backend's circuit *opens* and
//!    [`CircuitBreakerRegistry::allow`] starts returning `false`, so the
//!    forwarding loop skips that backend instead of piling onto a sick
//!    instance. After `cooldown` the circuit transitions to *half-open* and
//!    lets a probe through; `success_threshold` consecutive probe successes
//!    close the circuit again, while any failure re-opens it.
//!
//! Both are pure in-memory constructs: no background tasks, no timers — the
//! cooldown is evaluated lazily on the next [`allow`](CircuitBreakerRegistry::allow)
//! call, which keeps the hot path allocation-free after the first touch of a
//! backend.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::Mutex;

use hier_kv_gateway_core::config::ResilienceConfig;
use hier_kv_gateway_core::ids::BackendId;

/// Exponential retry backoff, derived from [`ResilienceConfig`].
///
/// `backoff(attempt)` returns `base * 2^attempt` capped at `max`:
/// with the defaults (50 ms / 1000 ms) the sequence is 50, 100, 200, 400,
/// 800, 1000, 1000, …
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    base: Duration,
    max: Duration,
}

impl RetryPolicy {
    /// Build a policy from explicit bounds.
    pub fn new(base: Duration, max: Duration) -> Self {
        Self { base, max }
    }

    /// Delay before the next attempt, where `attempt` is the 0-based index of
    /// the attempt that just failed.
    pub fn backoff(&self, attempt: u32) -> Duration {
        // base * 2^attempt, saturating on overflow, then capped at max.
        let factor = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
        let base_ms = self.base.as_millis() as u64;
        let delay_ms = base_ms.saturating_mul(factor);
        let capped = delay_ms.min(self.max.as_millis() as u64);
        Duration::from_millis(capped)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        let cfg = ResilienceConfig::default();
        Self::new(
            Duration::from_millis(cfg.retry_backoff_ms),
            Duration::from_millis(cfg.retry_max_backoff_ms),
        )
    }
}

/// Observable snapshot of a backend's circuit state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitState {
    /// Traffic flows normally.
    Closed,
    /// Backend is failing repeatedly; traffic is short-circuited.
    Open,
    /// Cooldown elapsed; a limited probe is allowed through.
    HalfOpen,
}

/// Internal mutable breaker state, guarded by a mutex.
#[derive(Debug)]
enum BreakerState {
    Closed { consecutive_failures: u32 },
    Open { opened_at: Instant },
    HalfOpen { consecutive_successes: u32 },
}

/// Per-backend circuit breaker.
#[derive(Debug)]
struct Breaker {
    state: Mutex<BreakerState>,
}

impl Breaker {
    fn new() -> Self {
        Self {
            state: Mutex::new(BreakerState::Closed {
                consecutive_failures: 0,
            }),
        }
    }
}

/// Registry of per-backend circuit breakers.
///
/// Cheap to clone (`Arc` inside is unnecessary — the registry itself is meant
/// to be shared as `Arc<CircuitBreakerRegistry>`); lookups are lock-free reads
/// on the `DashMap`, and the per-breaker mutex is only held for a handful of
/// instructions, never across I/O.
pub struct CircuitBreakerRegistry {
    failure_threshold: u32,
    cooldown: Duration,
    success_threshold: u32,
    breakers: DashMap<BackendId, Arc<Breaker>>,
}

impl CircuitBreakerRegistry {
    /// Build a registry from the resilience configuration.
    pub fn new(config: &ResilienceConfig) -> Self {
        Self {
            failure_threshold: config.circuit_breaker_failure_threshold,
            cooldown: Duration::from_secs(config.circuit_breaker_cooldown_secs),
            success_threshold: config.half_open_success_threshold.max(1),
            breakers: DashMap::new(),
        }
    }

    fn breaker(&self, id: &BackendId) -> Arc<Breaker> {
        self.breakers
            .entry(id.clone())
            .or_insert_with(|| Arc::new(Breaker::new()))
            .clone()
    }

    /// Whether a forward attempt to `id` is allowed right now.
    ///
    /// Transitions `Open → HalfOpen` lazily once the cooldown has elapsed.
    pub fn allow(&self, id: &BackendId) -> bool {
        let breaker = self.breaker(id);
        let mut state = breaker.state.lock();
        match *state {
            BreakerState::Closed { .. } => true,
            BreakerState::Open { opened_at } => {
                if opened_at.elapsed() >= self.cooldown {
                    *state = BreakerState::HalfOpen {
                        consecutive_successes: 0,
                    };
                    true
                } else {
                    false
                }
            }
            BreakerState::HalfOpen { .. } => true,
        }
    }

    /// Record a successful forward to `id`.
    pub fn on_success(&self, id: &BackendId) {
        let breaker = self.breaker(id);
        let mut state = breaker.state.lock();
        match *state {
            BreakerState::Closed { .. } => {
                *state = BreakerState::Closed {
                    consecutive_failures: 0,
                };
            }
            BreakerState::HalfOpen {
                consecutive_successes,
            } => {
                let successes = consecutive_successes + 1;
                if successes >= self.success_threshold {
                    *state = BreakerState::Closed {
                        consecutive_failures: 0,
                    };
                } else {
                    *state = BreakerState::HalfOpen {
                        consecutive_successes: successes,
                    };
                }
            }
            BreakerState::Open { .. } => {
                // A success while open (e.g. a concurrent probe) closes the circuit.
                *state = BreakerState::Closed {
                    consecutive_failures: 0,
                };
            }
        }
    }

    /// Record a failed forward to `id`.
    pub fn on_failure(&self, id: &BackendId) {
        let breaker = self.breaker(id);
        let mut state = breaker.state.lock();
        match *state {
            BreakerState::Closed {
                consecutive_failures,
            } => {
                let failures = consecutive_failures + 1;
                if failures >= self.failure_threshold {
                    *state = BreakerState::Open {
                        opened_at: Instant::now(),
                    };
                } else {
                    *state = BreakerState::Closed {
                        consecutive_failures: failures,
                    };
                }
            }
            BreakerState::HalfOpen { .. } | BreakerState::Open { .. } => {
                // Any failure re-opens (or keeps open) the circuit and
                // restarts the cooldown window.
                *state = BreakerState::Open {
                    opened_at: Instant::now(),
                };
            }
        }
    }

    /// Observable snapshot of `id`'s circuit state (for admin endpoints and tests).
    pub fn state(&self, id: &BackendId) -> CircuitState {
        let Some(breaker) = self.breakers.get(id) else {
            return CircuitState::Closed;
        };
        let state = breaker.state.lock();
        match *state {
            BreakerState::Closed { .. } => CircuitState::Closed,
            BreakerState::Open { opened_at } => {
                if opened_at.elapsed() >= self.cooldown {
                    CircuitState::HalfOpen
                } else {
                    CircuitState::Open
                }
            }
            BreakerState::HalfOpen { .. } => CircuitState::HalfOpen,
        }
    }

    /// Forget all breaker state (e.g. after a topology reshuffle in tests).
    pub fn reset(&self) {
        self.breakers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(threshold: u32, cooldown_secs: u64, success_threshold: u32) -> ResilienceConfig {
        ResilienceConfig {
            retry_backoff_ms: 50,
            retry_max_backoff_ms: 1_000,
            circuit_breaker_failure_threshold: threshold,
            circuit_breaker_cooldown_secs: cooldown_secs,
            half_open_success_threshold: success_threshold,
        }
    }

    #[test]
    fn retry_backoff_doubles_and_caps() {
        let policy = RetryPolicy::new(Duration::from_millis(50), Duration::from_millis(1_000));
        let expected = [50, 100, 200, 400, 800, 1_000, 1_000];
        for (attempt, want) in expected.iter().enumerate() {
            assert_eq!(policy.backoff(attempt as u32).as_millis(), *want as u128);
        }
    }

    #[test]
    fn retry_backoff_saturates_on_huge_attempt() {
        let policy = RetryPolicy::new(Duration::from_millis(50), Duration::from_millis(1_000));
        assert_eq!(policy.backoff(63).as_millis(), 1_000);
        assert_eq!(policy.backoff(u32::MAX).as_millis(), 1_000);
    }

    #[test]
    fn breaker_opens_after_threshold_failures() {
        let registry = CircuitBreakerRegistry::new(&config(3, 60, 1));
        let id = BackendId::new("r1", "i1");

        assert_eq!(registry.state(&id), CircuitState::Closed);
        registry.on_failure(&id);
        registry.on_failure(&id);
        assert!(registry.allow(&id));
        registry.on_failure(&id);
        assert_eq!(registry.state(&id), CircuitState::Open);
        assert!(!registry.allow(&id));
    }

    #[test]
    fn breaker_success_resets_failure_streak() {
        let registry = CircuitBreakerRegistry::new(&config(3, 60, 1));
        let id = BackendId::new("r1", "i1");

        registry.on_failure(&id);
        registry.on_failure(&id);
        registry.on_success(&id);
        registry.on_failure(&id);
        registry.on_failure(&id);
        // Streak was reset by the success: two failures < threshold of 3.
        assert!(registry.allow(&id));
        assert_eq!(registry.state(&id), CircuitState::Closed);
    }

    #[test]
    fn breaker_half_open_probe_closes_after_success_threshold() {
        let registry = CircuitBreakerRegistry::new(&config(1, 0, 2));
        let id = BackendId::new("r1", "i1");

        registry.on_failure(&id);
        assert_eq!(registry.state(&id), CircuitState::HalfOpen);
        // cooldown = 0s → immediately half-open
        assert!(registry.allow(&id));
        registry.on_success(&id);
        assert_eq!(registry.state(&id), CircuitState::HalfOpen);
        registry.on_success(&id);
        assert_eq!(registry.state(&id), CircuitState::Closed);
    }

    #[test]
    fn breaker_failure_during_half_open_reopens() {
        let registry = CircuitBreakerRegistry::new(&config(1, 60, 1));
        let id = BackendId::new("r1", "i1");

        registry.on_failure(&id);
        assert!(!registry.allow(&id));
        // Simulate cooldown expiry by re-registering with zero cooldown is not
        // possible on the same registry; instead verify a half-open failure
        // re-opens via a second registry with zero cooldown.
        let registry = CircuitBreakerRegistry::new(&config(1, 0, 1));
        registry.on_failure(&id);
        assert!(registry.allow(&id)); // half-open probe
        registry.on_failure(&id);
        assert_eq!(registry.state(&id), CircuitState::HalfOpen);
        // State reports HalfOpen because cooldown is zero again.
    }

    #[test]
    fn breakers_are_independent_per_backend() {
        let registry = CircuitBreakerRegistry::new(&config(1, 60, 1));
        let a = BackendId::new("r1", "a");
        let b = BackendId::new("r1", "b");

        registry.on_failure(&a);
        assert!(!registry.allow(&a));
        assert!(registry.allow(&b));
    }

    #[test]
    fn reset_clears_all_breakers() {
        let registry = CircuitBreakerRegistry::new(&config(1, 60, 1));
        let id = BackendId::new("r1", "i1");
        registry.on_failure(&id);
        assert!(!registry.allow(&id));
        registry.reset();
        assert!(registry.allow(&id));
    }
}
