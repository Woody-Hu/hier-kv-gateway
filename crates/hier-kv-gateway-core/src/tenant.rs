//! Multi-tenant resource types: quotas, priorities, admission decisions.
//!
//! This module defines the data structures used by the tenant-aware router to
//! enforce per-tenant rate limits, fair scheduling, and priority-based
//! admission control. It is intentionally pure data — the scheduling logic
//! lives in `hier_kv_gateway_routing::tenant_scheduler`.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::ids::TenantId;

/// Tenant priority level. Higher = more important.
///
/// When the system is saturated, higher-priority tenants get guaranteed
/// capacity first. Remaining capacity is distributed among lower-priority
/// tenants via weighted fair queuing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TenantPriority {
    /// Background / batch: lowest priority, shed first under load.
    Background = 0,
    /// Normal / default level.
    Normal = 1,
    /// Premium / SLA-guaranteed: reserved capacity, never starved.
    Premium = 2,
}

impl Default for TenantPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Per-tenant quota configuration.
///
/// All limits are optional — `None` means "unlimited for this dimension."
#[derive(Clone, Debug)]
pub struct TenantQuota {
    /// Tenant identifier.
    pub tenant_id: TenantId,
    /// Priority level.
    pub priority: TenantPriority,
    /// Maximum requests per second (RPS).
    pub max_rps: Option<f64>,
    /// Maximum input tokens per minute (TPM, input).
    pub max_input_tpm: Option<u64>,
    /// Maximum output tokens per minute (TPM, output).
    pub max_output_tpm: Option<u64>,
    /// Maximum concurrent requests (in-flight).
    pub max_concurrent: Option<u32>,
    /// Reserved fraction of total capacity (0.0–1.0).
    ///
    /// When the system is saturated, this fraction of capacity is reserved
    /// exclusively for this tenant. Only meaningful for `Premium` tenants.
    pub reserved_capacity_fraction: f64,
}

impl TenantQuota {
    /// Create a quota with default (unlimited) values for a tenant.
    pub fn new(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            priority: TenantPriority::default(),
            max_rps: None,
            max_input_tpm: None,
            max_output_tpm: None,
            max_concurrent: None,
            reserved_capacity_fraction: 0.0,
        }
    }

    /// Create a quota with the given priority.
    pub fn with_priority(tenant_id: TenantId, priority: TenantPriority) -> Self {
        Self {
            priority,
            ..Self::new(tenant_id)
        }
    }

    /// Create a premium tenant with reserved capacity.
    pub fn premium(tenant_id: TenantId, reserved_fraction: f64) -> Self {
        let tid = tenant_id.clone();
        Self {
            tenant_id,
            priority: TenantPriority::Premium,
            reserved_capacity_fraction: reserved_fraction,
            ..Self::new(tid)
        }
    }

    /// Whether this tenant has any finite limit.
    pub fn has_any_limit(&self) -> bool {
        self.max_rps.is_some()
            || self.max_input_tpm.is_some()
            || self.max_output_tpm.is_some()
            || self.max_concurrent.is_some()
    }
}

/// Per-tenant runtime state tracked by the scheduler.
#[derive(Debug)]
pub struct TenantState {
    /// Tenant identifier.
    pub tenant_id: TenantId,
    /// Priority level.
    pub priority: TenantPriority,
    /// Current number of in-flight requests.
    pub in_flight: u32,
    /// Accumulated input tokens in the current minute window.
    pub input_tokens_this_minute: u64,
    /// Accumulated output tokens in the current minute window.
    pub output_tokens_this_minute: u64,
    /// Accumulated requests in the current second window.
    pub requests_this_second: u64,
    /// Timestamp of the last request from this tenant.
    pub last_request_time: Instant,
    /// Timestamp when the token-per-minute window resets.
    pub tpm_window_start: Instant,
    /// Timestamp when the request-per-second window resets.
    pub rps_window_start: Instant,
}

impl TenantState {
    pub fn new(tenant_id: TenantId, priority: TenantPriority) -> Self {
        let now = Instant::now();
        Self {
            tenant_id,
            priority,
            in_flight: 0,
            input_tokens_this_minute: 0,
            output_tokens_this_minute: 0,
            requests_this_second: 0,
            last_request_time: now,
            tpm_window_start: now,
            rps_window_start: now,
        }
    }

    /// Reset the TPM window if it has elapsed.
    pub fn maybe_reset_tpm_window(&mut self, now: Instant) {
        if now.duration_since(self.tpm_window_start).as_secs() >= 60 {
            self.input_tokens_this_minute = 0;
            self.output_tokens_this_minute = 0;
            self.tpm_window_start = now;
        }
    }

    /// Reset the RPS window if it has elapsed.
    pub fn maybe_reset_rps_window(&mut self, now: Instant) {
        if now.duration_since(self.rps_window_start).as_secs() >= 1 {
            self.requests_this_second = 0;
            self.rps_window_start = now;
        }
    }
}

/// Result of an admission check for a single tenant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// Request is admitted; proceed to routing.
    Admitted,
    /// Rate limited: tenant has exceeded their quota. The HTTP layer should
    /// return 429 with a `Retry-After` hint.
    RateLimited {
        /// Suggested retry-after duration in seconds.
        retry_after_secs: u64,
        /// Human-readable reason.
        reason: String,
    },
    /// Queued: system is saturated but the tenant is within their quota.
    /// The request will be held until capacity is available.
    Queued,
}

/// Global saturation state used by the scheduler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaturationState {
    /// Fraction of total backend capacity currently in use (0.0–1.0).
    pub capacity_used: f64,
    /// Whether the system is considered saturated.
    pub is_saturated: bool,
    /// Total available capacity across all backends.
    pub total_capacity: u64,
    /// Total used capacity across all backends.
    pub used_capacity: u64,
}

impl SaturationState {
    /// Create a new saturation state.
    pub fn new(total_capacity: u64, used_capacity: u64) -> Self {
        let capacity_used = if total_capacity > 0 {
            used_capacity as f64 / total_capacity as f64
        } else {
            0.0
        };
        Self {
            capacity_used,
            is_saturated: capacity_used >= 0.8,
            total_capacity,
            used_capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_priority_ordering() {
        assert!(TenantPriority::Premium > TenantPriority::Normal);
        assert!(TenantPriority::Normal > TenantPriority::Background);
        assert_eq!(TenantPriority::default(), TenantPriority::Normal);
    }

    #[test]
    fn tenant_quota_has_any_limit() {
        let unlimited = TenantQuota::new(TenantId::from("t1"));
        assert!(!unlimited.has_any_limit());

        let mut limited = TenantQuota::new(TenantId::from("t2"));
        limited.max_rps = Some(10.0);
        assert!(limited.has_any_limit());
    }

    #[test]
    fn tenant_state_window_reset() {
        let mut state = TenantState::new(
            TenantId::from("t1"),
            TenantPriority::Normal,
        );
        state.input_tokens_this_minute = 100;
        state.requests_this_second = 5;

        // Advance time by 61 seconds
        state.tpm_window_start = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(61))
            .unwrap_or(state.tpm_window_start);
        state.rps_window_start = state.tpm_window_start;

        let now = std::time::Instant::now();
        state.maybe_reset_tpm_window(now);
        state.maybe_reset_rps_window(now);

        assert_eq!(state.input_tokens_this_minute, 0);
        assert_eq!(state.requests_this_second, 0);
    }

    #[test]
    fn saturation_state_is_saturated() {
        let s = SaturationState::new(100, 85);
        assert!(s.is_saturated);
        assert!((s.capacity_used - 0.85).abs() < 1e-9);

        let s2 = SaturationState::new(100, 50);
        assert!(!s2.is_saturated);
    }
}
