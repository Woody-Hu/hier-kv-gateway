//! Adaptive hybrid-weight feedback controller.
//!
//! The static [`StrategyWeights`] from the config file are a good prior, but
//! they cannot react to runtime conditions: a backend fleet under failure
//! pressure benefits from stronger load awareness, while a workload with a
//! consistently high KV hit ratio deserves a stronger KV signal.
//!
//! [`AdaptiveWeightController`] closes that loop. It consumes two signal
//! families and periodically recomputes the *effective* weights around the
//! configured base weights:
//!
//! 1. **Execution metrics** recorded by the forwarding loop:
//!    - per-backend forward success/failure (EMA of the success rate),
//!    - per-backend forward latency (EMA, kept for observability and future
//!      latency-driven adjustments),
//!    - the KV hit ratio of served requests (`kv_overlap / prompt_blocks`).
//! 2. **Broadcast node state**: the per-backend load snapshots in the
//!    [`MetadataStore`] — fed by local collectors *and* by peer gateways'
//!    `METRIC_BROADCAST` gossip messages — used to measure the load spread
//!    across the fleet.
//!
//! Adjustment rules (all bounded by [`AdaptiveConfig::max_adjustment`]):
//!
//! - **Failure pressure** (worst per-backend EMA success rate below 1.0)
//!   shifts weight toward *load*: when some backends are erroring out,
//!   queue-aware routing avoids them more reliably than KV or topology.
//! - **KV hit EMA above 0.5** boosts the *kv* weight proportionally: a
//!   demonstrably cache-friendly workload should lean harder on KV overlap.
//! - **Load spread** (coefficient of variation of `active_requests` across
//!   the fleet) boosts *load*: high variance means the load signal
//!   discriminates well between candidates.
//!
//! After boosting, weights are renormalized and clamped to
//! [`AdaptiveConfig::min_weight`] so no dimension can be starved out. When
//! no signal has been observed yet, the controller returns the base weights
//! unchanged, degrading gracefully to static behaviour.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use hier_kv_gateway_core::config::{AdaptiveConfig, StrategyWeights};
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_metadata::store::MetadataStore;

/// Per-backend execution feedback, fed by the forwarding loop.
#[derive(Clone, Copy, Debug)]
pub struct OutcomeStats {
    /// EMA of the forward success rate in `[0, 1]`.
    pub ema_success: f64,
    /// EMA of the forward attempt latency in milliseconds.
    pub ema_latency_ms: f64,
    /// Total attempts recorded.
    pub samples: u64,
}

impl Default for OutcomeStats {
    fn default() -> Self {
        Self {
            ema_success: 1.0,
            ema_latency_ms: 0.0,
            samples: 0,
        }
    }
}

/// Controller recomputing hybrid weights from runtime signals.
pub struct AdaptiveWeightController {
    /// Static base weights from the configuration.
    base: StrategyWeights,
    /// Tuning parameters.
    cfg: AdaptiveConfig,
    /// Per-backend forward outcome EMAs, updated by the forwarding loop.
    outcomes: DashMap<BackendId, OutcomeStats>,
    /// EMA of the KV hit ratio (`kv_overlap / prompt_blocks`) of served requests.
    kv_hit_ema: Mutex<Option<f64>>,
    /// Last recomputation time; recomputes at most every `adjust_interval_secs`.
    last_adjust: Mutex<Instant>,
    /// Cached effective weights used between recomputations.
    current: Mutex<StrategyWeights>,
}

impl AdaptiveWeightController {
    /// Create a controller. The cached weights start at the base weights.
    pub fn new(base: StrategyWeights, cfg: AdaptiveConfig) -> Self {
        Self {
            current: Mutex::new(base.clone()),
            base,
            cfg,
            outcomes: DashMap::new(),
            kv_hit_ema: Mutex::new(None),
            last_adjust: Mutex::new(Instant::now()),
        }
    }

    /// Record a successful forward attempt against `backend`.
    pub fn record_success(&self, backend: &BackendId, latency: Duration) {
        let alpha = self.cfg.ema_alpha.clamp(f64::EPSILON, 1.0);
        let mut entry = self.outcomes.entry(backend.clone()).or_default();
        let s = entry.value_mut();
        s.ema_success = ema(s.ema_success, 1.0, alpha, s.samples);
        s.ema_latency_ms = ema(s.ema_latency_ms, latency.as_secs_f64() * 1e3, alpha, s.samples);
        s.samples += 1;
    }

    /// Record a failed forward attempt against `backend`.
    pub fn record_failure(&self, backend: &BackendId) {
        let alpha = self.cfg.ema_alpha.clamp(f64::EPSILON, 1.0);
        let mut entry = self.outcomes.entry(backend.clone()).or_default();
        let s = entry.value_mut();
        s.ema_success = ema(s.ema_success, 0.0, alpha, s.samples);
        s.samples += 1;
    }

    /// Record the KV hit ratio of a served request.
    pub fn record_kv_overlap(&self, overlap: u32, prompt_blocks: u32) {
        if prompt_blocks == 0 {
            return;
        }
        let ratio = (overlap as f64 / prompt_blocks as f64).clamp(0.0, 1.0);
        let alpha = self.cfg.ema_alpha.clamp(f64::EPSILON, 1.0);
        let mut ema_slot = self.kv_hit_ema.lock().unwrap();
        *ema_slot = Some(match *ema_slot {
            Some(prev) => prev + alpha * (ratio - prev),
            None => ratio,
        });
    }

    /// The weights currently in effect (also used for telemetry snapshots).
    pub fn current_weights(&self) -> StrategyWeights {
        self.current.lock().unwrap().clone()
    }

    /// Outcome snapshot for a backend, if any attempts were recorded.
    pub fn outcome_stats(&self, backend: &BackendId) -> Option<OutcomeStats> {
        self.outcomes.get(backend).map(|e| *e.value())
    }

    /// Current KV hit-ratio EMA, if any served request carried KV context.
    pub fn kv_hit_ratio(&self) -> Option<f64> {
        *self.kv_hit_ema.lock().unwrap()
    }

    /// Return the effective weights, recomputing them when the adjustment
    /// interval has elapsed.
    ///
    /// The hot path is a cheap lock + clock check between recomputations.
    pub fn effective_weights(&self, meta: &MetadataStore) -> StrategyWeights {
        {
            let last = self.last_adjust.lock().unwrap();
            if last.elapsed() < Duration::from_secs(self.cfg.adjust_interval_secs) {
                return self.current.lock().unwrap().clone();
            }
        }

        let adjusted = self.compute(meta);

        let mut last = self.last_adjust.lock().unwrap();
        *last = Instant::now();
        *self.current.lock().unwrap() = adjusted.clone();
        adjusted
    }

    /// Recompute weights from the current signals.
    ///
    /// Split out from [`effective_weights`](Self::effective_weights) so unit
    /// tests can force a recompute without waiting out the interval.
    pub fn compute(&self, meta: &MetadataStore) -> StrategyWeights {
        let max_adj = self.cfg.max_adjustment.max(0.0);

        // Signal 1: failure pressure = 1 - min(ema_success) across backends
        // that have recorded attempts. No data → no pressure.
        let failure_pressure = self
            .outcomes
            .iter()
            .filter(|e| e.value().samples > 0)
            .map(|e| e.value().ema_success)
            .fold(None, |acc: Option<f64>, s| Some(acc.map_or(s, |m| m.min(s))))
            .map_or(0.0, |min_success| (1.0 - min_success).clamp(0.0, 1.0));

        // Signal 2: KV hit-ratio EMA. Only boosts beyond 0.5 — below that the
        // workload gives no evidence of being cache-friendly.
        let kv_boost = self
            .kv_hit_ratio()
            .map_or(0.0, |r| ((r - 0.5) * 2.0).clamp(0.0, 1.0) * max_adj);

        // Signal 3: load spread across the fleet, from gossip/local-fed load
        // snapshots. Uses the coefficient of variation of `active_requests`;
        // a spread of 1.0 (stddev == mean) maps to a full `max_adj` boost.
        // Failure pressure and spread both argue for stronger load awareness,
        // so the boost takes the stronger of the two.
        let load_boost = failure_pressure.max(load_spread(meta)).clamp(0.0, 1.0) * max_adj;

        let w_kv = self.base.kv * (1.0 + kv_boost);
        let w_load = self.base.load * (1.0 + load_boost);
        let w_topo = self.base.topology;
        // Cost is treated as an externality by the adaptive loop: pricing is
        // a static, exogenous signal, not something to feedback-adjust. We
        // preserve the configured base cost weight and let the normalizer
        // include it in the sum.
        let w_cost = self.base.cost;

        let sum = w_kv + w_load + w_topo + w_cost;
        if sum <= 0.0 {
            return self.base.clone();
        }
        let floor = self.cfg.min_weight.clamp(0.0, 1.0 / 4.0);
        let [kv, load, topology, cost] = normalize_with_floor(
            [w_kv / sum, w_load / sum, w_topo / sum, w_cost / sum],
            floor,
        );

        StrategyWeights {
            kv,
            load,
            topology,
            cost,
        }
    }
}

/// Normalize `w` to sum 1.0 while guaranteeing every element stays >= `floor`.
///
/// Iterative water-filling: elements below the floor are clamped to it, and
/// the remaining elements are scaled proportionally to fill the leftover
/// budget. When no element violates the floor the proportions are preserved
/// exactly, so the no-signal case returns the base ratios unchanged.
///
/// Generic over the array length so the same routine covers the 3-weight
/// (kv/load/topology) and 4-weight (with cost) shapes.
fn normalize_with_floor<const N: usize>(mut w: [f64; N], floor: f64) -> [f64; N] {
    let mut clamped = [false; N];
    loop {
        let fixed: f64 = w
            .iter()
            .zip(&clamped)
            .filter(|(_, c)| **c)
            .map(|(v, _)| v)
            .sum();
        let rest_sum: f64 = w
            .iter()
            .zip(&clamped)
            .filter(|(_, c)| !**c)
            .map(|(v, _)| v)
            .sum();
        let budget = 1.0 - fixed;
        let n_rest = clamped.iter().filter(|c| !**c).count() as f64;
        let mut newly_clamped = false;
        for i in 0..N {
            if clamped[i] {
                continue;
            }
            w[i] = if rest_sum > 0.0 {
                w[i] / rest_sum * budget
            } else {
                budget / n_rest.max(1.0)
            };
            if w[i] < floor {
                w[i] = floor;
                clamped[i] = true;
                newly_clamped = true;
            }
        }
        if !newly_clamped {
            break;
        }
    }
    w
}

/// EMA update: on the first sample adopt the value directly, then blend.
fn ema(prev: f64, value: f64, alpha: f64, samples: u64) -> f64 {
    if samples == 0 {
        value
    } else {
        prev + alpha * (value - prev)
    }
}

/// Coefficient of variation (stddev / mean) of `active_requests` across all
/// backends with load snapshots in the store.
///
/// Returns 0.0 when fewer than two snapshots exist or the mean is zero —
/// either way the load signal carries no discriminative information.
fn load_spread(meta: &MetadataStore) -> f64 {
    let values: Vec<f64> = meta
        .backends_all()
        .iter()
        .filter_map(|b| meta.load_get_metrics(&b.id))
        .map(|m| m.active_requests as f64)
        .collect();
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    var.sqrt() / mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::backend::{
        BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig,
        Protocol,
    };
    use hier_kv_gateway_core::ids::{IndexerDomainId, RegionId};
    use hier_kv_gateway_core::metrics::BackendMetrics;

    fn base_weights() -> StrategyWeights {
        StrategyWeights {
            kv: 0.35,
            load: 0.30,
            topology: 0.20,
            cost: 0.0,
        }
    }

    fn config(enabled: bool) -> AdaptiveConfig {
        AdaptiveConfig {
            enabled,
            ema_alpha: 0.5,
            max_adjustment: 0.25,
            min_weight: 0.05,
            adjust_interval_secs: 0, // recompute on every call
        }
    }

    fn register_backend(meta: &MetadataStore, region: &str, instance: &str) -> BackendId {
        let id = BackendId::new(region, instance);
        meta.register_backend(BackendInfo {
            id: id.clone(),
            backend_type: BackendType::VllmEngine,
            endpoint: Endpoint {
                url: format!("http://{instance}"),
                protocol: Protocol::Http,
            },
            models: Vec::new(),
            region: RegionId::new(region),
            indexer_domain: IndexerDomainId(0),
            capabilities: BackendCapabilities {
                supports_kv_events: false,
                supports_batching: true,
                max_batch_size: 0,
                gpu_count: 1,
                gpu_memory_gb: 24,
            },
            kv_config: KvConfig {
                block_size: 16,
                cache_namespace: String::new(),
                max_kv_blocks: 0,
            },
            status: BackendStatus::Healthy,
        });
        id
    }

    #[test]
    fn no_signals_returns_base_weights() {
        let ctl = AdaptiveWeightController::new(base_weights(), config(true));
        let meta = MetadataStore::new();
        let w = ctl.compute(&meta);
        // Base weights are not normalized in config; controller normalizes them.
        let sum = w.kv + w.load + w.topology + w.cost;
        assert!((sum - 1.0).abs() < 1e-9);
        // With no signals the kv/load ratio matches the base ratio. The
        // absolute values shift slightly because the cost weight's
        // min-weight floor (0.05) eats into the budget, but the *ratios*
        // between the non-cost weights stay anchored to the configured base.
        assert!((w.kv / w.load - 0.35 / 0.30).abs() < 1e-9);
        assert!((w.load / w.topology - 0.30 / 0.20).abs() < 1e-9);
        // Cost has a zero base, so it sits at the min-weight floor.
        assert!((w.cost - 0.05).abs() < 1e-9);
    }

    #[test]
    fn repeated_failures_boost_load_weight() {
        let ctl = AdaptiveWeightController::new(base_weights(), config(true));
        let meta = MetadataStore::new();
        let bad = BackendId::new("r1", "bad");
        for _ in 0..10 {
            ctl.record_failure(&bad);
        }
        let w = ctl.compute(&meta);
        let base = ctl.compute_base_normalized();
        assert!(
            w.load > base.load,
            "failure pressure should raise load weight: {:?} vs {:?}",
            w,
            base
        );
        assert!((w.kv + w.load + w.topology + w.cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn successes_keep_weights_near_base() {
        let ctl = AdaptiveWeightController::new(base_weights(), config(true));
        let meta = MetadataStore::new();
        let good = BackendId::new("r1", "good");
        for _ in 0..10 {
            ctl.record_success(&good, Duration::from_millis(20));
        }
        let w = ctl.compute(&meta);
        // All-success fleet → failure_pressure = 0 → load ratio unchanged.
        assert!((w.load / w.kv - 0.30 / 0.35).abs() < 1e-9);
        let stats = ctl.outcome_stats(&good).unwrap();
        assert!((stats.ema_success - 1.0).abs() < 1e-9);
        assert!(stats.ema_latency_ms > 0.0);
    }

    #[test]
    fn high_kv_hit_ratio_boosts_kv_weight() {
        let ctl = AdaptiveWeightController::new(base_weights(), config(true));
        let meta = MetadataStore::new();
        for _ in 0..10 {
            ctl.record_kv_overlap(9, 10); // 90% hit ratio
        }
        let w = ctl.compute(&meta);
        assert!(
            w.kv / w.topology > 0.35 / 0.20,
            "kv weight should outgrow its base ratio: {:?}",
            w
        );
    }

    #[test]
    fn low_kv_hit_ratio_does_not_penalize() {
        let ctl = AdaptiveWeightController::new(base_weights(), config(true));
        let meta = MetadataStore::new();
        for _ in 0..10 {
            ctl.record_kv_overlap(0, 10);
        }
        let w = ctl.compute(&meta);
        // No boost, no penalty: ratios equal base.
        assert!((w.kv / w.load - 0.35 / 0.30).abs() < 1e-9);
    }

    #[test]
    fn kv_overlap_recording_ignores_empty_prompts() {
        let ctl = AdaptiveWeightController::new(base_weights(), config(true));
        ctl.record_kv_overlap(3, 0);
        assert!(ctl.kv_hit_ratio().is_none());
    }

    #[test]
    fn load_spread_from_gossip_fed_snapshots_boosts_load() {
        let ctl = AdaptiveWeightController::new(base_weights(), config(true));
        let meta = MetadataStore::new();
        // Two backends with strongly uneven load (e.g. one near saturation as
        // reported via METRIC_BROADCAST from a peer gateway).
        let a = register_backend(&meta, "r1", "hot");
        let b = register_backend(&meta, "r1", "idle");
        let mut hot = BackendMetrics::default();
        hot.active_requests = 90;
        let idle = BackendMetrics::default(); // 0 active
        meta.load_update(a, hot);
        meta.load_update(b, idle);

        let spread = load_spread(&meta);
        assert!(spread > 0.9, "spread should be near 1.0, got {spread}");

        let w = ctl.compute(&meta);
        assert!(
            w.load / w.kv > 0.30 / 0.35,
            "load spread should raise the load ratio: {:?}",
            w
        );
    }

    #[test]
    fn min_weight_floor_holds_when_base_is_zero() {
        let weights = StrategyWeights {
            kv: 0.0,
            load: 1.0,
            topology: 0.0,
            cost: 0.0,
        };
        let ctl = AdaptiveWeightController::new(weights, config(true));
        let meta = MetadataStore::new();
        let w = ctl.compute(&meta);
        assert!(w.kv >= 0.05 - 1e-9);
        assert!(w.topology >= 0.05 - 1e-9);
        assert!(w.cost >= 0.05 - 1e-9);
        assert!((w.kv + w.load + w.topology + w.cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn effective_weights_caches_between_intervals() {
        let cfg = AdaptiveConfig {
            adjust_interval_secs: 3600, // no recompute inside the test
            ..config(true)
        };
        let ctl = AdaptiveWeightController::new(base_weights(), cfg);
        let meta = MetadataStore::new();
        let w1 = ctl.effective_weights(&meta);
        // Record failures; cached weights must NOT change within the interval.
        let bad = BackendId::new("r1", "bad");
        for _ in 0..10 {
            ctl.record_failure(&bad);
        }
        let w2 = ctl.effective_weights(&meta);
        assert_eq!(w1.kv, w2.kv);
        assert_eq!(w1.load, w2.load);
        // But a forced compute reflects them.
        let w3 = ctl.compute(&meta);
        assert!(w3.load > w2.load);
    }

    #[test]
    fn ema_first_sample_adopted_directly() {
        assert_eq!(ema(1.0, 0.0, 0.2, 0), 0.0);
        assert!((ema(0.5, 1.0, 0.2, 3) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn normalize_with_floor_preserves_ratios_when_no_clamping() {
        let w = normalize_with_floor([0.4, 0.35, 0.25], 0.05);
        assert!((w[0] / w[1] - 0.4 / 0.35).abs() < 1e-9);
        assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_with_floor_clamps_and_redistributes() {
        let w = normalize_with_floor([0.0, 1.0, 0.0], 0.05);
        assert!((w[0] - 0.05).abs() < 1e-9);
        assert!((w[2] - 0.05).abs() < 1e-9);
        assert!((w[1] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn normalize_with_floor_all_zero_falls_to_uniform() {
        let w = normalize_with_floor([0.0, 0.0, 0.0], 0.05);
        assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1e-9);
        assert!(w.iter().all(|v| *v >= 0.05));
    }

    impl AdaptiveWeightController {
        /// Base weights normalized the same way `compute` normalizes, for
        /// test comparisons.
        fn compute_base_normalized(&self) -> StrategyWeights {
            let sum = self.base.kv + self.base.load + self.base.topology + self.base.cost;
            StrategyWeights {
                kv: self.base.kv / sum,
                load: self.base.load / sum,
                topology: self.base.topology / sum,
                cost: self.base.cost / sum,
            }
        }
    }
}
