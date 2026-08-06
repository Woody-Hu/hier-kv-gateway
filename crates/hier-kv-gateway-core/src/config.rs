//! Gateway configuration model and TOML loading.
//!
//! Provides the ability to load [`GatewayConfig`] from a TOML file. Configuration
//! covers gateway instance identity, region, listen port, routing strategy,
//! cluster membership protocol, and backend connection info.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::backend::{BackendType, Endpoint, Quantization};
use crate::coalescing::CoalescingConfig;
use crate::cost::CostConfig;
use crate::error::{HierKvGatewayError, Result};
use crate::ids::{InstanceId, RegionId, RegionTier};
use crate::model_tier::ModelTierConfig;
use crate::tenant::TenantPriority;

/// Re-export of the KV-estimation config section.
///
/// `KvEstimateConfig` is defined in the leaf `hier-kv-gateway-kv-estimate`
/// crate (so the estimator stays reusable with no gateway dependency) and is
/// surfaced here so it parses as the `[kv_estimate]` section of
/// [`GatewayConfig`]. The *behaviour* half — `KvCapacityStrategy` — lives in
/// the `hier-kv-gateway-routing` crate.
pub use hier_kv_gateway_kv_estimate::KvEstimateConfig;

/// Top-level gateway configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Current gateway instance identifier.
    pub instance_id: InstanceId,
    /// Region configuration for the current gateway.
    pub region: RegionConfig,
    /// Listen configuration.
    pub listen: ListenConfig,
    /// Routing strategy configuration.
    pub routing: RoutingConfig,
    /// Cluster membership protocol configuration.
    pub cluster: ClusterConfig,
    /// Downstream (gateway → backend) forwarding behavior.
    #[serde(default)]
    pub forwarding: ForwardingConfig,
    /// Retry / circuit-breaker resilience behavior for backend forwarding.
    #[serde(default)]
    pub resilience: ResilienceConfig,
    /// Decision telemetry: how routing decision events are exported.
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// Configured backend list; empty by default.
    #[serde(default)]
    pub backends: Vec<BackendConfig>,
    /// Multi-tenant scheduling configuration.
    #[serde(default)]
    pub tenant: TenantConfig,
    /// Cost-model configuration for cost-aware routing. Off by default;
    /// existing configurations parse unchanged (`enabled = false`).
    #[serde(default)]
    pub cost: CostConfig,
    /// Large/small model tiering configuration. Off by default; existing
    /// configurations parse unchanged (`enabled = false`).
    #[serde(default)]
    pub model_tier: ModelTierConfig,
    /// KV-cache memory estimation & capacity-aware routing. Off by default;
    /// existing configurations parse unchanged (`enabled = false`). When
    /// enabled, the routing engine attaches a `KvCapacityStrategy` plugin
    /// that estimates each request's KV footprint (via the
    /// `hier-kv-gateway-kv-estimate` leaf crate) and scores backends by
    /// available KV / GPU-memory headroom.
    #[serde(default)]
    pub kv_estimate: KvEstimateConfig,
    /// Request coalescing (single-flight) configuration. Off by default;
    /// existing configurations parse unchanged (`enabled = false`).
    #[serde(default)]
    pub coalescing: CoalescingConfig,
}

/// Region configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionConfig {
    /// Region identifier.
    pub id: RegionId,
    /// Region tier.
    pub tier: RegionTier,
    /// Geographic coordinates, optional.
    pub geo: Option<GeoCoordConfig>,
    /// Network zone label.
    pub network_zone: String,
}

/// TOML-friendly geographic coordinates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoCoordConfig {
    /// Latitude.
    pub lat: f64,
    /// Longitude.
    pub lon: f64,
}

/// Listen address and port.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListenConfig {
    /// Listen address, e.g. `0.0.0.0`.
    pub addr: String,
    /// Listen port.
    pub port: u16,
}

/// Routing strategy type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum StrategyType {
    /// Hybrid strategy (combined KV + load + topology scoring).
    Hybrid,
    /// Route by KV cache hit rate only.
    Kv,
    /// Route by model availability only.
    Model,
    /// Route by load only.
    Load,
    /// Route by topology distance only.
    Topology,
    /// Round-robin baseline: rotate through candidates in order.
    ///
    /// The routing engine always keeps a round-robin terminal fallback so a
    /// decision is still produced when every metadata-driven strategy fails;
    /// selecting this variant makes round-robin the *primary* mechanism.
    RoundRobin,
}

/// Routing strategy weights.
///
/// The weights are typically normalized before participating in hybrid scoring.
///
/// `cost` is serde-defaulted to `0.0` so existing configurations that predate
/// cost-aware routing continue to parse unchanged; setting it to a positive
/// value (and `[cost] enabled = true`) opts the hybrid strategy into the
/// cost-aware sub-strategy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyWeights {
    /// KV cache hit weight.
    pub kv: f64,
    /// Load weight.
    pub load: f64,
    /// Topology distance weight.
    pub topology: f64,
    /// Cost (projected dollar cost) weight. Defaults to `0.0` so existing
    /// configs keep parsing; a positive value is only effective when
    /// [`CostConfig::enabled`] is `true`.
    #[serde(default)]
    pub cost: f64,
}

/// Routing-related configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Primary routing strategy.
    pub strategy: StrategyType,
    /// KV cache block size; must match the backend convention.
    pub kv_block_size: u32,
    /// Extra score credit given to candidate backends on hit overlap.
    pub overlap_score_credit: f64,
    /// Load scaling factor for the prefill phase.
    pub prefill_load_scale: f64,
    /// Routing temperature parameter (softmax); higher values mean more randomness.
    pub temperature: f64,
    /// Session affinity TTL (seconds).
    pub session_affinity_ttl_secs: u64,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Weights for the three dimensions under the hybrid strategy.
    pub weights: StrategyWeights,
    /// Adaptive weight feedback controller (disabled by default).
    #[serde(default)]
    pub adaptive: AdaptiveConfig,
}

/// Adaptive hybrid-weight feedback configuration.
///
/// When enabled, an EMA-based controller nudges the static
/// [`StrategyWeights`] at runtime using two signal families:
///
/// * **Execution metrics** collected by the gateway's own forwarding loop:
///   per-backend forward success rate and latency, plus the KV hit ratio of
///   served requests.
/// * **Broadcast node state**: the load snapshots peers publish over gossip
///   (they land in the shared [`crate::metrics::BackendMetrics`] store), used
///   to measure load spread across the fleet.
///
/// Adjustments are bounded around the configured base weights so the system
/// degrades gracefully to the static behaviour when signals disappear.
///
/// All fields carry defaults; `[routing.adaptive] enabled = true` opts in.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveConfig {
    /// Master switch. When `false` the hybrid strategy always uses the static
    /// base weights.
    pub enabled: bool,
    /// EMA smoothing factor in `(0, 1]`; higher values react faster but are
    /// noisier.
    pub ema_alpha: f64,
    /// Maximum relative adjustment applied to any base weight, e.g. `0.25`
    /// allows `weight * (1 ± 0.25)` before normalization.
    pub max_adjustment: f64,
    /// Floor applied to every normalized weight so no signal can be fully
    /// starved out.
    pub min_weight: f64,
    /// Minimum interval (seconds) between two weight recomputations; the hot
    /// path reuses the cached weights in between.
    pub adjust_interval_secs: u64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ema_alpha: 0.2,
            max_adjustment: 0.25,
            min_weight: 0.05,
            adjust_interval_secs: 10,
        }
    }
}

/// Decision telemetry export configuration.
///
/// The gateway emits one [`crate::decision_event::DecisionEvent`] per request.
/// This section selects where those events go; an in-memory ring buffer for
/// the `GET /admin/decision_events` endpoint is always maintained (its size is
/// configurable and can be disabled with `buffer_size = 0`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetryConfig {
    /// Export mode for the durable/streaming sink.
    pub mode: TelemetryMode,
    /// Output file for [`TelemetryMode::File`] (NDJSON, one event per line).
    pub file_path: String,
    /// Capacity of the in-memory ring buffer backing
    /// `GET /admin/decision_events`; `0` disables the buffer.
    pub buffer_size: usize,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            mode: TelemetryMode::None,
            file_path: "decision_events.ndjson".to_string(),
            buffer_size: 256,
        }
    }
}

/// Where decision events are exported beyond the in-memory buffer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryMode {
    /// No external export (in-memory buffer only).
    None,
    /// Emit events as structured `tracing` records on the
    /// `hier_kv_gateway.decision_events` target, letting the existing log
    /// pipeline (Loki, ELK, journald, ...) carry them.
    Tracing,
    /// Append events to `file_path` as NDJSON via a background writer task.
    File,
}

/// Cluster membership protocol configuration (based on SWIM/gossip).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Cluster bind address.
    pub bind_addr: String,
    /// List of seed node addresses.
    pub seed_peers: Vec<String>,
    /// Gossip interval (milliseconds).
    pub gossip_interval_ms: u64,
    /// Liveness probe timeout (milliseconds).
    pub probe_timeout_ms: u64,
    /// Suspect state timeout (seconds).
    pub suspect_timeout_secs: u64,
    /// Gossip fanout: number of alive members pinged per gossip round.
    ///
    /// Defaults to `3` (the SWIM paper's recommended value). Larger values
    /// spread membership/metadata changes faster at the cost of more
    /// per-round network traffic. Tunable at deploy time to trade off
    /// convergence speed vs bandwidth on slow edge links.
    #[serde(default = "default_gossip_fanout")]
    pub gossip_fanout: usize,
    /// Probe loop interval (milliseconds).
    ///
    /// Controls how often the probe loop scans the member list for
    /// `Alive → Suspect` and `Suspect → Dead` transitions. Independent of
    /// `gossip_interval_ms` to avoid coupling detection cadence to
    /// heartbeat cadence. Defaults to `500` ms.
    #[serde(default = "default_probe_interval_ms")]
    pub probe_interval_ms: u64,
}

fn default_gossip_fanout() -> usize {
    3
}

fn default_probe_interval_ms() -> u64 {
    500
}

/// Downstream forwarding behavior (gateway → backend).
///
/// All fields carry defaults so existing configurations keep working unchanged
/// (`emit_token_ids = false` preserves the historical text-based forwarding).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ForwardingConfig {
    /// When `true` and the incoming request carries a pre-tokenized
    /// `token_ids` sequence, the gateway forwards `prompt_token_ids` to the
    /// backend *instead of* re-serializing the chat `messages` text.
    ///
    /// This matches the disaggregated-serving pattern (vLLM / Dynamo accept
    /// `prompt_token_ids`): the gateway tokenizes (or receives tokens) once,
    /// computes KV block hashes from the same sequence used for routing, and
    /// the backend skips re-tokenization. Requests without `token_ids` fall
    /// back to the normal text path regardless of this flag.
    pub emit_token_ids: bool,
}

impl Default for ForwardingConfig {
    fn default() -> Self {
        Self {
            emit_token_ids: false,
        }
    }
}

/// Retry / circuit-breaker configuration for backend forwarding.
///
/// The retry count itself is governed by [`RoutingConfig::max_retries`]; this
/// section tunes the backoff shape and the per-backend circuit breaker that
/// short-circuits candidates with a recent failure streak.
///
/// All fields carry defaults so existing configurations keep working
/// unchanged.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ResilienceConfig {
    /// Base backoff between two forward attempts (milliseconds). The effective
    /// delay doubles per attempt: `base * 2^attempt`, capped by
    /// `retry_max_backoff_ms`.
    pub retry_backoff_ms: u64,
    /// Upper bound for the retry backoff (milliseconds).
    pub retry_max_backoff_ms: u64,
    /// Consecutive forward failures after which a backend's circuit opens and
    /// it is skipped by the forwarding loop.
    pub circuit_breaker_failure_threshold: u32,
    /// How long (seconds) an open circuit waits before allowing a half-open
    /// probe through.
    pub circuit_breaker_cooldown_secs: u64,
    /// Consecutive successful half-open probes required to fully close the
    /// circuit again.
    pub half_open_success_threshold: u32,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            retry_backoff_ms: 50,
            retry_max_backoff_ms: 1_000,
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_cooldown_secs: 30,
            half_open_success_threshold: 1,
        }
    }
}

/// Backend connection configuration (without runtime state).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend type.
    pub backend_type: BackendType,
    /// Backend endpoint.
    pub endpoint: Endpoint,
    /// List of model names hosted by this backend.
    pub models: Vec<String>,
    /// Region the backend resides in.
    pub region: RegionId,
    /// KV cache block size.
    pub kv_block_size: u32,
    /// Optional quantization method, used only for descriptive purposes on the configuration side.
    pub quantization: Option<Quantization>,
}

/// Multi-tenant scheduling configuration.
///
/// Controls per-tenant rate limits, priority-based admission, and fair
/// queuing when the system is saturated.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TenantConfig {
    /// Master switch for tenant-aware scheduling. When `false`, every
    /// request is treated as belonging to the implicit default tenant and
    /// no rate limiting or priority-based admission is applied.
    pub enabled: bool,
    /// Default maximum requests per second for tenants without an explicit
    /// per-tenant override. `None` means unlimited.
    pub default_max_rps: Option<f64>,
    /// Default maximum concurrent (in-flight) requests for tenants without
    /// an explicit per-tenant override. `None` means unlimited.
    pub default_max_concurrent: Option<u32>,
    /// Saturation threshold (0.0–1.0). When the fraction of backend capacity
    /// in use exceeds this value, the scheduler activates priority-based
    /// admission: Premium tenants get their reserved fraction first, then
    /// Normal, then Background.
    pub saturation_threshold: f64,
    /// Per-tenant quota overrides. Each entry defines the limits for one
    /// tenant; tenants not listed here receive the defaults.
    #[serde(default)]
    pub tenants: Vec<TenantQuotaConfig>,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_max_rps: None,
            default_max_concurrent: None,
            saturation_threshold: 0.8,
            tenants: Vec::new(),
        }
    }
}

/// Per-tenant quota entry in the gateway configuration file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TenantQuotaConfig {
    /// Tenant identifier (matches the `X-Tenant-Id` header).
    pub id: String,
    /// Priority level. Defaults to `"normal"`.
    #[serde(default = "default_tenant_priority")]
    pub priority: TenantPriority,
    /// Maximum requests per second. Overrides `default_max_rps`.
    pub max_rps: Option<f64>,
    /// Maximum concurrent requests. Overrides `default_max_concurrent`.
    pub max_concurrent: Option<u32>,
    /// Reserved fraction of total backend capacity (0.0–1.0). Only
    /// meaningful for `Premium` tenants; ignored for others.
    #[serde(default)]
    pub reserved_capacity_fraction: f64,
}

fn default_tenant_priority() -> TenantPriority {
    TenantPriority::Normal
}

/// Load [`GatewayConfig`] from a TOML file.
///
/// The file content must deserialize into `GatewayConfig`, otherwise
/// [`HierKvGatewayError::ConfigError`] is returned.
pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<GatewayConfig> {
    let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
        HierKvGatewayError::ConfigError(format!(
            "Failed to read config file {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;
    let cfg: GatewayConfig = toml::from_str(&content).map_err(|e| {
        HierKvGatewayError::ConfigError(format!(
            "Failed to parse config file {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Protocol;

    #[test]
    fn strategy_type_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&StrategyType::Kv).unwrap(),
            r#""kv""#
        );
        assert_eq!(
            serde_json::to_string(&StrategyType::Hybrid).unwrap(),
            r#""hybrid""#
        );
        assert_eq!(
            serde_json::to_string(&StrategyType::RoundRobin).unwrap(),
            r#""round_robin""#
        );
    }

    #[test]
    fn forwarding_and_resilience_default_when_absent() {
        let toml_text = r#"
instance_id = "g1"

[region]
id = "r1"
tier = "edge"
network_zone = "z1"

[listen]
addr = "127.0.0.1"
port = 9090

[routing]
strategy = "load"
kv_block_size = 16
overlap_score_credit = 0.0
prefill_load_scale = 1.0
temperature = 0.0
session_affinity_ttl_secs = 60
max_retries = 2

[routing.weights]
kv = 0.0
load = 1.0
topology = 0.0

[cluster]
bind_addr = "0.0.0.0:7946"
seed_peers = []
gossip_interval_ms = 200
probe_timeout_ms = 1000
suspect_timeout_secs = 5
"#;
        let cfg: GatewayConfig = toml::from_str(toml_text).unwrap();
        // Sections absent → defaults kick in (backwards compatible).
        assert!(!cfg.forwarding.emit_token_ids);
        let r = &cfg.resilience;
        assert_eq!(r.retry_backoff_ms, 50);
        assert_eq!(r.retry_max_backoff_ms, 1_000);
        assert_eq!(r.circuit_breaker_failure_threshold, 5);
        assert_eq!(r.circuit_breaker_cooldown_secs, 30);
        assert_eq!(r.half_open_success_threshold, 1);
    }

    #[test]
    fn forwarding_and_resilience_parse_explicit_values() {
        let toml_text = r#"
instance_id = "g1"

[region]
id = "r1"
tier = "edge"
network_zone = "z1"

[listen]
addr = "127.0.0.1"
port = 9090

[routing]
strategy = "round_robin"
kv_block_size = 16
overlap_score_credit = 0.0
prefill_load_scale = 1.0
temperature = 0.0
session_affinity_ttl_secs = 60
max_retries = 2

[routing.weights]
kv = 0.0
load = 1.0
topology = 0.0

[cluster]
bind_addr = "0.0.0.0:7946"
seed_peers = []
gossip_interval_ms = 200
probe_timeout_ms = 1000
suspect_timeout_secs = 5

[forwarding]
emit_token_ids = true

[resilience]
retry_backoff_ms = 10
retry_max_backoff_ms = 500
circuit_breaker_failure_threshold = 3
circuit_breaker_cooldown_secs = 15
half_open_success_threshold = 2
"#;
        let cfg: GatewayConfig = toml::from_str(toml_text).unwrap();
        assert!(cfg.forwarding.emit_token_ids);
        assert_eq!(cfg.routing.strategy, StrategyType::RoundRobin);
        let r = &cfg.resilience;
        assert_eq!(r.retry_backoff_ms, 10);
        assert_eq!(r.retry_max_backoff_ms, 500);
        assert_eq!(r.circuit_breaker_failure_threshold, 3);
        assert_eq!(r.circuit_breaker_cooldown_secs, 15);
        assert_eq!(r.half_open_success_threshold, 2);
    }

    #[test]
    fn telemetry_and_adaptive_default_when_absent() {
        // The minimal config from `parse_minimal_config` carries neither
        // section; both must fall back to their backwards-compatible defaults.
        let toml_text = r#"
instance_id = "g1"

[region]
id = "r1"
tier = "edge"
network_zone = "z1"

[listen]
addr = "127.0.0.1"
port = 9090

[routing]
strategy = "hybrid"
kv_block_size = 16
overlap_score_credit = 0.0
prefill_load_scale = 1.0
temperature = 0.0
session_affinity_ttl_secs = 60
max_retries = 2

[routing.weights]
kv = 0.35
load = 0.30
topology = 0.20

[cluster]
bind_addr = "0.0.0.0:7946"
seed_peers = []
gossip_interval_ms = 200
probe_timeout_ms = 1000
suspect_timeout_secs = 5
"#;
        let cfg: GatewayConfig = toml::from_str(toml_text).unwrap();
        assert!(!cfg.routing.adaptive.enabled);
        assert!((cfg.routing.adaptive.ema_alpha - 0.2).abs() < 1e-9);
        assert_eq!(cfg.telemetry.mode, TelemetryMode::None);
        assert_eq!(cfg.telemetry.buffer_size, 256);
        assert_eq!(cfg.telemetry.file_path, "decision_events.ndjson");
    }

    #[test]
    fn telemetry_and_adaptive_parse_explicit_values() {
        let toml_text = r#"
instance_id = "g1"

[region]
id = "r1"
tier = "edge"
network_zone = "z1"

[listen]
addr = "127.0.0.1"
port = 9090

[routing]
strategy = "hybrid"
kv_block_size = 16
overlap_score_credit = 0.0
prefill_load_scale = 1.0
temperature = 0.0
session_affinity_ttl_secs = 60
max_retries = 2

[routing.weights]
kv = 0.35
load = 0.30
topology = 0.20

[routing.adaptive]
enabled = true
ema_alpha = 0.3
max_adjustment = 0.4
min_weight = 0.1
adjust_interval_secs = 5

[cluster]
bind_addr = "0.0.0.0:7946"
seed_peers = []
gossip_interval_ms = 200
probe_timeout_ms = 1000
suspect_timeout_secs = 5

[telemetry]
mode = "file"
file_path = "/var/log/hier/decisions.ndjson"
buffer_size = 1024
"#;
        let cfg: GatewayConfig = toml::from_str(toml_text).unwrap();
        let a = &cfg.routing.adaptive;
        assert!(a.enabled);
        assert!((a.ema_alpha - 0.3).abs() < 1e-9);
        assert!((a.max_adjustment - 0.4).abs() < 1e-9);
        assert!((a.min_weight - 0.1).abs() < 1e-9);
        assert_eq!(a.adjust_interval_secs, 5);
        let t = &cfg.telemetry;
        assert_eq!(t.mode, TelemetryMode::File);
        assert_eq!(t.file_path, "/var/log/hier/decisions.ndjson");
        assert_eq!(t.buffer_size, 1024);
    }

    #[test]
    fn telemetry_mode_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&TelemetryMode::Tracing).unwrap(),
            r#""tracing""#
        );
        assert_eq!(
            serde_json::from_str::<TelemetryMode>(r#""file""#).unwrap(),
            TelemetryMode::File
        );
    }

    #[test]
    fn parse_minimal_config() {
        let toml_text = r#"
instance_id = "g1"

[region]
id = "r1"
tier = "edge"
network_zone = "z1"

[listen]
addr = "127.0.0.1"
port = 9090

[routing]
strategy = "load"
kv_block_size = 16
overlap_score_credit = 0.0
prefill_load_scale = 1.0
temperature = 0.0
session_affinity_ttl_secs = 60
max_retries = 2

[routing.weights]
kv = 0.0
load = 1.0
topology = 0.0

[cluster]
bind_addr = "0.0.0.0:7946"
seed_peers = []
gossip_interval_ms = 200
probe_timeout_ms = 1000
suspect_timeout_secs = 5
"#;
        let cfg: GatewayConfig = toml::from_str(toml_text).unwrap();
        assert_eq!(cfg.instance_id.as_str(), "g1");
        assert_eq!(cfg.region.id.as_str(), "r1");
        assert_eq!(cfg.listen.port, 9090);
        assert_eq!(cfg.routing.strategy, StrategyType::Load);
        assert!(cfg.backends.is_empty());
        assert!(cfg.region.geo.is_none());
    }

    #[test]
    fn parse_config_with_backends() {
        let toml_text = r#"
instance_id = "g1"

[region]
id = "r1"
tier = "cloud"
network_zone = "z1"

[region.geo]
lat = 1.0
lon = 2.0

[listen]
addr = "0.0.0.0"
port = 8080

[routing]
strategy = "kv"
kv_block_size = 32
overlap_score_credit = 0.1
prefill_load_scale = 0.5
temperature = 0.2
session_affinity_ttl_secs = 120
max_retries = 5

[routing.weights]
kv = 1.0
load = 0.0
topology = 0.0

[cluster]
bind_addr = "0.0.0.0:7946"
seed_peers = ["a:7946"]
gossip_interval_ms = 100
probe_timeout_ms = 500
suspect_timeout_secs = 10

[[backends]]
backend_type = "vllm_engine"
region = "r1"
kv_block_size = 32
quantization = "bf16"
models = ["m1", "m2"]

[backends.endpoint]
url = "http://10.0.0.1:8080"
protocol = "http"
"#;
        let cfg: GatewayConfig = toml::from_str(toml_text).unwrap();
        assert_eq!(cfg.backends.len(), 1);
        let b = &cfg.backends[0];
        assert_eq!(b.backend_type, BackendType::VllmEngine);
        assert_eq!(b.endpoint.protocol, Protocol::Http);
        assert_eq!(b.models, vec!["m1".to_string(), "m2".to_string()]);
        assert_eq!(b.quantization, Some(Quantization::Bf16));
        assert!(cfg.region.geo.is_some());
        let geo = cfg.region.geo.unwrap();
        assert!((geo.lat - 1.0).abs() < 1e-9);
        assert!((geo.lon - 2.0).abs() < 1e-9);
    }

    #[test]
    fn load_from_missing_file_returns_config_error() {
        let res = load_from_file("/nonexistent/path/does-not-exist.toml");
        assert!(matches!(res, Err(HierKvGatewayError::ConfigError(_))));
    }

    #[test]
    fn load_from_invalid_toml_returns_config_error() {
        let tmp = tempfile_lite();
        std::fs::write(&tmp, "not = valid = toml").unwrap();
        let res = load_from_file(&tmp);
        assert!(matches!(res, Err(HierKvGatewayError::ConfigError(_))));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_from_valid_file_succeeds() {
        let tmp = tempfile_lite();
        let toml_text = r#"
instance_id = "g1"

[region]
id = "r1"
tier = "device"
network_zone = "z1"

[listen]
addr = "0.0.0.0"
port = 8080

[routing]
strategy = "topology"
kv_block_size = 16
overlap_score_credit = 0.0
prefill_load_scale = 1.0
temperature = 0.0
session_affinity_ttl_secs = 60
max_retries = 1

[routing.weights]
kv = 0.0
load = 0.0
topology = 1.0

[cluster]
bind_addr = "0.0.0.0:7946"
seed_peers = []
gossip_interval_ms = 200
probe_timeout_ms = 1000
suspect_timeout_secs = 5
"#;
        std::fs::write(&tmp, toml_text).unwrap();
        let cfg = load_from_file(&tmp).unwrap();
        assert_eq!(cfg.routing.strategy, StrategyType::Topology);
        assert_eq!(cfg.region.tier, RegionTier::Device);
        let _ = std::fs::remove_file(&tmp);
    }

    /// Generate a unique temporary file path for testing.
    fn tempfile_lite() -> String {
        // Uses std::process::id() combined with an atomic counter to avoid depending on the tempfile crate.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        format!("/tmp/hier-kv-gateway-core-test-{}-{}.toml", pid, n)
    }
}
