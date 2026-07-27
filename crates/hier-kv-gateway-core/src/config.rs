//! Gateway configuration model and TOML loading.
//!
//! Provides the ability to load [`GatewayConfig`] from a TOML file. Configuration
//! covers gateway instance identity, region, listen port, routing strategy,
//! cluster membership protocol, and backend connection info.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::backend::{BackendType, Endpoint, Quantization};
use crate::error::{HierKvGatewayError, Result};
use crate::ids::{InstanceId, RegionId, RegionTier};

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
    /// Configured backend list; empty by default.
    #[serde(default)]
    pub backends: Vec<BackendConfig>,
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
}

/// Routing strategy weights.
///
/// The three weights are typically normalized before participating in hybrid scoring.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyWeights {
    /// KV cache hit weight.
    pub kv: f64,
    /// Load weight.
    pub load: f64,
    /// Topology distance weight.
    pub topology: f64,
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
