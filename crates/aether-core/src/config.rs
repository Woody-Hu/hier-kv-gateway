//! 网关配置模型与 TOML 加载。
//!
//! 提供从 TOML 文件加载 [`GatewayConfig`] 的能力，配置覆盖网关实例身份、
//! 区域、监听端口、路由策略、集群成员协议与后端连接信息。

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::backend::{BackendType, Endpoint, Quantization};
use crate::error::{AetherError, Result};
use crate::ids::{InstanceId, RegionId, RegionTier};

/// 网关顶层配置。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// 当前网关实例标识。
    pub instance_id: InstanceId,
    /// 当前网关所在区域配置。
    pub region: RegionConfig,
    /// 监听配置。
    pub listen: ListenConfig,
    /// 路由策略配置。
    pub routing: RoutingConfig,
    /// 集群成员协议配置。
    pub cluster: ClusterConfig,
    /// 已配置的后端列表，缺省时为空。
    #[serde(default)]
    pub backends: Vec<BackendConfig>,
}

/// 区域配置。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionConfig {
    /// 区域标识。
    pub id: RegionId,
    /// 区域层级。
    pub tier: RegionTier,
    /// 地理坐标，可选。
    pub geo: Option<GeoCoordConfig>,
    /// 网络区域标签。
    pub network_zone: String,
}

/// TOML 友好的地理坐标。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoCoordConfig {
    /// 纬度。
    pub lat: f64,
    /// 经度。
    pub lon: f64,
}

/// 监听地址与端口。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListenConfig {
    /// 监听地址，例如 `0.0.0.0`。
    pub addr: String,
    /// 监听端口。
    pub port: u16,
}

/// 路由策略类型。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum StrategyType {
    /// 混合策略（KV + 负载 + 拓扑综合评分）。
    Hybrid,
    /// 仅按 KV 缓存命中率路由。
    Kv,
    /// 仅按模型可用性路由。
    Model,
    /// 仅按负载路由。
    Load,
    /// 仅按拓扑距离路由。
    Topology,
}

/// 路由策略权重。
///
/// 三个权重通常归一化后参与混合评分。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyWeights {
    /// KV 缓存命中权重。
    pub kv: f64,
    /// 负载权重。
    pub load: f64,
    /// 拓扑距离权重。
    pub topology: f64,
}

/// 路由相关配置。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// 主路由策略。
    pub strategy: StrategyType,
    /// KV 缓存块大小，必须与后端约定一致。
    pub kv_block_size: u32,
    /// 命中重叠时给予候选后端的额外评分信用。
    pub overlap_score_credit: f64,
    /// prefill 阶段负载缩放系数。
    pub prefill_load_scale: f64,
    /// 路由温度参数（softmax），值越大随机性越强。
    pub temperature: f64,
    /// 会话亲和 TTL（秒）。
    pub session_affinity_ttl_secs: u64,
    /// 最大重试次数。
    pub max_retries: u32,
    /// 混合策略下三个维度的权重。
    pub weights: StrategyWeights,
}

/// 集群成员协议配置（基于 SWIM/gossip）。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// 集群绑定地址。
    pub bind_addr: String,
    /// 种子节点地址列表。
    pub seed_peers: Vec<String>,
    /// gossip 间隔（毫秒）。
    pub gossip_interval_ms: u64,
    /// 探活超时（毫秒）。
    pub probe_timeout_ms: u64,
    /// suspect 状态超时（秒）。
    pub suspect_timeout_secs: u64,
}

/// 后端连接配置（不含运行时状态）。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendConfig {
    /// 后端类型。
    pub backend_type: BackendType,
    /// 后端端点。
    pub endpoint: Endpoint,
    /// 该后端承载的模型名列表。
    pub models: Vec<String>,
    /// 所在区域。
    pub region: RegionId,
    /// KV 缓存块大小。
    pub kv_block_size: u32,
    /// 可选的量化方式，仅用于配置侧的描述。
    pub quantization: Option<Quantization>,
}

/// 从 TOML 文件加载 [`GatewayConfig`]。
///
/// 文件内容须能反序列化为 `GatewayConfig`，否则返回 [`AetherError::ConfigError`]。
pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<GatewayConfig> {
    let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
        AetherError::ConfigError(format!(
            "无法读取配置文件 {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;
    let cfg: GatewayConfig = toml::from_str(&content).map_err(|e| {
        AetherError::ConfigError(format!("配置文件解析失败 {}: {}", path.as_ref().display(), e))
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
        assert!(matches!(res, Err(AetherError::ConfigError(_))));
    }

    #[test]
    fn load_from_invalid_toml_returns_config_error() {
        let tmp = tempfile_lite();
        std::fs::write(&tmp, "not = valid = toml").unwrap();
        let res = load_from_file(&tmp);
        assert!(matches!(res, Err(AetherError::ConfigError(_))));
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

    /// 生成一个唯一的临时文件路径用于测试。
    fn tempfile_lite() -> String {
        // 使用 std::process::id() 与原子计数器组合，避免依赖 tempfile crate。
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        format!("/tmp/aether-core-test-{}-{}.toml", pid, n)
    }
}
