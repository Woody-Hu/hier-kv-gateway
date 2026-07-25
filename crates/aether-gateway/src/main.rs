//! Aether LLM Gateway 主二进制入口。
//!
//! 该 binary 加载 TOML 配置文件，初始化 tracing、MetadataStore、ConnectorRegistry、
//! RoutingEngine 与 HTTP API server，并支持 Ctrl-C 优雅关闭。
//!
//! 典型用法：
//! ```bash
//! aether --config /path/to/aether.toml
//! ```
//!
//! 启动后会监听配置中 `[listen]` 指定的地址，并对外提供 OpenAI 兼容的 HTTP API。

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use aether_api::handlers::AppState;
use aether_api::server;
use aether_core::config::{load_from_file, GatewayConfig};
use aether_core::topology::{GeoCoord, RegionInfo};
use aether_metadata::store::MetadataStore;
use aether_connector::registry::ConnectorRegistry;
use aether_routing::engine::RoutingEngine;
use aether_routing::hybrid::HybridStrategy;
use aether_routing::kv_aware::KvAwareStrategy;
use aether_routing::load_aware::LoadAwareStrategy;
use aether_routing::model_aware::ModelAwareStrategy;
use aether_routing::topology_aware::TopologyAwareStrategy;

/// 命令行参数。
#[derive(Parser, Debug)]
#[command(
    name = "aether",
    version,
    about = "Aether LLM Gateway - OpenAI 兼容的 LLM 网关"
)]
struct CliArgs {
    /// 配置文件路径（TOML）。
    #[arg(short, long, value_name = "PATH", default_value = "aether.toml")]
    config: String,
}

/// 程序入口：解析参数、初始化组件、启动 HTTP server。
#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();

    // 1) 加载配置文件
    let config: GatewayConfig = load_from_file(&args.config)
        .with_context(|| format!("加载配置文件 {} 失败", args.config))?;

    // 2) 初始化 tracing
    init_tracing(&config);
    print_startup_banner(&config);

    // 3) 创建 MetadataStore，并注册本网关所在 Region 到拓扑图
    let metadata = Arc::new(MetadataStore::new());
    register_self_region(&metadata, &config);

    // 4) 创建 ConnectorRegistry，并发现后端
    let connectors = Arc::new(ConnectorRegistry::from_configs(
        &config.backends,
        &config.region.id,
    ));
    discover_and_register_backends(&connectors, &metadata).await;

    // 5) 创建 RoutingEngine（混合策略 + 配置参数）
    let routing = Arc::new(build_routing_engine(&config));

    // 6) 组装 AppState 并启动 HTTP server（启用 graceful shutdown）
    let app_state = AppState {
        metadata: metadata.clone(),
        routing,
        connectors,
        routing_config: config.routing.clone(),
    };

    let listen_addr = format!("{}:{}", config.listen.addr, config.listen.port);
    info!(addr = %listen_addr, "启动 HTTP server");

    if let Err(e) = server::serve_with_graceful_shutdown(&listen_addr, app_state).await {
        error!(error = %e, "HTTP server 异常退出");
        return Err(anyhow::anyhow!("HTTP server 退出: {}", e));
    }

    info!("Aether gateway 已停止");
    Ok(())
}

/// 初始化 tracing：默认 INFO 级别，可通过 `RUST_LOG` 环境变量覆盖。
fn init_tracing(config: &GatewayConfig) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = fmt::layer().with_target(true);
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();
    let _ = config; // 预留：未来可基于配置覆盖默认 subscriber
}

/// 把网关自身所在的 RegionInfo 写入拓扑图，便于后续 RTT 估算与拓扑感知路由。
fn register_self_region(metadata: &MetadataStore, config: &GatewayConfig) {
    let geo = config.region.geo.as_ref().map(|g| GeoCoord {
        lat: g.lat,
        lon: g.lon,
    });
    let info = RegionInfo {
        id: config.region.id.clone(),
        tier: config.region.tier.clone(),
        geo,
        network_zone: config.region.network_zone.clone(),
        endpoints: Vec::new(),
    };
    metadata.topo_add_region(info);
}

/// 调用每个已注册连接器的 `discover()`，把得到的 BackendInfo 注册到 MetadataStore。
///
/// 失败的发现不会中断启动流程，仅记录 warning。
async fn discover_and_register_backends(
    connectors: &ConnectorRegistry,
    metadata: &MetadataStore,
) {
    let registered_types = connectors.registered_types();
    info!(
        backend_types = registered_types.len(),
        "开始发现后端"
    );

    for bt in &registered_types {
        let connector = match connectors.get(bt) {
            Some(c) => c,
            None => continue,
        };
        match connector.discover().await {
            Ok(infos) => {
                for info in infos {
                    info!(
                        backend = %info.id,
                        backend_type = ?info.backend_type,
                        models = ?info.models.iter().map(|m| m.model_name.as_str()).collect::<Vec<_>>(),
                        "注册后端"
                    );
                    metadata.register_backend(info);
                }
            }
            Err(e) => {
                warn!(backend_type = ?bt, error = %e, "后端发现失败（跳过）");
            }
        }
    }

    info!(backends = metadata.backends_len(), "后端发现完成");
}

/// 根据 GatewayConfig 构造 RoutingEngine（内嵌默认的混合策略）。
fn build_routing_engine(config: &GatewayConfig) -> RoutingEngine {
    let r = &config.routing;
    let kv = Box::new(KvAwareStrategy {
        overlap_score_credit: r.overlap_score_credit,
        prefill_load_scale: r.prefill_load_scale,
        ckf_false_positive_penalty: 0.0,
    });
    let model = Box::new(ModelAwareStrategy::default());
    let load = Box::new(LoadAwareStrategy::default());
    let topology = Box::new(TopologyAwareStrategy {
        w_rtt: 1.0,
        w_bw: 0.0,
        self_region: config.region.id.clone(),
    });
    let hybrid = HybridStrategy::new(
        kv,
        model,
        load,
        topology,
        r.weights.clone(),
        r.temperature,
    );
    let session_affinity_ttl = Duration::from_secs(r.session_affinity_ttl_secs);
    RoutingEngine::new(
        hybrid,
        session_affinity_ttl,
        r.max_retries,
        config.region.id.clone(),
    )
}

/// 打印启动横幅，包含实例身份、监听地址、Region 与路由策略等关键信息。
fn print_startup_banner(config: &GatewayConfig) {
    let listen_addr = format!("{}:{}", config.listen.addr, config.listen.port);
    info!("======================================");
    info!("Aether LLM Gateway");
    info!("--------------------------------------");
    info!(instance_id = %config.instance_id, "实例标识");
    info!(region = %config.region.id, tier = ?config.region.tier, "所在区域");
    info!(addr = %listen_addr, "HTTP 监听地址");
    info!(strategy = ?config.routing.strategy, "路由策略");
    info!(
        kv = config.routing.weights.kv,
        load = config.routing.weights.load,
        topology = config.routing.weights.topology,
        "策略权重"
    );
    info!(
        kv_block_size = config.routing.kv_block_size,
        max_retries = config.routing.max_retries,
        session_affinity_ttl_secs = config.routing.session_affinity_ttl_secs,
        "路由参数"
    );
    info!(
        backends_configured = config.backends.len(),
        "已配置后端数量"
    );
    info!(
        bind_addr = %config.cluster.bind_addr,
        seeds = config.cluster.seed_peers.len(),
        "集群配置"
    );
    info!("======================================");
}
