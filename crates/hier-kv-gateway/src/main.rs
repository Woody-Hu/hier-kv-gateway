//! Hier KV Gateway main binary entry point.
//!
//! This binary loads a TOML config file, initializes tracing, MetadataStore,
//! ConnectorRegistry, RoutingEngine, and the HTTP API server, and supports graceful
//! shutdown on Ctrl-C.
//!
//! Typical usage:
//! ```bash
//! hier-kv-gateway --config /path/to/hier-kv-gateway.toml
//! ```
//!
//! After startup, it listens on the address specified by `[listen]` in the config and
//! serves an OpenAI-compatible HTTP API.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use hier_kv_gateway_api::handlers::AppState;
use hier_kv_gateway_api::server;
use hier_kv_gateway_core::config::{load_from_file, GatewayConfig};
use hier_kv_gateway_core::topology::{GeoCoord, RegionInfo};
use hier_kv_gateway_metadata::store::MetadataStore;
use hier_kv_gateway_connector::registry::ConnectorRegistry;
use hier_kv_gateway_routing::engine::RoutingEngine;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

/// Command-line arguments.
#[derive(Parser, Debug)]
#[command(
    name = "hier-kv-gateway",
    version,
    about = "Hier KV Gateway - OpenAI-compatible LLM gateway"
)]
struct CliArgs {
    /// Configuration file path (TOML).
    #[arg(short, long, value_name = "PATH", default_value = "hier-kv-gateway.toml")]
    config: String,
}

/// Program entry point: parses arguments, initializes components, and starts the HTTP
/// server.
#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();

    // 1) Load the configuration file
    let config: GatewayConfig = load_from_file(&args.config)
        .with_context(|| format!("failed to load configuration file {}", args.config))?;

    // 2) Initialize tracing
    init_tracing(&config);
    print_startup_banner(&config);

    // 3) Create the MetadataStore, and register the Region where this gateway resides into the topology graph
    let metadata = Arc::new(MetadataStore::new());
    register_self_region(&metadata, &config);

    // 4) Create the ConnectorRegistry, and discover backends
    let connectors = Arc::new(ConnectorRegistry::from_configs(
        &config.backends,
        &config.region.id,
    ));
    discover_and_register_backends(&connectors, &metadata).await;

    // 5) Create the RoutingEngine (hybrid strategy + config parameters)
    let routing = Arc::new(build_routing_engine(&config));

    // 6) Assemble the AppState and start the HTTP server (with graceful shutdown enabled)
    let app_state = AppState {
        metadata: metadata.clone(),
        routing,
        connectors,
        routing_config: config.routing.clone(),
    };

    let listen_addr = format!("{}:{}", config.listen.addr, config.listen.port);
    info!(addr = %listen_addr, "starting HTTP server");

    if let Err(e) = server::serve_with_graceful_shutdown(&listen_addr, app_state).await {
        error!(error = %e, "HTTP server exited abnormally");
        return Err(anyhow::anyhow!("HTTP server exited: {}", e));
    }

    info!("Hier KV Gateway has stopped");
    Ok(())
}

/// Initialize tracing: default INFO level, can be overridden via the `RUST_LOG`
/// environment variable.
fn init_tracing(config: &GatewayConfig) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = fmt::layer().with_target(true);
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();
    let _ = config; // Reserved: in the future, the default subscriber may be overridden based on config
}

/// Write the gateway's own RegionInfo into the topology graph, for subsequent RTT
/// estimation and topology-aware routing.
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

/// Invoke each registered connector's `discover()` and register the resulting
/// BackendInfo into the MetadataStore.
///
/// Failed discoveries do not abort the startup flow; they only log a warning.
async fn discover_and_register_backends(
    connectors: &ConnectorRegistry,
    metadata: &MetadataStore,
) {
    let registered_types = connectors.registered_types();
    info!(
        backend_types = registered_types.len(),
        "starting backend discovery"
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
                        "registering backend"
                    );
                    metadata.register_backend(info);
                }
            }
            Err(e) => {
                warn!(backend_type = ?bt, error = %e, "backend discovery failed (skipped)");
            }
        }
    }

    info!(backends = metadata.backends_len(), "backend discovery completed");
}

/// Build the RoutingEngine from the GatewayConfig (with the default hybrid strategy
/// embedded).
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

/// Print the startup banner, including instance identity, listen address, Region, and
/// routing strategy, and other key information.
fn print_startup_banner(config: &GatewayConfig) {
    let listen_addr = format!("{}:{}", config.listen.addr, config.listen.port);
    info!("======================================");
    info!("Hier KV Gateway");
    info!("--------------------------------------");
    info!(instance_id = %config.instance_id, "instance identifier");
    info!(region = %config.region.id, tier = ?config.region.tier, "region");
    info!(addr = %listen_addr, "HTTP listen address");
    info!(strategy = ?config.routing.strategy, "routing strategy");
    info!(
        kv = config.routing.weights.kv,
        load = config.routing.weights.load,
        topology = config.routing.weights.topology,
        "strategy weights"
    );
    info!(
        kv_block_size = config.routing.kv_block_size,
        max_retries = config.routing.max_retries,
        session_affinity_ttl_secs = config.routing.session_affinity_ttl_secs,
        "routing parameters"
    );
    info!(
        backends_configured = config.backends.len(),
        "number of configured backends"
    );
    info!(
        bind_addr = %config.cluster.bind_addr,
        seeds = config.cluster.seed_peers.len(),
        "cluster configuration"
    );
    info!("======================================");
}
