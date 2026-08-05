//! Hier KV Gateway main binary entry point.
//!
//! This binary loads a TOML config file, initializes tracing, MetadataStore,
//! ConnectorRegistry, RoutingEngine, the cluster gossip engine, and the HTTP API
//! server, and supports graceful shutdown on Ctrl-C.
//!
//! Typical usage:
//! ```bash
//! hier-kv-gateway --config /path/to/hier-kv-gateway.toml
//! ```
//!
//! After startup, it listens on the address specified by `[listen]` in the config and
//! serves an OpenAI-compatible HTTP API. The `[cluster]` section additionally binds a
//! TCP transport for SWIM/gossip membership management; the `POST /cluster/peers`
//! endpoint can be used to dynamically attach external-Region gateways after startup.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::Parser;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use hier_kv_gateway_api::coalescer::RequestCoalescer;
use hier_kv_gateway_api::handlers::{AppState, PeerRegistrar};
use hier_kv_gateway_api::server;
use hier_kv_gateway_api::telemetry::build_telemetry;
use hier_kv_gateway_cluster::gossip::GossipEngine;
use hier_kv_gateway_cluster::member::MemberList;
use hier_kv_gateway_cluster::region_view::RegionMemberView;
use hier_kv_gateway_cluster::tcp_transport::TcpClusterTransport;
use hier_kv_gateway_cluster::transport::ClusterTransport;
use hier_kv_gateway_core::config::{load_from_file, GatewayConfig};
use hier_kv_gateway_core::topology::{GeoCoord, RegionInfo};
use hier_kv_gateway_metadata::store::MetadataStore;
use hier_kv_gateway_connector::registry::ConnectorRegistry;
use hier_kv_gateway_connector::resilience::{CircuitBreakerRegistry, RetryPolicy};
use hier_kv_gateway_routing::adaptive::AdaptiveWeightController;
use hier_kv_gateway_routing::engine::RoutingEngine;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::plugin::RoutingPlugin;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

mod cluster_bridge;

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
        &config.forwarding,
    ));
    discover_and_register_backends(&connectors, &metadata).await;

    // 5) Create the RoutingEngine (hybrid strategy + config parameters)
    let routing = Arc::new(build_routing_engine(&config));

    // 6) Start the cluster gossip engine (membership + Meet/Ping/Pong transport).
    //    The HTTP layer's PeerRegistrar is backed by GossipEngine::meet_peer, so the
    //    `POST /cluster/peers` endpoint only becomes functional once this step succeeds.
    //    The gossip handler bridges into MetadataStore so that metrics/topology/
    //    CKF/session-affinity broadcasts received from peers are applied locally.
    let cluster_state = start_cluster(&config, metadata.clone()).await;

    // 7) Assemble the AppState and start the HTTP server (with graceful shutdown enabled)
    let breakers = Arc::new(CircuitBreakerRegistry::new(&config.resilience));
    let retry_policy = RetryPolicy::new(
        Duration::from_millis(config.resilience.retry_backoff_ms),
        Duration::from_millis(config.resilience.retry_max_backoff_ms),
    );
    // Decision telemetry: ring buffer (admin endpoint) + optional tracing /
    // NDJSON-file sinks, driven by the `[telemetry]` section.
    let telemetry = build_telemetry(&config.telemetry).await;
    info!(
        mode = ?config.telemetry.mode,
        buffer_size = config.telemetry.buffer_size,
        adaptive_enabled = config.routing.adaptive.enabled,
        "decision telemetry configured"
    );
    let app_state = AppState {
        metadata: metadata.clone(),
        routing,
        connectors,
        routing_config: config.routing.clone(),
        breakers,
        retry_policy,
        peer_registrar: cluster_state
            .as_ref()
            .map(|cs| cs.peer_registrar.clone()),
        decision_sink: telemetry.sink,
        decision_buffer: telemetry.buffer,
        gateway_instance: config.instance_id.to_string(),
        gateway_region: config.region.id.to_string(),
        coalescer: RequestCoalescer::new(config.coalescing.clone()),
    };

    let listen_addr = format!("{}:{}", config.listen.addr, config.listen.port);
    info!(addr = %listen_addr, "starting HTTP server");

    if let Err(e) = server::serve_with_graceful_shutdown(&listen_addr, app_state).await {
        error!(error = %e, "HTTP server exited abnormally");
        // Still try to stop the cluster engine so the gossip transport does not leak.
        if let Some(cs) = cluster_state.as_ref() {
            let _ = cs.engine.stop().await;
        }
        return Err(anyhow::anyhow!("HTTP server exited: {}", e));
    }

    // 8) Graceful shutdown: stop the cluster engine after the HTTP server has exited.
    if let Some(cs) = cluster_state.as_ref() {
        info!("stopping cluster gossip engine");
        if let Err(e) = cs.engine.stop().await {
            warn!(error = %e, "cluster engine stop returned an error");
        }
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
/// Backends reported by a connector beyond its own anchor (e.g. workers behind
/// a Dynamo front-end) are additionally attached as registry aliases so the
/// forwarding loop can address them individually.
///
/// Failed discoveries do not abort the startup flow; they only log a warning.
async fn discover_and_register_backends(
    connectors: &ConnectorRegistry,
    metadata: &MetadataStore,
) {
    let all = connectors.all();
    info!(
        connectors = all.len(),
        backend_types = connectors.registered_types().len(),
        "starting backend discovery"
    );

    for connector in &all {
        match connector.discover().await {
            Ok(infos) => {
                for info in infos {
                    info!(
                        backend = %info.id,
                        backend_type = ?info.backend_type,
                        models = ?info.models.iter().map(|m| m.model_name.as_str()).collect::<Vec<_>>(),
                        "registering backend"
                    );
                    // Make the discovered id addressable even when it differs
                    // from the connector's anchor id (alias is a no-op then).
                    connectors.register_alias(info.id.clone(), connector);
                    metadata.register_backend(info);
                }
            }
            Err(e) => {
                warn!(backend_type = ?connector.backend_type(), error = %e, "backend discovery failed (skipped)");
            }
        }
    }

    info!(backends = metadata.backends_len(), "backend discovery completed");
}

/// Build the RoutingEngine from the GatewayConfig.
///
/// The hybrid strategy is always embedded (it backs `trace_sub_scores` and the
/// degradation fallback); the configured `routing.strategy` then decides which
/// strategy becomes the *primary* scorer:
///
/// - `hybrid` — the hybrid itself (default);
/// - `round_robin` — the metadata-free rotation baseline;
/// - `kv` / `model` / `load` / `topology` — the corresponding single
///   sub-strategy, lifted out of the hybrid ensemble.
///
/// When `[routing.adaptive] enabled = true`, an EMA-based
/// [`AdaptiveWeightController`] is attached to the hybrid strategy so its
/// weights react to forward outcomes and gossip-fed load state at runtime.
///
/// ## Plugin extension
///
/// Two extra sub-strategies are attached as [`RoutingPlugin`]s when their
/// config sections are enabled, contributing a weighted score term to the
/// hybrid ensemble without forking the engine:
///
/// * `[cost] enabled = true` — [`CostAwareStrategy`] scores backends by
///   projected dollar cost (LiteLLM `cost-based-routing` analogue).
/// * `[model_tier] enabled = true` with `policy.type = "pick"` —
///   [`ModelTierStrategy`] scores backends by large/small tier match
///   (Portkey "conditional routing" analogue).
///
/// `[model_tier] policy.type = "fallback"` is *not* attached as a hybrid
/// plugin — it is meant to be the primary strategy so the forwarding loop's
/// retry realizes "small then large" ordering. When the operator sets
/// `routing.strategy = "hybrid"` (default) and chooses `fallback`, we install
/// it as the primary scorer instead.
fn build_routing_engine(config: &GatewayConfig) -> RoutingEngine {
    use hier_kv_gateway_core::config::StrategyType;
    use hier_kv_gateway_core::model_tier::TierRoutingPolicy;
    use hier_kv_gateway_routing::cost_aware::CostAwareStrategy;
    use hier_kv_gateway_routing::model_tier::ModelTierStrategy;
    use hier_kv_gateway_routing::round_robin::RoundRobinStrategy;
    use hier_kv_gateway_routing::strategy::RoutingStrategy;

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
    let mut hybrid = HybridStrategy::new(
        kv,
        model,
        load,
        topology,
        r.weights.clone(),
        r.temperature,
    );
    if r.adaptive.enabled {
        let controller = AdaptiveWeightController::new(r.weights.clone(), r.adaptive.clone());
        hybrid = hybrid.with_adaptive(Arc::new(controller));
    }

    // Attach cost-aware sub-strategy as a plugin when enabled. The plugin's
    // weight comes from `[cost] weight`, participating in the same hybrid
    // normalization pass as kv/load/topology.
    if config.cost.enabled {
        let cost_model = Arc::new(config.cost.build_model());
        let cost_strategy = CostAwareStrategy::new(cost_model, config.cost.clone());
        hybrid = hybrid.with_plugin(RoutingPlugin::from_strategy(Arc::new(cost_strategy)));
    }

    // Attach model-tier sub-strategy as a plugin when enabled AND the policy
    // is `Pick` (a soft sub-strategy). The `Fallback` policy is installed as
    // the primary scorer below so the forwarding loop's retry realizes the
    // "small then large" chain.
    if config.model_tier.enabled {
        if matches!(config.model_tier.policy, TierRoutingPolicy::Pick { .. }) {
            let tier_strategy =
                ModelTierStrategy::new(Arc::new(config.model_tier.clone()));
            hybrid = hybrid.with_plugin(RoutingPlugin::from_strategy(Arc::new(tier_strategy)));
        }
    }

    let session_affinity_ttl = Duration::from_secs(r.session_affinity_ttl_secs);
    let engine = RoutingEngine::new(
        hybrid,
        session_affinity_ttl,
        r.max_retries,
        config.region.id.clone(),
    );

    // Resolve the primary strategy. The default `Hybrid` keeps the hybrid
    // ensemble as the primary scorer. The single-strategy variants
    // (`Kv`/`Load`/`Topology`/`Model`/`RoundRobin`) are realized via
    // `with_primary_strategy`, exactly as before. The `model_tier` `Fallback`
    // policy is *also* realized here — when enabled with `policy = fallback`,
    // it overrides the configured `strategy` to become the primary so the
    // ranked candidate list becomes "small first, then large" and the
    // forwarding loop's retry realizes the fallback chain for free.
    let primary: Option<Box<dyn RoutingStrategy>> =
        if config.model_tier.enabled
            && matches!(config.model_tier.policy, TierRoutingPolicy::Fallback)
        {
            Some(Box::new(ModelTierStrategy::new(Arc::new(
                config.model_tier.clone(),
            ))))
        } else {
            match r.strategy {
                StrategyType::Hybrid => None,
                StrategyType::RoundRobin => Some(Box::new(RoundRobinStrategy::new())),
                StrategyType::Kv => Some(Box::new(KvAwareStrategy {
                    overlap_score_credit: r.overlap_score_credit,
                    prefill_load_scale: r.prefill_load_scale,
                    ckf_false_positive_penalty: 0.0,
                })),
                StrategyType::Model => Some(Box::new(ModelAwareStrategy::default())),
                StrategyType::Load => Some(Box::new(LoadAwareStrategy::default())),
                StrategyType::Topology => Some(Box::new(TopologyAwareStrategy {
                    w_rtt: 1.0,
                    w_bw: 0.0,
                    self_region: config.region.id.clone(),
                })),
            }
        };
    match primary {
        Some(p) => engine.with_primary_strategy(p),
        None => engine,
    }
}

/// Owned handle to a running cluster engine, kept around so the main binary can
/// stop it cleanly on shutdown.
struct ClusterState {
    /// The gossip engine itself; shared with the `PeerRegistrar` impl.
    engine: Arc<GossipEngine>,
    /// Backs the `POST /cluster/peers` HTTP endpoint.
    peer_registrar: Arc<dyn PeerRegistrar>,
    /// Region-grouped view over the member list; kept so future HTTP
    /// diagnostic endpoints (`/admin/cluster` etc.) can answer
    /// "which foreign Regions are currently represented?" without rebuilding
    /// the view from scratch.
    #[allow(dead_code)]
    region_view: RegionMemberView,
}

/// [`PeerRegistrar`] implementation backed by [`GossipEngine::meet_peer`].
///
/// Each `POST /cluster/peers` call results in a single `Meet` message being sent
/// to the requested peer address; the standard gossip loop then propagates the
/// new member to the rest of the cluster.
struct GossipPeerRegistrar {
    engine: Arc<GossipEngine>,
}

#[async_trait]
impl PeerRegistrar for GossipPeerRegistrar {
    async fn meet_peer(&self, peer_addr: &str) -> std::result::Result<(), String> {
        self.engine
            .meet_peer(peer_addr)
            .await
            .map_err(|e| e.to_string())
    }
}

/// Bring up the cluster gossip engine: bind the TCP transport, start the gossip
/// / probe / message loops, and send `Meet` to every configured seed peer.
///
/// Returns `None` (and logs a warning) when the transport fails to bind — the
/// HTTP server can still serve requests in single-node mode, but the
/// `POST /cluster/peers` endpoint will return 503.
///
/// `bind_addr` is also used as the address advertised in `Meet` messages; if
/// it is `0.0.0.0:port`, peers will not be able to dial back. Operators should
/// set `cluster.bind_addr` to a reachable IP for cross-Region deployments.
async fn start_cluster(
    config: &GatewayConfig,
    metadata: Arc<MetadataStore>,
) -> Option<ClusterState> {
    let cluster_cfg = &config.cluster;
    let members = Arc::new(MemberList::new());
    let transport: Arc<dyn ClusterTransport> = Arc::new(TcpClusterTransport::new(members.clone()));

    let handler = Arc::new(cluster_bridge::MetadataGossipHandler::new(
        metadata,
        members.clone(),
    ));

    let region_view = RegionMemberView::new(members.clone());

    let engine = Arc::new(GossipEngine::new(
        config.instance_id.clone(),
        config.region.id.clone(),
        cluster_cfg.bind_addr.clone(),
        members,
        transport,
        cluster_cfg.clone(),
        handler,
    ));

    if let Err(e) = engine.start().await {
        warn!(
            bind_addr = %cluster_cfg.bind_addr,
            error = %e,
            "cluster engine failed to start; running in single-node mode (POST /cluster/peers will return 503)"
        );
        return None;
    }

    info!(
        bind_addr = %cluster_cfg.bind_addr,
        seeds = cluster_cfg.seed_peers.len(),
        gossip_interval_ms = cluster_cfg.gossip_interval_ms,
        gossip_fanout = cluster_cfg.gossip_fanout,
        probe_interval_ms = cluster_cfg.probe_interval_ms,
        local_region = %config.region.id,
        "cluster gossip engine started"
    );

    // Best-effort join: send Meet to each seed peer. Failures are logged but
    // do not abort startup — the gossip loop will retry via subsequent Pings.
    if !cluster_cfg.seed_peers.is_empty() {
        if let Err(e) = engine.join_cluster(&cluster_cfg.seed_peers).await {
            warn!(error = %e, "join_cluster encountered an error (continuing)");
        }
    }

    // Log the initial region-grouped membership snapshot. At startup this is
    // typically just ourselves, but after `join_cluster` lands the seed-peers
    // will appear here on the next gossip round.
    let local = region_view.local_alive(&config.region.id);
    let foreign = region_view.foreign_alive(&config.region.id);
    info!(
        local_alive = local.len(),
        foreign_alive = foreign.len(),
        foreign_regions = ?region_view
            .foreign_regions(&config.region.id)
            .iter()
            .map(|r| r.as_str().to_string())
            .collect::<Vec<_>>(),
        "region-grouped membership snapshot"
    );

    let peer_registrar: Arc<dyn PeerRegistrar> = Arc::new(GossipPeerRegistrar {
        engine: engine.clone(),
    });

    Some(ClusterState {
        engine,
        peer_registrar,
        region_view,
    })
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
        gossip_interval_ms = config.cluster.gossip_interval_ms,
        gossip_fanout = config.cluster.gossip_fanout,
        probe_interval_ms = config.cluster.probe_interval_ms,
        probe_timeout_ms = config.cluster.probe_timeout_ms,
        suspect_timeout_secs = config.cluster.suspect_timeout_secs,
        "cluster configuration"
    );
    info!("======================================");
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::backend::BackendType;
    use hier_kv_gateway_core::config::TelemetryMode;

    /// Path to the workspace-root `examples/` directory.
    fn example_path(name: &str) -> String {
        format!("{}/../../examples/{}", env!("CARGO_MANIFEST_DIR"), name)
    }

    /// Every shipped example config must parse into `GatewayConfig`.
    #[test]
    fn example_configs_parse() {
        for name in [
            "hier-kv-gateway.toml",
            "multi-backend.toml",
            "sglang-backend.toml",
        ] {
            let path = example_path(name);
            load_from_file(&path).unwrap_or_else(|e| panic!("{} failed to parse: {}", path, e));
        }
    }

    /// The SGLang example exercises the new sections end to end:
    /// `sglang_engine` backends, `[forwarding]`, `[routing.adaptive]`,
    /// and `[telemetry]`.
    #[test]
    fn sglang_example_wires_new_sections() {
        let cfg = load_from_file(example_path("sglang-backend.toml")).unwrap();

        assert_eq!(cfg.backends.len(), 2);
        assert!(cfg
            .backends
            .iter()
            .all(|b| b.backend_type == BackendType::SglangEngine));
        assert!(cfg.forwarding.emit_token_ids);
        assert!(cfg.routing.adaptive.enabled);
        assert_eq!(cfg.telemetry.mode, TelemetryMode::File);
        assert_eq!(cfg.telemetry.buffer_size, 1024);
    }

    /// Building the routing engine from the SGLang example must attach an
    /// adaptive controller (weights snapshot reflects controller state).
    #[test]
    fn sglang_example_builds_adaptive_engine() {
        let cfg = load_from_file(example_path("sglang-backend.toml")).unwrap();
        let engine = build_routing_engine(&cfg);
        let w = engine.weight_snapshot();
        // Weights normalize to <= 1 and stay positive.
        assert!(w.kv > 0.0 && w.load > 0.0 && w.topology > 0.0);
        assert!(w.kv <= 1.0 && w.load <= 1.0 && w.topology <= 1.0);
    }

    /// The multi-backend example must keep building a static-weight engine.
    #[test]
    fn multi_backend_example_builds_static_engine() {
        let cfg = load_from_file(example_path("multi-backend.toml")).unwrap();
        assert!(!cfg.routing.adaptive.enabled);
        let engine = build_routing_engine(&cfg);
        let w = engine.weight_snapshot();
        assert!((w.kv - 0.35).abs() < 1e-9);
        assert!((w.load - 0.30).abs() < 1e-9);
        assert!((w.topology - 0.20).abs() < 1e-9);
    }
}
