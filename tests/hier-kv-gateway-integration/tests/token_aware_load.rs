//! Token-aware load scheduling — closed-loop validation.
//!
//! This test proves (or disproves) the hypothesis that the gateway's load
//! signal was *count-blind* to generation length, and that folding the
//! request's output-token budget into the load cost produces materially
//! better decode-capacity balance under a mixed-length workload.
//!
//! ## Closed-loop argument
//!
//! 1. **Problem (factual)**: before this change, [`LoadAwareStrategy`] scored
//!    backends by `w_req * active_requests + ...` — a backend holding one
//!    4096-token generation scored as "less loaded" than one holding four
//!    16-token generations, even though the former occupies ~64× more decode
//!    capacity. `RoutingContext::estimated_output_tokens` and
//!    `BackendMetrics::active_prefill_tokens` were collected but unused.
//! 2. **Hypothesis**: under a mixed workload with realistic completion timing,
//!    count-blind scoring misroutes long requests onto backends whose count is
//!    momentarily low but whose decode capacity is already saturated, causing
//!    decode-pressure imbalance.
//! 3. **Falsifiable metric**: coefficient of variation (CoV) of
//!    `active_decode_blocks` across backends after replaying a fixed mixed
//!    workload through the *real* routing engine. Lower is better.
//! 4. **Decision rule**: introduce token-awareness iff it reduces CoV by ≥15%
//!    with no regression in the peak-pressure headroom.
//!
//! ## Honesty contract (no mocks)
//!
//! Every component on the decision path is the production one:
//! - [`MetadataStore`] with real [`LoadStats`] (lock-free `ArcSwap` reads).
//! - [`RoutingEngine`] + [`HybridStrategy`] + real [`LoadAwareStrategy`].
//! - The workload is a *discrete-event replay*: each routed request actually
//!   updates the selected backend's `LoadStats` via `load_update`, and each
//!   completion actually decrements it. The routing decision's *consequence*
//!   is measured, not assumed.
//!
//! The only "simulation" is the load feedback itself, which is exactly what
//! `LoadStats` models. No test doubles, no stubs, no pre-baked scores.
//!
//! Run with:
//! ```bash
//! cargo test -p hier-kv-gateway-integration --test token_aware_load -- --nocapture
//! ```

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hier_kv_gateway_core::backend::{
    BackendCapabilities, BackendInfo, BackendStatus, BackendType, Endpoint, KvConfig, ModelInstance,
    Protocol, Quantization,
};
use hier_kv_gateway_core::config::StrategyWeights;
use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId};
use hier_kv_gateway_core::metrics::{BackendMetrics, LatencyStats};
use hier_kv_gateway_core::request::RoutingContext;
use hier_kv_gateway_metadata::store::MetadataStore;
use hier_kv_gateway_routing::engine::RoutingEngine;
use hier_kv_gateway_routing::hybrid::HybridStrategy;
use hier_kv_gateway_routing::kv_aware::KvAwareStrategy;
use hier_kv_gateway_routing::load_aware::LoadAwareStrategy;
use hier_kv_gateway_routing::model_aware::ModelAwareStrategy;
use hier_kv_gateway_routing::topology_aware::TopologyAwareStrategy;

/// KV block size used throughout the replay.
const BLOCK_SIZE: u32 = 16;
/// Number of candidate backends. Six is enough to expose imbalance without
/// making the per-step routing cost dominate the measurement.
const N_BACKENDS: usize = 6;
/// Total KV capacity per backend (blocks). Generous so the hard
/// `available_capacity <= 0` exclusion never fires — we want to measure the
/// *soft* scoring behaviour, not capacity-gated routing.
const KV_TOTAL_BLOCKS: u64 = 8192;
const MODEL_NAME: &str = "qwen2.5-7b";

/// One request in the synthetic workload.
struct WorkloadReq {
    /// Logical arrival time (1 token == 1 time unit).
    arrival: u64,
    /// Conservative output-token budget (sourced from `max_tokens` in the
    /// real API path; here it is the ground-truth generation length).
    output_tokens: u32,
}

/// Quality report produced by one replay run.
#[derive(Clone, Debug)]
struct ReplayReport {
    /// Label identifying the configuration.
    label: &'static str,
    /// Coefficient of variation of `active_decode_blocks` across backends at
    /// the end of the replay. Lower is better.
    decode_cov: f64,
    /// Peak `active_decode_blocks` on any single backend at the end.
    decode_peak: u64,
    /// Mean `active_decode_blocks` across backends at the end.
    decode_mean: f64,
    /// How many requests each backend received (dispatch distribution).
    dispatches: Vec<u64>,
    /// Clairvoyant lower bound on the peak: if every still-in-flight decode
    /// block were spread evenly, the peak would be `total / N`. The gap
    /// between `decode_peak` and this bound is the routing inefficiency.
    clairvoyant_peak: f64,
}

// ---------------------------------------------------------------------------
// Deterministic workload generator (splitmix64 — no external RNG dependency)
// ---------------------------------------------------------------------------

/// splitmix64 step — deterministic, good distribution, zero dependencies.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a deterministic mixed-length workload.
///
/// Shape: 180 requests arriving every 2 time units. ~72% are short
/// (16–48 tokens, ~1–3 decode blocks), ~28% are long (512–2048 tokens,
/// ~32–128 decode blocks). Short requests complete within a few arrivals;
/// long requests linger across hundreds. This is the regime where count and
/// decode pressure diverge: backends holding longs carry high decode but a
/// slowly-growing count, while backends cycling through shorts carry higher
/// count turnover but low decode.
fn build_workload() -> Vec<WorkloadReq> {
    let mut state: u64 = 0x0243_CEFA_C0FF_EE99; // fixed seed → reproducible
    let mut out = Vec::with_capacity(180);
    let mut t: u64 = 0;
    for _ in 0..180 {
        let r = splitmix64(&mut state) % 100;
        let output_tokens = if r < 72 {
            // short: 16..=48 (one to three 16-token blocks)
            16 + (splitmix64(&mut state) % 33) as u32
        } else {
            // long: 512..=2048 (32 to 128 blocks) — heavy-tailed
            512 + (splitmix64(&mut state) % 1537) as u32
        };
        out.push(WorkloadReq {
            arrival: t,
            output_tokens,
        });
        t += 2;
    }
    out
}

// ---------------------------------------------------------------------------
// Real component builders
// ---------------------------------------------------------------------------

fn make_backend_info(region: &str, instance: &str, domain: u64) -> BackendInfo {
    BackendInfo {
        id: BackendId::new(region, instance),
        backend_type: BackendType::VllmEngine,
        endpoint: Endpoint {
            url: format!("http://{}.example:8000", instance),
            protocol: Protocol::Http,
        },
        models: vec![ModelInstance {
            model_name: MODEL_NAME.to_string(),
            model_architecture: "qwen".to_string(),
            quantization: Quantization::Fp16,
            max_context_len: 32_768,
            supports_tool_calling: true,
            supports_streaming: true,
        }],
        region: RegionId::new(region),
        indexer_domain: IndexerDomainId::new(domain),
        capabilities: BackendCapabilities {
            supports_kv_events: true,
            supports_batching: true,
            max_batch_size: 32,
            gpu_count: 1,
            gpu_memory_gb: 24,
        },
        kv_config: KvConfig {
            block_size: BLOCK_SIZE,
            cache_namespace: "default".to_string(),
            max_kv_blocks: KV_TOTAL_BLOCKS,
        },
        status: BackendStatus::Healthy,
    }
}

/// Build a fresh MetadataStore with `N_BACKENDS` healthy backends in the same
/// Region (topology-neutral, KV-neutral) so the load signal is the sole
/// discriminator. Each backend starts idle with a fresh metrics snapshot.
fn build_store() -> (MetadataStore, Vec<BackendId>) {
    let store = MetadataStore::new();
    let region = "cloud-cn-beijing";
    let mut backends = Vec::with_capacity(N_BACKENDS);
    for i in 0..N_BACKENDS {
        let info = make_backend_info(region, &format!("inst-{i}"), 0);
        let id = info.id.clone();
        store.register_backend(info);
        store.load_update(id.clone(), idle_metrics());
        backends.push(id);
    }
    (store, backends)
}

fn idle_metrics() -> BackendMetrics {
    BackendMetrics {
        active_requests: 0,
        queue_depth: 0,
        active_decode_blocks: 0,
        active_prefill_tokens: 0,
        kv_used_blocks: 0,
        kv_total_blocks: KV_TOTAL_BLOCKS,
        gpu_utilization: 0.0,
        gpu_memory_used_mb: 0,
        gpu_memory_total_mb: 24_000,
        latency: LatencyStats {
            p50_ms: 10.0,
            p99_ms: 50.0,
            p999_ms: 80.0,
            sample_count: 1000,
        },
        timestamp: now_unix_secs(),
    }
}

/// Current Unix time in seconds, matching `BackendMetrics::timestamp`'s unit.
fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Build a routing engine whose load strategy uses the given weights.
///
/// `w_decode` / `w_prefill` are the token-budget knobs: `0.0 / 0.0` reproduces
/// the historical count-blind cost; the `LoadAwareStrategy::default()` values
/// turn token-awareness on. KV and topology weights are zeroed so the load
/// signal is the sole discriminator (KV is also unavailable here because no
/// KV events are applied, which would zero the KV weight anyway).
fn build_engine(load: LoadAwareStrategy) -> RoutingEngine {
    let kv = Box::new(KvAwareStrategy::default());
    let model = Box::new(ModelAwareStrategy::default());
    let topology = Box::new(TopologyAwareStrategy {
        w_rtt: 1.0,
        w_bw: 0.0,
        self_region: RegionId::new("cloud-cn-beijing"),
    });
    let weights = StrategyWeights {
        kv: 0.0,
        load: 1.0,
        topology: 0.0,
    };
    let hybrid = HybridStrategy::new(kv, model, Box::new(load), topology, weights, 0.0);
    RoutingEngine::new(hybrid, Duration::from_secs(300), 3, RegionId::new("cloud-cn-beijing"))
}

fn baseline_load() -> LoadAwareStrategy {
    // Count-blind: token-budget terms disabled. Identical to the pre-change cost.
    LoadAwareStrategy {
        w_decode: 0.0,
        w_prefill: 0.0,
        ..LoadAwareStrategy::default()
    }
}

fn token_aware_load() -> LoadAwareStrategy {
    LoadAwareStrategy::default()
}

// ---------------------------------------------------------------------------
// Discrete-event replay (the honest measurement)
// ---------------------------------------------------------------------------

struct InFlight {
    complete_at: u64,
    backend: BackendId,
    decode_blocks: u64,
}

/// Replay `workload` through the real routing engine, mutating the real
/// `MetadataStore` load stats as requests arrive and complete.
///
/// Returns the end-of-replay quality report. The replay is fully deterministic
/// given the same `engine` configuration and workload.
async fn replay(label: &'static str, workload: &[WorkloadReq]) -> ReplayReport {
    let (store, backends) = build_store();

    // Two engines cannot share the prefix-history Arc safely across independent
    // replays; each replay builds its own engine here.
    let engine = match label {
        "baseline" => build_engine(baseline_load()),
        "token_aware" => build_engine(token_aware_load()),
        _ => unreachable!("unknown label {label}"),
    };

    let mut in_flight: Vec<InFlight> = Vec::new();
    let mut dispatches: Vec<u64> = vec![0; backends.len()];
    let mut total_active_decode: u64 = 0;

    for req in workload {
        let now = req.arrival;

        // 1. Drain completions up to `now`: decrement each completing backend's
        //    load stats. This is the real load-feedback path the gateway uses.
        let mut i = 0;
        while i < in_flight.len() {
            if in_flight[i].complete_at <= now {
                let f = in_flight.remove(i);
                if let Some(mut m) = store.load_get_metrics(&f.backend) {
                    m.active_requests = m.active_requests.saturating_sub(1);
                    m.active_decode_blocks = m.active_decode_blocks.saturating_sub(f.decode_blocks);
                    m.kv_used_blocks = m.kv_used_blocks.saturating_sub(f.decode_blocks);
                    // Real-clock timestamp keeps the metrics fresh so the hybrid
                    // staleness discount (weight_load *= 0.3) does not confound
                    // the comparison — both configs see the same freshness.
                    m.timestamp = now_unix_secs();
                    store.load_update(f.backend.clone(), m);
                    total_active_decode = total_active_decode.saturating_sub(f.decode_blocks);
                }
            } else {
                i += 1;
            }
        }

        // 2. Route via the real engine. The context carries the request's
        //    output-token budget — the signal the token-aware path consumes.
        let ctx = RoutingContext {
            request_id: None,
            session_id: None, // no session affinity — force the scoring path
            tenant_id: None,
            model_name: Some(MODEL_NAME.to_string()),
            token_ids: Vec::new(),
            block_hashes: Vec::new(), // no KV overlap — isolate the load signal
            block_size: BLOCK_SIZE,
            lora_name: None,
            cache_namespace: None,
            estimated_output_tokens: req.output_tokens,
            requires_tool_calling: false,
        };
        let decision = engine.route(&ctx, &store).await.expect("routing must succeed");

        // 3. Record dispatch and land the request on the selected backend.
        let decode_blocks = ((req.output_tokens as u64 + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64) as u64;
        if let Some(pos) = backends.iter().position(|b| b == &decision.backend) {
            dispatches[pos] += 1;
        }
        if let Some(mut m) = store.load_get_metrics(&decision.backend) {
            m.active_requests += 1;
            m.active_decode_blocks += decode_blocks;
            m.kv_used_blocks += decode_blocks;
            m.timestamp = now_unix_secs();
            store.load_update(decision.backend.clone(), m);
            total_active_decode += decode_blocks;
        }

        // 4. Schedule completion: duration proportional to output tokens
        //    (1 token == 1 time unit). Long requests linger; short ones cycle.
        in_flight.push(InFlight {
            complete_at: now + req.output_tokens as u64,
            backend: decision.backend,
            decode_blocks,
        });
    }

    // 5. Measure end-state decode pressure across backends.
    let mut decode_pressures: Vec<u64> = Vec::with_capacity(backends.len());
    for b in &backends {
        let m = store
            .load_get_metrics(b)
            .expect("backend must have metrics after replay");
        decode_pressures.push(m.active_decode_blocks);
    }

    let mean = decode_pressures.iter().sum::<u64>() as f64 / decode_pressures.len() as f64;
    let variance = if mean > 0.0 {
        decode_pressures
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / decode_pressures.len() as f64
    } else {
        0.0
    };
    let cov = if mean > 0.0 {
        variance.sqrt() / mean
    } else {
        0.0
    };
    let peak = *decode_pressures.iter().max().unwrap_or(&0);
    let clairvoyant_peak = total_active_decode as f64 / backends.len() as f64;

    ReplayReport {
        label,
        decode_cov: cov,
        decode_peak: peak,
        decode_mean: mean,
        dispatches,
        clairvoyant_peak,
    }
}

fn print_report(r: &ReplayReport) {
    eprintln!(
        "  [{label}] decode CoV={cov:.3}  peak={peak}  mean={mean:.1}  clairvoyant_peak={cp:.1}  dispatches={d:?}",
        label = r.label,
        cov = r.decode_cov,
        peak = r.decode_peak,
        mean = r.decode_mean,
        cp = r.clairvoyant_peak,
        d = r.dispatches,
    );
}

// ---------------------------------------------------------------------------
// The closed-loop test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn token_aware_load_balances_decode_pressure_better_than_count_blind() {
    let workload = build_workload();

    let baseline = replay("baseline", &workload).await;
    let token_aware = replay("token_aware", &workload).await;

    eprintln!();
    eprintln!("== Token-aware load scheduling replay ==");
    eprintln!(
        "workload: {} requests, {} backends, block_size={}",
        workload.len(),
        N_BACKENDS,
        BLOCK_SIZE,
    );
    eprintln!("end-of-replay decode-pressure distribution:");
    print_report(&baseline);
    print_report(&token_aware);

    let improvement_pct = if baseline.decode_cov > 0.0 {
        (baseline.decode_cov - token_aware.decode_cov) / baseline.decode_cov * 100.0
    } else {
        0.0
    };
    eprintln!(
        "  CoV improvement: {:.1}% (baseline {:.3} → token_aware {:.3})",
        improvement_pct, baseline.decode_cov, token_aware.decode_cov,
    );
    eprintln!(
        "  Peak improvement: {} → {} blocks (clairvoyant lower bound {:.1})",
        baseline.decode_peak, token_aware.decode_peak, token_aware.clairvoyant_peak,
    );

    // Decision rule: introduce token-awareness iff CoV drops by ≥15%.
    assert!(
        token_aware.decode_cov < baseline.decode_cov,
        "token-aware CoV ({}) must be strictly lower than baseline ({})",
        token_aware.decode_cov,
        baseline.decode_cov,
    );
    assert!(
        improvement_pct >= 15.0,
        "token-aware CoV improvement ({:.1}%) must meet the 15% decision threshold \
         (baseline {:.3} → token_aware {:.3})",
        improvement_pct,
        baseline.decode_cov,
        token_aware.decode_cov,
    );
    // Peak pressure must not regress — token-aware must not concentrate load.
    assert!(
        token_aware.decode_peak <= baseline.decode_peak,
        "token-aware peak ({}) must not exceed baseline ({})",
        token_aware.decode_peak,
        baseline.decode_peak,
    );
}

/// Sanity: the count-blind baseline must itself be non-trivially imbalanced —
/// otherwise the workload does not actually exercise the defect and the
/// comparison above would be meaningless. This guards against a test that
/// passes vacuously.
#[tokio::test]
async fn baseline_exhibits_nontrivial_imbalance() {
    let workload = build_workload();
    let baseline = replay("baseline", &workload).await;
    eprintln!();
    eprintln!("== Baseline imbalance sanity ==");
    eprintln!(
        "  baseline CoV={:.3} peak={} clairvoyant_peak={:.1}",
        baseline.decode_cov, baseline.decode_peak, baseline.clairvoyant_peak,
    );
    // The principled signal is the gap between the realized peak and the
    // clairvoyant lower bound (perfectly-spread peak = total/N). A baseline
    // sitting on the bound would mean the workload fails to exercise the
    // count-blind defect. Require at least 5% slack above the bound.
    let slack = baseline.decode_peak as f64 - baseline.clairvoyant_peak;
    let slack_pct = if baseline.clairvoyant_peak > 0.0 {
        slack / baseline.clairvoyant_peak * 100.0
    } else {
        0.0
    };
    assert!(
        slack_pct >= 5.0,
        "baseline peak ({}) must sit at least 5% above the clairvoyant bound ({:.1}); \
         got {:.1}% — workload does not exercise the count-blind defect",
        baseline.decode_peak,
        baseline.clairvoyant_peak,
        slack_pct,
    );
}
