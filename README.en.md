> English | [中文](README.md)

# Hier KV Gateway

> An automatic scheduling gateway system for LLM requests across cloud-edge-device collaboration

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-430%20passed-brightgreen.svg)](#tests)

## Overview

Hier KV Gateway is an automatic scheduling gateway for LLM inference requests in **cloud-edge-device collaborative** scenarios. It unifies inference backends distributed across different geographic locations, scales, and engines (cloud-side K8s clusters, edge-side lightweight clusters, device-side vLLM/llama.cpp single processes) into a single inference service entry point, and automatically routes requests to the most appropriate backend through intelligent routing strategies.

### Core Capabilities

| Capability | Description |
|------|------|
| **KV-aware routing** | Sense the KV Cache prefix overlap of each backend and route to the backend with the highest cache hit rate, reducing prefill computation |
| **KV-capacity-aware routing** | Estimate a request's KV Cache memory footprint and score backends by their remaining KV-block / GPU-memory headroom, excluding backends that cannot fit the request (load shedding) |
| **Model-aware routing** | Route based on exact/compatible matching of the loaded model, version, and quantization of backends |
| **Load-aware routing** | Collect backend metrics such as queue depth, GPU utilization, and KV usage in real time for load balancing |
| **Topology-aware routing** | Preferentially route to nearby backends based on the network latency topology, reducing end-to-end latency |
| **Hybrid intelligent routing** | The default strategy, fusing the four aspects above via weighted scoring for an aggregate decision |
| **KV memory estimation** | A standalone leaf crate that estimates KV Cache memory with analytical formulas (not simulation): zero-allocation hot path, pluggable extension, builtin mainstream models |
| **Gossip cluster communication** | Gateway instance groups across clusters discover each other and synchronize metadata via the Gossip protocol |
| **CKF cross-domain KV projection** | A compact Cuckoo Filter projection of cross-Region KV state enables cross-domain KV-aware routing |
| **Pluggable architecture** | Routing strategies, backend connectors, and cluster communication are all extensible |
| **Service degradation** | Automatically degrades to the next-best strategy when metadata is unavailable, ultimately falling back to basic load balancing |
| **OpenAI-compatible API** | Fully compatible with the OpenAI Chat Completions API, supporting streaming SSE |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client (OpenAI API)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP
┌────────────────────────────────▼────────────────────────────────────┐
│                     Hier KV Gateway Process                         │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ HTTP/API │→ │ Routing      │  │ Metadata   │  │  Gossip      │  │
│  │  Server  │  │ Engine       │  │ Store      │  │  Layer       │  │
│  │          │  │ (5 strategies│  │ (in-memory │  │ (cluster     │  │
│  │          │  │  + Hybrid)   │  │  cache)    │  │  comm)       │  │
│  └──────────┘  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘  │
│                       │                │                 │          │
│  ┌────────────────────▼────────────────▼─────────────────▼───────┐  │
│  │                    Backend Connectors (plugins)               │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │  │
│  │  │  LLM-D   │ │ vLLM     │ │llama.cpp │ │  Generic OpenAI   │  │  │
│  │  │ Cluster  │ │ Engine   │ │ Engine   │ │  Compatible      │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          ▲ Gossip                    ▲ Gossip
          │                            │
┌─────────┴──────────┐       ┌─────────┴──────────┐
│  Gateway (Cloud)   │◄─────►│  Gateway (Edge)    │
│  ┌───────────────┐ │       │ ┌────────────────┐ │
│  │  LLM-D Cluster│ │       │ │  vLLM Engine    │ │
│  │  (K8s + multi)│ │       │ │  (single proc)  │ │
│  └───────────────┘ │       │ └────────────────┘ │
└───────────────────┘       └────────────────────┘
```

### KV-Aware Routing (Two-Stage Architecture)

```
┌─── Region A (Cloud) ─────────┐     ┌─── Region B (Edge) ─────────┐
│  Workers → KV Events          │     │  Engine → KV Events          │
│      ▼                        │     │      ▼                        │
│  Local KV Relay               │     │  Local KV Relay               │
│  (exact ownership + refcount) │     │  (exact ownership + refcount) │
│      ▼ CKF projection         │     │      ▼ CKF projection         │
│  ┌──────── Gossip Bus ──────────────────────────────────┐         │
│  │  CKF Publish: barrier snapshot + sequenced delta     │         │
│  └──────────────────────────────────────────────────────┘         │
│      ▼                        │     │      ▼                        │
│  Global CKF Consumer          │◄───►│  Global CKF Consumer          │
│  (transposed, multi-lane concurrent)│  (transposed, multi-lane concurrent)│
│      ▼                        │     │      ▼                        │
│  KV Aware Router              │     │  KV Aware Router              │
└───────────────────────────────┘     └───────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.75+ (1.85+ recommended)
- Cargo

### Build

```bash
cargo build --release
```

### Run Tests

```bash
# All tests (430)
cargo test --workspace

# KV estimation crate only (includes the zero-allocation test)
cargo test -p hier-kv-gateway-kv-estimate

# Integration tests only
cargo test -p hier-kv-gateway-integration

# End-to-end tests only
cargo test --test end_to_end -p hier-kv-gateway-integration -- --nocapture
```

### Start the Gateway

```bash
# Using the example configuration
cargo run --release -- --config examples/hier-kv-gateway.toml
```

Example configuration ([examples/hier-kv-gateway.toml](examples/hier-kv-gateway.toml)):

```toml
instance_id = "gateway-1"

[region]
id = "cloud-cn-beijing"
tier = "cloud"
geo = { lat = 39.9, lon = 116.4 }

[listen]
addr = "0.0.0.0"
port = 8080

[routing]
strategy = "hybrid"
kv_block_size = 16
overlap_score_credit = 1.0
prefill_load_scale = 1.0
temperature = 0.0
session_affinity_ttl_secs = 300
max_retries = 3

[routing.weights]
kv = 0.35
load = 0.30
topology = 0.20

[cluster]
bind_addr = "0.0.0.0:7946"
seed_peers = []
gossip_interval_ms = 1000
probe_timeout_ms = 5000
suspect_timeout_secs = 10

[[backends]]
type = "vllm_engine"
endpoint = { url = "http://localhost:8000", protocol = "http" }
models = ["qwen2.5-7b"]
region = "cloud-cn-beijing"
kv_block_size = 16
```

### Send a Request

```bash
# Chat Completions (fully compatible with the OpenAI API)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'

# Inspect routing decision info (response headers)
curl -v http://localhost:8080/v1/chat/completions ...
# X-Hier-KV-Gateway-Backend: cloud-beijing-worker-3
# X-Hier-KV-Gateway-Strategy: hybrid
# X-Hier-KV-Gateway-KV-Overlap: 8
```

## Routing Strategies

### 1. KV-Aware Routing (KV Aware)

Cost function:

```
total_overlap = device_overlap(local RadixTree) + ckf_overlap(cross-domain CKF)
prefill_blocks = max(len(hashes) - total_overlap, 0)
decode_blocks = backend.active_decode_blocks
cost = prefill_load_scale * prefill_blocks + decode_blocks
score = 1.0 / (1.0 + cost)
```

**Ownership refcount four branches**:
- First owner of a hash → insert a CKF fingerprint
- Another owner of the same hash → refcount++ only
- One of several removes → refcount-- only
- Final owner removes → delete the CKF fingerprint

### 2. Model-Aware Routing (Model Aware)

Acts as a **hard filter**, excluding non-matching backends:
- exact match (model_name + version + quant): score = 1.0
- model_match (same name, different version): score = 0.7
- compatible_match (same architecture): score = 0.3
- no_match: excluded
- Additional checks: max_context_len, tool_calling support

### 3. Load-Aware Routing (Load Aware)

```
load_cost = w_req * active_requests
          + w_queue * queue_depth
          + w_lat * (p99_latency / 100)
          + w_gpu * gpu_utilization
          + w_kv * kv_cache_usage
score = 1.0 / (1.0 + load_cost)
```

### 4. Topology-Aware Routing (Topology Aware)

```
rtt = latency_matrix.rtt_ms(self_region, backend.region)
network_cost = w_rtt * rtt + w_bw * bandwidth_penalty
score = 1.0 / (1.0 + network_cost / 100.0)
```

Latency matrix sources: configuration + active probing + Gossip propagation + geographic-distance estimation.

### 5. Hybrid Intelligent Routing (Hybrid, default)

```
1. Model Aware filters the candidate set
2. For each available sub-strategy S, compute scores_S
3. Dynamic weight adjustment:
   - KV unavailable → weight_kv = 0
   - Load stale (>10s) → weight_load *= 0.3
   - Normalize so the sum = 1.0
4. hybrid_score(b) = Σ(weight_S * normalize(score_S[b]))
5. temperature > 0: softmax sampling; temperature = 0: greedy
```

**Default weights**: KV=0.35, Load=0.30, Topology=0.20

### 6. KV-Capacity-Aware Routing (KV Capacity, optional plugin)

When `[kv_estimate] enabled = true`, this attaches as a `RoutingPlugin` to Hybrid. It first estimates the KV Cache memory footprint of the request, then scores each backend by its **remaining capacity**, excluding backends that cannot fit the request — this is the load-shedding / admission-control decision. It complements `KvAwareStrategy` (which scores by prefix-overlap hit to reduce prefill work): one decides "how much prefill to skip," the other decides "whether it fits at all."

```
per_token = f(num_layers, num_kv_heads, head_dim, dtype, attention family)  // MLA uses kv_lora_rank+qk_rope_head_dim, no factor 2
seq_len   = input_tokens + max_tokens                                       // output uses max_tokens as a conservative upper bound
effective = min(seq_len, sliding_window)                                    // sliding-window truncation
blocks    = ceil(effective / block_size) * batch_size
bytes     = per_token * batch_size * (blocks * block_size)                  // block-padded

available_bytes =
  (kv_total_blocks - kv_used_blocks) * per_block_bytes                 // KV-block path (exact, preferred)
  or (gpu_memory_total_mb - gpu_memory_used_mb) * 1e6 * safety_frac    // GPU-memory fallback (conservative)

if available_bytes <= 0 or bytes > available_bytes: exclude (raw_cost=∞, score=0)   // load shedding
else: ratio = bytes/available_bytes; raw_cost=ratio; score = 1/(1+ratio)
```

**Key design decisions**: output uses `max_tokens` as a conservative upper bound (never underestimates); exclusion uses `f64::INFINITY` (recognized by `normalize_costs` via `!is_finite()`); the GPU fallback only claims "free memory × `gpu_mem_safety_fraction`" (KV is not the sole GPU-memory consumer); unknown specs default to neutral (`exclude_on_unknown_spec=false`) and defer to other sub-strategies.

For full formula derivations and an end-to-end worked example, see [KV Memory Estimation Architecture](docs/en/05-kv-estimation.md) and [Routing Algorithms §9](docs/en/02-routing-algorithms.md).

### Service Degradation Chain

```
Hybrid (KV+Load+Topo)
  │ If KV Index is unavailable
  ▼
Model + Load + Topo
  │ If Load Stats are unavailable
  ▼
Model + Topo
  │ If cross-cluster communication is broken
  ▼
Local Load Aware
  │ If no local Backend is available
  ▼
Return 503
```

## KV Memory Estimation

A standalone leaf crate, [hier-kv-gateway-kv-estimate](crates/hier-kv-gateway-kv-estimate), estimates a single inference's KV Cache memory footprint using **analytical formulas** (not simulation), aligned with the KV-size computation in vLLM / SGLang / Mooncake / llm-d. `KvCapacityStrategy` uses this for capacity-aware routing (section 6 above).

### Features

- **Analytical formulas**: Direct integer multiply-add over model architecture parameters (layers / KV-head count / head dim / dtype / attention family) plus request shape (batch / input length / output length / block size) to yield bytes.
- **Zero-allocation hot path**: `ModelSpec` is `Copy`, the model name is kept as the catalog key, and `contains_ascii_ci` does in-place lowercase matching — no `String`/`HashMap` value clones on the hot path. A counting-allocator test asserts 0 bytes allocated (see `tests/alloc_free.rs`).
- **Nanosecond-scale**: full hot path `registry.estimate` is 45–91 ns; the formula itself is ~10 ns (independent of input length).
- **Pluggable extension (two paths)**:
  - **Add a model spec (data)**: one `[[kv_estimate.models]]` TOML line whose fields map 1:1 to HuggingFace `config.json`; the Standard formula covers it. This is the path for the vast majority of new models.
  - **Add a custom estimator (code)**: implement the `KvEstimator` trait and register via `with_estimator` / `with_estimator_front`. Used for architectures the standard formula cannot express (extra Cross-Attention caches, Mamba/SSM state).
- **Builtin mainstream models**: Llama-2/3, Qwen2/2.5, Mistral/Mixtral, Gemma-2, DeepSeek-V2/V3/R1 (MLA), ChatGLM3 — covering all four families MHA / GQA / MQA / MLA / sliding window.

### Analytical formulas

```
Standard (MHA/GQA/MQA): per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
MLA (DeepSeek):         per_token = layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes   // no factor 2
Sliding window:         effective = min(seq_len, sliding_window)
Block paging:           blocks = ceil(effective / block_size) * batch
                        bytes  = per_token * batch * (blocks * block_size)
```

### Configuration

```toml
[kv_estimate]
enabled = true                       # when off, the plugin is not attached (default false)
weight = 0.20                        # Hybrid weight
gpu_mem_safety_fraction = 0.5        # claimable fraction for the GPU-memory fallback
exclude_on_unknown_spec = false      # unknown spec: false=neutral, true=exclude

# Optional: register a private model spec (fields map 1:1 to HuggingFace config.json)
[[kv_estimate.models]]
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

### Benchmark

| Scenario | Latency |
|------|------|
| `estimate_kv` (the formula itself) | ~10 ns |
| `registry.estimate` (full hot path, name → spec → formula) | 45–91 ns |
| `KvCapacityStrategy::evaluate` (10 backends) | 2.57 µs |
| Hybrid end-to-end overhead (10 backends, with vs without the plugin) | ~3.85 µs (~42%) |

The full benchmark report is at [docs/benchmarks/kv-estimation.md](docs/benchmarks/kv-estimation.md), with anti-cheat assertions (hand-computed expected values, `raw_cost.is_finite()`, `score ∈ (0,1]`, zero-allocation zero-tolerance).

## Metadata Caching

```
L1: Request-Local   - block hashes, overlap scores (per request)
L2: Hot Metadata    - RadixTree, LoadStats, CKF Consumer (resident in memory)
L3: Warm Metadata   - ModelRegistry(TTL 60s), Topology(TTL 30s), Discovery(TTL 15s)
L4: Cold Metadata   - RoutingHistory(TTL 300s), DegradationStats(TTL 60s)
```

Concurrency-safety design:
- RadixTree: dedicated background thread + mpsc channel, avoiding lock contention
- LoadStats: `DashMap<BackendId, ArcSwap<Metrics>>`, lock-free reads
- CKF Consumer: bucket-level atomic u64, no lane-wide lock

## Gossip Cluster Communication

| Message type | Purpose |
|---------|------|
| PING/PONG | Heartbeat, carries a metadata digest |
| MEET | A new node joins the cluster |
| SYNC | Request a full state sync |
| CKF_PUBLISH | Cross-Region KV projection publication (barrier + delta) |
| METRIC_BROADCAST | Load/latency metric broadcast |

Behavior:
- Every second, randomly select P instances and send PING; PONG carries a metadata digest
- PING timeout → suspect → confirmed dead
- Metadata is tagged with a version vector; items that are behind request SYNC

## Plugin Mechanism

### Routing Strategy Plugin

```rust
#[async_trait]
pub trait RoutingStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    async fn evaluate(&self, ctx: &RoutingContext, candidates: &[BackendId],
                      meta: &MetadataStore) -> Result<Vec<ScoredBackend>>;
    fn is_available(&self, meta: &MetadataStore) -> bool;
    fn weight(&self) -> f64;
}
```

### Backend Connector Plugin

```rust
#[async_trait]
pub trait BackendConnector: Send + Sync {
    fn backend_type(&self) -> BackendType;
    async fn discover(&self) -> Result<Vec<BackendInfo>>;
    async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus>;
    async fn forward(&self, backend: &BackendId, request: &InferenceRequest)
        -> Result<BoxStream<'static, InferenceChunk>>;
    fn supports_kv_events(&self) -> bool;
    async fn subscribe_kv_events(&self, backend: &BackendId)
        -> Result<BoxStream<'static, KvCacheEvent>>;
    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics>;
}
```

### Cluster Communication Plugin

```rust
#[async_trait]
pub trait ClusterTransport: Send + Sync {
    async fn start(&self, self_id: &InstanceId, region: &RegionId, addr: &str) -> Result<()>;
    async fn broadcast(&self, msg: &ClusterMessage) -> Result<()>;
    async fn send(&self, target: &str, msg: &ClusterMessage) -> Result<()>;
    fn messages(&self) -> mpsc::Receiver<ClusterMessage>;
    fn members(&self) -> Vec<ClusterMember>;
}
```

## API Endpoints

| Method | Path | Description |
|------|------|------|
| POST | `/v1/chat/completions` | OpenAI-compatible Chat Completions |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |
| GET | `/admin/backends` | Backend list and status |
| GET | `/admin/backends/:id/metrics` | Backend metrics |

Response headers carry routing information: `X-Hier-KV-Gateway-Backend`, `X-Hier-KV-Gateway-Strategy`, `X-Hier-KV-Gateway-KV-Overlap`

## Project Structure

```
hier-kv-gateway/
├── crates/
│   ├── hier-kv-gateway-core/           # core types: IDs, BackendInfo, Metrics, Config
│   ├── hier-kv-gateway-metadata/       # metadata: RadixTree, CKF, LoadStats, ModelRegistry
│   ├── hier-kv-gateway-routing/        # routing engine: 5 strategies + Hybrid + KV-capacity plugin
│   ├── hier-kv-gateway-kv-estimate/    # KV memory estimation (standalone leaf crate, analytical formulas + plugins)
│   ├── hier-kv-gateway-cluster/        # Gossip cluster communication + CKF Relay
│   ├── hier-kv-gateway-connector/      # backend connectors: OpenAI-compatible/vLLM
│   ├── hier-kv-gateway-api/            # HTTP API Server (OpenAI compatible)
│   └── hier-kv-gateway/                # main binary
├── tests/
│   └── hier-kv-gateway-integration/    # integration tests (no mocks)
├── docs/
│   ├── 01-architecture.md              # architecture design document
│   ├── 02-routing-algorithms.md        # routing algorithm design document (incl. KV-capacity §9)
│   ├── 03-interfaces-data-models.md    # interface and data model document
│   ├── 05-kv-estimation.md             # KV memory estimation module architecture
│   ├── benchmarks/                     # benchmark reports
│   └── session-logs/                   # development session logs
└── examples/
    ├── hier-kv-gateway.toml             # single-backend example config
    └── multi-backend.toml               # multi-backend example (with [kv_estimate] section)
```

## Tests

All **430 tests pass**, with no mocks and no cheating:

| Test suite | Count | Description |
|---------|------|------|
| hier-kv-gateway-core | 77 | Type serialization, config parsing, geographic distance computation |
| hier-kv-gateway-metadata | 30 | RadixTree event handling, CKF insert/delete/lookup |
| hier-kv-gateway-routing | 78 | Hybrid strategy scoring, softmax sampling, KV-capacity-aware routing |
| hier-kv-gateway-kv-estimate | 77 | KV analytical formulas (MHA/GQA/MLA/sliding window), catalog matching, zero-allocation hot path |
| hier-kv-gateway-cluster | 45 | Gossip member management, CKF Relay publication |
| hier-kv-gateway-connector | 32 | Prometheus metric parsing, connector registration |
| hier-kv-gateway-api | 41 | HTTP routing, SSE conversion, response headers |
| hier-kv-gateway (main binary) | 13 | Routing-engine construction, KV-capacity plugin attachment |
| **Integration tests** | **36** | **Real-component end-to-end** |
| **Total** | **430** | |

Integration tests include:
- **radix_tree_kv_events**: KV event lifecycle of a real RadixTree
- **ckf_producer_consumer**: cross-domain projection by a real CKF Producer/Consumer
- **hybrid_routing**: scoring and path selection by the real hybrid routing strategy
- **gossip_cluster**: real Gossip member management
- **end_to_end**: start a real axum HTTP backend → discover → route → forward full chain

## Design Documents

- [Architecture Design](docs/en/01-architecture.md)
- [Routing Algorithm Design](docs/en/02-routing-algorithms.md) (incl. KV-capacity-aware routing §9)
- [Interface and Data Model Design](docs/en/03-interfaces-data-models.md)
- [KV Memory Estimation Module Architecture](docs/en/05-kv-estimation.md)
- [KV Estimation Benchmark Report](docs/benchmarks/kv-estimation.md)
- [Development Session Logs](docs/session-logs/) (incl. the [KV estimation session log](docs/session-logs/2026-08-06-kv-estimation.md))

## Tech Stack

- **Language**: Rust (performance, memory safety)
- **Async runtime**: tokio
- **HTTP framework**: axum
- **HTTP client**: reqwest
- **Serialization**: serde / serde_json / toml
- **Hashing**: xxhash-rust (xxh3)
- **Concurrency**: dashmap, arc-swap, parking_lot

## License

Apache-2.0
