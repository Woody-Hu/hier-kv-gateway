> English | [中文](README.md)

# Hier KV Gateway

> An automatic scheduling gateway system for LLM requests across cloud-edge-device collaboration

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-139%20passed-brightgreen.svg)](#tests)

## Overview

Hier KV Gateway is an automatic scheduling gateway for LLM inference requests in **cloud-edge-device collaborative** scenarios. It unifies inference backends distributed across different geographic locations, scales, and engines (cloud-side K8s clusters, edge-side lightweight clusters, device-side vLLM/llama.cpp single processes) into a single inference service entry point, and automatically routes requests to the most appropriate backend through intelligent routing strategies.

### Core Capabilities

| Capability | Description |
|------|------|
| **KV-aware routing** | Sense the KV Cache prefix overlap of each backend and route to the backend with the highest cache hit rate, reducing prefill computation |
| **Model-aware routing** | Route based on exact/compatible matching of the loaded model, version, and quantization of backends |
| **Load-aware routing** | Collect backend metrics such as queue depth, GPU utilization, and KV usage in real time for load balancing |
| **Topology-aware routing** | Preferentially route to nearby backends based on the network latency topology, reducing end-to-end latency |
| **Hybrid intelligent routing** | The default strategy, fusing the four aspects above via weighted scoring for an aggregate decision |
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
# All tests (139)
cargo test --workspace

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
│   ├── hier-kv-gateway-routing/        # routing engine: 5 strategies + Hybrid
│   ├── hier-kv-gateway-cluster/        # Gossip cluster communication + CKF Relay
│   ├── hier-kv-gateway-connector/      # backend connectors: OpenAI-compatible/vLLM
│   ├── hier-kv-gateway-api/            # HTTP API Server (OpenAI compatible)
│   └── hier-kv-gateway/                # main binary
├── tests/
│   └── hier-kv-gateway-integration/    # integration tests (no mocks)
├── docs/
│   ├── 01-architecture.md      # architecture design document
│   ├── 02-routing-algorithms.md # routing algorithm design document
│   └── 03-interfaces-data-models.md # interface and data model document
└── examples/
    └── hier-kv-gateway.toml             # example configuration
```

## Tests

All **139 tests pass**, with no mocks and no cheating:

| Test suite | Count | Description |
|---------|------|------|
| hier-kv-gateway-core | 48 | Type serialization, config parsing, geographic distance computation |
| hier-kv-gateway-metadata | 22 | RadixTree event handling, CKF insert/delete/lookup |
| hier-kv-gateway-routing | 3 | Hybrid strategy scoring, softmax sampling |
| hier-kv-gateway-cluster | 26 | Gossip member management, CKF Relay publication |
| hier-kv-gateway-connector | 6 | Prometheus metric parsing, connector registration |
| hier-kv-gateway-api | 20 | HTTP routing, SSE conversion, response headers |
| **Integration tests** | **14** | **Real-component end-to-end** |
| **Total** | **139** | |

Integration tests include:
- **radix_tree_kv_events**: KV event lifecycle of a real RadixTree
- **ckf_producer_consumer**: cross-domain projection by a real CKF Producer/Consumer
- **hybrid_routing**: scoring and path selection by the real hybrid routing strategy
- **gossip_cluster**: real Gossip member management
- **end_to_end**: start a real axum HTTP backend → discover → route → forward full chain

## Design Documents

- [Architecture Design](docs/en/01-architecture.md)
- [Routing Algorithm Design](docs/en/02-routing-algorithms.md)
- [Interface and Data Model Design](docs/en/03-interfaces-data-models.md)

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
