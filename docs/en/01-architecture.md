# Hier KV Gateway Architecture Design Document

> English | [中文](../01-architecture.md)

> An automatic scheduling gateway system for LLM requests across cloud-edge-device collaboration

## 1. Background and Goals

### 1.1 Problem Domain

In cloud-edge-device collaborative LLM inference scenarios, inference resources are distributed across:

- **Cloud side (Cloud)**: Full clusters with K8s + distributed inference systems (e.g., llm-d), multi-node multi-GPU, KV Cache can be shared across nodes.
- **Edge side (Edge)**: Resource-constrained clusters or single-node multi-GPU, may have lightweight scheduling.
- **Device side (Device)**: Single-process inference engines (vLLM / llama.cpp), no cluster scheduling.

These backends differ greatly in **geographic location, latency, capacity, KV Cache state, and model version**. Clients need a unified entry point that automatically routes requests to the most appropriate backend.

### 1.2 Design Goals

1. **Cross-cluster/process KV-aware routing**: Sense the KV Cache prefix overlap of each backend and route requests to the backend with the highest hit rate.
2. **Cross-cluster model-information-aware routing**: Route based on the loaded model, version, and quantization of backends.
3. **Routing based on per-instance latency and load statistics**: Collect backend load metrics in real time for load balancing.
4. **Routing based on geographic topology**: Preferentially route to nearby backends based on the network latency topology.
5. **Hybrid intelligent routing**: The default strategy that fuses the four aspects above.
6. **Distributed system**: Gateway instance groups across clusters communicate via Gossip; multiple instances within a cluster provide high availability.
7. **Single-process high availability and service degradation**: Fall back to basic load balancing when prediction is inaccurate.
8. **Pluggability**: Strategies, backend connectors, and inter-instance communication are all extensible.

## 2. Top-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client (OpenAI API)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP/gRPC
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
│  Gateway instance  │◄─────►│  Gateway instance  │
│  (Cloud)           │       │  (Edge)            │
│  ┌───────────────┐ │       │ ┌────────────────┐ │
│  │  LLM-D Cluster│ │       │ │  vLLM Engine    │ │
│  │  (K8s + multi)│ │       │ │  (single proc)  │ │
│  └───────────────┘ │       │ └────────────────┘ │
└───────────────────┘       └────────────────────┘
```

## 3. Core Concepts

### 3.1 Region

A Region corresponds to a logical deployment domain (similar to a data center, DC):

```
RegionId = a stable string identifier (e.g., "cloud-cn-beijing", "edge-shanghai", "device-rpi-01")
```

- Each Region has one or more Gateway instances (multiple instances on the cloud side for HA, a single instance on the device side).
- RegionId remains stable across Gateway restarts.
- A Region has topology attributes: geographic coordinates, network latency matrix, tier (cloud/edge/device).

### 3.2 Backend

An inference service instance within a Region; the target of routing:

```
BackendId = (RegionId, instance identifier)
```

Backend types:
- **Cluster Backend**: Connects to distributed inference systems such as llm-d; multiple workers; supports KV events.
- **Engine Backend**: Connects to single-process engines such as vLLM / llama.cpp; may or may not support KV events.

### 3.3 Indexer Domain

Identifies a set of caches that can be compared as a single logical routing namespace:

```
IndexerDomainId = (hash of model architecture + tokenizer + KV block size + quantization config)
```

Backends under the same IndexerDomain can have their KV Cache compared with one another. Backends with different models/configurations belong to different domains.

### 3.4 Pool

A Pool is defined as `PoolId = (IndexerDomainId, RegionId)`:

```
PoolId = (IndexerDomainId, RegionId)
```

A Pool corresponds to a set of Backends within a Region that share the same IndexerDomain. KV-aware routing performs cross-Region comparisons at the Pool granularity.

## 4. Layered Architecture

### 4.1 Layered Structure Inside a Gateway Process

```
┌─────────────────────────────────────────────────────┐
│  API Layer (HTTP Server, OpenAI compatible)         │
├─────────────────────────────────────────────────────┤
│  Routing Layer                                       │
│   ├── Strategy: KV Aware                             │
│   ├── Strategy: Model Aware                         │
│   ├── Strategy: Load Aware                          │
│   ├── Strategy: Topology Aware                      │
│   ├── Plugin: KV Capacity (optional, [kv_estimate]) │
│   └── Strategy: Hybrid (default)                    │
├─────────────────────────────────────────────────────┤
│  Metadata Layer                                      │
│   ├── KV Index (RadixTree + CKF projection)         │
│   ├── Model Registry (model/version/capabilities)   │
│   ├── Load Stats (latency/queue/capacity)           │
│   ├── Topology Graph (latency matrix between Regions)│
│   └── Routing History (session affinity / degradation stats) │
├─────────────────────────────────────────────────────┤
│  Cluster Layer (Gossip)                             │
│   ├── Member Discovery (SWIM-like)                  │
│   ├── State Sync (metadata broadcast/sync)          │
│   └── CKF Relay (cross-Region KV projection publication) │
├─────────────────────────────────────────────────────┤
│  Connector Layer (plugins)                          │
│   ├── LLM-D Connector (NATS/HTTP KV event)          │
│   ├── vLLM Connector (ZMQ/HTTP KV event)            │
│   ├── llama.cpp Connector (no KV event, degraded)   │
│   └── Generic OpenAI Connector (no KV, degraded)    │
└─────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

```
Client request
  │
  ▼
[API Layer] Parse request → extract token_ids / model / parameters
  │
  ▼
[Routing Layer] Hybrid strategy evaluation
  │  1. Compute block hashes
  │  2. Query local KV Index → device overlap
  │  3. Query cross-Region CKF → remote KV overlap
  │  4. Query Model Registry → candidate Backend set
  │  5. Query Load Stats → per-Backend cost
  │  6. Query Topology → network latency penalty
  │  7. Aggregate score → pick the best Backend
  ▼
[Connector Layer] Forward the request to the selected Backend
  │  ├── Success → stream the response back → update Load Stats / KV Index
  │  └── Failure → retry/degrade → pick the next-best Backend
  ▼
Response returned to the client
```

## 5. Distributed Architecture

### 5.1 Gossip Protocol

Gateway instance groups across clusters communicate via Gossip:

**Message types**:
- `PING / PONG`: Heartbeat, carries the sender's metadata digest.
- `MEET`: A new node joins the cluster.
- `SYNC`: Request a full state sync (new node or repair).
- `CKF_PUBLISH`: Cross-Region KV projection publication (barrier snapshot + sequenced delta).
- `METRIC_BROADCAST`: Load/latency metric broadcast.

**Gossip behavior**:
- Each Gateway instance maintains a member list (Region → instance address + heartbeat time + metadata version).
- Every second, randomly select P instances and send PING; PONG carries the latest metadata digest.
- If PING times out (N consecutive failures), mark the instance as suspect → confirmed down.
- New instances join via `MEET`; the instance that receives MEET adds it to the member list and propagates it in subsequent Gossip rounds.

**Metadata synchronization**:
- Metadata is tagged with a version number (version vector); PONG carries a digest (Region → version).
- The receiver compares with its local version and requests `SYNC` for items that are behind.
- Large state (such as CKF projections) uses barrier snapshot + sequenced delta and is not placed in PING.

### 5.2 Intra-Cluster High Availability

Multiple Gateway instances can be deployed within a Region on the cloud side:

- **Leader election**: Based on Raft (a leader is elected from the Gossip members for coordination) or simple primary-backup (based on etcd lease).
- **Stateless routing**: Routing decisions are stateless (based on current metadata); any instance can route independently. The Leader only coordinates metadata versions and avoids duplicate CKF publications.
- **Request retry**: Clients can retry to another Gateway instance.
- **Session affinity**: Routing history is shared via Gossip (with TTL) to provide cross-instance session affinity.

### 5.3 Single-Process High Availability on the Device Side

The device side has only one Gateway process, which uses:

- **Health self-check**: Periodic self-checks; if memory/connection anomalies occur, restart.
- **Degradation mode**: If the KV Index is unavailable → fall back to Load Aware; if Load Stats are unavailable → fall back to Topology Aware (nearest); if all are unavailable → Round Robin.
- **Local persistence**: Key metadata (member list, topology) is written to a local file and restored after restart.

### 5.4 Service Degradation Mechanism

```
Normal: Hybrid strategy (KV + Model + Load + Topology aggregate score)
  │ If KV Index is unavailable
  ▼
Degradation 1: Model + Load + Topology strategy
  │ If Load Stats are unavailable
  ▼
Degradation 2: Model + Topology strategy
  │ If cross-cluster communication is broken (only the local Region)
  ▼
Degradation 3: Load Aware within the local Region
  │ If no Backend is available locally
  ▼
Degradation 4: Return 503 + cached list of last healthy backends
```

## 6. KV-Aware Routing (Core)

### 6.1 Two-Stage Architecture

```
┌─── Region A (Cloud) ─────────┐     ┌─── Region B (Edge) ─────────┐
│  Workers → KV Events          │     │  Engine → KV Events          │
│      │                        │     │      │                        │
│      ▼                        │     │      ▼                        │
│  Local KV Relay               │     │  Local KV Relay               │
│  (exact ownership + refcount) │     │  (exact ownership + refcount) │
│      │                        │     │      │                        │
│      ▼ CKF projection         │     │      ▼ CKF projection         │
│  ┌──────── Gossip Bus ──────────────────────────────────┐         │
│  │  CKF Publish: barrier snapshot + sequenced delta     │         │
│  └──────────────────────────────────────────────────────┘         │
│      │                        │     │      │                        │
│      ▼                        │     │      ▼                        │
│  Global CKF Consumer          │◄───►│  Global CKF Consumer          │
│  (transposed, multi-lane concurrent)│  (transposed, multi-lane concurrent)│
│      │                        │     │      │                        │
│      ▼                        │     │      ▼                        │
│  KV Aware Router              │     │  KV Aware Router              │
└───────────────────────────────┘     └───────────────────────────────┘
```

### 6.2 Stage 1: Backend KV Events → Local KV Relay

The Gateway instance (or Relay-role instance) within each Region:

1. **Discovers backends**: Discovers this Region's Backends via the Connector plugin.
2. **Consumes KV Events**: Backends report KV Cache events (block stored / removed) through the connector.
3. **Maintains exact state**:
   - `full_hash → Set<(backend_id, dp_rank)>`: which backends/ranks own each full block hash.
   - `full_hash → refcount`: DC/Region-wide reference count.
4. **Ownership change handling**:
   - First owner of a full hash → insert a CKF fingerprint
   - Another owner of same hash → refcount++ only
   - One of several removes → refcount-- only
   - Final owner removes → delete the CKF fingerprint

### 6.3 Stage 2: CKF Projection → Global Consumer

Each Region's Relay publishes its local CKF projection to the cross-Region Gossip Bus:

- **Barrier Snapshot**: Full CKF state + terminal publication sequence.
- **Sequenced Delta**: Absolute image of the packed buckets changed since the last publication.
- **Lease**: Binds a consumer instance to a lane.

The Global Consumer runs in every Gateway instance:
- Maintains a transposed CKF layout (bucket-major, one atomic packed word per lane).
- Concurrent queries: a single prefix query searches all Region lanes simultaneously.
- Failure recovery: when a lane disconnects, it is excluded; on reconnection a new barrier snapshot is installed.

### 6.4 Cuckoo Filter Design

CKF design highlights:

- **Fingerprint**: A short fingerprint of the block hash (e.g., 16 bits), lossy.
- **Bucket**: Each bucket holds multiple fingerprints (e.g., 4 × 16 bits = 64-bit packed word).
- **Transposed Layout**: On the consumer side, organized by bucket; each bucket is an atomic u64 across lanes, supporting concurrent reads.
- **Capacity**: Supports up to 16 Region lanes for concurrent queries.
- **False positives**: CKF may return false positives (a Region appears to have a block that it actually does not); this is corrected by subsequent exact queries / request results.

## 7. Language Choice: Rust

**Rust** was chosen as the implementation language for the following reasons:

1. **Performance**: The Gateway is on the critical path of the data plane and is latency-sensitive. Rust's zero-cost abstractions + no GC make it suitable for high concurrency and low latency.
2. **Ecosystem consistency**: The core language is consistent with similar inference infrastructure projects (such as llm-d), making it easy to reuse data structures and algorithmic ideas.
3. **Memory safety**: Routing metadata (KV index, CKF, load stats) is accessed concurrently; Rust's ownership model guarantees thread safety.
4. **Ecosystem**: tokio (async runtime), axum (HTTP), serde (serialization), dashmap (concurrent map), etc., are mature.

## 8. Plugin and Interface Mechanism

### 8.1 Strategy Plugin

```rust
pub trait RoutingStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;
    
    /// Evaluate the candidate Backend list and return a sorted result with scores
    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>>;
    
    /// Whether this strategy is available (for degradation decisions)
    fn is_available(&self, meta: &MetadataStore) -> bool;
}
```

Built-in strategies implement this trait; users can register custom strategies.

### 8.2 Backend Connector Plugin

```rust
#[async_trait]
pub trait BackendConnector: Send + Sync {
    /// Connector type name
    fn backend_type(&self) -> &str;
    
    /// Discover backend instances of this type
    async fn discover(&self) -> Result<Vec<BackendInfo>>;
    
    /// Health check
    async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus>;
    
    /// Forward an inference request (streaming)
    async fn forward(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<BoxStream<'static, InferenceChunk>>;
    
    /// Whether KV Cache events are supported
    fn supports_kv_events(&self) -> bool;
    
    /// Subscribe to the KV Cache event stream (if supported)
    async fn subscribe_kv_events(
        &self,
        backend: &BackendId,
    ) -> Result<BoxStream<'static, KvCacheEvent>>;
    
    /// Collect load metrics
    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics>;
}
```

### 8.3 Inter-Instance Communication Plugin

```rust
#[async_trait]
pub trait ClusterTransport: Send + Sync {
    /// Start communication
    async fn start(&self, self_id: &InstanceId) -> Result<()>;
    
    /// Broadcast a message
    async fn broadcast(&self, msg: &ClusterMessage) -> Result<()>;
    
    /// Send to a specific instance
    async fn send(&self, target: &InstanceId, msg: &ClusterMessage) -> Result<()>;
    
    /// Receive a message stream
    fn messages(&self) -> BoxStream<'static, ClusterMessage>;
}
```

A Gossip implementation is provided by default and can be replaced with others (such as NATS, gRPC mesh).

## 9. Metadata Cache Mechanism

### 9.1 In-Memory Cache Tiers

```
┌─────────────────────────────────────────────┐
│ L1: Request-Local Cache (per request)       │
│   - block hashes for this request           │
│   - overlap scores for this request         │
│   Lifecycle: single request                 │
├─────────────────────────────────────────────┤
│ L2: Hot Metadata Cache (hot data)           │
│   - RadixTree (local KV prefix tree)        │
│   - Load Stats (sliding window, TTL 5s)     │
│   - CKF Consumer (cross-Region projection)  │
│   Lifecycle: resident in memory, real-time  │
│              updates                        │
├─────────────────────────────────────────────┤
│ L3: Warm Metadata Cache (warm data)         │
│   - Model Registry (model info, TTL 60s)    │
│   - Topology Graph (latency matrix, TTL 30s)│
│   - Backend Discovery (backend list, TTL 15s)│
│   Lifecycle: periodic refresh               │
├─────────────────────────────────────────────┤
│ L4: Cold Metadata Store (cold data)         │
│   - Routing History (session affinity, TTL 300s) │
│   - Degradation Stats (TTL 60s)             │
│   Lifecycle: on-demand query + periodic     │
│              cleanup                        │
└─────────────────────────────────────────────┘
```

### 9.2 Concurrency Safety

- RadixTree: dedicated background thread + mpsc channel, avoiding lock contention.
- Load Stats: `DashMap<BackendId, ArcSwap<Metrics>>`, lock-free reads, CAS writes.
- CKF Consumer: bucket-level atomic u64, no lane-wide lock.
- Model Registry: `Arc<RwLock<...>>`, read-heavy write-light.

## 10. Failure Recovery

Failure recovery boundaries are designed around the "narrowest state boundary" principle:

| Failure | Recovery boundary | Behavior |
|------|---------|------|
| Backend event gap | That backend's rank state | Recover from the backend's event history, or install the current tree state |
| Backend replacement | All state of that backend | Rebuild from the new source after the completion barrier |
| CKF delivery gap | The affected consumer lane | Retire that lane; install a new barrier snapshot on reconnection |
| Gateway instance crash | That instance's local state | Other instances detect via Gossip and take over routing; the new instance SYNCs full state |
| Region isolation | That Region's lane | Routing excludes that Region; the lane is reactivated after recovery |

## 11. Directory Structure Plan

```
hier-kv-gateway/
├── Cargo.toml                 # workspace root
├── crates/
│   ├── hier-kv-gateway-core/           # core types: BackendId, RegionId, metadata models
│   ├── hier-kv-gateway-metadata/       # metadata store: RadixTree, CKF, LoadStats, ModelRegistry
│   ├── hier-kv-gateway-routing/        # routing engine: 5 strategies + Hybrid
│   ├── hier-kv-gateway-cluster/         # Gossip cluster communication + CKF Relay
│   ├── hier-kv-gateway-connector/       # backend connector trait + built-in implementations
│   ├── hier-kv-gateway-kv-estimate/     # KV memory estimation (standalone leaf crate, analytical formulas + plugins)
│   ├── hier-kv-gateway-api/             # HTTP API server (OpenAI compatible)
│   └── hier-kv-gateway/                # main binary: assembles all components
├── tests/                     # integration tests (real backends, no mocks)
├── docs/                      # design documents
└── examples/                  # configuration examples
```
