# Aether 接口与数据模型设计

> 详细的 Rust 数据结构、Trait 接口、对内对外 API 设计

## 1. 核心标识类型

```rust
// crates/aether-core/src/ids.rs

/// 区域标识，稳定字符串，跨重启不变
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct RegionId(pub Arc<str>);

/// 区域层级
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RegionTier {
    Cloud,
    Edge,
    Device,
}

/// 后端实例标识
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendId {
    pub region: RegionId,
    pub instance: BackendInstanceId,
}

/// 后端实例标识（region 内唯一）
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendInstanceId(pub Arc<str>);

/// 索引域标识（模型兼容性分组）
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct IndexerDomainId(pub u64);

/// 池标识 = (IndexerDomainId, RegionId)
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PoolId {
    pub domain: IndexerDomainId,
    pub region: RegionId,
}

/// Gateway 实例标识
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct InstanceId(pub Arc<str>);

/// 请求 ID
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct RequestId(pub Arc<str>);

/// 会话 ID（用于会话亲和）
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SessionId(pub Arc<str>);

/// Worker + DP Rank（集群后端内部并行维度）
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerWithRank {
    pub worker_id: u64,
    pub dp_rank: u32,
}
```

## 2. 元数据模型

### 2.1 后端信息

```rust
// crates/aether-core/src/backend.rs

/// 后端完整描述
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendInfo {
    pub id: BackendId,
    pub backend_type: BackendType,
    pub endpoint: Endpoint,
    pub models: Vec<ModelInstance>,
    pub region: RegionId,
    pub indexer_domain: IndexerDomainId,
    pub capabilities: BackendCapabilities,
    pub kv_config: Option<KvConfig>,
    pub status: BackendStatus,
}

/// 后端类型
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendType {
    DynamoCluster,
    LlmDCluster,
    VllmEngine,
    LlamaCppEngine,
    GenericOpenAI,
}

/// 网络端点
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Endpoint {
    pub url: String,
    pub protocol: Protocol,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Protocol {
    Http,
    Grpc,
    Nats,
}

/// 模型实例信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInstance {
    pub model_name: String,
    pub model_architecture: String,  // e.g. "Qwen2ForCausalLM"
    pub quantization: Quantization,
    pub max_context_len: u32,
    pub supports_tool_calling: bool,
    pub supports_streaming: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Quantization {
    Fp16,
    Bf16,
    Int8,
    Int4,
    Awq,
    Gptq,
}

/// 后端能力
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub supports_kv_events: bool,
    pub supports_gpu_utilization: bool,
    pub supports_batching: bool,
    pub max_batch_size: Option<u32>,
    pub gpu_count: u32,
    pub gpu_memory_gb: f64,
}

/// KV 配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvConfig {
    pub block_size: u32,           // KV block token 数
    pub cache_namespace: String,   // 缓存命名空间
    pub max_kv_blocks: u64,
}

/// 后端运行状态
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}
```

### 2.2 负载指标

```rust
// crates/aether-core/src/metrics.rs

/// 后端实时负载指标
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackendMetrics {
    pub active_requests: u32,
    pub queue_depth: u32,
    pub active_decode_blocks: u64,
    pub active_prefill_tokens: u64,
    pub kv_used_blocks: u64,
    pub kv_total_blocks: u64,
    pub gpu_utilization: f32,       // 0.0 - 1.0
    pub gpu_memory_used_mb: u64,
    pub gpu_memory_total_mb: u64,
    pub latency: LatencyStats,
    pub timestamp: u64,              // unix millis
}

/// 延迟统计（滑动窗口）
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub p999_ms: f64,
    pub sample_count: u32,
}

/// 派生指标（路由用）
impl BackendMetrics {
    pub fn kv_cache_usage(&self) -> f64 {
        if self.kv_total_blocks == 0 { return 0.0; }
        self.kv_used_blocks as f64 / self.kv_total_blocks as f64
    }
    
    pub fn gpu_memory_usage(&self) -> f64 {
        if self.gpu_memory_total_mb == 0 { return 0.0; }
        self.gpu_memory_used_mb as f64 / self.gpu_memory_total_mb as f64
    }
    
    pub fn available_capacity(&self) -> i64 {
        // 估算剩余可接受请求数
        let kv_avail = (self.kv_total_blocks - self.kv_used_blocks) as i64;
        let gpu_room = ((1.0 - self.gpu_utilization) * 100.0) as i64;
        kv_avail.min(gpu_room)
    }
}
```

### 2.3 KV Cache 事件

```rust
// crates/aether-core/src/kv_event.rs

/// KV Cache 事件（参考 Dynamo KvCacheEvent）
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KvCacheEvent {
    Stored {
        worker: WorkerWithRank,
        block_hashes: Vec<u64>,
        parent_hash: Option<u64>,
        num_block_tokens: Vec<u64>,
    },
    Removed {
        worker: WorkerWithRank,
        block_hashes: Vec<u64>,
    },
    Clear {
        worker: WorkerWithRank,
    },
    /// Worker 重置（generation fence）
    Reset {
        worker: WorkerWithRank,
        generation: u64,
    },
}

/// Block hash 计算输入
#[derive(Clone, Debug)]
pub struct BlockHashInput<'a> {
    pub token_ids: &'a [u32],
    pub block_size: u32,
    pub lora_name: Option<&'a str>,
    pub cache_namespace: Option<&'a str>,
}

/// 计算 block hashes
pub fn compute_block_hashes(input: &BlockHashInput) -> Vec<u64> {
    // 参考 Dynamo compute_block_hash_for_seq
    let mut hashes = Vec::new();
    for chunk in input.token_ids.chunks(input.block_size as usize) {
        let mut h = xxhash_rust::xxh3_64(chunk);
        if let Some(ns) = input.cache_namespace {
            h = xxhash_rust::xxh3_64_with_seed(
                &[h.to_le_bytes().as_slice(), ns.as_bytes()].concat(),
                0,
            );
        }
        if let Some(lora) = input.lora_name {
            h ^= xxhash_rust::xxh3_64(lora.as_bytes());
        }
        hashes.push(h);
    }
    hashes
}
```

### 2.4 拓扑信息

```rust
// crates/aether-core/src/topology.rs

/// 区域信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionInfo {
    pub id: RegionId,
    pub tier: RegionTier,
    pub geo: Option<GeoCoord>,
    pub network_zone: Option<String>,
    pub endpoints: Vec<String>,  // 该 region 的 gateway 地址
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GeoCoord {
    pub lat: f64,
    pub lon: f64,
}

/// 两区域间延迟估计
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LatencyEstimate {
    pub rtt_p50_ms: f64,
    pub rtt_p99_ms: f64,
    pub bandwidth_mbps: f64,
    pub last_updated_unix: u64,
}

/// 延迟矩阵
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LatencyMatrix {
    pub entries: HashMap<(RegionId, RegionId), LatencyEstimate>,
}

impl LatencyMatrix {
    /// 获取两区域间延迟，无数据时用地理距离估算
    pub fn rtt_ms(&self, a: &RegionId, b: &RegionId, regions: &HashMap<RegionId, RegionInfo>) -> f64 {
        if a == b { return 0.0; }
        if let Some(est) = self.entries.get(&(a.clone(), b.clone())) {
            return est.rtt_p50_ms;
        }
        // 地理距离估算
        if let (Some(ra), Some(rb)) = (regions.get(a), regions.get(b)) {
            if let (Some(ga), Some(gb)) = (ra.geo, rb.geo) {
                let dist_km = haversine_km(ga, gb);
                return dist_km / 200.0; // 光纤 ~200km/ms
            }
        }
        100.0 // 默认 100ms
    }
}
```

## 3. Trait 接口设计

### 3.1 路由策略接口

```rust
// crates/aether-routing/src/strategy.rs

/// 路由上下文（每次请求构建）
#[derive(Clone, Debug)]
pub struct RoutingContext {
    pub request_id: RequestId,
    pub session_id: Option<SessionId>,
    pub model_name: String,
    pub token_ids: Vec<u32>,
    pub block_hashes: Vec<u64>,
    pub block_size: u32,
    pub lora_name: Option<String>,
    pub cache_namespace: Option<String>,
    pub estimated_output_tokens: u32,
    pub requires_tool_calling: bool,
}

/// 带分数的后端
#[derive(Clone, Debug)]
pub struct ScoredBackend {
    pub backend_id: BackendId,
    pub score: f64,        // [0, 1], 1.0 = 最优
    pub raw_cost: f64,    // 策略原始成本
    pub meta_version: u64,
}

/// 路由策略 trait
#[async_trait]
pub trait RoutingStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    
    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>>;
    
    fn is_available(&self, meta: &MetadataStore) -> bool;
    
    /// 策略权重（Hybrid 用）
    fn weight(&self) -> f64;
}
```

### 3.2 元数据存储接口

```rust
// crates/aether-metadata/src/store.rs

/// 元数据存储（所有路由策略的统一数据源）
pub struct MetadataStore {
    kv_index: KvIndex,
    ckf_consumer: CkfConsumer,
    model_registry: ModelRegistry,
    load_stats: LoadStats,
    topology: TopologyGraph,
    routing_history: RoutingHistory,
}

impl MetadataStore {
    // === KV 相关 ===
    pub fn kv_find_local_overlap(&self, hashes: &[u64], backend: &BackendId) -> u32;
    pub fn kv_find_global_overlap(&self, hashes: &[u64], region: &RegionId) -> u32;
    pub fn kv_apply_event(&self, event: KvCacheEvent, backend: &BackendId);
    pub fn kv_confidence(&self) -> f64;  // CKF 置信度
    
    // === Model 相关 ===
    pub fn model_match_score(&self, backend: &BackendId, model: &str) -> f64;
    pub fn model_get_instances(&self, backend: &BackendId) -> &[ModelInstance];
    pub fn model_find_backends(&self, model: &str) -> Vec<BackendId>;
    
    // === Load 相关 ===
    pub fn load_get_metrics(&self, backend: &BackendId) -> Option<BackendMetrics>;
    pub fn load_update(&self, backend: &BackendId, metrics: BackendMetrics);
    pub fn load_freshness(&self, backend: &BackendId) -> Option<Duration>;
    
    // === Topology 相关 ===
    pub fn topo_rtt_ms(&self, from: &RegionId, to: &RegionId) -> f64;
    pub fn topo_get_region(&self, region: &RegionId) -> Option<&RegionInfo>;
    
    // === Session Affinity ===
    pub fn session_get(&self, session: &SessionId) -> Option<SessionAffinity>;
    pub fn session_set(&self, session: &SessionId, affinity: SessionAffinity);
    
    // === Backend Discovery ===
    pub fn backends_all(&self) -> Vec<&BackendInfo>;
    pub fn backends_by_region(&self, region: &RegionId) -> Vec<&BackendInfo>;
    pub fn backends_by_domain(&self, domain: &IndexerDomainId) -> Vec<&BackendInfo>;
}

/// 会话亲和记录
#[derive(Clone, Debug)]
pub struct SessionAffinity {
    pub backend: BackendId,
    pub last_used_unix: u64,
    pub kv_overlap_at_route: u32,
}
```

### 3.3 后端连接器接口

```rust
// crates/aether-connector/src/connector.rs

#[async_trait]
pub trait BackendConnector: Send + Sync {
    fn backend_type(&self) -> BackendType;
    
    async fn discover(&self) -> Result<Vec<BackendInfo>>;
    
    async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus>;
    
    async fn forward(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<BoxStream<'static, InferenceChunk>>;
    
    fn supports_kv_events(&self) -> bool;
    
    async fn subscribe_kv_events(
        &self,
        backend: &BackendId,
    ) -> Result<BoxStream<'static, KvCacheEvent>>;
    
    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics>;
}

#[derive(Clone, Debug)]
pub struct HealthStatus {
    pub status: BackendStatus,
    pub healthy_since_unix: u64,
    pub error_count: u32,
}

/// 推理请求（内部表示）
#[derive(Clone, Debug)]
pub struct InferenceRequest {
    pub request_id: RequestId,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub token_ids: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stream: bool,
    pub tools: Vec<ToolDefinition>,
    pub lora_name: Option<String>,
}

/// 推理响应块（流式）
#[derive(Clone, Debug)]
pub enum InferenceChunk {
    /// 文本块
    Delta { text: String, finish_reason: Option<String> },
    /// 工具调用块
    ToolCall { id: String, function: String, args: String },
    /// 请求完成
    Done { backend_id: BackendId, latency: Duration },
    /// 错误
    Error { code: u16, message: String },
}
```

### 3.4 集群通信接口

```rust
// crates/aether-cluster/src/transport.rs

/// 集群消息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ClusterMessage {
    /// Gossip 心跳
    Ping { sender: InstanceId, meta_digest: MetaDigest },
    Pong { sender: InstanceId, meta_digest: MetaDigest },
    /// 新成员加入
    Meet { sender: InstanceId, region: RegionId, addr: String },
    /// 状态同步请求
    SyncRequest { sender: InstanceId, keys: Vec<MetaKey> },
    SyncResponse { entries: Vec<MetaEntry> },
    /// CKF 发布
    CkfBarrierSnapshot { pool: PoolId, sequence: u64, buckets: Vec<PackedBucket> },
    CkfDelta { pool: PoolId, sequence: u64, dirty_buckets: Vec<PackedBucket> },
    /// 负载指标广播
    MetricsBroadcast { region: RegionId, backends: Vec<(BackendId, BackendMetrics)> },
    /// 拓扑更新
    TopologyUpdate { matrix: LatencyMatrix },
    /// 会话亲和共享
    SessionAffinityBroadcast { session: SessionId, affinity: SessionAffinity },
}

/// 元数据摘要（PONG 携带）
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MetaDigest {
    pub kv_version: u64,
    pub model_version: u64,
    pub load_version: u64,
    pub topology_version: u64,
    pub members_version: u64,
}

#[async_trait]
pub trait ClusterTransport: Send + Sync {
    async fn start(&self, self_id: &InstanceId) -> Result<()>;
    async fn stop(&self) -> Result<()>;
    async fn broadcast(&self, msg: &ClusterMessage) -> Result<()>;
    async fn send(&self, target: &InstanceId, msg: &ClusterMessage) -> Result<()>;
    fn messages(&self) -> BoxStream<'static, ClusterMessage>;
    fn members(&self) -> Vec<ClusterMember>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterMember {
    pub instance_id: InstanceId,
    pub region: RegionId,
    pub addr: String,
    pub last_pong_unix: u64,
    pub status: MemberStatus,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemberStatus {
    Alive,
    Suspect,
    Dead,
}
```

## 4. 对外 API（HTTP / OpenAI 兼容）

### 4.1 Chat Completions

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "qwen2.5-7b",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true
}
```

响应与 OpenAI API 完全兼容（流式 SSE 或非流式 JSON）。

### 4.2 管理端点

```
GET  /health                          # 健康检查
GET  /v1/models                       # 列出可用模型
GET  /admin/backends                   # 列出后端及状态
GET  /admin/backends/:id/metrics       # 后端指标
GET  /admin/regions                    # 列出区域
GET  /admin/topology                   # 拓扑矩阵
GET  /admin/routing/history            # 路由历史
GET  /admin/routing/strategy           # 当前策略与权重
GET  /admin/kv/index/stats             # KV Index 统计
GET  /admin/kv/ckf/snapshot            # CKF 快照（诊断）
POST /admin/routing/strategy           # 动态切换策略
POST /admin/backends                   # 手动注册后端
DELETE /admin/backends/:id             # 移除后端
```

### 4.3 路由响应头

每个响应携带路由信息头（便于调试）：

```
X-Aether-Backend: cloud-beijing-worker-3
X-Aether-Region: cloud-cn-beijing
X-Aether-Strategy: hybrid
X-Aether-KV-Overlap: 8
X-Aether-Route-Latency-Ms: 2
```

## 5. KV Index 数据结构

### 5.1 RadixTree

```rust
// crates/aether-metadata/src/radix_tree.rs

use std::collections::HashMap;

/// RadixTree 节点
struct Node {
    hash: u64,
    owners: HashSet<(BackendId, u32)>,  // (backend_id, dp_rank)
    children: HashMap<u64, Box<Node>>,
    ref_count: u32,
}

/// RadixTree（线程安全，通过 channel 序列化访问，参考 Dynamo）
pub struct RadixTree {
    // 通过 mpsc channel 串行化所有操作
    cmd_tx: mpsc::Sender<RadixCommand>,
}

enum RadixCommand {
    ApplyEvent {
        backend: BackendId,
        event: KvCacheEvent,
        done: oneshot::Sender<Result<()>>,
    },
    FindMatches {
        hashes: Vec<u64>,
        backend: BackendId,
        done: oneshot::Sender<u32>,  // overlap count
    },
    FindAllMatches {
        hashes: Vec<u64>,
        done: oneshot::Sender<HashMap<BackendId, u32>>,
    },
    RemoveBackend {
        backend: BackendId,
        done: oneshot::Sender<()>,
    },
    Stats {
        done: oneshot::Sender<RadixStats>,
    },
}

#[derive(Debug, Default)]
pub struct RadixStats {
    pub total_nodes: usize,
    pub total_blocks: usize,
    pub backends: usize,
    pub depth: usize,
}
```

### 5.2 Cuckoo Filter

```rust
// crates/aether-metadata/src/cuckoo_filter.rs

const FINGERPRINT_BITS: usize = 16;
const BUCKETS_PER_LANE: usize = 1 << 16;  // 65536 buckets
const FP_PER_BUCKET: usize = 4;
const MAX_KICKS: usize = 500;

/// 单个 fingerprint（16 bit）
type Fp = u16;

/// Packed bucket: 4 × 16bit = 64bit
type PackedBucket = u64;

/// CKF Producer（本地，每 pool 一个）
pub struct CkfProducer {
    buckets: Vec<PackedBucket>,
    num_items: usize,
    dirty_buckets: BitSet,       // 脏 bucket 集合
    pub_seq: u64,                // publication sequence
    // 精确状态（refcount）
    hash_refcount: HashMap<u64, u32>,  // full_hash -> refcount
    hash_owners: HashMap<u64, HashSet<WorkerWithRank>>,
}

impl CkfProducer {
    /// 应用 KV 事件，更新 CKF
    pub fn apply_event(&mut self, event: &KvCacheEvent, backend: &BackendId) {
        match event {
            KvCacheEvent::Stored { worker, block_hashes, .. } => {
                for &hash in block_hashes {
                    let owners = self.hash_owners.entry(hash).or_default();
                    let key = (backend.clone(), worker.dp_rank);
                    if owners.insert(worker.clone_key(&key)) {
                        // first owner of this (backend, hash)
                        let rc = self.hash_refcount.entry(hash).or_insert(0);
                        if *rc == 0 {
                            // first owner overall → insert fingerprint
                            self.insert_fingerprint(hash);
                        }
                        *rc += 1;
                    }
                }
            }
            KvCacheEvent::Removed { worker, block_hashes } => {
                for &hash in block_hashes {
                    let key = (backend.clone(), worker.dp_rank);
                    if let Some(owners) = self.hash_owners.get_mut(&hash) {
                        if owners.remove(&key) {
                            let rc = self.hash_refcount.get_mut(&hash).unwrap();
                            *rc -= 1;
                            if *rc == 0 {
                                // final owner removed → delete fingerprint
                                self.delete_fingerprint(hash);
                                self.hash_refcount.remove(&hash);
                            }
                        }
                    }
                    // unknown removal = no-op (参考 Dynamo)
                }
            }
            KvCacheEvent::Clear { worker } => { /* ... */ }
            KvCacheEvent::Reset { .. } => { /* generation fence */ }
        }
    }
    
    fn insert_fingerprint(&mut self, hash: u64) {
        let fp = (hash & 0xFFFF) as Fp;
        if fp == 0 { return; }  // 0 = empty slot
        let idx1 = (hash >> 16) as usize % BUCKETS_PER_LANE;
        let idx2 = alt_index(idx1, fp);
        if try_insert(&mut self.buckets[idx1], fp) {
            self.dirty_buckets.set(idx1);
        } else if try_insert(&mut self.buckets[idx2], fp) {
            self.dirty_buckets.set(idx2);
        } else {
            // cuckoo eviction
            self.relocate(idx1, idx2, fp);
        }
        self.num_items += 1;
    }
    
    fn delete_fingerprint(&mut self, hash: u64) {
        let fp = (hash & 0xFFFF) as Fp;
        if fp == 0 { return; }
        let idx1 = (hash >> 16) as usize % BUCKETS_PER_LANE;
        let idx2 = alt_index(idx1, fp);
        if try_delete(&mut self.buckets[idx1], fp) {
            self.dirty_buckets.set(idx1);
        } else if try_delete(&mut self.buckets[idx2], fp) {
            self.dirty_buckets.set(idx2);
        }
        // false negative on delete is acceptable for CKF
        self.num_items -= 1;
    }
    
    /// 发布 barrier snapshot
    pub fn snapshot(&mut self) -> CkfSnapshot {
        self.dirty_buckets.clear();
        CkfSnapshot {
            sequence: self.pub_seq,
            buckets: self.buckets.clone(),
        }
    }
    
    /// 发布 sequenced delta
    pub fn delta(&mut self) -> Option<CkfDelta> {
        if self.dirty_buckets.is_empty() { return None; }
        let dirty: Vec<(usize, PackedBucket)> = self.dirty_buckets
            .iter()
            .map(|i| (i, self.buckets[i]))
            .collect();
        self.dirty_buckets.clear();
        self.pub_seq += 1;
        Some(CkfDelta {
            sequence: self.pub_seq,
            prev_sequence: self.pub_seq - 1,
            buckets: dirty,
        })
    }
}

fn alt_index(idx: usize, fp: Fp) -> usize {
    // partial-key cuckoo: idx2 = idx1 ^ hash(fp)
    let h = (fp as u64).wrapping_mul(0x9E3779B97F4A7C15) as usize;
    (idx ^ h) % BUCKETS_PER_LANE
}

/// 在 packed bucket 中插入 fingerprint
fn try_insert(bucket: &mut PackedBucket, fp: Fp) -> bool {
    for i in 0..FP_PER_BUCKET {
        let shift = i * FINGERPRINT_BITS;
        let slot = (*bucket >> shift) & 0xFFFF;
        if slot == 0 {
            *bucket |= (fp as u64) << shift;
            return true;
        }
    }
    false
}

/// 在 packed bucket 中删除 fingerprint
fn try_delete(bucket: &mut PackedBucket, fp: Fp) -> bool {
    for i in 0..FP_PER_BUCKET {
        let shift = i * FINGERPRINT_BITS;
        let slot = (*bucket >> shift) & 0xFFFF;
        if slot == fp as u64 {
            *bucket &= !(0xFFFFu64 << shift);
            return true;
        }
    }
    false
}

/// 在 packed bucket 中查找 fingerprint
fn bucket_contains(bucket: PackedBucket, fp: Fp) -> bool {
    for i in 0..FP_PER_BUCKET {
        let slot = (bucket >> (i * FINGERPRINT_BITS)) & 0xFFFF;
        if slot == fp as u64 { return true; }
    }
    false
}
```

### 5.3 CKF Consumer（Transposed Layout）

```rust
// crates/aether-metadata/src/ckf_consumer.rs

const MAX_LANES: usize = 16;

/// Transposed CKF Consumer
/// 按 bucket 组织，每个 bucket 跨 lane 是一个 atomic u64
pub struct CkfConsumer {
    /// bucket_major: [bucket][lane] → AtomicU64
    buckets: Vec<[AtomicU64; MAX_LANES]>,
    /// lane → region 映射
    lane_regions: RwLock<HashMap<usize, RegionId>>,
    /// lane 状态
    lane_status: [AtomicLaneStatus; MAX_LANES],
}

#[derive(Clone, Copy)]
enum LaneStatus {
    Active,
    Retired,
}

impl CkfConsumer {
    /// 估算请求在目标 region 的 KV overlap
    pub fn estimate_overlap(&self, hashes: &[u64], region: &RegionId) -> u32 {
        let lane = self.lane_regions.read().get(region)?;
        if self.lane_status[lane] != LaneStatus::Active { return 0; }
        
        let mut overlap = 0u32;
        for &hash in hashes {
            let fp = (hash & 0xFFFF) as u16;
            if fp == 0 { break; }
            let idx1 = ((hash >> 16) as usize) % self.buckets.len();
            let idx2 = alt_index(idx1, fp);
            
            let bucket = self.buckets[idx1][lane].load(Relaxed);
            if bucket_contains(bucket, fp) 
               || bucket_contains(self.buckets[idx2][lane].load(Relaxed), fp) {
                overlap += 1;
            } else {
                break;  // 前缀中断
            }
        }
        overlap
    }
    
    /// 安装 barrier snapshot（lane 重连）
    pub fn install_snapshot(&self, lane: usize, snapshot: &CkfSnapshot) {
        self.lane_status[lane].store(LaneStatus::Retired, Relaxed);
        for (idx, &bucket) in snapshot.buckets.iter().enumerate() {
            self.buckets[idx][lane].store(bucket, Relaxed);
        }
        self.lane_status[lane].store(LaneStatus::Active, Relaxed);
    }
    
    /// 应用 delta
    pub fn apply_delta(&self, lane: usize, delta: &CkfDelta) {
        for &(idx, bucket) in &delta.buckets {
            self.buckets[idx][lane].store(bucket, Relaxed);
        }
    }
}
```

## 6. 路由引擎

```rust
// crates/aether-routing/src/engine.rs

pub struct RoutingEngine {
    strategies: Vec<Box<dyn RoutingStrategy>>,
    hybrid: HybridStrategy,
    session_affinity_ttl: Duration,
    max_retries: u32,
    temperature: f64,
}

impl RoutingEngine {
    /// 路由决策主入口
    pub async fn route(
        &self,
        ctx: &RoutingContext,
        meta: &MetadataStore,
    ) -> Result<RouteDecision> {
        // 1. 会话亲和检查
        if let Some(session_id) = &ctx.session_id {
            if let Some(affinity) = meta.session_get(session_id) {
                if self.is_backend_healthy(&affinity.backend, meta).await 
                   && affinity.kv_overlap_at_route > 0 {
                    return Ok(RouteDecision {
                        backend: affinity.backend.clone(),
                        strategy: "session_affinity".into(),
                        kv_overlap: affinity.kv_overlap_at_route,
                    });
                }
            }
        }
        
        // 2. Hybrid 策略评估
        let decision = self.hybrid.evaluate(ctx, &candidates, meta).await?;
        
        // 3. 更新会话亲和
        if let Some(session_id) = &ctx.session_id {
            meta.session_set(session_id, SessionAffinity {
                backend: decision.backend.clone(),
                last_used_unix: now(),
                kv_overlap_at_route: decision.kv_overlap,
            });
        }
        
        Ok(decision)
    }
}

#[derive(Debug)]
pub struct RouteDecision {
    pub backend: BackendId,
    pub strategy: String,
    pub kv_overlap: u32,
}
```

## 7. 配置模型

```rust
// crates/aether-core/src/config.rs

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub instance_id: InstanceId,
    pub region: RegionConfig,
    pub listen: ListenConfig,
    pub routing: RoutingConfig,
    pub cluster: ClusterConfig,
    pub backends: Vec<BackendConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionConfig {
    pub id: RegionId,
    pub tier: RegionTier,
    pub geo: Option<GeoCoord>,
    pub network_zone: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub strategy: StrategyType,       // hybrid | kv | model | load | topology
    pub kv_block_size: u32,
    pub overlap_score_credit: f64,
    pub prefill_load_scale: f64,
    pub temperature: f64,
    pub session_affinity_ttl_secs: u64,
    pub max_retries: u32,
    pub weights: StrategyWeights,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum StrategyType {
    Hybrid, Kv, Model, Load, Topology,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyWeights {
    pub kv: f64,
    pub load: f64,
    pub topology: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub bind_addr: String,           // gossip 监听地址
    pub seed_peers: Vec<String>,     // 种子节点
    pub gossip_interval_ms: u64,
    pub probe_timeout_ms: u64,
    pub suspect_timeout_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendConfig {
    #[serde(rename = "type")]
    pub backend_type: BackendType,
    pub endpoint: String,
    pub models: Vec<String>,
    pub region: Option<RegionId>,
    pub kv_block_size: Option<u32>,
}
```
