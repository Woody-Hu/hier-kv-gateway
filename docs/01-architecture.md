# Hier KV Gateway 架构设计文档

> 中文 | [English](en/01-architecture.md)

> 云边端协同的 LLM 请求自动调度 Gateway 系统

## 1. 背景与目标

### 1.1 问题域

在云边端协同的 LLM 推理场景中，推理资源分布在：

- **云侧（Cloud）**：具备 K8s + 分布式推理系统（如 llm-d）的完整集群，多机多卡，KV Cache 可跨节点共享。
- **边侧（Edge）**：资源受限的集群或单机多卡，可能有轻量调度。
- **端侧（Device）**：单进程推理引擎（vLLM / llama.cpp），无集群调度。

这些后端在**地理位置、延迟、容量、KV Cache 状态、模型版本**上差异巨大。客户端需要一个统一入口，将请求自动路由到最合适的后端。

### 1.2 设计目标

1. **跨集群/进程的 KV 感知路由**：感知各后端 KV Cache 前缀重叠，将请求路由到命中率最高的后端。
2. **跨集群的模型信息感知路由**：根据后端加载的模型、版本、量化方式做路由。
3. **基于实例延迟与负载统计的路由**：实时收集后端负载指标，做负载均衡。
4. **基于地理位置拓扑的路由**：根据网络延迟拓扑优先就近路由。
5. **混合智能路由**：融合上述四项的默认策略。
6. **分布式系统**：Gateway 实例组跨集群通过 Gossip 通信；集群内多实例高可用。
7. **单进程高可用与服务降级**：预测不准时回退到基础负载均衡。
8. **插件化**：策略、后端连接器、实例间通信均可扩展。

## 2. 顶层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         客户端 (OpenAI API)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP/gRPC
┌────────────────────────────────▼────────────────────────────────────┐
│                     Hier KV Gateway 进程                            │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ HTTP/API │→ │  路由引擎     │  │ 元数据存储  │  │  Gossip 层   │  │
│  │  Server  │  │ (5种策略+混合) │  │ (内存缓存)  │  │ (集群通信)   │  │
│  └──────────┘  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘  │
│                       │                │                 │          │
│  ┌────────────────────▼────────────────▼─────────────────▼───────┐  │
│  │                    后端连接器 (插件)                           │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │  │
│  │  │  LLM-D   │ │ vLLM     │ │llama.cpp │ │  Generic OpenAI   │  │  │
│  │  │ Cluster  │ │ Engine   │ │ Engine   │ │  Compatible      │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          ▲ Gossip                    ▲ Gossip
          │                            │
┌─────────┴──────────┐       ┌─────────┴──────────┐
│  Gateway 实例 (云)  │◄─────►│  Gateway 实例 (边)  │
│  ┌───────────────┐ │       │ ┌────────────────┐ │
│  │  LLM-D Cluster│ │       │ │  vLLM Engine    │ │
│  │  (K8s + 多机)  │ │       │ │  (单进程)       │ │
│  └───────────────┘ │       │ └────────────────┘ │
└───────────────────┘       └────────────────────┘
```

## 3. 核心概念

### 3.1 Region（区域）

一个 Region 对应一个逻辑部署域（类似数据中心 DC 概念）：

```
RegionId = 稳定字符串标识（如 "cloud-cn-beijing", "edge-shanghai", "device-rpi-01"）
```

- 每个 Region 内有一个或多个 Gateway 实例（云侧多实例 HA，端侧单实例）。
- RegionId 跨 Gateway 重启保持稳定。
- Region 有拓扑属性：地理位置坐标、网络延迟矩阵、层级（cloud/edge/device）。

### 3.2 Backend（后端）

一个 Region 内的推理服务实例，是路由的目标：

```
BackendId = (RegionId, 实例标识)
```

后端类型：
- **Cluster Backend**：对接 llm-d 等分布式推理系统，多 worker，支持 KV event。
- **Engine Backend**：对接 vLLM / llama.cpp 单进程引擎，可能支持 KV event 或不支持。

### 3.3 Indexer Domain（索引域）

标识可作为一个逻辑路由命名空间比较的缓存集合：

```
IndexerDomainId = (模型架构 + 分词器 + KV block size + 量化配置 的哈希)
```

同一 IndexerDomain 下的 Backend 的 KV Cache 可以相互比较。不同模型/配置的 Backend 属于不同 domain。

### 3.4 Pool（池）

Pool 定义为 `PoolId = (IndexerDomainId, RegionId)`：

```
PoolId = (IndexerDomainId, RegionId)
```

一个 Pool 对应一个 Region 内、同一 IndexerDomain 的一组 Backend。KV 感知路由在 Pool 粒度做跨 Region 比较。

## 4. 分层架构

### 4.1 Gateway 进程内部分层

```
┌─────────────────────────────────────────────────────┐
│  API Layer (HTTP Server, OpenAI 兼容)               │
├─────────────────────────────────────────────────────┤
│  Routing Layer                                       │
│   ├── Strategy: KV Aware                             │
│   ├── Strategy: Model Aware                         │
│   ├── Strategy: Load Aware                          │
│   ├── Strategy: Topology Aware                      │
│   ├── Plugin: KV Capacity (可选, [kv_estimate])      │
│   └── Strategy: Hybrid (默认)                       │
├─────────────────────────────────────────────────────┤
│  Metadata Layer                                      │
│   ├── KV Index (RadixTree + CKF projection)         │
│   ├── Model Registry (模型/版本/能力)                │
│   ├── Load Stats (延迟/队列/容量)                    │
│   ├── Topology Graph (Region 间延迟矩阵)            │
│   └── Routing History (会话亲和 / 降级统计)           │
├─────────────────────────────────────────────────────┤
│  Cluster Layer (Gossip)                             │
│   ├── Member Discovery (SWIM-like)                  │
│   ├── State Sync (元数据广播/同步)                   │
│   └── CKF Relay (跨 Region KV 投影发布)             │
├─────────────────────────────────────────────────────┤
│  Connector Layer (插件)                              │
│   ├── LLM-D Connector (NATS/HTTP KV event)          │
│   ├── vLLM Connector (ZMQ/HTTP KV event)            │
│   ├── llama.cpp Connector (无 KV event, 降级)       │
│   └── Generic OpenAI Connector (无 KV, 降级)        │
└─────────────────────────────────────────────────────┘
```

### 4.2 数据流

```
客户端请求
  │
  ▼
[API Layer] 解析请求 → 提取 token_ids / model / 参数
  │
  ▼
[Routing Layer] 混合策略评估
  │  1. 计算 block hashes
  │  2. 查询本地 KV Index → device overlap
  │  3. 查询跨 Region CKF → remote KV overlap
  │  4. 查询 Model Registry → 候选 Backend 集合
  │  5. 查询 Load Stats → 各 Backend cost
  │  6. 查询 Topology → 网络延迟惩罚
  │  7. 综合评分 → 选最优 Backend
  ▼
[Connector Layer] 转发请求到选中 Backend
  │  ├── 成功 → 流式返回响应 → 更新 Load Stats / KV Index
  │  └── 失败 → 重试/降级 → 选次优 Backend
  ▼
响应返回客户端
```

## 5. 分布式架构

### 5.1 Gossip 协议

Gateway 实例组跨集群通过 Gossip 通信：

**消息类型**：
- `PING / PONG`：心跳，携带发送方的元数据摘要。
- `MEET`：新节点加入集群。
- `SYNC`：请求全量状态同步（新节点或修复）。
- `CKF_PUBLISH`：跨 Region 的 KV 投影发布（barrier snapshot + sequenced delta）。
- `METRIC_BROADCAST`：负载/延迟指标广播。

**Gossip 行为**：
- 每个 Gateway 实例维护一个成员列表（Region → 实例地址 + 心跳时间 + 元数据版本）。
- 每秒随机选 P 个实例发 PING，PONG 携带最新元数据摘要。
- 若 PING 超时（N 次连续失败），标记该实例为疑似下线 (suspect) → 确认下线 (down)。
- 新实例通过 `MEET` 加入，收到 MEET 的实例将其加入成员列表并在后续 Gossip 中传播。

**元数据同步**：
- 元数据用版本号 (version vector) 标记，PONG 中携带摘要（Region → version）。
- 接收方对比本地版本，对落后的项请求 `SYNC` 获取增量。
- 大状态（如 CKF 投影）用 barrier snapshot + sequenced delta，不放在 PING 中。

### 5.2 集群内高可用

云侧一个 Region 内可部署多个 Gateway 实例：

- **Leader 选举**：基于 Raft（Gossip 成员中选 leader 负责协调）或简单的主备（基于 etcd lease）。
- **无状态路由**：路由决策是无状态的（基于当前元数据），任何实例都能独立路由。Leader 仅负责协调元数据版本、避免重复 CKF 发布。
- **请求重试**：客户端可重试到另一个 Gateway 实例。
- **会话亲和**：通过 Gossip 共享 routing history（带 TTL），实现跨实例的会话亲和。

### 5.3 端侧单进程高可用

端侧只有一个 Gateway 进程，采用：

- **健康自检**：定期自检，若内存/连接异常则重启。
- **降级模式**：若 KV Index 不可用 → 回退到 Load Aware；若 Load Stats 不可用 → 回退到 Topology Aware（就近）；若全部不可用 → Round Robin。
- **本地持久化**：关键元数据（成员列表、拓扑）写入本地文件，重启后恢复。

### 5.4 服务降级机制

```
正常：Hybrid 策略（KV + Model + Load + Topology 综合评分）
  │ 若 KV Index 不可用
  ▼
降级1：Model + Load + Topology 策略
  │ 若 Load Stats 不可用
  ▼
降级2：Model + Topology 策略
  │ 若跨集群通信断开（只有本地 Region）
  ▼
降级3：本地 Region 内 Load Aware
  │ 若本地无可用 Backend
  ▼
降级4：返回 503 + 缓存的上一次健康后端列表
```

## 6. KV 感知路由（核心）

### 6.1 两阶段架构

```
┌─── Region A (云) ───────────┐     ┌─── Region B (边) ───────────┐
│  Workers → KV Events         │     │  Engine → KV Events          │
│      │                       │     │      │                       │
│      ▼                       │     │      ▼                       │
│  Local KV Relay              │     │  Local KV Relay              │
│  (精确所有权 + refcount)      │     │  (精确所有权 + refcount)      │
│      │                       │     │      │                       │
│      ▼ CKF 投影               │     │      ▼ CKF 投影               │
│  ┌──────── Gossip Bus ──────────────────────────────────┐         │
│  │  CKF Publish: barrier snapshot + sequenced delta     │         │
│  └──────────────────────────────────────────────────────┘         │
│      │                       │     │      │                       │
│      ▼                       │     │      ▼                       │
│  Global CKF Consumer         │◄───►│  Global CKF Consumer         │
│  (transposed, 多 lane 并发)  │     │  (transposed, 多 lane 并发)  │
│      │                       │     │      │                       │
│      ▼                       │     │      ▼                       │
│  KV Aware Router             │     │  KV Aware Router             │
└──────────────────────────────┘     └──────────────────────────────┘
```

### 6.2 Stage 1: Backend KV Events → Local KV Relay

每个 Region 内的 Gateway 实例（或 Relay 角色实例）：

1. **发现后端**：通过 Connector 插件发现本 Region 的 Backend。
2. **消费 KV Events**：Backend 通过 connector 上报 KV Cache 事件（block stored / removed）。
3. **维护精确状态**：
   - `full_hash → Set<(backend_id, dp_rank)>`：每个 full block hash 被哪些 backend/rank 拥有。
   - `full_hash → refcount`：DC/Region-wide 的引用计数。
4. **所有权变化处理**：
   - First owner of a full hash → 插入一个 CKF fingerprint
   - Another owner of same hash → refcount++ only
   - One of several removes → refcount-- only
   - Final owner removes → 删除 CKF fingerprint

### 6.3 Stage 2: CKF Projection → Global Consumer

每个 Region 的 Relay 将本地 CKF 投影发布到跨 Region 的 Gossip Bus：

- **Barrier Snapshot**：全量 CKF 状态 + terminal publication sequence。
- **Sequenced Delta**：自上次发布后变更的 packed bucket 绝对镜像。
- **Lease**：绑定一个 consumer 实例和一个 lane。

Global Consumer 在每个 Gateway 实例内运行：
- 维护 transposed CKF layout（bucket-major，每 lane 一个 atomic packed word）。
- 并发查询：一次前缀查询同时搜索所有 Region lane。
- 故障恢复：lane 断开时排除该 lane，重连时安装新 barrier snapshot。

### 6.4 Cuckoo Filter 设计

CKF 设计要点：

- **Fingerprint**：block hash 的短指纹（如 16 bit），有损。
- **Bucket**：每个 bucket 存放多个 fingerprint（如 4 个 × 16 bit = 64 bit packed word）。
- **Transposed Layout**：consumer 侧按 bucket 组织，每个 bucket 跨 lane 是一个 atomic u64，支持并发读。
- **容量**：支持最多 16 个 Region lane 并发查询。
- **假阳性**：CKF 可能返回 false positive（某 Region 似乎有该 block，实际没有），由后续精确查询 / 请求结果校正。

## 7. 语言选型：Rust

选择 **Rust** 作为实现语言，理由：

1. **性能**：Gateway 是数据面关键路径，延迟敏感。Rust 零成本抽象 + 无 GC，适合高并发低延迟。
2. **生态一致**：与同类推理基础设施项目（如 llm-d）核心语言一致，便于复用数据结构和算法思路。
3. **内存安全**：路由元数据（KV index、CKF、load stats）并发访问频繁，Rust 的所有权模型保证线程安全。
4. **生态**：tokio（异步运行时）、axum（HTTP）、serde（序列化）、dashmap（并发 map）等成熟。

## 8. 插件与接口机制

### 8.1 策略插件

```rust
pub trait RoutingStrategy: Send + Sync {
    /// 策略名称
    fn name(&self) -> &str;
    
    /// 评估候选 Backend 列表，返回带分数的排序结果
    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>>;
    
    /// 该策略是否可用（降级判断）
    fn is_available(&self, meta: &MetadataStore) -> bool;
}
```

内置策略实现该 trait，用户可自定义策略注册。

### 8.2 后端连接器插件

```rust
#[async_trait]
pub trait BackendConnector: Send + Sync {
    /// 连接器类型名
    fn backend_type(&self) -> &str;
    
    /// 发现该类型的后端实例
    async fn discover(&self) -> Result<Vec<BackendInfo>>;
    
    /// 健康检查
    async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus>;
    
    /// 转发推理请求（流式）
    async fn forward(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<BoxStream<'static, InferenceChunk>>;
    
    /// 是否支持 KV Cache 事件
    fn supports_kv_events(&self) -> bool;
    
    /// 订阅 KV Cache 事件流（若支持）
    async fn subscribe_kv_events(
        &self,
        backend: &BackendId,
    ) -> Result<BoxStream<'static, KvCacheEvent>>;
    
    /// 收集负载指标
    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics>;
}
```

### 8.3 实例间通信插件

```rust
#[async_trait]
pub trait ClusterTransport: Send + Sync {
    /// 启动通信
    async fn start(&self, self_id: &InstanceId) -> Result<()>;
    
    /// 广播消息
    async fn broadcast(&self, msg: &ClusterMessage) -> Result<()>;
    
    /// 发送给特定实例
    async fn send(&self, target: &InstanceId, msg: &ClusterMessage) -> Result<()>;
    
    /// 接收消息流
    fn messages(&self) -> BoxStream<'static, ClusterMessage>;
}
```

默认提供 Gossip 实现，可替换为其他（如 NATS、gRPC mesh）。

## 9. 元数据缓存机制

### 9.1 内存缓存层级

```
┌─────────────────────────────────────────────┐
│ L1: Request-Local Cache (请求级)              │
│   - 本次请求的 block hashes                    │
│   - 本次请求的 overlap scores                 │
│   生命周期: 单次请求                            │
├─────────────────────────────────────────────┤
│ L2: Hot Metadata Cache (热数据)              │
│   - RadixTree (本地 KV prefix tree)          │
│   - Load Stats (滑动窗口, TTL 5s)            │
│   - CKF Consumer (跨 Region 投影)            │
│   生命周期: 常驻内存, 实时更新                  │
├─────────────────────────────────────────────┤
│ L3: Warm Metadata Cache (温数据)             │
│   - Model Registry (模型信息, TTL 60s)       │
│   - Topology Graph (延迟矩阵, TTL 30s)       │
│   - Backend Discovery (后端列表, TTL 15s)    │
│   生命周期: 定期刷新                            │
├─────────────────────────────────────────────┤
│ L4: Cold Metadata Store (冷数据)             │
│   - Routing History (会话亲和, TTL 300s)      │
│   - Degradation Stats (降级统计, TTL 60s)    │
│   生命周期: 按需查询 + 定期清理                 │
└─────────────────────────────────────────────┘
```

### 9.2 并发安全

- RadixTree：专用后台线程 + mpsc channel，避免锁竞争。
- Load Stats：`DashMap<BackendId, ArcSwap<Metrics>>`，读无锁、写 CAS。
- CKF Consumer：bucket 级 atomic u64，无 lane-wide lock。
- Model Registry：`Arc<RwLock<...>>`，读多写少。

## 10. 故障恢复

按 "narrowest state boundary" 原则设计故障恢复边界：

| 故障 | 恢复边界 | 行为 |
|------|---------|------|
| Backend 事件 gap | 该 backend 的 rank 状态 | 从 backend 事件历史恢复，或安装当前 tree state |
| Backend 替换 | 该 backend 所有状态 | completion barrier 后从新 source 重建 |
| CKF delivery gap | 受影响的 consumer lane | 退役该 lane，重连时安装新 barrier snapshot |
| Gateway 实例崩溃 | 该实例的本地状态 | 其他实例通过 Gossip 感知，接管路由；新实例 SYNC 全量状态 |
| Region 隔离 | 该 Region 的 lane | 路由排除该 Region；恢复后 lane 重新激活 |

## 11. 目录结构规划

```
hier-kv-gateway/
├── Cargo.toml                 # workspace 根
├── crates/
│   ├── hier-kv-gateway-core/           # 核心类型: BackendId, RegionId, 元数据模型
│   ├── hier-kv-gateway-metadata/       # 元数据存储: RadixTree, CKF, LoadStats, ModelRegistry
│   ├── hier-kv-gateway-routing/        # 路由引擎: 5种策略 + Hybrid
│   ├── hier-kv-gateway-cluster/         # Gossip 集群通信 + CKF Relay
│   ├── hier-kv-gateway-connector/       # 后端连接器 trait + 内置实现
│   ├── hier-kv-gateway-kv-estimate/     # KV 显存估算（独立叶子 crate，解析公式 + 插件）
│   ├── hier-kv-gateway-api/             # HTTP API server (OpenAI 兼容)
│   └── hier-kv-gateway/                # 主二进制: 组装所有组件
├── tests/                     # 集成测试 (真实后端, 无 mock)
├── docs/                      # 设计文档
└── examples/                  # 配置示例
```
