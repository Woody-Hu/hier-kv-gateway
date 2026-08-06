# Hier KV Gateway

> 中文 | [English](README.en.md)

> 云边端协同的 LLM 请求自动调度 Gateway 系统

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-430%20passed-brightgreen.svg)](#测试)

## 概述

Hier KV Gateway 是一个面向**云边端协同**场景的 LLM 推理请求自动调度网关。它将分布在不同地理位置、不同规模、不同引擎的推理后端（云侧 K8s 集群、边侧轻量集群、端侧 vLLM/llama.cpp 单进程）统一为一个推理服务入口，通过智能路由策略将请求自动调度到最合适的后端。

### 核心能力

| 能力 | 说明 |
|------|------|
| **KV 感知路由** | 感知各后端 KV Cache 前缀重叠，路由到缓存命中率最高的后端，减少 prefill 计算 |
| **KV 容量感知路由** | 估算请求 KV Cache 显存占用，按后端剩余 KV block / GPU 显存余量打分，排除放不下的后端（load shedding） |
| **模型感知路由** | 根据后端加载的模型、版本、量化方式做精确/兼容匹配路由 |
| **负载感知路由** | 实时收集后端队列深度、GPU 利用率、KV 使用率等指标做负载均衡 |
| **拓扑感知路由** | 根据网络延迟拓扑优先就近路由，降低端到端延迟 |
| **混合智能路由** | 默认策略，融合上述四项的加权评分综合决策 |
| **KV 显存估算** | 独立叶子 crate，解析公式（非仿真）估算 KV Cache 显存，零分配热路径、插件化扩展、预制主流模型 |
| **Gossip 集群通信** | 跨集群 Gateway 实例组通过 Gossip 协议互相发现与同步元数据 |
| **CKF 跨域 KV 投影** | Cuckoo Filter 紧凑投影跨 Region KV 状态，实现跨域 KV 感知路由 |
| **插件化架构** | 路由策略、后端连接器、集群通信均可扩展 |
| **服务降级** | 元数据不可用时自动降级到次优策略，最终回退到基础负载均衡 |
| **OpenAI 兼容 API** | 完全兼容 OpenAI Chat Completions API，支持流式 SSE |

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         客户端 (OpenAI API)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP
┌────────────────────────────────▼────────────────────────────────────┐
│                     Hier KV Gateway 进程                             │
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
│  Gateway (云)       │◄─────►│  Gateway (边)      │
│  ┌───────────────┐ │       │ ┌────────────────┐ │
│  │  LLM-D Cluster│ │       │ │  vLLM Engine    │ │
│  │  (K8s + 多机)  │ │       │ │  (单进程)       │ │
│  └───────────────┘ │       │ └────────────────┘ │
└───────────────────┘       └────────────────────┘
```

### KV 感知路由（两阶段架构）

```
┌─── Region A (云) ───────────┐     ┌─── Region B (边) ───────────┐
│  Workers → KV Events         │     │  Engine → KV Events          │
│      ▼                       │     │      ▼                       │
│  Local KV Relay              │     │  Local KV Relay              │
│  (精确所有权 + refcount)      │     │  (精确所有权 + refcount)      │
│      ▼ CKF 投影               │     │      ▼ CKF 投影               │
│  ┌──────── Gossip Bus ──────────────────────────────────┐         │
│  │  CKF Publish: barrier snapshot + sequenced delta     │         │
│  └──────────────────────────────────────────────────────┘         │
│      ▼                       │     │      ▼                       │
│  Global CKF Consumer         │◄───►│  Global CKF Consumer         │
│  (transposed, 多 lane 并发)  │     │  (transposed, 多 lane 并发)  │
│      ▼                       │     │      ▼                       │
│  KV Aware Router             │     │  KV Aware Router             │
└──────────────────────────────┘     └──────────────────────────────┘
```

## 快速开始

### 前置要求

- Rust 1.75+ (推荐 1.85+)
- Cargo

### 编译

```bash
cargo build --release
```

### 运行测试

```bash
# 全部测试（430 个）
cargo test --workspace

# 仅 KV 估算模块（含零分配测试）
cargo test -p hier-kv-gateway-kv-estimate

# 仅集成测试
cargo test -p hier-kv-gateway-integration

# 仅端到端测试
cargo test --test end_to_end -p hier-kv-gateway-integration -- --nocapture
```

### 启动 Gateway

```bash
# 使用示例配置
cargo run --release -- --config examples/hier-kv-gateway.toml
```

示例配置 ([examples/hier-kv-gateway.toml](examples/hier-kv-gateway.toml)):

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

### 发送请求

```bash
# Chat Completions（与 OpenAI API 完全兼容）
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'

# 查看路由决策信息（响应头）
curl -v http://localhost:8080/v1/chat/completions ...
# X-Hier-KV-Gateway-Backend: cloud-beijing-worker-3
# X-Hier-KV-Gateway-Strategy: hybrid
# X-Hier-KV-Gateway-KV-Overlap: 8
```

## 路由策略

### 1. KV 感知路由 (KV Aware)

成本函数：

```
total_overlap = device_overlap(本地 RadixTree) + ckf_overlap(跨域 CKF)
prefill_blocks = max(len(hashes) - total_overlap, 0)
decode_blocks = backend.active_decode_blocks
cost = prefill_load_scale * prefill_blocks + decode_blocks
score = 1.0 / (1.0 + cost)
```

**所有权 refcount 四分支**：
- First owner of a hash → 插入 CKF fingerprint
- Another owner of same hash → refcount++ only
- One of several removes → refcount-- only
- Final owner removes → 删除 CKF fingerprint

### 2. 模型感知路由 (Model Aware)

作为**硬性过滤器**，排除不匹配的后端：
- exact match (model_name + version + quant): score = 1.0
- model_match (同名不同版本): score = 0.7
- compatible_match (同架构): score = 0.3
- no_match: 排除
- 额外检查：max_context_len、tool_calling 支持

### 3. 负载感知路由 (Load Aware)

```
load_cost = w_req * active_requests
          + w_queue * queue_depth
          + w_lat * (p99_latency / 100)
          + w_gpu * gpu_utilization
          + w_kv * kv_cache_usage
score = 1.0 / (1.0 + load_cost)
```

### 4. 拓扑感知路由 (Topology Aware)

```
rtt = latency_matrix.rtt_ms(self_region, backend.region)
network_cost = w_rtt * rtt + w_bw * bandwidth_penalty
score = 1.0 / (1.0 + network_cost / 100.0)
```

延迟矩阵来源：配置 + 主动探测 + Gossip 传播 + 地理距离估算。

### 5. 混合智能路由 (Hybrid, 默认)

```
1. Model Aware 过滤候选集
2. 对每个可用子策略 S 计算 scores_S
3. 动态权重调整:
   - KV 不可用 → weight_kv = 0
   - Load 过期(>10s) → weight_load *= 0.3
   - 归一化使总和 = 1.0
4. hybrid_score(b) = Σ(weight_S * normalize(score_S[b]))
5. temperature > 0: softmax 采样; temperature = 0: 贪心
```

**默认权重**：KV=0.35, Load=0.30, Topology=0.20

### 6. KV 容量感知路由 (KV Capacity, 可选插件)

`[kv_estimate] enabled = true` 时作为 `RoutingPlugin` 挂到 Hybrid。先估算本次请求的 KV Cache 显存占用，再按各后端**剩余容量**打分，把放不下的后端排除 —— 这是 load shedding / admission control 决策。与 `KvAwareStrategy`（按前缀命中重叠打分，减少 prefill 工作）互补：一个决定「少做多少 prefill」，一个决定「放不放得下」。

```
per_token = f(num_layers, num_kv_heads, head_dim, dtype, attention族)   // MLA 用 kv_lora_rank+qk_rope_head_dim，无因子 2
seq_len   = input_tokens + max_tokens                                  // 输出用 max_tokens 作保守上界
effective = min(seq_len, sliding_window)                               // 滑动窗口截断
blocks    = ceil(effective / block_size) * batch_size
bytes     = per_token * batch_size * (blocks * block_size)             // block-padded

available_bytes =
  (kv_total_blocks - kv_used_blocks) * per_block_bytes              // KV-block 路径（精确，优先）
  or (gpu_memory_total_mb - gpu_memory_used_mb) * 1e6 * safety_frac // GPU 显存 fallback（保守）

if available_bytes <= 0 or bytes > available_bytes: 排除 (raw_cost=∞, score=0)   // load shedding
else: ratio = bytes/available_bytes; raw_cost=ratio; score = 1/(1+ratio)
```

**关键设计**：output 用 `max_tokens` 作保守上界（永不低估）；排除用 `f64::INFINITY`（由 `normalize_costs` 的 `!is_finite()` 识别）；GPU fallback 仅声明「空闲显存 × `gpu_mem_safety_fraction`」（KV 不是唯一 GPU 内存消费者）；未知 spec 默认中立（`exclude_on_unknown_spec=false`），交由其他子策略决定。

详细公式推导与端到端算例见 [KV 显存估算架构](docs/05-kv-estimation.md) 与 [路由算法 §9](docs/02-routing-algorithms.md)。

### 服务降级链

```
Hybrid (KV+Load+Topo)
  │ 若 KV Index 不可用
  ▼
Model + Load + Topo
  │ 若 Load Stats 不可用
  ▼
Model + Topo
  │ 若跨集群通信断开
  ▼
本地 Load Aware
  │ 若本地无可用 Backend
  ▼
返回 503
```

## KV 显存估算

独立的叶子 crate [hier-kv-gateway-kv-estimate](crates/hier-kv-gateway-kv-estimate) 用**解析公式**（非仿真）估算一次推理的 KV Cache 显存占用，与 vLLM / SGLang / Mooncake / llm-d 的 KV 大小计算一致。`KvCapacityStrategy` 据此做容量感知路由（见上文第 6 节）。

### 特性

- **解析公式**：直接用模型架构参数（层数 / KV head 数 / head 维度 / dtype / 注意力族）+ 请求形状（batch / input length / output length / block size）做整数乘加，得出字节数。
- **零分配热路径**：`ModelSpec` 是 `Copy`、模型名留作 catalog key、`contains_ascii_ci` 在线小写匹配，热路径全程无 `String`/`HashMap` 值克隆。计数分配器测试 0 字节分配（见 `tests/alloc_free.rs`）。
- **纳秒级**：完整热路径 `registry.estimate` 45–91 ns，公式本身 ~10 ns（与输入长度无关）。
- **插件化扩展（两条路径）**：
  - **加模型 spec（数据）**：`[[kv_estimate.models]]` 一行 TOML，字段对应 HuggingFace `config.json`，Standard 公式覆盖。绝大多数新模型走这条路。
  - **加自定义 estimator（代码）**：实现 `KvEstimator` trait，经 `with_estimator` / `with_estimator_front` 注册。用于标准公式无法表达的架构（Cross-Attention 额外缓存、Mamba/SSM 状态）。
- **预制主流模型**：内置 Llama-2/3、Qwen2/2.5、Mistral/Mixtral、Gemma-2、DeepSeek-V2/V3/R1（MLA）、ChatGLM3，覆盖 MHA / GQA / MQA / MLA / 滑动窗口四族。

### 解析公式

```
Standard (MHA/GQA/MQA): per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
MLA (DeepSeek):         per_token = layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes   // 无因子 2
滑动窗口:               effective = min(seq_len, sliding_window)
block 分页:             blocks = ceil(effective / block_size) * batch
                        bytes  = per_token * batch * (blocks * block_size)
```

### 配置

```toml
[kv_estimate]
enabled = true                       # 关闭时不挂载策略（默认 false）
weight = 0.20                        # Hybrid 权重
gpu_mem_safety_fraction = 0.5        # GPU 显存 fallback 可声明比例
exclude_on_unknown_spec = false      # 未知 spec: false=中立, true=排除

# 可选：注册私有模型 spec（字段对应 HuggingFace config.json）
[[kv_estimate.models]]
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

### Benchmark

| 场景 | 延迟 |
|------|------|
| `estimate_kv`（公式本身） | ~10 ns |
| `registry.estimate`（完整热路径，名称→spec→公式） | 45–91 ns |
| `KvCapacityStrategy::evaluate`（10 后端） | 2.57 µs |
| Hybrid 端到端开销（10 后端，with vs without 插件） | ~3.85 µs（~42%） |

完整 benchmark 报告见 [docs/benchmarks/kv-estimation.md](docs/benchmarks/kv-estimation.md)，含反作弊断言（手算期望值、`raw_cost.is_finite()`、`score ∈ (0,1]`、零分配 0 容忍）。

## 元数据缓存

```
L1: Request-Local   - block hashes, overlap scores (单次请求)
L2: Hot Metadata    - RadixTree, LoadStats, CKF Consumer (常驻内存)
L3: Warm Metadata   - ModelRegistry(TTL 60s), Topology(TTL 30s), Discovery(TTL 15s)
L4: Cold Metadata   - RoutingHistory(TTL 300s), DegradationStats(TTL 60s)
```

并发安全设计：
- RadixTree：专用后台线程 + mpsc channel，避免锁竞争
- LoadStats：`DashMap<BackendId, ArcSwap<Metrics>>`，读无锁
- CKF Consumer：bucket 级 atomic u64，无 lane-wide lock

## Gossip 集群通信

| 消息类型 | 用途 |
|---------|------|
| PING/PONG | 心跳，携带元数据摘要 |
| MEET | 新节点加入集群 |
| SYNC | 请求全量状态同步 |
| CKF_PUBLISH | 跨 Region KV 投影发布（barrier + delta） |
| METRIC_BROADCAST | 负载/延迟指标广播 |

行为：
- 每秒随机选 P 个实例发 PING，PONG 携带元数据摘要
- PING 超时 → suspect → 确认 dead
- 元数据用 version vector 标记，落后的项请求 SYNC

## 插件机制

### 路由策略插件

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

### 后端连接器插件

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

### 集群通信插件

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

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat/completions` | OpenAI 兼容 Chat Completions |
| GET | `/v1/models` | 列出可用模型 |
| GET | `/health` | 健康检查 |
| GET | `/admin/backends` | 后端列表与状态 |
| GET | `/admin/backends/:id/metrics` | 后端指标 |

响应头携带路由信息：`X-Hier-KV-Gateway-Backend`, `X-Hier-KV-Gateway-Strategy`, `X-Hier-KV-Gateway-KV-Overlap`

## 项目结构

```
hier-kv-gateway/
├── crates/
│   ├── hier-kv-gateway-core/           # 核心类型: IDs, BackendInfo, Metrics, Config
│   ├── hier-kv-gateway-metadata/       # 元数据: RadixTree, CKF, LoadStats, ModelRegistry
│   ├── hier-kv-gateway-routing/        # 路由引擎: 5种策略 + Hybrid + KV 容量插件
│   ├── hier-kv-gateway-kv-estimate/    # KV 显存估算（独立叶子 crate，解析公式 + 插件）
│   ├── hier-kv-gateway-cluster/        # Gossip 集群通信 + CKF Relay
│   ├── hier-kv-gateway-connector/      # 后端连接器: OpenAI兼容/vLLM
│   ├── hier-kv-gateway-api/            # HTTP API Server (OpenAI 兼容)
│   └── hier-kv-gateway/                # 主二进制
├── tests/
│   └── hier-kv-gateway-integration/    # 集成测试（无 mock）
├── docs/
│   ├── 01-architecture.md              # 架构设计文档
│   ├── 02-routing-algorithms.md        # 路由算法设计文档（含 KV 容量感知 §9）
│   ├── 03-interfaces-data-models.md    # 接口与数据模型文档
│   ├── 05-kv-estimation.md             # KV 显存估算模块架构文档
│   ├── benchmarks/                     # Benchmark 报告
│   └── session-logs/                   # 开发会话日志
└── examples/
    ├── hier-kv-gateway.toml             # 单后端配置示例
    └── multi-backend.toml               # 多后端示例（含 [kv_estimate] 段）
```

## 测试

全部 **430 个测试通过**，无 mock，无作弊：

| 测试套件 | 数量 | 说明 |
|---------|------|------|
| hier-kv-gateway-core | 77 | 类型序列化、配置解析、地理距离计算 |
| hier-kv-gateway-metadata | 30 | RadixTree 事件处理、CKF insert/delete/lookup |
| hier-kv-gateway-routing | 78 | 混合策略评分、softmax 采样、KV 容量感知路由 |
| hier-kv-gateway-kv-estimate | 77 | KV 解析公式（MHA/GQA/MLA/滑动窗口）、catalog 匹配、零分配热路径 |
| hier-kv-gateway-cluster | 45 | Gossip 成员管理、CKF Relay 发布 |
| hier-kv-gateway-connector | 32 | Prometheus 指标解析、连接器注册 |
| hier-kv-gateway-api | 41 | HTTP 路由、SSE 转换、响应头 |
| hier-kv-gateway (主二进制) | 13 | 路由引擎构建、KV 容量插件挂载 |
| **集成测试** | **36** | **真实组件端到端** |
| **合计** | **430** | |

集成测试包括：
- **radix_tree_kv_events**: 真实 RadixTree 的 KV event 生命周期
- **ckf_producer_consumer**: 真实 CKF Producer/Consumer 的跨域投影
- **hybrid_routing**: 真实混合路由策略的评分与选路
- **gossip_cluster**: 真实 Gossip 成员管理
- **end_to_end**: 启动真实 axum HTTP 后端 → discover → route → forward 全链路

## 设计文档

- [架构设计](docs/01-architecture.md)
- [路由算法设计](docs/02-routing-algorithms.md)（含 KV 容量感知路由 §9）
- [接口与数据模型设计](docs/03-interfaces-data-models.md)
- [KV 显存估算模块架构](docs/05-kv-estimation.md)
- [KV 估算 Benchmark 报告](docs/benchmarks/kv-estimation.md)
- [开发会话日志](docs/session-logs/)（含 [KV 估算 session log](docs/session-logs/2026-08-06-kv-estimation.md)）

## 技术栈

- **语言**: Rust (性能、内存安全)
- **异步运行时**: tokio
- **HTTP 框架**: axum
- **HTTP 客户端**: reqwest
- **序列化**: serde / serde_json / toml
- **哈希**: xxhash-rust (xxh3)
- **并发**: dashmap, arc-swap, parking_lot

## License

Apache-2.0
