# Session Log: 成本模型 · 请求去重 · 大小模型分层 · 插件架构

> 开发会话日志 · 2026-08-05

## 概述

| 项 | 内容 |
|----|------|
| **主题** | Token Gateway 增强：成本感知路由 + 并发请求去重 + 大小模型协调 + 插件化路由扩展 |
| **类型** | 调研 + 设计 + 实施 + Benchmark |
| **结论** | 三项能力全部实现并通过测试；插件机制统一了路由策略扩展入口；Benchmark 含反作弊断言 |
| **判据** | 见各节 Benchmark 结果；所有测试 0 失败 |

---

## 1. 问题分析与任务分解

用户提出三项需求：

1. **成本模型策略**：调研主流 Token Gateway，通过插件与接口形式引入成本感知路由
2. **并发请求去重**：短时间内并发相同请求的处理方案（调研、接口设计、测试、Benchmark）
3. **大小模型协调**：调研主流方案，引入大小模型无感切换能力，设计插件机制，扩展路由策略

共同要求：详细方案调研 → 架构/接口设计 → 充足测试与 Benchmark（不可作弊）→ 实施

---

## 2. 开源系统调研

### 2.1 成本感知路由

| 项目 | 机制 | 可借鉴点 |
|------|------|----------|
| **LiteLLM** | `cost-based-routing`：从 `model_prices_and_context_window.json` 读取静态价格表，选择最便宜的可用模型 | 静态价格表格式、`cost` 路由模式 |
| **Langfuse** | `/pricing` API：价格数据与消费者解耦（gen-ai tracing、dashboard 各自消费） | 数据/消费者分离的 trait 设计 |
| **OpenRouter** | 模型按 price × capability × latency 排序，暴露 `ranking` 字段 | per-token 价格标准化 |

**选型决策**：`CostModel` trait + `StaticCostModel` 实现，分离价格数据与路由逻辑。未来可通过实现 trait 接入 OpenRouter/Langfuse HTTP API。

### 2.2 并发请求去重（Single-Flight）

| 项目 | 机制 | 适用层 |
|------|------|--------|
| **Go `singleflight`** | `Group.Do(key, fn)`：相同 key 的并发调用只执行一次 `fn`，其他等待者共享结果 | 通用并发去重 |
| **nginx `proxy_cache_lock`** | `on` 时，相同 cache key 的并发请求只转发一个，其他等待 | HTTP 代理层 |
| **Envoy `request mirroring`** | 请求镜像 + cache 合并 | 服务网格层 |
| **LiteLLM `cache`** | Redis/内存缓存：按 request hash 去重，但是**持久化缓存**而非在途去重 | LLM 网关层 |
| **vLLM scheduler** | 连续批处理中合并相同 sequence（`--enable-prefix-caching` + sequence dedup） | 推理引擎层 |

**选型决策**：Go `singleflight` 模式 — 在途请求去重（非持久化缓存）。使用 Rust `futures::future::Shared` 实现共享 future，`DashMap` 管理在途条目。仅对非流式请求生效（流式无法透明共享）。

### 2.3 大小模型协调

| 项目 | 机制 | 类型 |
|------|------|------|
| **LiteLLM `fallbacks`** | 转发层链式降级：试模型组 A，失败则回退到 B | 转发层 |
| **LiteLLM `cost-based-routing`** | 路由层：从模型组中选择最便宜的可用模型 | 路由层 |
| **Portkey / Helicone** | 条件路由规则引擎：if prompt_len > N or tools present → 路由到模型 B | 路由层 |
| **OpenRouter** | 模型按 price × capability 排序，客户端选择 tier | 客户端选择 |
| **vLLM / SGLang** | 单模型服务器，分层是网关职责 | 引擎层 |

**选型决策**：两种策略，一个 trait：
- `Pick`（复杂度感知评分）：软子策略，作为混合策略的插件。短 prompt + 无 tools → 偏好小模型；长 prompt 或 tools → 偏好大模型
- `Fallback`（无条件小优先排序）：主策略，引擎的候选列表变为"先小后大"，转发循环的重试逻辑自动实现降级链

**诚实声明**：基于响应质量的"无缝"降级（如小模型回答置信度低则切换大模型）需要评估响应内容，超出路由层策略的范围。本次实现路由时分层（复杂度感知 Pick + 排序 Fallback）；响应质量驱动的降级列为未来工作。

---

## 3. 架构设计

### 3.1 插件化路由架构

```
                    ┌──────────────────────────────────────────┐
                    │           RoutingEngine                   │
                    │  ┌────────────┐  ┌────────────────────┐ │
                    │  │  Primary   │  │   HybridStrategy    │ │
  config.strategy → │  │  Strategy  │  │  ┌─────┐ ┌───────┐ │ │
                    │  │  (optional)│  │  │ KV  │ │ Model │ │ │
                    │  └────────────┘  │  ├─────┤ ├───────┤ │ │
                    │                  │  │Load │ │Topo   │ │ │
                    │                  │  ├─────┤ ├───────┤ │ │
                    │                  │  │Plugin│ │Plugin │ │ │
                    │                  │  │(cost)│ │(tier) │ │ │
                    │                  │  └─────┘ └───────┘ │ │
                    │                  └────────────────────┘ │
                    └──────────────────────────────────────────┘
```

**插件机制设计**：
- `RoutingPlugin`：包装 `Arc<dyn RoutingStrategy>` + 元数据（id）
- `HybridStrategy::with_plugin(plugin)`：注册插件子策略
- 插件权重来自 `RoutingStrategy::weight()`，参与混合策略的统一归一化
- 插件的 `is_available()` / `evaluate()` 与内置子策略走相同路径

**两种路由通过插件扩展**：
1. **成本感知路由**：`CostAwareStrategy` 作为插件注入混合策略，权重来自 `[cost] weight`
2. **大小模型分层路由**（Pick 策略）：`ModelTierStrategy` 作为插件注入混合策略，权重来自 `[model_tier] weight`
3. **大小模型降级路由**（Fallback 策略）：`ModelTierStrategy` 作为主策略，通过 `with_primary_strategy` 安装，候选列表"先小后大"

### 3.2 请求去重架构

```
  Client A ─┐
  Client B ─┼─→ [RequestCoalescer] ──→ Leader: route + forward + aggregate
  Client C ─┘         │                   │
                      │                   ▼
                      │              CoalescedResponse (Arc<[u8]>)
                      │                   │
                      └───────────────────┼──→ A: clone
                                          ├──→ B: clone
                                          └──→ C: clone
```

- 非流式请求经过 `RequestCoalescer`
- 第一个请求（Leader）执行完整 route → forward → aggregate 管线
- 并发等待者（Waiter）附加到 Leader 的 `Shared` future，获得响应克隆
- `ttl_ms` 控制完成后缓存窗口：窗口内的迟到请求仍命中缓存
- 流式请求**始终绕过**去重（SSE 流无法透明共享）

### 3.3 配置结构

```toml
[cost]
enabled = true
weight = 0.15
output_cost_scale = 1.0
exclude_on_unknown_price = false
[[cost.prices]]
model = "qwen2.5-7b"
input_per_1m = 0.15
output_per_1m = 0.60

[model_tier]
enabled = true
weight = 0.20
[model_tier.policy]
type = "pick"  # 或 "fallback"
prompt_token_threshold = 2048
max_token_threshold = 1024
prefer_large_for_tools = true
[[model_tier.tiers]]
model = "qwen2.5-7b"
tier = "small"
[[model_tier.tiers]]
model = "qwen2.5-72b"
tier = "large"

[coalescing]
enabled = true
ttl_ms = 50
max_inflight = 1024
```

---

## 4. 接口设计

### 4.1 CostModel trait

```rust
/// 价格目录 + 成本投影。可由静态 TOML 表或未来的 HTTP API 实现。
pub trait CostModel: Send + Sync {
    fn price_for(&self, model: &str) -> Option<ModelPrice>;
    fn projected_cost(&self, model: &str, prompt_tokens: u32, est_output_tokens: u32) -> Option<f64>;
}
```

### 4.2 RoutingPlugin

```rust
/// 路由插件：带标签的、加权的、可选启用的子策略。
pub struct RoutingPlugin {
    pub strategy: Arc<dyn RoutingStrategy>,
    pub id: &'static str,
}

impl RoutingPlugin {
    pub fn from_strategy(strategy: Arc<dyn RoutingStrategy>) -> Self;
    pub fn with_id(strategy: Arc<dyn RoutingStrategy>, id: &'static str) -> Self;
}
```

### 4.3 RequestCoalescer

```rust
/// 非流式 chat completions 的 single-flight 去重器。
pub struct RequestCoalescer {
    inner: Arc<RequestCoalescerInner>,  // DashMap + config + stats
}

impl RequestCoalescer {
    pub fn new(cfg: CoalescingConfig) -> Self;
    pub fn enabled(&self) -> bool;
    pub fn coalesce<F, Fut>(&self, key: u64, produce: F) -> Result<CoalescedResponse, CoalesceError>;
    pub fn stats(&self) -> &CoalesceStats;  // leaders / waiters / bypassed / forwards_saved
}

pub fn request_key(req: &OpenAIChatRequest) -> u64;  // 语义 hash
```

### 4.4 TierRoutingPolicy

```rust
pub enum TierRoutingPolicy {
    Pick { prompt_token_threshold: u32, max_token_threshold: u32, prefer_large_for_tools: bool },
    Fallback,
}
```

---

## 5. 实施详情

### 5.1 新增文件

| 文件 | 用途 |
|------|------|
| `crates/hier-kv-gateway-core/src/cost.rs` | `CostModel` trait + `StaticCostModel` + `CostConfig` |
| `crates/hier-kv-gateway-core/src/coalescing.rs` | `CoalescingConfig` |
| `crates/hier-kv-gateway-core/src/model_tier.rs` | `ModelTierConfig` + `TierRoutingPolicy` + `ModelTier` |
| `crates/hier-kv-gateway-routing/src/cost_aware.rs` | `CostAwareStrategy` |
| `crates/hier-kv-gateway-routing/src/model_tier.rs` | `ModelTierStrategy` |
| `crates/hier-kv-gateway-routing/src/plugin.rs` | `RoutingPlugin` |
| `crates/hier-kv-gateway-api/src/coalescer.rs` | `RequestCoalescer` |

### 5.2 修改文件

| 文件 | 变更 |
|------|------|
| `crates/hier-kv-gateway-routing/src/hybrid.rs` | 添加 `with_plugin()` / `plugins` 字段 / 插件权重归一化 |
| `crates/hier-kv-gateway-routing/src/lib.rs` | 导出 `cost_aware` / `model_tier` / `plugin` 模块 |
| `crates/hier-kv-gateway-api/src/handlers.rs` | 集成 coalescer 到 `chat_completions`；`AppState` 添加 `coalescer` 字段 |
| `crates/hier-kv-gateway/src/main.rs` | `build_routing_engine` 接入 cost/tier 插件；`AppState` 初始化 coalescer |
| `crates/hier-kv-gateway-core/src/config.rs` | `GatewayConfig` 添加 `cost` / `model_tier` / `coalescing` 字段 |
| `crates/hier-kv-gateway-core/src/config.rs` | `StrategyWeights` 添加 `cost` 字段 |

### 5.3 关键设计决策

1. **`f64::INFINITY` 而非 `f64::MAX`**：排除策略使用 `f64::INFINITY`（非有限），因为 `HybridStrategy::normalize_costs` 通过 `!is_finite()` 识别排除候选。`f64::MAX` 是有限的，会被误认为"非常贵但有效"。

2. **Fallback 策略作为主策略而非插件**：`TierRoutingPolicy::Fallback` 通过 `with_primary_strategy` 安装，使引擎的候选列表直接变为"先小后大"排序。转发循环的现有重试逻辑自动实现降级链 — 无需新增重试代码。这镜像了 LiteLLM 的 `fallbacks` 特性，但在路由层实现而非转发层。

3. **`resolve_tier` 的"实际服务"守卫**：解析后端 tier 时，只有当后端**实际服务**了所请求的模型时，才使用该模型的 tier。没有这个守卫，每个候选都会继承请求模型的 tier，使小/大区分失效。

4. **流式请求绕过去重**：SSE 流无法透明共享（需要缓冲+重放整个流给迟到的等待者，改变尾延迟且有内存风险）。非流式请求可以安全共享已序列化的 JSON 响应。

5. **`forwards_saved` 反作弊计数器**：每个 `coalesce_concurrent` benchmark 迭代断言 `forwards_saved == N - 1`。如果去重器损坏（每个请求独立转发），`forwards_saved == 0`，benchmark 会 panic，数字无法被伪造。

---

## 6. Benchmark 结果

### 6.1 成本感知 + 大小模型分层插件

**运行命令**：`cargo bench -p hier-kv-gateway-routing --bench cost_tier_plugins`

#### 独立策略评估延迟

| 后端数 | cost_aware_evaluate | model_tier_evaluate |
|--------|---------------------|---------------------|
| 2 | 407 ns | 333 ns |
| 10 | 1.82 µs | 1.28 µs |
| 50 | 8.61 µs | 6.89 µs |

#### 混合策略插件开销（baseline vs with_plugins）

| 后端数 | Baseline | With Plugins | 绝对开销 | 相对开销 |
|--------|----------|--------------|----------|----------|
| 2 | 1.51 µs | 2.54 µs | ~1.0 µs | ~67% |
| 10 | 5.44 µs | 8.89 µs | ~3.5 µs | ~63% |
| 20 | 11.12 µs | 16.62 µs | ~5.5 µs | ~49% |

**结论**：两个插件子策略的绝对开销为 ~1–5.5 µs，相对于 LLM 推理延迟（100ms+）可忽略不计。百分比开销较高是因为每次 evaluate 新增两次策略调用 + 两次 `normalize_costs` HashMap 分配，但绝对值在可接受范围内。

### 6.2 请求去重（Single-Flight Coalescer）

**运行命令**：`cargo bench -p hier-kv-gateway-api --bench request_coalescer`

#### 相同请求并发去重（50ms 模拟转发）

| 并发数 | 总延迟 | 预期（无去重） | 加速比 |
|--------|--------|----------------|--------|
| 2 | 51.7 ms | ~100 ms | ~1.9× |
| 8 | 52.0 ms | ~400 ms | ~7.7× |
| 32 | 51.8 ms | ~1600 ms | ~30.9× |

**反作弊断言**：每次迭代断言 `forward_calls == 1` 且 `forwards_saved == N - 1`。全部通过。

#### 不同请求并发（控制组，5ms 模拟转发）

| 并发数 | 总延迟 | forwards_saved |
|--------|--------|----------------|
| 2 | 6.75 ms | 0 |
| 8 | 6.47 ms | 0 |
| 32 | 7.08 ms | 0 |

**结论**：N 个并发相同请求在 ~1× 转发延迟内完成（而非 N×）。不同请求不被去重，各自独立转发。

#### 请求 Key 计算开销

| 消息数 | 延迟 |
|--------|------|
| 1 | 360 ns |
| 4 | 739 ns |
| 16 | 2.16 µs |

**结论**：Key 计算（canonical JSON 序列化 + hash）在亚微秒到微秒级，相对于转发延迟可忽略。

---

## 7. 测试覆盖

### 7.1 单元测试

| 模块 | 测试数 | 覆盖点 |
|------|--------|--------|
| `cost_aware` | 5 | 便宜模型评分更高、未知价格排除/中立、output_cost_scale 放大、配置构建 |
| `model_tier` | 9 | Pick 简单/复杂/tools 偏好、未知 tier 中立、请求模型 tier 优先、Fallback 排序、可用性/权重 |
| `coalescer` | 7 | 并发相同请求一次转发、不同 key 独立、TTL 窗口缓存、容量上限绕过、错误传播、key 排除 request_id |
| `plugin` | 2 | from_strategy 派生 id、with_id 覆盖 |
| `cost` (core) | 5 | 投影成本计算、未知价格 None、配置解析 |
| `coalescing` (core) | 3 | 默认关闭、显式值解析、缺省 section |
| `model_tier` (core) | 5 | 默认关闭、Pick/Fallback 解析、缺省 section、阈值默认 |

### 7.2 全量测试结果

```
hier-kv-gateway-core:       77 passed, 0 failed
hier-kv-gateway-routing:    66 passed, 0 failed
hier-kv-gateway-api:        41 passed, 0 failed
hier-kv-gateway:            11 passed, 0 failed
hier-kv-gateway-integration: all passed
```

---

## 8. 未来工作

1. **响应质量驱动的降级**：小模型回答置信度低时自动切换大模型。需要响应评估层（超出路由层范围）。
2. **HTTP-fetched 价格目录**：实现 `CostModel` trait 的 OpenRouter/Langfuse HTTP API 版本，支持动态价格更新。
3. **流式请求部分去重**：通过缓冲+重放实现 SSE 流共享（需要内存限制和尾延迟权衡）。
4. **更多插件路由策略**：延迟 SLO 感知、canary pinning、A/B 测试路由等，均可通过 `RoutingPlugin` 接入。
5. **跨实例去重**：当前去重是单实例内的；跨 gateway 实例的去重需要分布式缓存（如 Redis）。

---

## 9. 文件索引

| 文件 | 说明 |
|------|------|
| [cost.rs](file:///workspace/crates/hier-kv-gateway-core/src/cost.rs) | CostModel trait + StaticCostModel + CostConfig |
| [coalescing.rs](file:///workspace/crates/hier-kv-gateway-core/src/coalescing.rs) | CoalescingConfig |
| [model_tier.rs](file:///workspace/crates/hier-kv-gateway-core/src/model_tier.rs) | ModelTierConfig + TierRoutingPolicy |
| [cost_aware.rs](file:///workspace/crates/hier-kv-gateway-routing/src/cost_aware.rs) | CostAwareStrategy |
| [model_tier.rs](file:///workspace/crates/hier-kv-gateway-routing/src/model_tier.rs) | ModelTierStrategy |
| [plugin.rs](file:///workspace/crates/hier-kv-gateway-routing/src/plugin.rs) | RoutingPlugin |
| [hybrid.rs](file:///workspace/crates/hier-kv-gateway-routing/src/hybrid.rs) | HybridStrategy (with_plugin) |
| [coalescer.rs](file:///workspace/crates/hier-kv-gateway-api/src/coalescer.rs) | RequestCoalescer |
| [handlers.rs](file:///workspace/crates/hier-kv-gateway-api/src/handlers.rs) | chat_completions (coalescer 集成) |
| [main.rs](file:///workspace/crates/hier-kv-gateway/src/main.rs) | build_routing_engine (插件接入) |
| [cost_tier_plugins.rs](file:///workspace/crates/hier-kv-gateway-routing/benches/cost_tier_plugins.rs) | 成本/分层插件 Benchmark |
| [request_coalescer.rs](file:///workspace/crates/hier-kv-gateway-api/benches/request_coalescer.rs) | 去重器 Benchmark |
