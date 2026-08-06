# Session Log: KV 显存估算 · 容量感知路由

> 开发会话日志 · 2026-08-06

## 概述

| 项 | 内容 |
|----|------|
| **主题** | KV Cache 显存估算独立模块 + KvCapacityStrategy 容量感知路由策略 |
| **类型** | 调研 + 设计 + 实施 + 测试 + Benchmark + 文档 |
| **结论** | 独立叶子 crate 实现解析公式估算（零分配热路径）；KvCapacityStrategy 作为插件接入 Hybrid；测试全绿、benchmark 含反作弊断言 |
| **判据** | 见各节 Benchmark 结果与零分配测试；所有测试 0 失败 |

---

## 1. 问题分析与任务分解

用户需求：对一次推理所需的 KV 显存大小进行估算（参考 Mooncake 等开源实现），若网关能获取各集群资源情况则据此判断路由。具体要求：

1. 先梳理现有架构、接口与测试体系
2. KV 估算查询各种开源项目
3. KV 估算是**独立模块**，性能足够好（本地计算），提供**插件机制**扩展新模型，**预制主流模型**；用 head 数/层数等结合 batch size / input length 计算，**不是仿真**
4. 逐步添加测试与 benchmark 并调优，**测试不能伪造与作弊**
5. 维护文档体系，新增架构/方案/session log 文档
6. 更新 README，逐步 commit

---

## 2. 开源系统调研

| 项目 | KV 大小机制 | 可借鉴点 |
|------|------------|----------|
| **vLLM** | `cache_block_size = 2 * num_layers * num_kv_heads * head_size * block_size * dtype_size`（`vllm/worker/worker.py`） | Standard 注意力 per-block 公式；block 分页 |
| **SGLang** | `model_config.py: get_kv_cache_bytes` | 与 vLLM 一致的 Standard 公式 |
| **Mooncake** | 分页 KV block 分配，按 block_size 粒度管理 | block 分页语义；网关层 KV 感知 |
| **DeepSeek-V2/V3** | MLA：单潜向量 `c_kv`（`kv_lora_rank`）+ RoPE 解耦 `k_pe`（`qk_rope_head_dim`），无因子 2（论文 §3.1） | MLA 公式；~57× 压缩 |
| **llm-d** | KV block 总量上报 + 调度器按 block 准入 | 后端上报 `kv_total_blocks`/`kv_used_blocks` 的精确路径 |

**选型决策**：解析公式（非仿真），与 vLLM/SGLang/Mooncake 一致。覆盖四族注意力：MHA / GQA / MQA（统一 Standard 公式，仅 `num_kv_heads` 参数不同）/ MLA（DeepSeek 独立公式）/ 滑动窗口（截断有效缓存长度）/ block 分页（向上取整）。

**诚实声明**：本模块算的是「从零缓存到完整长度」的占用上界，用于容量准入。**不**预测前缀命中后的增量占用（那是 `KvAwareStrategy` 的领域），**不**模拟调度器行为（不预估 batch 内 token 复用/抢占/换出）。

---

## 3. 架构设计

### 3.1 独立叶子 crate

`hier-kv-gateway-kv-estimate` 不依赖任何 gateway 类型，只依赖 `serde`。可被任意推理路由器/调度器/容量规划工具复用。热路径零分配、纳秒级。

```
┌──────────────────────────────────────────────────────────┐
│                  KvEstimationRegistry                     │
│   (builtin StandardEstimator + custom specs + plugins)   │
│   spec_for(model) ──► 按注册顺序首个识别 model 的 estimator │
│   estimate(model, input) ──► 该 estimator 的公式          │
└──────────────────────────────────────────────────────────┘
        ▲                              ▲
  ┌─────┴──────┐              ┌────────┴────────┐
  │  Standard  │              │  用户插件        │
  │ Estimator  │              │ KvEstimator impl│
  └─────┬──────┘              └─────────────────┘
  ┌─────┴──────────────────────┐
  │  SpecCatalog               │
  │  custom specs (TOML)       │
  │  + builtin pattern table   │
  └────────────────────────────┘
```

### 3.2 插件化扩展（两条路径）

1. **加模型 spec（数据）**：`[[kv_estimate.models]]` TOML 一行，字段对应 HuggingFace `config.json`。Standard 公式覆盖。绝大多数新模型走这条路。
2. **加自定义 estimator（代码）**：实现 `KvEstimator` trait，经 `with_estimator` / `with_estimator_front` 注册。用于标准公式无法表达的架构（Cross-Attention 额外缓存、Mamba/SSM 状态）。

镜像 vLLM：标准模型用 `config.json` 参数化，异构情况由代码覆盖。

### 3.3 与路由集成

- **数据半**（`KvEstimateConfig`、spec catalog）在叶子 crate，经 `GatewayConfig::kv_estimate` 暴露。
- **行为半**（`KvCapacityStrategy`）在路由 crate，`kv_estimate.enabled=true` 时作为 `RoutingPlugin` 挂到 Hybrid。

---

## 4. 接口设计

### 4.1 ModelSpec（Copy 值类型）

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub attention: AttentionKind,   // Standard | Mla
    pub dtype: KvDtype,             // Fp32|Fp16|Bf16|Fp8|Int8
    pub kv_lora_rank: u32,          // MLA only
    pub qk_rope_head_dim: u32,      // MLA only
    pub sliding_window: u32,        // 0 = none
}
```

模型名**不**在 `ModelSpec` 内（作为 catalog key 单独存放）—— 这是热路径能 `Copy`、零分配的关键。

### 4.2 KvEstimator trait

```rust
pub trait KvEstimator: Send + Sync {
    fn name(&self) -> &str;
    fn spec_for(&self, model: &str) -> Option<ModelSpec>;
    fn estimate(&self, spec: &ModelSpec, input: &EstimateInput) -> KvEstimate;
}
```

### 4.3 解析公式

```rust
// Standard (MHA/GQA/MQA): per_token = 2 * layers * kv_heads * head_dim * dtype
// MLA (DeepSeek):          per_token = layers * (kv_lora_rank + qk_rope_head_dim) * dtype  // 无因子 2
// 滑动窗口: effective = min(seq_len, sliding_window)
// block 分页: blocks = ceil(effective / block_size) * batch; bytes = per_token * batch * (blocks * block_size)
pub fn estimate_kv(spec: &ModelSpec, input: &EstimateInput) -> KvEstimate;
```

### 4.4 配置

```toml
[kv_estimate]
enabled = true
weight = 0.20
gpu_mem_safety_fraction = 0.5
exclude_on_unknown_spec = false

[[kv_estimate.models]]
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

---

## 5. 实施详情

### 5.1 新增文件

| 文件 | 用途 |
|------|------|
| `crates/hier-kv-gateway-kv-estimate/src/spec.rs` | `ModelSpec` / `AttentionKind` / `KvDtype` / `NamedModelSpec` |
| `crates/hier-kv-gateway-kv-estimate/src/catalog.rs` | 内置模型表 + 零分配 `contains_ascii_ci` 匹配 |
| `crates/hier-kv-gateway-kv-estimate/src/estimate.rs` | 解析公式 `estimate_kv` 等 |
| `crates/hier-kv-gateway-kv-estimate/src/plugin.rs` | `KvEstimator` trait / `StandardEstimator` / `SpecCatalog` |
| `crates/hier-kv-gateway-kv-estimate/src/registry.rs` | `KvEstimationRegistry` 复合 estimator |
| `crates/hier-kv-gateway-kv-estimate/src/config.rs` | `KvEstimateConfig` TOML 段 |
| `crates/hier-kv-gateway-kv-estimate/src/lib.rs` | crate 入口 + 架构总览 |
| `crates/hier-kv-gateway-kv-estimate/tests/alloc_free.rs` | 计数分配器零分配证明 |
| `crates/hier-kv-gateway-kv-estimate/benches/kv_estimate.rs` | 估算热路径 benchmark（反作弊） |
| `crates/hier-kv-gateway-routing/src/kv_capacity.rs` | `KvCapacityStrategy` |
| `crates/hier-kv-gateway-routing/benches/kv_capacity.rs` | 容量感知策略 benchmark（反作弊） |

### 5.2 修改文件

| 文件 | 变更 |
|------|------|
| `crates/hier-kv-gateway-core/src/config.rs` | `GatewayConfig` 加 `kv_estimate: KvEstimateConfig`（re-export） |
| `crates/hier-kv-gateway-routing/src/lib.rs` | 导出 `kv_capacity` 模块 |
| `crates/hier-kv-gateway-routing/Cargo.toml` | 依赖 `hier-kv-gateway-kv-estimate` + `[[bench]] kv_capacity` |
| `crates/hier-kv-gateway/src/main.rs` | `build_routing_engine` 接入 kv_capacity 插件；新增插件挂载测试 |
| `crates/hier-kv-gateway/Cargo.toml` | 依赖 `hier-kv-gateway-kv-estimate` |
| `examples/multi-backend.toml` | 新增 `[kv_estimate]` 示例段 |
| `Cargo.toml` (workspace) | 注册 `hier-kv-gateway-kv-estimate` 成员 |

### 5.3 关键设计决策

1. **`ModelSpec` 是 `Copy`，无 `String`**：模型名留作 catalog key。`spec_for` 返回 `Copy`、热路径零分配。早期实现曾把 `name: String` 放进 `ModelSpec` 导致 `Copy` 派生失败 —— 拆出 `NamedModelSpec` 解决。

2. **`contains_ascii_ci` 零分配匹配**：不调用 `to_ascii_lowercase()`（每次查找分配 `String`），逐字节在线小写比较。代价是 builtin miss 需扫描全表（~457 ns），但属冷路径。

3. **`f64::INFINITY` 而非 `f64::MAX`**：排除用 `∞`（非有限），由 `HybridStrategy::normalize_costs` 通过 `!is_finite()` 识别。`f64::MAX` 有限，会被误判为「很贵但有效」。

4. **GPU 显存 fallback 安全比例**：后端只报 GPU 显存时，仅「空闲显存 × `gpu_mem_safety_fraction`」可被 KV 声明（KV 非唯一 GPU 内存消费者）。

5. **未知 spec 默认中立**：`exclude_on_unknown_spec=false` 让未知模型后端交由其他子策略决定，避免没把握时饿死确有余量的后端。

6. **`div_ceil` 用 std 实现**：早期 `(a + b - 1) / b` 在 `a` 接近 `u64::MAX` 时溢出；改用 `a.div_ceil(b)`（const-stable，不溢出），消除 clippy `manual_div_ceil` 警告。

7. **`score_capacity` 字节级统一判断**：KV-block 路径与 GPU 显存路径都用 `est.bytes > available_bytes` 判断排除，避免 block 级 vs 字节级两种判断的不一致。KV-block 路径下等价于 `est.blocks > free_blocks`。

---

## 6. Benchmark 结果

### 6.1 估算热路径（叶子 crate）

`cargo bench -p hier-kv-gateway-kv-estimate --bench kv_estimate`

#### 解析公式 `estimate_kv`

| spec | 延迟 |
|------|------|
| llama3_8b_gqa (GQA) | 9.92 ns |
| deepseek_v3_mla (MLA) | 9.97 ns |
| mistral_7b_sliding (GQA + 滑动窗口) | 9.91 ns |

输入长度无关（512 / 4096 / 32_768 / 131_072 均 ~9.9 ns）—— 公式是常数次整数乘加。`per_token_bytes` 1.18 ns。

#### 完整热路径 `registry.estimate`（名称 → spec → 公式）

| 模型 | 延迟 |
|------|------|
| Qwen2.5-7B | 44.8 ns |
| mistral-7b | 51.2 ns |
| Llama-3-8B | 59.3 ns |
| deepseek-v3 | 91.3 ns |

差异来自 pattern 表扫描位置。

#### spec 查找 / 自定义 catalog

| 场景 | 延迟 |
|------|------|
| `spec_for` builtin_hit | 45.2 ns |
| `spec_for` builtin_miss | 457.2 ns |
| custom_hit (HashMap) | 28.3 ns |
| builtin_fallback_through_custom | 56.3 ns |

**结论**：完整热路径 **45–91 ns**，相对推理延迟（100ms+）可忽略；公式本身 ~10 ns；零分配由 `alloc_free` 测试证明。

### 6.2 容量感知策略（路由 crate）

`cargo bench -p hier-kv-gateway-routing --bench kv_capacity`

#### 独立策略评估

| 后端数 | 延迟 |
|--------|------|
| 2 | 557 ns |
| 10 | 2.57 µs |
| 50 | 13.04 µs |

#### Hybrid 端到端开销（baseline vs with_kv_capacity）

| 后端数 | Baseline | With kv_capacity | 绝对开销 | 相对开销 |
|--------|----------|------------------|---------|---------|
| 2 | 1.85 µs | 2.70 µs | ~0.85 µs | ~46% |
| 10 | 9.06 µs | 12.91 µs | ~3.85 µs | ~42% |
| 20 | 18.10 µs | 26.09 µs | ~7.99 µs | ~44% |

**结论**：插件绝对开销 **~0.85–8 µs**，相对推理延迟可忽略；与既有 cost/tier 插件开销量级一致。

**反作弊**：估算 bench 每次迭代 `assert_eq!(r.bytes, expected)`（手算期望值）；kv_capacity bench 每次迭代断言评分非空、`raw_cost.is_finite()`、`score ∈ (0,1]`；`alloc_free` 计数分配器零容忍断言热路径 0 字节。

---

## 7. 测试覆盖

### 7.1 单元测试

| 模块 | 测试数 | 覆盖点 |
|------|--------|--------|
| `estimate` | 19 | MHA/GQA/MLA/滑动窗口/block padding/batch 缩放/dtype/饱和加/MB 转换 |
| `spec` | 9 | dtype bytes、serde snake_case、standard/mla builder、Copy、TOML 往返、flattened named spec |
| `catalog` | 17 | 四族覆盖、大小写不敏感匹配、specific-before-generic、索引越界、`contains_ascii_ci` 边界、独立 copy |
| `plugin` | 10 | builtin 解析、自定义覆盖/新增、`from_specs` builder、insert 替换、clone 共享 Arc |
| `registry` | 8 | 解析顺序、前置覆盖、自定义公式被尊重、不遮蔽 builtin 其他模型、clone 共享 |
| `config` | 7 | 默认关闭、TOML 解析、MLA entry、catalog 分层、custom_specs 迭代、往返 |
| `kv_capacity` | 12 | 余量多者得分高、超容排除、exact fit 准入、未知 spec 中立/排除、无指标中立、GPU fallback 准入+排除、自定义 catalog、单调性、KV-block 溢出排除 |

### 7.2 零分配测试

`tests/alloc_free.rs`：1 个 `#[test]`，9 个测量窗口（`estimate_kv` / `per_token_bytes` / `registry.spec_for` 命中+未命中 / `registry.estimate` 完整热路径 / 跨注意力族轮换 / `StandardEstimator.spec_for` / 自定义 catalog 命中+fallback），各 10 000 次断言 0 字节分配。

### 7.3 全量测试结果

```
hier-kv-gateway-kv-estimate: 77 passed (75 unit + 1 alloc_free + 1 doctest), 0 failed
hier-kv-gateway-routing:     79 passed (78 unit + 1 doctest), 0 failed
hier-kv-gateway:             13 passed (含 multi_backend_example_attaches_kv_capacity_plugin / disabled_kv_estimate_attaches_no_plugin), 0 failed
hier-kv-gateway-integration: 36 passed, 0 failed
全 workspace 合计: 430 passed, 0 failed
```

clippy：仅 `degradation.rs` / `engine.rs` 既有警告（与本工作无关）；kv-estimate crate 与 `kv_capacity.rs` 零警告。

---

## 8. 文档产出

| 文件 | 类型 |
|------|------|
| `docs/05-kv-estimation.md` (+ `en/05-kv-estimation.md`) | 架构文档（新增） |
| `docs/02-routing-algorithms.md` §9 (+ en mirror) | 方案文档（容量感知路由策略） |
| `docs/01-architecture.md` (+ en mirror) | 目录结构与路由层图更新 |
| `docs/benchmarks/kv-estimation.md` | Benchmark 报告（新增） |
| `docs/session-logs/2026-08-06-kv-estimation.md` | 本 session log |
| `README.md` (+ `README.en.md`) | 新增 KV 估算章节、测试数更新 |

---

## 9. 未来工作

1. **更多模型 spec**：Gemma-3、Phi-3、Command-R、DBRX 等，按需追加 builtin 或走 TOML。
2. **后端真实 KV 余量上报**：当前依赖 `BackendMetrics::kv_total_blocks`/`kv_used_blocks`；推动 connector 从 vLLM/SGLang `/metrics` 拉取真实 block 占用，替代 GPU 显存 fallback。
3. **prefix-aware 增量估算**：结合 `KvAwareStrategy` 的 overlap，估算「扣掉命中前缀后的增量 KV」，做更精确的 admission。
4. **MLA 吸收成本（absorbed）模式**：DeepSeek V2.5 的吸收态进一步压缩 KV，可作为自定义 estimator 实现。
5. **估算结果缓存**：同一 (model, input_shape) 的估算可短 TTL 缓存（公式本身已 ~10 ns，缓存收益有限，仅在超大规模候选数时考虑）。

---

## 10. 文件索引

| 文件 | 说明 |
|------|------|
| [lib.rs](../../crates/hier-kv-gateway-kv-estimate/src/lib.rs) | crate 入口 + 架构总览 |
| [spec.rs](../../crates/hier-kv-gateway-kv-estimate/src/spec.rs) | `ModelSpec` 值类型 |
| [catalog.rs](../../crates/hier-kv-gateway-kv-estimate/src/catalog.rs) | 内置模型表 + 零分配匹配 |
| [estimate.rs](../../crates/hier-kv-gateway-kv-estimate/src/estimate.rs) | 解析公式 |
| [plugin.rs](../../crates/hier-kv-gateway-kv-estimate/src/plugin.rs) | `KvEstimator` trait + `StandardEstimator` |
| [registry.rs](../../crates/hier-kv-gateway-kv-estimate/src/registry.rs) | `KvEstimationRegistry` |
| [config.rs](../../crates/hier-kv-gateway-kv-estimate/src/config.rs) | `[kv_estimate]` TOML 段 |
| [alloc_free.rs](../../crates/hier-kv-gateway-kv-estimate/tests/alloc_free.rs) | 零分配证明 |
| [kv_estimate.rs (bench)](../../crates/hier-kv-gateway-kv-estimate/benches/kv_estimate.rs) | 估算热路径 benchmark |
| [kv_capacity.rs](../../crates/hier-kv-gateway-routing/src/kv_capacity.rs) | `KvCapacityStrategy` |
| [kv_capacity.rs (bench)](../../crates/hier-kv-gateway-routing/benches/kv_capacity.rs) | 容量感知策略 benchmark |
| [05-kv-estimation.md](../05-kv-estimation.md) | 估算模块架构文档 |
| [02-routing-algorithms.md §9](../02-routing-algorithms.md) | 容量感知路由方案文档 |
| [benchmarks/kv-estimation.md](../benchmarks/kv-estimation.md) | Benchmark 报告 |
