# KV 显存估算模块架构设计

> 中文 | [English](en/05-kv-estimation.md)

> 独立、插件化、零分配热路径的 LLM 推理 KV Cache 显存估算

## 1. 背景与目标

### 1.1 问题域

网关要把一个推理请求路由到「放得下它」的后端，必须先回答一个问题：

> 给定模型的架构（层数 / KV head 数 / head 维度 / dtype / 注意力族）与请求形状（batch size / 输入长度 / 输出长度 / block size），这次推理的 KV Cache 会占多少 GPU 显存？

这是**容量感知路由**（admission control / load shedding）的前提：只有知道请求的 KV 占用，才能与后端剩余的 KV block 或 GPU 显存余量比较，把放不下的后端排除掉。

### 1.2 设计目标

1. **解析公式，而非仿真**：直接用模型架构参数 + 请求形状做整数乘加得出字节数，与 vLLM / SGLang / Mooncake / llm-d 的 KV 大小计算一致。不引入任何排队论或调度仿真。
2. **独立叶子 crate**：`hier-kv-gateway-kv-estimate` 不依赖任何 gateway 类型，只依赖 `serde`。可被任意推理路由器 / 调度器 / 容量规划工具复用。
3. **插件化扩展**：新模型优先用**数据**（一行 TOML）注册；新注意力机制用**代码**（实现 `KvEstimator` trait）注册。两条路径都覆盖。
4. **预制主流模型**：内置 Llama-2/3、Qwen2/2.5、Mistral/Mixtral、Gemma-2、DeepSeek-V2/V3/R1、ChatGLM3，覆盖 MHA / GQA / MQA / MLA / 滑动窗口四族。
5. **热路径零分配 + 纳秒级**：`spec_for → estimate` 全程 `Copy`、无 `String`、无 `HashMap` 值克隆，由计数分配器测试证明（见 §6）。
6. **可配置 + 可关闭**：`[kv_estimate]` 段默认关闭，开启时不影响既有配置解析。

### 1.3 非目标

- **不预测命中前缀后的实际增量占用**：那是 `KvAwareStrategy`（前缀重叠打分）的领域；本模块算的是「从零缓存到完整长度」的占用上界，用于容量准入。
- **不模拟调度器行为**：不预估 batch 内部的 token 复用、抢占、换出。
- **不取代后端自报的 KV block 总量**：后端上报 `kv_total_blocks`/`kv_used_blocks` 时，路由用精确的 block 路径；只有后端只报 GPU 显存时才走估算的 fallback。

---

## 2. 顶层架构

```
┌──────────────────────────────────────────────────────────┐
│                  KvEstimationRegistry                     │
│   (builtin StandardEstimator + custom specs + plugins)   │
│                                                          │
│   spec_for(model) ──► 按注册顺序首个识别 model 的 estimator │
│   estimate(model, input) ──► 该 estimator 的公式          │
└──────────────────────────────────────────────────────────┘
        ▲                              ▲
        │                              │
  ┌─────┴──────┐              ┌────────┴────────┐
  │  Standard  │              │  用户插件        │
  │ Estimator  │              │ KvEstimator impl│
  │ (解析公式)  │              │ (自定义公式)     │
  └─────┬──────┘              └─────────────────┘
        │
  ┌─────┴──────────────────────┐
  │  SpecCatalog               │
  │  custom specs (config TOML)│
  │  + builtin pattern table   │
  │    (Llama/Qwen/Mistral/    │
  │     Gemma/DeepSeek/GLM…)   │
  └────────────────────────────┘
```

模块组成：

| 文件 | 职责 |
|------|------|
| [spec.rs](../crates/hier-kv-gateway-kv-estimate/src/spec.rs) | `ModelSpec` / `AttentionKind` / `KvDtype` / `NamedModelSpec` — 决定 KV 占用的架构参数（`Copy` 值类型） |
| [catalog.rs](../crates/hier-kv-gateway-kv-estimate/src/catalog.rs) | 内置模型表 + 零分配大小写不敏感子串匹配 `lookup_builtin` |
| [estimate.rs](../crates/hier-kv-gateway-kv-estimate/src/estimate.rs) | 纯解析公式 `estimate_kv` / `per_token_bytes` / `per_block_bytes` |
| [plugin.rs](../crates/hier-kv-gateway-kv-estimate/src/plugin.rs) | `KvEstimator` trait / `StandardEstimator` / `SpecCatalog` |
| [registry.rs](../crates/hier-kv-gateway-kv-estimate/src/registry.rs) | `KvEstimationRegistry` — 复合 estimator，解析顺序 = 注册顺序 |
| [config.rs](../crates/hier-kv-gateway-kv-estimate/src/config.rs) | `KvEstimateConfig` — `[kv_estimate]` TOML 段 |

---

## 3. 解析公式（与开源实现对齐）

### 3.1 Standard 注意力（MHA / GQA / MQA）

Llama / Qwen / Mistral / Gemma / GLM 等都缓存完整 K 和 V 张量，单 token 占用：

```
per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
```

- 因子 `2` = K + V。
- `num_kv_heads` 已区分 MHA（= query heads）/ GQA（更少）/ MQA（= 1），公式不变，只是参数不同。

这是 **vLLM**（`vllm/worker/worker.py`: `cache_block_size = 2 * num_layers * num_kv_heads * head_size * block_size * dtype_size`）与 **SGLang**（`sglang/srt/configs/model_config.py`: `get_kv_cache_bytes`）使用的公式。

### 3.2 MLA — Multi-head Latent Attention（DeepSeek-V2 / V3 / R1）

MLA 把 KV Cache 压缩成单个潜向量 `c_kv`（维度 `kv_lora_rank`），注意力时分别上投影重建 K 与 V；另缓存一个 RoPE 解耦的小 key `k_pe`（维度 `qk_rope_head_dim`）。单 token 占用：

```
per_token_bytes = num_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
```

**没有因子 2**：一个潜向量同时重建 K 和 V。对 DeepSeek-V3（61 层，`kv_lora_rank=512`，`qk_rope_head_dim=64`，BF16）：`61 * 576 * 2 = 70 272` B/token —— 比等价的全 K/V 布局小约 57×（DeepSeek-V2 论文 §3.1）。

### 3.3 滑动窗口注意力

`sliding_window > 0` 时，持久缓存每序列最多保留 `sliding_window` 个 token（更早 token 的 KV 被驱逐）。有效缓存序列长 = `min(seq_len, sliding_window)`。Mistral（4096）、Gemma-2 使用。

### 3.4 Block 分页

分页注意力引擎（vLLM / SGLang / Mooncake）按 `block_size` 个 token 为单位分配 KV 显存，请求占用向上取整到整 block：

```
padded_seq_len = ceil(effective_seq_len / block_size) * block_size
total_bytes    = per_token_bytes * batch_size * padded_seq_len
total_blocks   = ceil(effective_seq_len / block_size) * batch_size
```

`EstimateInput::block_size = 0` 关闭 padding，返回精确（未 padding）占用。

### 3.5 端到端示例

Llama-3-8B（32 层，8 KV heads，head_dim 128，BF16），4096 prompt + 1024 output，block_size 16，batch 1：

```
per_token = 2 * 32 * 8 * 128 * 2 = 131_072 B (128 KiB/token)
effective_seq_len = 4096 + 1024 = 5120
blocks = ceil(5120/16) = 320
padded_seq_len = 320 * 16 = 5120
total_bytes = 131_072 * 5120 = 671_088_640 B (≈ 640 MiB)
```

---

## 4. 插件化扩展机制（两条路径）

### 4.1 路径 A：加模型 spec（数据，无代码）

绝大多数新模型走这条路 —— 任何 MHA/GQA/MQA/MLA 架构都是几个 config 字段，不是新公式：

```toml
[[kv_estimate.models]]
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

字段与 HuggingFace `config.json` 一一对应：

| `ModelSpec` 字段 | HuggingFace `config.json` 字段 |
|------------------|-------------------------------|
| `num_layers` | `num_hidden_layers` |
| `num_kv_heads` | `num_key_value_heads`（MHA 时 = `num_attention_heads`） |
| `head_dim` | `head_dim`（或 `hidden_size / num_attention_heads`） |
| `attention` | 默认 `standard`；DeepSeek-V2/V3 用 `mla` |
| `dtype` | `torch_dtype` |
| `kv_lora_rank` | `kv_lora_rank`（仅 MLA） |
| `qk_rope_head_dim` | `qk_rope_head_dim`（仅 MLA） |
| `sliding_window` | `sliding_window`（0 = 无） |

自定义 spec 优先于同名内置 spec（可用来纠正内置 dtype 或补漏的滑动窗口）。

### 4.2 路径 B：加自定义 estimator（代码）

当某架构的 KV 占用**无法**用标准公式表达（如 Cross-Attention 额外缓存、Mamba/SSM 状态、混合方案），实现 `KvEstimator` trait：

```rust
pub trait KvEstimator: Send + Sync {
    fn name(&self) -> &str;
    fn spec_for(&self, model: &str) -> Option<ModelSpec>;
    fn estimate(&self, spec: &ModelSpec, input: &EstimateInput) -> KvEstimate;
}
```

通过 `KvEstimationRegistry::with_estimator`（追加，低优先级）或 `with_estimator_front`（前置，高优先级，可覆盖内置）注册。registry 按注册顺序问每个 estimator `spec_for(model)`，**首个**识别 model 的 estimator 同时提供 spec 与（可能自定义的）公式 —— 自定义公式被完整尊重。

这镜像了 vLLM 的做法：标准模型用 `config.json` 字段参数化，异构情况由引擎代码覆盖。

---

## 5. 内置模型目录

18 条 spec 覆盖四族注意力，每条参数转录自对应模型 HuggingFace `config.json`：

| 模型族 | 注意力族 | 代表模型 |
|--------|---------|---------|
| Llama-2 | MHA / GQA | Llama-2-7B/13B/70B |
| Llama-3 / 3.1 / 3.3 | GQA | Llama-3-8B/70B, 3.1-405B |
| Qwen2 / 2.5 | GQA | Qwen2.5-7B/14B/32B/72B, Qwen2-7B/72B |
| Mistral / Mixtral | GQA + 滑动窗口 4096 | Mistral-7B, Mixtral-8x7B/8x22B |
| Gemma-2 | GQA + 滑动窗口 | Gemma-2-9B/27B |
| DeepSeek | MLA | DeepSeek-V2-Lite, V3, R1 |
| ChatGLM3 | GQA（kv_heads=2） | ChatGLM3-6B |

查找是**大小写不敏感子串匹配**，按「最具体到最一般」排序（如 `llama-3.1-405b` 排在 `llama-3` 之前，确保 405B 解析到 126 层 spec 而非 8B 的 32 层）。

---

## 6. 零分配热路径设计

热路径 = 「已知模型名 → 解析 spec → 算占用」，每次请求、每个候选后端都跑。设计上保证**零堆分配**：

1. **`ModelSpec` 是 `Copy`**：纯整数 + 两个 `#[derive(Copy)]` 枚举，无 `String`、无 `Arc`。模型名不进 `ModelSpec`（作为 catalog 的 key 单独存放），这正是热路径能 `Copy` 的关键。
2. **`lookup_builtin` 零分配**：用 `contains_ascii_ci` 做大小写不敏感子串匹配，**不**调用 `to_ascii_lowercase()`（那会每次查找分配一个 `String`）。
3. **自定义 spec 查找借 `Borrow<str>`**：`HashMap<String, ModelSpec>::get` 接受 `&str` 键，查找键是借用的 `&str`，无 clone。
4. **`estimate_kv` 纯整数运算**：常数次整数乘加，返回 `Copy` 结构。

证明见 [tests/alloc_free.rs](../crates/hier-kv-gateway-kv-estimate/tests/alloc_free.rs)：安装计数全局分配器，对 `estimate_kv` / `per_token_bytes` / `registry.spec_for`（命中/未命中）/ `registry.estimate`（完整热路径）/ 自定义 catalog 查找各跑 10 000 次，断言窗口内分配字节数 = **0**（零容忍）。所有检查集中在**单个** `#[test]` 函数里，避免 `cargo test` 并行线程间分配计数串扰。

---

## 7. 配置接口

```toml
[kv_estimate]
enabled = true            # 主开关，默认 false
weight = 0.20             # 在 Hybrid 中的权重
gpu_mem_safety_fraction = 0.5  # GPU 显存 fallback 时可声明的安全比例
exclude_on_unknown_spec = false  # 未知 spec 时排除(true)还是中立(false)

[[kv_estimate.models]]    # 可选：自定义模型 spec
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

| 字段 | 默认 | 说明 |
|------|------|------|
| `enabled` | `false` | 关闭时不挂载 `KvCapacityStrategy`，估算器不进路由路径 |
| `weight` | `0.20` | `[0,1]`；`0.0` 保持挂载（可用性探针仍跑）但不贡献分数 |
| `gpu_mem_safety_fraction` | `0.5` | 后端只报 GPU 显存时，仅「当前空闲显存 × 此比例」可被 KV 占用（KV 不是唯一 GPU 内存消费者） |
| `exclude_on_unknown_spec` | `false` | `true`：未知 spec 后端 `raw_cost=∞` 排除；`false`：中立让其他子策略决定（更安全，避免饿死确有余量的后端） |

---

## 8. 与路由的集成

估算模块本身只算字节数；把它变成路由分数的是 `KvCapacityStrategy`（路由 crate，详见 [02-routing-algorithms.md §9](02-routing-algorithms.md)）。集成方式：

```
[kv_estimate] enabled=true
       │ build_routing_engine
       ▼
KvEstimationRegistry (Arc, 启动时构造一次)
       │
       ▼
KvCapacityStrategy (plugin)
       │ HybridStrategy::with_plugin
       ▼
Hybrid 评分（与 KV/Load/Topology 各自独立归一化）
```

- **数据半**（`KvEstimateConfig`、spec catalog）在本叶子 crate，经 `GatewayConfig::kv_estimate` 暴露。
- **行为半**（`KvCapacityStrategy` 把估算转成路由分数）在路由 crate，作为 `RoutingPlugin` 挂到 Hybrid。

两者解耦：估算模块可独立复用，路由策略可独立测试。

---

## 9. 测试与反作弊

| 层 | 文件 | 覆盖 |
|----|------|------|
| 公式单元测试 | [estimate.rs](../crates/hier-kv-gateway-kv-estimate/src/estimate.rs) `#[cfg(test)]` | MHA/GQA/MLA/滑动窗口/block padding/batch 缩放/dtype/饱和加（19 个） |
| spec/catalog 单元测试 | [spec.rs](../crates/hier-kv-gateway-kv-estimate/src/spec.rs) / [catalog.rs](../crates/hier-kv-gateway-kv-estimate/src/catalog.rs) `#[cfg(test)]` | TOML 往返、大小写不敏感匹配、四族覆盖、索引越界检查 |
| registry/plugin 单元测试 | [registry.rs](../crates/hier-kv-gateway-kv-estimate/src/registry.rs) / [plugin.rs](../crates/hier-kv-gateway-kv-estimate/src/plugin.rs) `#[cfg(test)]` | 解析顺序、前置覆盖、自定义公式被尊重、clone 共享 Arc |
| 零分配证明 | [tests/alloc_free.rs](../crates/hier-kv-gateway-kv-estimate/tests/alloc_free.rs) | 计数分配器断言热路径 0 字节 |
| Benchmark 反作弊 | [benches/kv_estimate.rs](../crates/hier-kv-gateway-kv-estimate/benches/kv_estimate.rs) | 每次迭代断言手算期望字节数；`black_box` 防编译器消除 |

**反作弊原则**：benchmark 每次迭代 `assert_eq!(r.bytes, expected)`，若有人把公式「优化」成 no-op（或改坏），断言 panic、bench 失败 —— 数字无法伪造。零分配测试用计数全局分配器拦截测试二进制内**每一次** `alloc`，非零 delta 即真实回归。

---

## 10. 文件索引

| 文件 | 说明 |
|------|------|
| [lib.rs](../crates/hier-kv-gateway-kv-estimate/src/lib.rs) | crate 入口 + 架构总览 |
| [spec.rs](../crates/hier-kv-gateway-kv-estimate/src/spec.rs) | `ModelSpec` 值类型 |
| [catalog.rs](../crates/hier-kv-gateway-kv-estimate/src/catalog.rs) | 内置模型表 + 零分配匹配 |
| [estimate.rs](../crates/hier-kv-gateway-kv-estimate/src/estimate.rs) | 解析公式 |
| [plugin.rs](../crates/hier-kv-gateway-kv-estimate/src/plugin.rs) | `KvEstimator` trait + `StandardEstimator` |
| [registry.rs](../crates/hier-kv-gateway-kv-estimate/src/registry.rs) | `KvEstimationRegistry` 复合 estimator |
| [config.rs](../crates/hier-kv-gateway-kv-estimate/src/config.rs) | `[kv_estimate]` TOML 段 |
| [kv_capacity.rs](../crates/hier-kv-gateway-routing/src/kv_capacity.rs) | `KvCapacityStrategy`（行为半，路由 crate） |
