# KV 显存估算 · 容量感知路由 Benchmark 报告

## 1. 背景

新增 KV 显存估算模块（`hier-kv-gateway-kv-estimate`）与 `KvCapacityStrategy` 容量感知路由策略。本报告测量两件事：

1. **估算热路径延迟**：`estimate_kv`（解析公式）与 `registry.estimate`（名称 → spec → 公式）的每次调用成本。这是路由决策循环里「每请求 × 每候选后端」都要跑的代码。
2. **容量感知策略的路由开销**：`KvCapacityStrategy::evaluate` 独立成本，以及挂到 Hybrid 后相对 baseline 的端到端开销。

新增信号必须通过可证伪检验：估算必须**准**（公式与 vLLM/SGLang/Mooncake 一致，由单元测试断言）且**快**（热路径零分配、纳秒级，由计数分配器测试与 benchmark 证明）。

## 2. 测试环境

- CPU: Linux x86_64 (sandbox)
- Rust: `cargo bench` release profile (`--opt-level=3`)
- Benchmark 框架: criterion 0.5
- 数据: 真实 `KvEstimationRegistry` + 内置 catalog + 真实 `MetadataStore` + `HybridStrategy`，**无 mock、无 stub、无预烘焙分数**

## 3. 诚实契约（无 mock 作弊）

### 3.1 估算正确性

每次 benchmark 迭代 `assert_eq!(r.bytes, expected)`，`expected` 是**手算**的期望字节数（如 Llama-3-8B 4096 token = `131_072 × 4096 = 536_870_912`）。若有人把公式「优化」成 no-op 或改坏，断言 panic、bench 失败 —— 数字无法伪造。`black_box` 防编译器消除。

### 3.2 零分配证明

`tests/alloc_free.rs` 安装计数全局分配器，对 `estimate_kv` / `per_token_bytes` / `registry.spec_for`（命中/未命中）/ `registry.estimate`（完整热路径）/ 自定义 catalog 查找各跑 10 000 次，断言窗口内分配字节数 = **0**（零容忍）。所有检查集中在单个 `#[test]`，避免并行线程分配计数串扰。

### 3.3 路由策略反作弊

`kv_capacity` bench 每次迭代断言：评分列表非空、每个 `raw_cost.is_finite()`（该场景所有后端都应被准入）、`score ∈ (0, 1]`。若策略被意外短路成空/常量结果，bench 自身 panic。

## 4. 结果一：估算热路径延迟（叶子 crate）

`cargo bench -p hier-kv-gateway-kv-estimate --bench kv_estimate`

### 4.1 解析公式 `estimate_kv`

| spec | 延迟 |
|------|------|
| llama3_8b_gqa (GQA) | 9.92 ns |
| deepseek_v3_mla (MLA) | 9.97 ns |
| mistral_7b_sliding (GQA + 滑动窗口) | 9.91 ns |

**输入长度无关性**（Llama-3-8B，block_size 16）：

| 输入 tokens | 延迟 |
|------------|------|
| 512 | 9.92 ns |
| 4096 | 9.91 ns |
| 32_768 | 9.88 ns |
| 131_072 | 9.90 ns |

公式是常数次整数乘加，不随序列长度变化。`per_token_bytes`（架构级常量）仅 1.18 ns。

### 4.2 完整热路径 `registry.estimate`（名称 → spec → 公式）

| 模型 | 延迟 |
|------|------|
| Qwen2.5-7B | 44.8 ns |
| mistral-7b | 51.2 ns |
| Llama-3-8B | 59.3 ns |
| deepseek-v3 | 91.3 ns |

差异来自内置 catalog 的大小写不敏感子串扫描（`contains_ascii_ci`）：`deepseek-v3` 在 pattern 表中位置靠后，需扫描更多 pattern 才命中。

### 4.3 spec 查找 `registry.spec_for`

| 场景 | 延迟 |
|------|------|
| builtin_hit (Llama-3-8B) | 45.2 ns |
| builtin_miss (扫描全表后 None) | 457.2 ns |

### 4.4 自定义 catalog 查找（`HashMap` via `Borrow<str>`）

| 场景 | 延迟 |
|------|------|
| custom_hit (64 条表中查中) | 28.3 ns |
| builtin_fallback_through_custom | 56.3 ns |

自定义 `HashMap` 命中（28 ns）比内置 pattern 扫描（45 ns）更快 —— 配置驱动部署的热路径同样廉价。

**结论**：完整热路径在 **45–91 ns** 量级，相对 LLM 推理延迟（100ms+）完全可忽略。公式本身 ~10 ns。零分配由独立测试证明。

## 5. 结果二：容量感知策略路由开销（路由 crate）

`cargo bench -p hier-kv-gateway-routing --bench kv_capacity`

### 5.1 独立策略评估 `kv_capacity_evaluate`

| 候选后端数 | 延迟 |
|-----------|------|
| 2 | 557 ns |
| 10 | 2.57 µs |
| 50 | 13.04 µs |

每后端成本 ≈ 260 ns（含 MetadataStore 指标查询 + 模型解析 + 估算 + 容量打分）。50 后端仍 13 µs，远低于推理延迟。

### 5.2 Hybrid 端到端开销（baseline vs with_kv_capacity）

| 后端数 | Baseline (kv/load/topo) | With kv_capacity | 绝对开销 | 相对开销 |
|--------|------------------------|------------------|---------|---------|
| 2 | 1.85 µs | 2.70 µs | ~0.85 µs | ~46% |
| 10 | 9.06 µs | 12.91 µs | ~3.85 µs | ~42% |
| 20 | 18.10 µs | 26.09 µs | ~7.99 µs | ~44% |

**结论**：挂载 kv_capacity 插件的绝对开销为 **~0.85–8 µs**，相对 LLM 推理延迟（100ms+）可忽略。百分比开销 (~42–46%) 来自每次 evaluate 新增一次策略调用 + 一次 `normalize_costs` 分配，但绝对值在可接受范围 —— 与既有的 cost/tier 插件开销量级一致（见 [cost_tier_plugins](2026-08-05-cost-dedup-tier-plugins.md) 的 ~1–5.5 µs）。

## 6. 设计决策

### 6.1 为什么用解析公式而非仿真？

公式与 vLLM / SGLang / Mooncake 的 KV 大小计算完全一致，是**确定性的精确值**（给定架构 + 请求形状）。仿真（排队论 / 调度模拟）引入建模假设与误差，且无法在纳秒级热路径完成。用户明确要求「利用 head 数/层数等结合 batch size、input length 计算，不是仿真机制」。

### 6.2 为什么 output 用 `max_tokens` 作上界？

`estimated_output_tokens` 源自客户端 `max_tokens` —— 生成的硬上界。投影 KV 占用因此**永不低估**实际增长。悲观估计避免把请求路由到「估算放得下、实际放不下」的后端，符合 admission control 的安全取向。

### 6.3 为什么 GPU 显存 fallback 用安全比例？

后端只报 GPU 显存（无 KV block 总量）时，`gpu_mem_safety_fraction=0.5` 表示「仅当前空闲显存的一半可被 KV 声明」。KV 不是唯一 GPU 内存消费者（权重、激活），若把整卡空闲都算给 KV 会高估可用容量、导致误准入。

### 6.4 为什么 `ModelSpec` 是 `Copy` 且不含 `String`？

模型名不是公式参数，留在 catalog 作 key。`ModelSpec` 纯整数 + `#[derive(Copy)]` 枚举 → `spec_for` 返回 `Copy`、热路径零分配。这是 45–91 ns 热路径的前提（`String` clone 会引入堆分配）。

### 6.5 为什么内置匹配用 `contains_ascii_ci` 而非 `to_ascii_lowercase`？

`to_ascii_lowercase()` 每次查找分配一个 `String`，破坏零分配保证。`contains_ascii_ci` 逐字节在线小写比较，零分配。代价是 builtin miss 需扫描全表（457 ns），但这是冷路径（未知模型走 `exclude_on_unknown_spec` 策略，且可被自定义 spec 覆盖避免）。

## 7. 运行方式

```bash
# 估算热路径 benchmark（叶子 crate）
cargo bench -p hier-kv-gateway-kv-estimate --bench kv_estimate

# 容量感知策略 benchmark（路由 crate）
cargo bench -p hier-kv-gateway-routing --bench kv_capacity

# 零分配证明（计数分配器，零容忍断言）
cargo test -p hier-kv-gateway-kv-estimate --test alloc_free -- --nocapture

# HTML 报告
# target/criterion/estimate_kv/index.html
# target/criterion/kv_capacity_evaluate/index.html
```

## 8. 文件索引

| 文件 | 说明 |
|------|------|
| [kv_estimate.rs (bench)](../../crates/hier-kv-gateway-kv-estimate/benches/kv_estimate.rs) | 估算热路径 Criterion benchmark（含反作弊断言） |
| [kv_capacity.rs (bench)](../../crates/hier-kv-gateway-routing/benches/kv_capacity.rs) | 容量感知策略 benchmark（含反作弊断言） |
| [alloc_free.rs (test)](../../crates/hier-kv-gateway-kv-estimate/tests/alloc_free.rs) | 计数分配器零分配证明 |
| [estimate.rs](../../crates/hier-kv-gateway-kv-estimate/src/estimate.rs) | 解析公式实现 |
| [kv_capacity.rs](../../crates/hier-kv-gateway-routing/src/kv_capacity.rs) | `KvCapacityStrategy` 实现 |
| [05-kv-estimation.md](../05-kv-estimation.md) | 估算模块架构文档 |
| [02-routing-algorithms.md §9](../02-routing-algorithms.md) | 容量感知路由策略方案文档 |
