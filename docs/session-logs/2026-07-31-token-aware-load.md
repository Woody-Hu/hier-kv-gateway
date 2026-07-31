# Session Log: Token-aware Load 调度调研与引入

> 开发会话日志 · 2026-07-31

## 概述

| 项 | 内容 |
|----|------|
| **主题** | Token 网关与感知调度机制调研 → token-aware load 路由引入 |
| **类型** | 开放性调研 + 闭环验证 + 功能引入 |
| **结论** | 引入 token-aware load 调度（投影 decode 压力 + prefill 压力），通过严格闭环论证 |
| **判据** | CoV 改善 ≥15%、峰值不回升、路由延迟开销 <10% — 全部满足 |
| **诚实约束** | 全程真实组件，无 mock / stub / 预烘焙分数 |

## 1. 调研问题

> 「各类的 token 网关与感知调度机制，进行调研，看看有没有可参考的思路」

### 1.1 现状审计

审查 `crates/hier-kv-gateway-routing/src/load_aware.rs` 与 `crates/hier-kv-gateway-core/src/{request,metrics}.rs` 后发现：

- `LoadAwareStrategy::evaluate` 的 `load_cost` 仅由 `active_requests / queue_depth / p99 / gpu_util / kv_cache_usage` 五项构成——**全部是计数或利用率，对生成长度无感**。
- `RoutingContext::estimated_output_tokens`（来自客户端 `max_tokens`）与 `BackendMetrics::active_prefill_tokens` **已被采集但未被任何策略消费**——这是明显的「信号已铺到门口却没接进屋里」。

### 1.2 业界思路参考

调研 LLM 推理网关 / 调度器的 token 感知思路，归纳出可参考方向：

| 思路 | 来源 | 对本系统的适用性 |
|------|------|----------------|
| **输出预算投影** | vLLM/SGLang 的 continuous batching admission control 用 `max_tokens` 估算 decode 占用 | ✅ 信号已存在（`estimated_output_tokens`），直接可接 |
| **保守上界 vs 点估计** | 调度理论：长尾任务用点估计会系统性低估导致饥饿 | ✅ `max_tokens` 是硬上界，永不低估 |
| **活跃 decode blocks** | vLLM `EngineStats.num_running_tokens` / SGLang radix cache 压力 | ✅ `active_decode_blocks` 已采集 |
| **Prefill 压力** | vLLM `waiting_tokens` / SGLang prefill queue | ✅ `active_prefill_tokens` 已采集 |
| **KV overlap 投影 prefill** | DistServe / Splitwise 分离 prefill/decode | ❌ 属于 `KvAwareStrategy` 领域，跨策略合并会破坏 Hybrid 归一化语义 |

**结论**：最高 ROI 的改动是**闭合已采集但未消费的信号缺口**——把 `estimated_output_tokens` 投影为 decode 块数、把 `active_prefill_tokens` 作为软成本项纳入 `load_cost`。无需新增采集路径，无需跨策略耦合。

## 2. 闭环论证设计

引入新信号必须通过可证伪检验，避免「看起来更好」的主观判断：

```
1. 问题（事实）  : count-blind 打分对生成长度无感，两个 token 信号已采集未消费
2. 假设          : 混合长度工作负载下，count-blind 把长请求堆到「计数低但 decode 饱和」的 backend
3. 可证伪指标    : 跨 backend 的 active_decode_blocks 变异系数（CoV），越低越好
4. 决策规则      : CoV 改善 ≥15% 且峰值不回升 → 引入；路由延迟开销 <10% → 引入
```

**关键防作弊设计**：

- 离散事件回放：每个被路由的请求**真实地**通过 `load_update` 更新 `LoadStats`，每个完成事件**真实地**递减——测量路由决策的*后果*而非假设。
- baseline 用 `w_decode=0, w_prefill=0` **逐字节复现**改动前成本，确保对照公平。
- 基线非平凡性检验：单独断言 baseline 峰值距 clairvoyant 下界 ≥5%，防止工作负载没触发缺陷的伪通过。

## 3. 实现

### 3.1 代码改动

| 文件 | 改动 |
|------|------|
| [load_aware.rs](../crates/hier-kv-gateway-routing/src/load_aware.rs) | 新增 `w_decode` / `w_prefill` 字段；`evaluate` 增加 `projected_decode` 与 `active_prefill_tokens` 两项；默认权重 `0.02 / 0.001` |
| [token_aware_load.rs (test)](../tests/hier-kv-gateway-integration/tests/token_aware_load.rs) | 新增离散事件回放 + 闭环断言（2 个测试） |
| [token_aware_load.rs (bench)](../crates/hier-kv-gateway-routing/benches/token_aware_load.rs) | 新增 Criterion 路由延迟 benchmark |
| [Cargo.toml (routing)](../crates/hier-kv-gateway-routing/Cargo.toml) | 注册 `[[bench]] token_aware_load` |

### 3.2 核心算法

```
req_decode_blocks = ceil(ctx.estimated_output_tokens / ctx.block_size)
projected_decode  = m.active_decode_blocks + req_decode_blocks

load_cost = w_req    * m.active_requests
          + w_queue  * m.queue_depth
          + w_lat    * (m.p99_ms / 100)
          + w_gpu    * m.gpu_utilization
          + w_kv     * m.kv_cache_usage
          + w_decode * projected_decode        # 新增：投影 decode 压力
          + w_prefill * m.active_prefill_tokens # 新增：当前 prefill 压力
```

向后兼容开关：`w_decode=0, w_prefill=0` 逐字节复现改动前行为。

## 4. 验证结果

### 4.1 质量指标（离散事件回放）

```
workload: 180 requests, 6 backends, block_size=16
[baseline]     CoV=0.070  peak=690  mean=616.7  clairvoyant=616.7
[token_aware]  CoV=0.048  peak=651  mean=616.7  clairvoyant=616.7

CoV 改善: 31.8% (≥15% 判据 ✓)
峰值改善: 690 → 651 (不回升判据 ✓，向 clairvoyant 616.7 收敛)
基线非平凡性: peak 690 距 clairvoyant 616.7 达 11.9% (≥5% 判据 ✓)
```

### 4.2 性能指标（Criterion 路由延迟）

```
n=20 候选 (生产规模):
  baseline    ≈ 40 µs
  token_aware ≈ 38 µs
  开销: 低于测量噪声底 (criterion p > 0.05, "未检测到性能变化")
  (<10% 判据 ✓)
```

两项判据全部满足，引入决策通过。

## 5. 文档更新

| 文档 | 更新内容 |
|------|---------|
| [04-algorithms.md §7.4](../docs/04-algorithms.md) | 扩展负载感知路由章节，新增 §7.4.1 token-budget 感知小节，更新默认权重表 |
| [token-aware-load.md](../docs/benchmarks/token-aware-load.md) | 新增 benchmark 报告（匹配 `load-encoding.md` 风格） |

## 6. 经验沉淀

### 6.1 什么有效

- **先审计「已采集未消费」信号**：比新增采集路径 ROI 高一个数量级。本次两个信号都在 `BackendMetrics` 里躺着，只差接入 `load_cost`。
- **逐字节复现的 baseline**：`w_decode=0, w_prefill=0` 让对照绝对公平，消除「是不是别的改动也影响了」的疑虑。
- **基线非平凡性检验**：防止工作负载选错导致伪通过。这是闭环论证里最容易被忽略但最关键的一环。
- **保守上界而非点估计**：`max_tokens` 永不低估，避免长尾请求上的系统性饥饿。

### 6.2 什么需要警惕

- **CoV 不是唯一指标**：CoV 下降但峰值回升等于「靠集中负载换均衡」，不可接受。所以决策规则里同时锁了峰值。
- **跨策略耦合的诱惑**：把 prompt 投影进 prefill 看似更精确，但会破坏 Hybrid 归一化语义——load 策略拿不到 KV overlap，强行投影会高估。保持策略独立。
- **benchmark 噪声**：n=12 时方差极大，单次运行不可信。需要多次运行看 median 区间是否重叠。criterion 的 `p > 0.05` 是诚实表达，不要强行解读为「提升了」。

## 7. 后续可能方向（未实施，仅记录）

| 方向 | 评估 |
|------|------|
| EWMA 历史长度点估计 | ✗ 与保守上界冲突，长尾低估，不引入 |
| 跨策略 KV-aware + load 联合投影 prefill | 需重构 Hybrid 归一化语义，风险高，暂不引入 |
| 自适应 `w_decode` 权重（类似 `adaptive.rs`） | 可作为后续调研项，需先证明固定权重不足 |
| Token-aware session affinity | 当前 session affinity 只看 overlap，可考虑 decode 压力，需独立闭环 |

## 8. 文件清单

新增：
- `tests/hier-kv-gateway-integration/tests/token_aware_load.rs`
- `crates/hier-kv-gateway-routing/benches/token_aware_load.rs`
- `docs/benchmarks/token-aware-load.md`
- `docs/session-logs/2026-07-31-token-aware-load.md`（本文件）

修改：
- `crates/hier-kv-gateway-routing/src/load_aware.rs`
- `crates/hier-kv-gateway-routing/Cargo.toml`
- `docs/04-algorithms.md`
