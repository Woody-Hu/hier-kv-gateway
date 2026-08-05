# Session Log: 多租户负载平衡调度机制

> 开发会话日志 · 2026-08-04

## 概述

| 项 | 内容 |
|----|------|
| **主题** | 多租户网关负载平衡调度：Token Bucket 限流 + 优先级准入控制 |
| **类型** | 调研 + 设计 + 实施 + Benchmark |
| **结论** | 实现 `TenantScheduler`，基于 Token Bucket 做 RPS 限流，饱和时按 Premium > Normal > Background 优先级准入 |
| **判据** | 热路径 ~325ns/op，禁用时 ~2.9ns/op，多租户规模无性能退化 |

---

## 1. 问题分析

### 1.1 核心问题

> 一个租户建立巨大的洪峰请求占满所有资源，优先级应该重新调度。

多租户网关面临的关键挑战：

1. **洪峰抢占**（Noisy Neighbor）：一个租户的突发流量耗尽所有后端资源，其他租户饿死
2. **公平性缺失**：无优先级区分，批量任务和实时服务争抢同一资源池
3. **不可预测性**：缺少准入控制，系统过载时行为不可预测

### 1.2 现有系统状态

审计发现当前系统**完全缺乏多租户支持**：
- 无 `TenantId` 类型定义
- 无租户级配额或限流
- 无优先级区分
- `RoutingContext` 无租户标识字段

---

## 2. 开源系统调研

### 2.1 LiteLLM — 动态配额

**核心机制**：`/team/update` API 动态调整团队配额，基于 `model_id` 精细化控制。

**可借鉴**：动态配额 API、多维度限流（RPM/TPM/预算）、模型级粒度。

### 2.2 Envoy Proxy — 令牌桶限流

**核心机制**：`RateLimitFilter` + 外部限流服务，支持全局和本地限流。

**可借鉴**：令牌桶算法本身、请求头提取租户、Descriptor 机制。

### 2.3 AWS Bedrock — 租户隔离

**核心机制**：通过 IAM 策略为每个租户分配独立的模型访问配额，通过 `Provisioned Throughput` 实现预留容量。

**可借鉴**：预留容量（Reserved Capacity）概念、优先级分层。

### 2.4 选型决策

**选择**：Token Bucket（限流）+ 优先级准入（饱和保护）。理由：
- 实现简单，仅 ~300 行 Rust
- 热路径快（~325ns）
- 覆盖核心场景（洪峰限流 + 优先级保证）

---

## 3. 设计方案

### 3.1 配置项设计

```toml
[tenant]
enabled = true
default_max_rps = 100.0
default_max_concurrent = 10
saturation_threshold = 0.8

[[tenant.tenants]]
id = "premium-org"
priority = "premium"
max_rps = 500.0
max_concurrent = 50
reserved_capacity_fraction = 0.3

[[tenant.tenants]]
id = "batch-jobs"
priority = "background"
max_rps = 5.0
```

设计原则：
- **极简配置**：每个租户仅需 3-5 行
- **默认继承**：未配置的字段自动继承全局默认值
- **优先级语义**：Premium > Normal > Background

### 3.2 调度策略

```
              请求到达
                 │
          ┌──────▼──────┐
          │ Token Bucket │── 不足 ──→ 429 RateLimited
          │ 令牌充足？   │
          └──────┬──────┘
                 │ 充足
          ┌──────▼──────┐
          │ 并发限制？   │── 满 ──→ 429 RateLimited
          └──────┬──────┘
                 │ 未满
          ┌──────▼──────┐
          │ 系统饱和？   │── 否 ──→ Admitted
          └──────┬──────┘
                 │ 是
          ┌──────▼──────┐
          │ 租户优先级？ │
          ├─────────────┤
          │ Premium  ──→ Admitted (预留容量)
          │ Normal   ──→ Admitted / Queued
          │ Background ─→ Queued
          └─────────────┘
```

---

## 4. 实施细节

### 4.1 新增/修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `crates/hier-kv-gateway-core/src/ids.rs` | 修改 | 新增 `TenantId` 类型 |
| `crates/hier-kv-gateway-core/src/tenant.rs` | 新建 | 核心数据结构 |
| `crates/hier-kv-gateway-core/src/config.rs` | 修改 | 新增 `TenantConfig` |
| `crates/hier-kv-gateway-core/src/request.rs` | 修改 | `RoutingContext` 新增 `tenant_id` |
| `crates/hier-kv-gateway-routing/src/tenant_scheduler.rs` | 新建 | `TenantScheduler` 实现 |
| `crates/hier-kv-gateway-routing/benches/tenant_scheduler.rs` | 新建 | Benchmark 套件 |

---

## 5. Benchmark 结果

| 场景 | 延迟 |
|------|------|
| 禁用快速路径 | **2.9 ns** |
| 准入检查 (任意租户数) | **~325 ns** |
| Token Bucket 消费 | **~282 ns** (3.5M ops/s) |

租户数量不影响延迟（DashMap O(1)），准入开销仅占路由决策的 ~0.27%。

---

## 6. 后续工作

1. **HTTP 层集成**：从 `X-Tenant-Id` 请求头提取 `TenantId`，调用 `check_admission()`
2. **动态配额 API**：提供 `PUT /admin/tenants/:id/quota` 端点
3. **TPM 限流**：扩展 Token Per Minute 维度
4. **公平队列**：`Queued` 决策接入请求队列实现真正的公平排队
5. **Prometheus 指标**：暴露 `tenant_requests_total{tenant,decision}` 等指标
