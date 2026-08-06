# Hier KV Gateway 路由算法设计

> 中文 | [English](en/02-routing-algorithms.md)

> 五种路由策略 + Hybrid 混合策略的详细算法

## 1. 通用评分模型

所有策略最终产出统一的 `ScoredBackend` 结构，由 Hybrid 策略综合：

```rust
struct ScoredBackend {
    backend_id: BackendId,
    /// 归一化到 [0, 1] 的得分，1.0 = 最优
    score: f64,
    /// 该策略给出的成本（越低越好，用于 Hybrid 加权）
    raw_cost: f64,
    /// 评分依据的元数据快照版本
    meta_version: u64,
}
```

每个策略独立产出 `raw_cost`，Hybrid 策略对各策略的 cost 做归一化后加权求和。

---

## 2. 策略一：KV Aware Routing（KV 感知路由）

### 2.1 目标

将请求路由到 KV Cache 前缀重叠最大的后端，最大化缓存复用，减少 prefill 计算量。

### 2.2 成本模型

KV Router 成本函数：

```
adjusted_prefill_blocks = max(
    prefill_blocks
      - overlap_score_credit * device_overlap_blocks
      - host_cache_hit_weight * host_overlap_blocks
      - disk_cache_hit_weight * disk_overlap_blocks
      - shared_cache_multiplier * shared_beyond_blocks,
    0,
)
cost = prefill_load_scale * adjusted_prefill_blocks + decode_blocks
```

### 2.3 Hier KV Gateway 适配算法

云边端环境下，KV Cache 可能存在于：
- 本地 Region 的 Backend（精确，通过 RadixTree 查询）
- 远程 Region 的 Backend（近似，通过 CKF 查询）

```
对每个候选 Backend b:
  1. 计算请求的 block_hashes = compute_block_hashes(token_ids, block_size)
  2. 查询本地 RadixTree（若 b 在本地 Region）:
       device_overlap = radix_tree.find_matches(block_hashes, b)  // 精确
  3. 查询跨 Region CKF Consumer（若 b 在远程 Region）:
       ckf_overlap = ckf_consumer.estimate_overlap(block_hashes, b.region)  // 近似
  4. total_overlap = device_overlap + ckf_overlap
  5. prefill_blocks = len(block_hashes) - total_overlap
  6. decode_blocks = b.active_decode_blocks (来自 Load Stats)
  7. cost = prefill_load_scale * prefill_blocks + decode_blocks
  8. score = 1.0 / (1.0 + cost)  // 归一化
```

### 2.4 Block Hash 计算

按 `compute_block_hash_for_seq` 思路：

```
对 token 序列按 block_size 分块:
  对每个 block:
    block_content = tokens[start..start+block_size]
    hash = xxhash64(block_content, seed=cache_namespace_hash)
    若有 LoRA: hash = xxhash64(hash || lora_name)
    若有多模态: hash = xxhash64(hash || mm_info)
  返回 block_hashes 数组
```

### 2.5 RadixTree（本地精确查询）

```
RadixTree:
  root: Node
  Node:
    hash: u64               // 该节点的 block hash
    children: HashMap<u64, Node>  // 子节点
    owners: Set<(backend_id, dp_rank)>  // 哪些 backend 拥有此 block
    is_terminal: bool       // 是否是一个完整缓存路径的终点

find_matches(block_hashes, target_backend):
  node = root
  overlap = 0
  for hash in block_hashes:
    if hash in node.children:
      node = node.children[hash]
      if target_backend in node.owners:
        overlap += 1
      else:
        break  // 该 backend 不拥有此后缀，停止
    else:
      break  // 无匹配前缀
  return overlap
```

### 2.6 CKF Consumer（跨 Region 近似查询）

transposed CKF 实现：

```
CKFConsumer:
  lanes: [CKFLane; MAX_REGIONS]  // 每 Region 一个 lane
  num_buckets: usize

  estimate_overlap(block_hashes, target_region):
    lane = lanes[target_region.lane_index]
    overlap = 0
    for hash in block_hashes:
      fp = fingerprint(hash)       // 取低 16 位
      bucket_idx = hash % num_buckets
      alt_bucket_idx = alt_hash(fp, bucket_idx) % num_buckets
      if lane.bucket_contains(bucket_idx, fp) 
         or lane.bucket_contains(alt_bucket_idx, fp):
        overlap += 1
      else:
        break  // 前缀中断
    return overlap  // 可能因 CKF 假阳性偏高
```

### 2.7 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kv_block_size` | 16 | KV block token 数 |
| `overlap_score_credit` | 1.0 | device overlap 信用倍数 |
| `prefill_load_scale` | 1.0 | prefill 成本缩放 |
| `ckf_false_positive_penalty` | 0.3 | CKF 假阳性的惩罚系数 |

---

## 3. 策略二：Model Aware Routing（模型感知路由）

### 3.1 目标

根据请求所需的模型，路由到加载了兼容模型的后端。考虑模型版本、量化、能力。

### 3.2 算法

```
对每个候选 Backend b:
  1. 查询 Model Registry: b 加载了哪些模型?
  2. 匹配度计算:
     - exact_match (model_name + version + quant): score = 1.0
     - model_match (同名不同版本): score = 0.7
     - compatible_match (同架构不同名, 如 Qwen2.5-7B vs Qwen2.5-14B): score = 0.3
     - no_match: score = 0.0 (排除该候选)
  3. 额外加分:
     - 量化偏好: 若请求偏好高精度, fp16 > int8 > int4
     - 上下文长度: 若请求 token 数 > b.max_context, 排除
     - 工具调用能力: 若请求需 function_calling, 检查 b 是否支持
  4. cost = 1.0 - match_score
```

### 3.3 模型兼容性矩阵

```
兼容性判定（参考 HuggingFace model config）:
  - architecture: 同 transformer 架构 (如 Qwen2, Llama)
  - vocab_size: 兼容的分词器
  - hidden_size / num_layers: 可不同（但 KV Cache 不共享）
  - 量化: 不影响兼容性判定，但影响质量评分
```

---

## 4. 策略三：Load Aware Routing（负载感知路由）

### 4.1 目标

根据后端实时负载（队列深度、GPU 利用率、活跃请求数）做负载均衡，避免热点。

### 4.2 关键指标

`active_decode_blocks`、`potential_prefill_tokens`、queue policy。

### 4.3 算法

```
对每个候选 Backend b:
  1. 查询 Load Stats: 获取 b 的最近指标
     - active_requests: 当前活跃请求数
     - queue_depth: 排队请求数
     - avg_p50_latency / avg_p99_latency: 延迟统计
     - gpu_utilization: GPU 利用率 (0-1)
     - kv_cache_usage: KV Cache 使用率 (0-1)
     - available_capacity: 剩余容量
  2. 计算负载成本:
     load_cost = w_req * active_requests
               + w_queue * queue_depth
               + w_lat * normalize(avg_p99_latency)
               + w_gpu * gpu_utilization
               + w_kv * kv_cache_usage
  3. 容量检查:
     if available_capacity <= 0: 排除该候选 (score = 0)
  4. cost = load_cost
  5. score = 1.0 / (1.0 + load_cost)
```

### 4.4 滑动窗口统计

```
LoadStats 维护每个 Backend 的滑动窗口:
  - 窗口大小: 60 秒
  - 采样间隔: 1 秒
  - 存储: RingBuffer<Metrics>
  - 计算: p50/p99 用近似算法 (如 t-digest), 避免存储全量数据

  metrics 更新:
    - 请求开始: active_requests += 1
    - 请求结束: active_requests -= 1, 记录 latency
    - 定期采集: 从 connector 拉取 gpu_utilization / kv_cache_usage
```

### 4.5 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `w_req` | 1.0 | 活跃请求数权重 |
| `w_queue` | 2.0 | 队列深度权重 |
| `w_lat` | 1.5 | 延迟权重 |
| `w_gpu` | 0.5 | GPU 利用率权重 |
| `w_kv` | 0.8 | KV 使用率权重 |
| `stats_window_secs` | 60 | 统计窗口 |
| `stats_sample_interval` | 1 | 采样间隔(秒) |

---

## 5. 策略四：Topology Aware Routing（拓扑感知路由）

### 5.1 目标

根据网络延迟拓扑，优先路由到就近的后端，降低端到端延迟。

### 5.2 数据结构

```
TopologyGraph:
  regions: HashMap<RegionId, RegionInfo>
  latency_matrix: HashMap<(RegionId, RegionId), LatencyEstimate>
  
RegionInfo:
  region_id: RegionId
  tier: Cloud | Edge | Device      // 层级
  geo: (lat: f64, lon: f64)        // 地理坐标
  network_zone: String             // 网络区域

LatencyEstimate:
  rtt_p50: Duration
  rtt_p99: Duration
  bandwidth_mbps: f64
  last_updated: Instant
```

### 5.3 延迟矩阵构建

```
延迟矩阵来源:
  1. 配置: 静态配置已知 Region 间延迟
  2. 主动探测: Gateway 实例间互发 ping, 测量 RTT
  3. Gossip 传播: 探测结果通过 Gossip 共享
  
更新:
  - 每 30 秒主动探测一次相邻 Region
  - 探测结果取最近 5 次的 p50
  - 未探测的 Region 对用地理距离估算: 
    rtt_estimate = distance_km / 200km_per_ms (光纤)
```

### 5.4 算法

```
对每个候选 Backend b:
  1. 查询 TopologyGraph: 获取 (self_region, b.region) 的延迟
  2. network_cost = rtt_p50_ms(b) * w_rtt
                  + bandwidth_penalty(b) * w_bw
  3. 层级偏好:
     if self.tier == Device and b.tier == Edge:
       network_cost *= 0.8  // 端侧优先用边侧
     if self.tier == Device and b.tier == Cloud:
       network_cost *= 1.5  // 端侧避免用云侧（延迟高）
  4. cost = network_cost
  5. score = 1.0 / (1.0 + network_cost / 100.0)  // 100ms 基准
```

### 5.5 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `w_rtt` | 1.0 | RTT 权重 |
| `w_bw` | 0.3 | 带宽惩罚权重 |
| `topology_refresh_secs` | 30 | 拓扑刷新间隔 |
| `geo_latency_factor` | 200 | km/ms 转换因子 |

---

## 6. 策略五：Hybrid Routing（混合智能路由，默认策略）

### 6.1 目标

融合 KV / Model / Load / Topology 四种策略，用加权评分综合决策。

### 6.2 算法

```
Hybrid 策略:
  1. 收集候选 Backend 集合 C = Model Aware 过滤后的候选集
     (Model Aware 作为硬性过滤器，不匹配的排除)
  
  2. 对每个可用策略 S ∈ {KV, Load, Topology}:
     若 S.is_available():
       scores_S = S.evaluate(ctx, C, meta)  // 每个 backend 一个 score
  
  3. 对每个候选 b ∈ C:
     hybrid_score(b) = Σ_S( weight_S * normalize(scores_S[b]) )
     
     其中 normalize 将各策略的 raw_cost 归一化到 [0, 1]:
       normalize(cost) = (cost - min_cost_S) / (max_cost_S - min_cost_S)
       
     权重动态调整:
       weight_KV = base_kv * kv_confidence
         kv_confidence = 1.0 - (ckf_false_positive_rate)
       weight_Load = base_load * load_freshness
         load_freshness = exp(-(now - last_update).secs / 10)
       weight_Topology = base_topology
  
  4. 选择 hybrid_score 最高的 b
     若 temperature > 0: 用 softmax 采样（router_temperature）
     否则: 贪心选择
```

### 6.3 权重默认值

| 策略 | 基础权重 | 说明 |
|------|---------|------|
| KV | 0.35 | KV 复用对 TTFT 影响最大 |
| Load | 0.30 | 负载均衡避免热点 |
| Topology | 0.20 | 网络延迟影响端到端 |
| Model | 1.0 (过滤器) | 作为硬性过滤，不参与加权 |

### 6.4 自适应权重调整

```
运行时根据降级状态自动调整权重:

if KV Index 不可用:
    weight_KV = 0; weight_Load += 0.15; weight_Topology += 0.10
elif KV confidence < 0.5:
    weight_KV *= 0.5; weight_Load += 0.10

if Load Stats 过期 (last_update > 10s):
    weight_Load *= 0.3; weight_Topology += 0.10

归一化: 确保所有权重之和 = 1.0
```

### 6.5 降级链

```
is_available() 检查:
  Hybrid:
    - Model Registry 可用? (必须有, 否则无法路由)
    - 至少一个子策略可用? (是 → 降级模式)
  
  降级顺序:
    Hybrid (KV+Load+Topo)
      → 若 KV 不可用: Model+Load+Topo
      → 若 Load 不可用: Model+Topo
      → 若 Topo 不可用: Model+Load
      → 若仅 Model 可用: Model (退化为就近+随机)
      → 若 Model 不可用: 返回 503
```

### 6.6 Softmax 采样

```
当 temperature > 0:
  logits = [-hybrid_score(b) / temperature for b in candidates]
  probs = softmax(logits)
  selected = sample(probs)
  
temperature = 0: 贪心（选最高分）
temperature → ∞: 均匀随机（退化为 round robin）
```

---

## 7. 会话亲和（Session Affinity）

### 7.1 目标

同一会话/对话的连续请求倾向路由到同一后端，最大化 KV Cache 复用。

### 7.2 实现

```
RoutingHistory:
  session_id → (backend_id, last_used, kv_overlap_at_route_time)
  TTL: 300 秒

路由时:
  if session_id in routing_history:
    last_backend, last_time, last_overlap = routing_history[session_id]
    if last_backend still healthy and last_overlap > threshold:
      return last_backend  // 直接复用，跳过完整评估
    else:
      走正常 Hybrid 评估
  
  评估完成后:
    routing_history[session_id] = (selected_backend, now, overlap_score)
```

### 7.3 跨实例共享

通过 Gossip 广播 routing history 更新（带 TTL），使不同 Gateway 实例能维持会话亲和。

---

## 8. 重试与故障转移

```
forward(backend, request):
  try:
    stream = connector.forward(backend, request)
    return stream
  except BackendError:
    1. 标记 backend 为 degraded (降级统计 +1)
    2. 从候选列表中移除该 backend
    3. 若重试次数 < max_retries:
       重新走 Hybrid 评估（排除已失败的 backend）
       forward(new_backend, request)
    4. else:
       return 503 Service Unavailable

max_retries = 3
```

---

## 9. 策略六：KV Capacity Aware Routing（KV 容量感知路由）

> 作为 `RoutingPlugin` 挂到 Hybrid（见 [05-kv-estimation.md](05-kv-estimation.md) 的数据半）。

### 9.1 目标

估算本次请求的 KV Cache 显存占用，按各后端**剩余容量**打分，把放不下的后端排除掉 —— 这是容量准入 / load shedding 决策。与 `KvAwareStrategy`（按前缀命中重叠打分，减少 prefill 工作）互补：`KvAwareStrategy` 决定「少做多少 prefill」，`KvCapacityStrategy` 决定「放不放得下」。

### 9.2 估算来源

请求占用来自独立叶子 crate `hier-kv-gateway-kv-estimate` 的解析公式（非仿真，与 vLLM/SGLang/Mooncake 一致）：

```
per_token = f(num_layers, num_kv_heads, head_dim, dtype, attention族)  // MLA 用 kv_lora_rank+qk_rope_head_dim
seq_len   = input_tokens + estimated_output_tokens   // output 用客户端 max_tokens 作保守上界
effective = min(seq_len, sliding_window)              // 滑动窗口截断
blocks    = ceil(effective / block_size) * batch_size
bytes     = per_token * batch_size * (blocks * block_size)   // block-padded
```

### 9.3 容量信号选择与打分

对每个候选后端：

```
1. 解析后端实际服务的模型（优先 ctx.model_name 精确匹配，否则后端首个模型）
2. registry.estimate(model, input) → None 时按 exclude_on_unknown_spec 处理
3. 读取后端资源余量，选容量信号：
   - KV-block 路径（精确，优先）：kv_total_blocks>0 且 block_size>0
       available_bytes = (kv_total_blocks - kv_used_blocks) * per_block_bytes
   - GPU 显存路径（保守 fallback）：gpu_memory_total_mb>0
       available_bytes = (gpu_memory_total_mb - gpu_memory_used_mb) * 1e6 * gpu_mem_safety_fraction
   - 无信号：中立 (raw_cost=0, score=1)，让其他子策略决定
4. 准入判断：
   if available_bytes <= 0 or bytes > available_bytes:
       排除 (raw_cost=∞, score=0)            // load shedding
   else:
       ratio = bytes / available_bytes ∈ [0,1]
       raw_cost = ratio                       // 余量越多 cost 越低
       score = 1 / (1 + ratio)
```

### 9.4 关键设计决策

1. **output 用 `max_tokens` 作保守上界**：估算的 KV 增长永不低估，镜像 `LoadAwareStrategy::w_decode` 与 `CostAwareStrategy` 的输出投影。
2. **`f64::INFINITY` 而非 `f64::MAX`**：排除用 `∞`（非有限），由 `HybridStrategy::normalize_costs` 通过 `!is_finite()` 识别。`f64::MAX` 是有限的，会被误判为「很贵但有效」。
3. **GPU 显存 fallback 用安全比例**：KV 不是唯一 GPU 内存消费者（还有权重、激活），仅「当前空闲显存 × `gpu_mem_safety_fraction`」可被声明，避免把整卡空闲都算给 KV。
4. **未知 spec 默认中立**：`exclude_on_unknown_spec=false` 时未知模型后端让其他子策略决定，避免在没把握时饿死确有余量的后端。
5. **与 `KvAwareStrategy` 独立归一化**：两策略在 Hybrid 中是独立子策略，各自 `normalize_costs`，互不搅语义 —— 这与 `LoadAwareStrategy` vs `CostAwareStrategy` 的关系一致。

### 9.5 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 关闭时不挂载策略 |
| `weight` | `0.20` | Hybrid 权重 |
| `gpu_mem_safety_fraction` | `0.5` | GPU 显存 fallback 可声明比例 |
| `exclude_on_unknown_spec` | `false` | 未知 spec 排除(true)/中立(false) |

### 9.6 配置示例

```toml
[kv_estimate]
enabled = true
weight = 0.20
gpu_mem_safety_fraction = 0.5
exclude_on_unknown_spec = false

# 可选：注册私有模型 spec（字段对应 HuggingFace config.json）
[[kv_estimate.models]]
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

### 9.7 端到端算例

Llama-3-8B（per_token=131_072 B），4096 prompt，block_size 16：

```
blocks_needed = ceil(4096/16) = 256
后端 A: kv_total=1000, kv_used=0   → free=1000, ratio=256/1000=0.256  (admitted)
后端 B: kv_total=1000, kv_used=700 → free=300,  ratio=256/300=0.853  (admitted, 更高 cost)
后端 C: kv_total=1000, kv_used=995 → free=5,    256>5                → 排除 (raw_cost=∞)
```
