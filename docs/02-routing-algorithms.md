# Aether 路由算法设计

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

### 2.2 参考来源

Dynamo KV Router 成本函数：

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

### 2.3 Aether 适配算法

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

参考 Dynamo 的 `compute_block_hash_for_seq`：

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

参考 Dynamo 的 transposed CKF：

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

### 4.2 参考来源

Dynamo 的 `active_decode_blocks`、`potential_prefill_tokens`、queue policy。

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
     若 temperature > 0: 用 softmax 采样（参考 Dynamo router_temperature）
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

### 6.6 Softmax 采样（参考 Dynamo）

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
