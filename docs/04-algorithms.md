# Hier KV Gateway 算法设计文档

> 中文 | [English](en/04-algorithms.md)

> 本文档是系统的算法专档，覆盖 KV block 哈希、RadixTree、Cuckoo Filter、CKF Producer/Consumer、Gossip 协议、路由策略与混合评分等核心算法的实现细节。
> 所有常量与代码引用均来自 `crates/hier-kv-gateway-*` 的实际实现。

## 0. 算法总览

| 算法 | 位置 | 用途 |
|------|------|------|
| Block Hash | [hier-kv-gateway-core/src/kv_event.rs](../crates/hier-kv-gateway-core/src/kv_event.rs) | 把 token 序列切成 block，每块算 XXH3 哈希 |
| RadixTree | [hier-kv-gateway-metadata/src/radix_tree.rs](../crates/hier-kv-gateway-metadata/src/radix_tree.rs) | 本地精确 KV block 前缀索引 |
| Cuckoo Filter 原语 | [hier-kv-gateway-metadata/src/cuckoo_filter.rs](../crates/hier-kv-gateway-metadata/src/cuckoo_filter.rs) | fingerprint 寻址与 packed bucket 操作 |
| CKF Producer | [hier-kv-gateway-metadata/src/ckf_producer.rs](../crates/hier-kv-gateway-metadata/src/ckf_producer.rs) | 本地精确所有权 + 紧凑投影发布 |
| CKF Consumer | [hier-kv-gateway-metadata/src/ckf_consumer.rs](../crates/hier-kv-gateway-metadata/src/ckf_consumer.rs) | 跨 Region 近似 KV 索引（transposed 布局） |
| Gossip | [hier-kv-gateway-cluster/src/gossip.rs](../crates/hier-kv-gateway-cluster/src/gossip.rs) | 跨集群成员发现与元数据传播 |
| 路由策略 | [hier-kv-gateway-routing/src/*.rs](../crates/hier-kv-gateway-routing) | 5 种策略 + Hybrid 综合 |

---

## 1. Block Hash 计算

### 1.1 目标

把变长 token 序列切分为定长 block，每个 block 计算一个 64-bit 哈希。该哈希是后续 RadixTree 与 CKF 的基本寻址单元。

### 1.2 算法

实现于 [kv_event.rs](../crates/hier-kv-gateway-core/src/kv_event.rs) 的 `compute_block_hashes`：

```
输入: tokens: &[u32], kv_block_size: u32, cache_namespace: Option<&str>, lora_name: Option<&str>

1. seed = compute_seed(cache_namespace, lora_name)
2. 按 kv_block_size 把 tokens 切成不重叠窗口
   - 最后一个不完整块被丢弃
3. 对每个窗口:
   - 把窗口内 token 以小端字节写入缓冲
   - hash = xxh3_64_with_seed(bytes, seed)
4. 返回 hashes: Vec<u64>
```

### 1.3 种子派生

`compute_seed` 把命名空间与 LoRA 适配器独立混合进 XXH3 种子，保证不同租户/适配器下相同 token 序列产生不同哈希：

```
XXH3_SEED  = 1337
NS_SALT    = 0x4E53_5F4C_4F5F_4C4F
LORA_SALT  = 0x4C52_4F5F_4C4F_5F4C

seed = XXH3_SEED
if cache_namespace 非空:
    seed = seed.wrapping_add(xxh3_64_with_seed(ns_bytes, NS_SALT))
    seed ^= NS_SALT
if lora_name 非空:
    seed = seed.wrapping_add(xxh3_64_with_seed(lora_bytes, LORA_SALT))
    seed ^= LORA_SALT
```

**关键性质**：
- 空字符串视为未提供（与 `None` 等价）
- 命名空间与 LoRA 用独立 salt，两者影响相互独立且不可抵消
- 同名 namespace 与 lora_name（如都叫 "foo"）仍产生不同哈希

### 1.4 边界

- `kv_block_size == 0` → 返回空向量
- `tokens.len() < kv_block_size` → 返回空向量（无完整块）

---

## 2. RadixTree（本地精确 KV 索引）

### 2.1 数据结构

实现于 [radix_tree.rs](../crates/hier-kv-gateway-metadata/src/radix_tree.rs)。每个非根节点表示序列前缀中的一个 block hash：

```rust
struct Node {
    hash: u64,                              // 该节点的 block hash（根为 0）
    owners: HashSet<(BackendId, u32)>,      // 持有该 block 的 (backend, rank) 对
    children: HashMap<u64, Node>,           // 子节点，按 block hash 索引
    ref_count: u32,                         // owners 数量缓存
}
```

### 2.2 并发模型：后台线程 + mpsc channel

所有写操作通过专用后台线程串行化执行，读操作通过 `mpsc::Sender<RadixCommand>` + `oneshot` 同步返回结果：

```
调用方 (async)                      后台线程
   │                                  │
   ├─ RadixCommand::ApplyEvent ──────►│ apply_event()
   │  (oneshot::Sender)               │   ├─ Stored → add_owner
   │                                  │   ├─ Removed → remove_owner
   │◄─ done.send(Result) ─────────────┤   ├─ Clear → clear_backend
   │                                  │   └─ Reset → clear_backend (代际 fence)
   │                                  │
   ├─ RadixCommand::FindMatches ─────►│ find_matches()
   │◄─ done.send(u32) ────────────────┤
```

后台线程名 `hier-kv-gateway-radix-tree`，channel 容量 4096。这种设计：
- 内部无锁，实现简单
- 异步上下文安全调用（`async fn` 接口）
- Drop 时尽力 `try_send(Shutdown)`，若仍有 clone 持有 sender 则忽略

### 2.3 find_matches 算法

查询指定 backend 对给定 hash 序列的前缀重叠长度：

```
find_matches(hashes, backend):
    current = root
    overlap = 0
    for hash in hashes:
        child = current.children.get(hash)
        if child is None: break           // 无匹配前缀
        if not child.is_owned_by(backend): break  // 该 backend 不持有此后缀
        overlap += 1
        current = child
    return overlap
```

**关键性质**：前缀中断 — 一旦某 backend 不持有第 k 个块，即使它持有第 k+1 个块也不计入（因为推理 prefill 必须从前往后连续）。

### 2.4 find_all_matches

沿前缀路径收集每个 backend 的最大重叠长度：

```
find_all_matches(hashes):
    scores = {}
    current = root
    for hash in hashes:
        child = current.children.get(hash)
        if child is None: break
        if child.owners.is_empty(): break
        for (backend, _) in child.owners:
            scores[backend] += 1
        current = child
    return scores
```

### 2.5 事件应用

**Stored { block_hashes }**：把 `block_hashes` 当作从根出发的一条前缀路径，沿路径在每个节点为 `(backend, 0)` 添加 ownership（rank 默认 0）。

**Removed { block_hashes }**：`block_hashes` 是一组**独立**块哈希（非前缀路径）。在整棵树中搜索 hash 匹配的节点并移除该 backend 的 ownership。这是因为 content-addressed 块在缓存中可跨前缀共享。

**Clear { worker }**：递归移除该 backend 在所有节点上的 ownership。

**Reset { generation }**：代际 fence，语义等价于 Clear — 清空该 backend 的全部所有权。触发原因通常是 worker 重启或代际切换。

### 2.6 节点回收

自底向上回收空节点：当 `node.ref_count == 0 && node.children.is_empty()` 时，从父节点的 children 中删除该节点。根节点（hash=0）不参与回收，保证树结构始终保留。

---

## 3. Cuckoo Filter 基础原语

实现于 [cuckoo_filter.rs](../crates/hier-kv-gateway-metadata/src/cuckoo_filter.rs)。该模块只提供无状态的桶操作与寻址函数，上层 Producer/Consumer 组合这些原语。

### 3.1 常量

```
FINGERPRINT_BITS  = 16       // 指纹位数
FP_PER_BUCKET     = 4        // 每 bucket 的指纹数
MAX_KICKS         = 500      // 单次插入最大踢出次数
BUCKETS_PER_LANE  = 65536    // 单 lane 的 bucket 数（必须为 2 的幂）
BUCKET_MASK       = 0xFFFF   // bucket 索引掩码
ALT_MIX_DOMAIN    = 0x9E37_79B9_7F4A_7C15  // alt_index 混合常量
```

### 3.2 PackedBucket

一个 `u64` 打包 4 个 16-bit 指纹，slot 0 在低位：

```
| slot 3 | slot 2 | slot 1 | slot 0 |
| 63..48 | 47..32 | 31..16 | 15..0  |
```

`Fp = u16`，`0` 保留为"空槽"哨兵。

### 3.3 Partial-key Cuckoo Hashing

使用 partial-key cuckoo 寻址，只存储指纹不存储完整键：

```
probe(hash):
    mixed = xxh3_64_with_seed(hash.to_le_bytes(), 0)
    fp    = (mixed as u16) | 1        // 最低位置 1，避免生成 0
    bucket = ((mixed >> 16) as usize) & BUCKET_MASK
    return (fp, bucket)

alt_index(idx, fp):
    mixed = xxh3_64_with_seed(fp.to_le_bytes(), ALT_MIX_DOMAIN)
    delta = (mixed as usize) & BUCKET_MASK
    delta = 1 if delta == 0 else delta   // 避免 delta=0 导致两候选重合
    return (idx ^ delta) & BUCKET_MASK
```

**关键性质**：`alt_index` 是对合（involution）：`alt_index(alt_index(idx, fp), fp) == idx`。这保证插入与查找使用相同的两个候选 bucket。

### 3.4 SIMD-friendly bucket_contains

不逐 slot 比较，而是用位运算同时检测 4 个 slot：

```
bucket_contains(bucket, fp):
    repeated  = u64(fp) * 0x0001_0001_0001_0001   // 把 fp 复制到 4 个 slot
    different = bucket ^ repeated
    high_bits = 0x8000_8000_8000_8000              // 每 slot 的最高位
    return (different.wrapping_sub(0x0001_0001_0001_0001)
            & !different
            & high_bits) != 0
```

该公式利用"无借位减法 + 异或"判断是否存在等于 fp 的 slot，编译器可向量化。

### 3.5 桶操作

```
try_insert(bucket, fp): 找空槽写入，满返回 false
try_delete(bucket, fp): 找匹配槽置 0，未找到返回 false
first_match(bucket, fp): 返回首个匹配槽位
first_empty(bucket): 返回首个空槽
```

---

## 4. CKF Producer

实现于 [ckf_producer.rs](../crates/hier-kv-gateway-metadata/src/ckf_producer.rs)。每个 pool 一个 Producer，维护本地精确所有权 + 紧凑 CKF 投影。

### 4.1 状态

```rust
struct CkfProducer {
    buckets: Vec<PackedBucket>,              // lane 内所有 bucket
    num_items: u64,                           // 已插入 fingerprint 数
    dirty_buckets: HashSet<usize>,            // 自上次发布以来变动的 bucket
    pub_seq: u64,                             // 已发布的最大序列号
    hash_refcount: HashMap<u64, HashEntry>,   // hash → (refcount, owners)
    worker_hashes: HashMap<BackendId, HashSet<u64>>,  // backend → 持有的 hash 集合
    rng_state: u64,                           // splitmix64 PRNG 状态
}

struct HashEntry {
    refcount: u32,
    owners: HashSet<BackendId>,
}
```

### 4.2 所有权 4-分支规则

应用 `Stored { block_hashes }` 时，对每个 hash：

```
apply_stored(hash, worker):
    entry = hash_refcount.entry(hash).or_default()
    first_owner = entry.owners.is_empty()
    if not entry.owners.insert(worker):    // 已被该 worker 持有
        return                             // 去重，不增 refcount 也不插 fingerprint
    entry.refcount += 1
    worker_hashes[worker].insert(hash)
    if first_owner:
        insert_fingerprint(hash)           // 首个 owner → 插 fingerprint
```

应用 `Removed { block_hashes }` 时，对每个 hash：

```
apply_removed(hash, worker):
    entry = hash_refcount.get_mut(hash) or return
    if not entry.owners.remove(worker): return
    worker_hashes[worker].remove(hash)
    entry.refcount -= 1
    if entry.refcount == 0:
        delete_fingerprint(hash)           // 最后一个 owner → 删 fingerprint
        hash_refcount.remove(hash)
```

**4 分支总结**：

| 场景 | 行为 |
|------|------|
| First owner of a hash | 插入 fingerprint |
| Another owner of same hash | 仅 refcount++ |
| One of several removes | 仅 refcount-- |
| Final owner removes | 删除 fingerprint |

应用 `Clear { worker }`：迭代 `worker_hashes[worker]`，对每个 hash 走"one of several removes / final owner removes"分支。

应用 `Reset`：清空整个 producer 状态（代际 fence）。

### 4.3 Cuckoo 插入（含踢出与回滚）

```
insert_fingerprint(hash):
    (fp, bucket_a) = probe(hash)
    bucket_b = alt_index(bucket_a, fp)
    
    // 先尝试直接插入两个候选 bucket
    if try_insert(buckets[bucket_a], fp): mark dirty; num_items++; return true
    if try_insert(buckets[bucket_b], fp): mark dirty; num_items++; return true
    
    // 进入踢出循环
    touched = []
    current_bucket = bucket_a or bucket_b (随机)
    current_fp = fp
    for _ in 0..MAX_KICKS:
        before = buckets[current_bucket]
        slot_idx = next_random() & 0x3
        evicted = slot(before, slot_idx)
        buckets[current_bucket] = with_slot(before, slot_idx, current_fp)
        touched.push((current_bucket, before))
        current_fp = evicted
        current_bucket = alt_index(current_bucket, current_fp)
        if try_insert(buckets[current_bucket], current_fp):
            mark all touched + current_bucket dirty
            num_items++; return true
    
    // 达到 MAX_KICKS 仍未成功 → 回滚所有踢出
    for (idx, before) in touched.rev():
        buckets[idx] = before
    return false
```

**关键点**：
- 踢出过程中记录所有 touched bucket 的原始值，失败时按 LIFO 回滚
- splitmix64 PRNG 确定性但分布良好，避免引入额外依赖
- 失败（lane 满）时上层仅记 warning，不阻塞 ingestion

### 4.4 Barrier Snapshot + Sequenced Delta

```
snapshot():
    pub_seq += 1
    dirty_buckets.clear()
    return CkfSnapshot { sequence: pub_seq, buckets: buckets.clone() }

delta():
    if dirty_buckets.is_empty(): return None
    prev = pub_seq
    pub_seq += 1
    buckets = dirty_buckets.iter().map(|idx| (idx, buckets[idx])).collect()
    buckets.sort_by_idx()           // 排序便于 consumer 应用与诊断
    dirty_buckets.clear()
    return CkfDelta { sequence: pub_seq, prev_sequence: prev, buckets }
```

**语义**：
- Snapshot 是全量绝对镜像，consumer 可独立安装
- Delta 只含 dirty bucket 的当前绝对值（非补丁），consumer 直接覆盖
- 序列号单调递增（`wrapping_add`），consumer 可检测乱序

---

## 5. CKF Consumer（Transposed Layout）

实现于 [ckf_consumer.rs](../crates/hier-kv-gateway-metadata/src/ckf_consumer.rs)。每个 Gateway 实例内运行一个 Consumer，跟踪多个 Region 的 CKF 投影。

### 5.1 Transposed 布局

```
LANE_COUNT = 16   // 最多同时跟踪 16 个 Region

buckets: Vec<[AtomicU64; 16]>   // bucket-major
                                  // buckets[i][lane] 是 bucket i 在 lane 上的 packed 值
```

**为什么 transposed**：按 bucket 组织而非按 lane 组织，使得一次前缀查询（连续访问多个 bucket）在同一 lane 上沿 `buckets[0..k][lane]` 推进，缓存友好；同时多个 lane 的同一 bucket 共享缓存行，便于并发 probe。

### 5.2 Lane 状态机

```
LANE_ACTIVE  = 0
LANE_RETIRED = 1

lane_status: [AtomicU8; 16]
```

- `Active`：查询可见
- `Retired`：查询不可见（lane 重连中或已退役）

### 5.3 estimate_overlap 算法

```
estimate_overlap(hashes, region):
    lane = lane_of(region) or return 0
    if lane_status[lane].load(Acquire) != LANE_ACTIVE: return 0
    
    overlap = 0
    for hash in hashes:
        (fp, bucket_idx) = probe(hash)
        packed = buckets[bucket_idx][lane].load(Acquire)
        if bucket_contains(packed, fp):
            overlap += 1
        else:
            break                          // 前缀中断
    return overlap
```

**关键性质**：
- 前缀中断：与 RadixTree 一致，第 k 块未命中则停止
- 无 lane-wide lock：每个 bucket 独立 atomic，读端无锁
- 假阳性：CKF 可能返回 false positive（某 Region 似乎有该 block），由后续精确查询或请求结果校正

### 5.4 Snapshot 安装（retired → write → active）

```
install_snapshot(lane, snapshot):
    lane_status[lane].store(RETIRED, Release)      // 1. 屏蔽读
    for (i, value) in snapshot.buckets.iter().enumerate():
        buckets[i][lane].store(value, Relaxed)     // 2. 写所有 bucket
    lane_status[lane].store(ACTIVE, Release)       // 3. 恢复读
```

三步顺序保证：读端要么看到完整的旧快照，要么看到完整的新快照，不会看到中间状态。

### 5.5 Delta 应用（弱一致）

```
apply_delta(lane, delta):
    for (bucket_idx, value) in delta.buckets:
        buckets[bucket_idx][lane].store(value, Release)
```

Delta 是多 bucket 的弱一致写入，读端可能观察到部分应用的状态。这是 CKF 的设计取舍 — 假阳性容忍使得不需要 seqlock 或重试。

### 5.6 Lane 生命周期

```
assign_lane(lane, region)   // 绑定 lane 到 Region
activate_lane(lane)         // 标记 Active
retire_lane(lane)           // 标记 Retired（查询排除）
unassign_lane(lane)         // 解绑（Region 迁出）
```

故障恢复：lane 断开时 `retire_lane`，重连时 `install_snapshot` 安装新 barrier snapshot 后 `activate_lane`。

---

## 6. Gossip 协议

实现于 [hier-kv-gateway-cluster/src/gossip.rs](../crates/hier-kv-gateway-cluster/src/gossip.rs)。

### 6.1 消息类型

| 消息 | 用途 |
|------|------|
| `PING / PONG` | 心跳，携带发送方的元数据摘要 |
| `MEET` | 新节点加入集群 |
| `SYNC` | 请求全量状态同步（新节点或修复） |
| `CKF_PUBLISH` | 跨 Region KV 投影发布（barrier + delta） |
| `METRIC_BROADCAST` | 负载/延迟指标广播 |

### 6.2 Gossip 行为

```
每个 Gateway 实例维护:
    members: HashMap<InstanceId, ClusterMember>
        ClusterMember = { instance_id, region, addr, last_pong_unix, status }

每秒:
    1. 随机选 P 个 alive 成员发 PING
    2. PONG 携带最新元数据摘要 (MetaDigest)
    3. 若 PING 超时 → suspect_count++
    4. 连续 N 次失败 → status = Suspect → 确认 Dead
```

### 6.3 元数据版本同步

```
MetaDigest = {
    kv_version: u64,
    model_version: u64,
    load_version: u64,
    topology_version: u64,
    members_version: u64,
}

接收 PONG:
    for each (region, version) in pong.digest:
        if local_version < version:
            发送 SYNC 请求该 region 的增量
```

大状态（CKF 投影）不放在 PING 中，单独走 barrier snapshot + sequenced delta。

### 6.4 成员状态机

```
Alive ──PING 超时──► Suspect ──确认──► Dead
   ▲                                       │
   └─────────重新 PONG─────────────────────┘
```

新成员通过 `MEET` 加入：收到 MEET 的实例将其加入 members 并在后续 Gossip 中传播。

---

## 7. 路由策略算法

### 7.1 通用评分模型

所有策略产出统一的 `ScoredBackend`：

```rust
struct ScoredBackend {
    backend_id: BackendId,
    score: f64,        // [0, 1]，1.0 = 最优
    raw_cost: f64,    // 策略原始成本（越低越好）
    meta_version: u64,
}
```

### 7.2 KV Aware（KV 感知路由）

实现于 [kv_aware.rs](../crates/hier-kv-gateway-routing/src/kv_aware.rs)。

```
对每个候选 backend b:
    local_overlap  = RadixTree.find_matches(hashes, b)         // 本地精确
    remote_overlap = CkfConsumer.estimate_overlap(hashes, b.region)  // 跨域近似
    
    effective_remote = remote_overlap * (1 - ckf_false_positive_penalty)
    total_overlap = local_overlap + effective_remote
    
    prefill_blocks = max(len(hashes) - total_overlap, 0)
    decode_blocks  = b.active_decode_blocks   // 来自 LoadStats
    
    cost  = prefill_load_scale * prefill_blocks + decode_blocks
    score = 1.0 / (1.0 + cost) + overlap_score_credit * total_overlap
```

**默认参数**：
- `overlap_score_credit = 1.0`
- `prefill_load_scale = 1.0`
- `ckf_false_positive_penalty = 0.0`（可在配置中启用）
- 权重 `weight() = 0.35`

**可用性判断**：`meta.kv_confidence() > 0.0`，否则策略不可用（触发降级）。

### 7.3 Model Aware（模型感知路由，硬过滤器）

实现于 [model_aware.rs](../crates/hier-kv-gateway-routing/src/model_aware.rs)。作为硬性过滤器，剔除不匹配的候选：

```
对每个候选 backend b:
    score = match_degree(b, request.model):
        exact_match      (model_name + version + quant)  → 1.0
        model_match      (同名不同版本)                   → 0.7
        compatible_match (同架构不同名)                   → 0.3
        no_match                                         → 0.0 (排除)
    
    额外检查:
        - max_context_len >= request.token_count ?
        - supports_tool_calling >= request.requires_tool_calling ?
    
    cost = 1.0 - score
```

`score == 0.0` 的候选被 Hybrid 策略过滤掉。

### 7.4 Load Aware（负载感知路由）

实现于 [load_aware.rs](../crates/hier-kv-gateway-routing/src/load_aware.rs)。

```
对每个候选 backend b:
    m = LoadStats.get_metrics(b)

    # token-budget 项（保守上界，详见 7.4.1）
    req_decode_blocks = ceil(ctx.estimated_output_tokens / ctx.block_size)   # 本请求将占用的 decode 块
    projected_decode  = m.active_decode_blocks + req_decode_blocks           # 落在此 backend 后的 decode 压力
    prefill_pressure  = m.active_prefill_tokens

    load_cost = w_req    * m.active_requests
              + w_queue  * m.queue_depth
              + w_lat    * (m.p99_latency / 100)
              + w_gpu    * m.gpu_utilization
              + w_kv     * m.kv_cache_usage
              + w_decode * projected_decode        # 投影 decode 压力
              + w_prefill * prefill_pressure        # 当前 prefill 压力

    容量检查: if available_capacity <= 0: 排除

    cost  = load_cost
    score = 1.0 / (1.0 + load_cost)
```

**默认权重**：`w_req=1.0, w_queue=1.0, w_lat=0.01, w_gpu=1.0, w_kv=1.0, w_decode=0.02, w_prefill=0.001`

> 设 `w_decode = 0` 且 `w_prefill = 0` 即可逐字节复现改动前的 count-blind 成本（向后兼容开关）。

**滑动窗口**：60 秒窗口，1 秒采样间隔，p50/p99 用近似算法。

#### 7.4.1 Token-budget 感知（投影 decode / prefill 压力）

改动前 `load_cost` 仅按 `active_requests` 计数，对生成长度**无感**：持有 1 个 4096-token 请求的 backend 会被判为比持有 4 个 16-token 请求的 backend「更空闲」，尽管前者占用约 64× 的 decode 容量。`RoutingContext::estimated_output_tokens` 与 `BackendMetrics::active_prefill_tokens` 此前已被采集但未被任何策略消费。

新增两项闭合该缺口：

- **投影 decode 压力**（`w_decode`）：backend 当前 `active_decode_blocks` *加上* 本请求的输出预算会新增的块数。`estimated_output_tokens` 源自客户端 `max_tokens`（生成的硬上界），因此投影**永不低估** decode 占用——遵循业界对输出长度估计采用保守上界而非点估计的结论（避免饥饿与热点）。
- **Prefill 压力**（`w_prefill`）：backend 的 `active_prefill_tokens`，一个此前已采集但未入软成本项的信号。不与本请求的 prompt 投影合并，因为 load 策略拿不到 KV overlap（那是 KV 策略的领域）；保持两策略独立以维护 Hybrid 归一化语义。

**验证**：见 [token-aware-load.md](benchmarks/token-aware-load.md)。在真实 `MetadataStore` + `RoutingEngine` 的离散事件回放（180 请求混合短/长生成、含完成事件）下，token-aware 相对 count-blind 基线：decode 压力跨 backend 的 CoV 下降 **31.8%**（0.070 → 0.048），峰值 690 → 651 blocks（clairvoyant 下界 616.7）；n=20 候选时路由延迟与基线**统计不可区分**（baseline ≈ 40 µs、token_aware ≈ 39 µs，多次运行 median 互有高低，开销低于测量噪声底），远低于 10% 阈值。两项均满足引入判据（CoV 改善 ≥15%、延迟开销 <10%）。

### 7.5 Topology Aware（拓扑感知路由）

实现于 [topology_aware.rs](../crates/hier-kv-gateway-routing/src/topology_aware.rs)。

```
对每个候选 backend b:
    rtt = LatencyMatrix.rtt_ms(self_region, b.region)
    
    network_cost = w_rtt * rtt + w_bw * bandwidth_penalty
    
    层级偏好:
        if self.tier == Device and b.tier == Edge:  network_cost *= 0.8
        if self.tier == Device and b.tier == Cloud: network_cost *= 1.5
    
    cost  = network_cost
    score = 1.0 / (1.0 + network_cost / 100.0)   // 100ms 基准
```

**RTT 来源**：配置 + 主动探测 + Gossip 传播 + 地理距离估算（`rtt ≈ distance_km / 200`，光纤）。

---

## 8. Hybrid 混合路由（默认策略）

实现于 [hybrid.rs](../crates/hier-kv-gateway-routing/src/hybrid.rs)。

### 8.1 算法流程

```
1. Model Aware 做硬性过滤
    filtered = [b for b in candidates if model.evaluate(b).score > 0]
    if filtered.is_empty(): return RoutingFailed

2. 动态权重调整
    weight_kv  = kv.is_available(meta) ? weights.kv : 0
    load_stale = any(|c| load_freshness(c) > 10s for c in filtered)
    weight_load = load_stale ? weights.load * 0.3 : weights.load
    weight_topo = weights.topology
    
    归一化: total = weight_kv + weight_load + weight_topo
            if total > 0: 三者各自 /= total
            else:          三者 = 1/3 (兜底均匀)

3. 各子策略对 filtered 评分
    kv_scores   = kv.evaluate(...)        if weight_kv  > 0 else []
    load_scores = load.evaluate(...)      if weight_load > 0 else []
    topo_scores = topology.evaluate(...)  if weight_topo > 0 else []

4. 各策略 raw_cost 归一化到 [0, 1]
    normalize_costs(scores):
        min = min(s.raw_cost for s in scores if finite)
        max = max(s.raw_cost for s in scores if finite)
        span = max - min
        for s in scores:
            if not finite(s.raw_cost): norm = 0     // 不满足约束
            elif span > 0: norm = (s.raw_cost - min) / span
            else: norm = 0
            s.normalized = 1.0 - norm                // 成本越低分越高

5. 加权求和
    for c in filtered:
        hybrid_score(c) = weight_kv  * kv_norm[c]
                        + weight_load * load_norm[c]
                        + weight_topo * topo_norm[c]
        raw_cost = -hybrid_score   // 保持 "raw_cost 越低越好" 语义

6. 按 hybrid_score 降序排序
```

### 8.2 关键常量

```
STALE_LOAD_THRESHOLD_SECS = 10   // 负载指标过期阈值
```

### 8.3 权重默认值

| 策略 | 基础权重 | 角色 |
|------|---------|------|
| KV | 0.35 | 加权 |
| Load | 0.30 | 加权 |
| Topology | 0.20 | 加权 |
| Model | 1.0 (过滤) | 硬过滤器，不参与加权 |

### 8.4 Softmax 采样（在路由引擎层）

```
if temperature > 0:
    logits = [-hybrid_score(b) / temperature for b in candidates]
    probs  = softmax(logits)
    selected = sample(probs)
else:
    selected = argmax(hybrid_score)   // 贪心
```

- `temperature = 0`：贪心选最高分
- `temperature → ∞`：均匀随机（退化为 round robin）

### 8.5 降级链

```
Hybrid (KV + Load + Topo)
  │ KV Index 不可用 (kv_confidence == 0)
  ▼
Model + Load + Topo       (weight_kv = 0)
  │ Load Stats 全部过期 (>10s)
  ▼
Model + Topo              (weight_load *= 0.3)
  │ 跨集群通信断开
  ▼
本地 Load Aware
  │ 本地无可用 Backend
  ▼
返回 503
```

---

## 9. 会话亲和（Session Affinity）

实现于 [routing_history.rs](../crates/hier-kv-gateway-metadata/src/routing_history.rs) 与路由引擎。

```
路由时:
    if session_id in routing_history:
        (last_backend, last_time, last_overlap) = routing_history[session_id]
        if last_backend still healthy and last_overlap > 0:
            return last_backend          // 直接复用，跳过完整评估
        else:
            走正常 Hybrid 评估

评估完成后:
    routing_history[session_id] = (selected_backend, now, overlap_score)

TTL: 300 秒
```

跨实例共享：通过 Gossip 广播 routing history 更新（带 TTL），使不同 Gateway 实例维持会话亲和。

---

## 10. 重试与故障转移

```
forward(backend, request):
    try:
        stream = connector.forward(backend, request)
        return stream
    except BackendError:
        1. 标记 backend 为 degraded (降级统计 +1)
        2. 从候选列表中移除该 backend
        3. if 重试次数 < max_retries (默认 3):
               重新走 Hybrid 评估（排除已失败的 backend）
               forward(new_backend, request)
        4. else:
               return 503 Service Unavailable
```

---

## 11. 故障恢复（Narrowest State Boundary）

按"最窄状态边界"原则设计故障恢复，每种故障只影响最小的状态子集：

| 故障 | 恢复边界 | 行为 |
|------|---------|------|
| Backend 事件 gap | 该 backend 的 rank 状态 | 从 backend 事件历史恢复，或安装当前 tree state |
| Backend 替换 | 该 backend 所有状态 | completion barrier 后从新 source 重建 |
| CKF delivery gap | 受影响的 consumer lane | `retire_lane`，重连时 `install_snapshot` 安装新 barrier |
| Gateway 实例崩溃 | 该实例的本地状态 | 其他实例通过 Gossip 感知，接管路由；新实例 SYNC 全量状态 |
| Region 隔离 | 该 Region 的 lane | 路由排除该 Region；恢复后 lane 重新激活 |

---

## 12. 元数据缓存层级

| 层级 | 内容 | 生命周期 |
|------|------|---------|
| L1 Request-Local | block hashes, overlap scores | 单次请求 |
| L2 Hot | RadixTree, LoadStats (TTL 5s), CKF Consumer | 常驻内存，实时更新 |
| L3 Warm | ModelRegistry (TTL 60s), Topology (TTL 30s), Discovery (TTL 15s) | 定期刷新 |
| L4 Cold | RoutingHistory (TTL 300s), DegradationStats (TTL 60s) | 按需查询 + 定期清理 |

### 并发安全实现

| 组件 | 并发策略 |
|------|---------|
| RadixTree | 专用后台线程 + mpsc channel（无锁） |
| LoadStats | `DashMap<BackendId, ArcSwap<Metrics>>`，读无锁 |
| CKF Consumer | bucket 级 `AtomicU64`，无 lane-wide lock |
| ModelRegistry | `Arc<RwLock<...>>`，读多写少 |

---

## 13. 算法复杂度参考

| 操作 | 时间复杂度 | 备注 |
|------|----------|------|
| `compute_block_hashes` | O(n / block_size) | n = token 数 |
| `RadixTree.find_matches` | O(k) | k = block 数（前缀中断平均更优） |
| `RadixTree.apply_stored` | O(k) | k = block_hashes 长度 |
| `RadixTree.apply_removed` | O(N) 最坏 | N = 树节点数（全局搜索匹配 hash） |
| `CkfProducer.insert_fingerprint` | O(1) 平均，O(MAX_KICKS) 最坏 | 500 次踢出后回滚 |
| `CkfConsumer.estimate_overlap` | O(k) | k = block 数，每 bucket 一次 atomic load |
| `Hybrid.evaluate` | O(C × S) | C = 候选数，S = 子策略数（≤3） |
