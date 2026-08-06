# Hier KV Gateway Routing Algorithm Design

> English | [中文](../02-routing-algorithms.md)

> Detailed algorithms for the five routing strategies + the Hybrid mixed strategy

## 1. Unified Scoring Model

All strategies ultimately produce a unified `ScoredBackend` structure, which the Hybrid strategy aggregates:

```rust
struct ScoredBackend {
    backend_id: BackendId,
    /// Score normalized to [0, 1], 1.0 = optimal
    score: f64,
    /// Cost given by this strategy (lower is better; used for Hybrid weighting)
    raw_cost: f64,
    /// Metadata snapshot version on which the score is based
    meta_version: u64,
}
```

Each strategy produces its own `raw_cost` independently; the Hybrid strategy normalizes each strategy's cost and then takes a weighted sum.

---

## 2. Strategy 1: KV Aware Routing

### 2.1 Goal

Route requests to the backend with the largest KV Cache prefix overlap, maximizing cache reuse and reducing prefill computation.

### 2.2 Cost Model

KV Router cost function:

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

### 2.3 Hier KV Gateway Adaptation Algorithm

In a cloud-edge-device environment, KV Cache may exist in:
- A local Region's Backend (exact, queried via RadixTree)
- A remote Region's Backend (approximate, queried via CKF)

```
For each candidate Backend b:
  1. Compute the request's block_hashes = compute_block_hashes(token_ids, block_size)
  2. Query the local RadixTree (if b is in the local Region):
       device_overlap = radix_tree.find_matches(block_hashes, b)  // exact
  3. Query the cross-Region CKF Consumer (if b is in a remote Region):
       ckf_overlap = ckf_consumer.estimate_overlap(block_hashes, b.region)  // approximate
  4. total_overlap = device_overlap + ckf_overlap
  5. prefill_blocks = len(block_hashes) - total_overlap
  6. decode_blocks = b.active_decode_blocks (from Load Stats)
  7. cost = prefill_load_scale * prefill_blocks + decode_blocks
  8. score = 1.0 / (1.0 + cost)  // normalized
```

### 2.4 Block Hash Computation

Following the `compute_block_hash_for_seq` approach:

```
Chunk the token sequence by block_size:
  For each block:
    block_content = tokens[start..start+block_size]
    hash = xxhash64(block_content, seed=cache_namespace_hash)
    If LoRA: hash = xxhash64(hash || lora_name)
    If multimodal: hash = xxhash64(hash || mm_info)
  Return the block_hashes array
```

### 2.5 RadixTree (Local Exact Query)

```
RadixTree:
  root: Node
  Node:
    hash: u64               // block hash of this node
    children: HashMap<u64, Node>  // child nodes
    owners: Set<(backend_id, dp_rank)>  // which backends own this block
    is_terminal: bool       // whether this is the endpoint of a complete cache path

find_matches(block_hashes, target_backend):
  node = root
  overlap = 0
  for hash in block_hashes:
    if hash in node.children:
      node = node.children[hash]
      if target_backend in node.owners:
        overlap += 1
      else:
        break  // this backend does not own this suffix; stop
    else:
      break  // no matching prefix
  return overlap
```

### 2.6 CKF Consumer (Cross-Region Approximate Query)

Transposed CKF implementation:

```
CKFConsumer:
  lanes: [CKFLane; MAX_REGIONS]  // one lane per Region
  num_buckets: usize

  estimate_overlap(block_hashes, target_region):
    lane = lanes[target_region.lane_index]
    overlap = 0
    for hash in block_hashes:
      fp = fingerprint(hash)       // take the low 16 bits
      bucket_idx = hash % num_buckets
      alt_bucket_idx = alt_hash(fp, bucket_idx) % num_buckets
      if lane.bucket_contains(bucket_idx, fp) 
         or lane.bucket_contains(alt_bucket_idx, fp):
        overlap += 1
      else:
        break  // prefix break
    return overlap  // may be inflated due to CKF false positives
```

### 2.7 Parameters

| Parameter | Default | Description |
|------|--------|------|
| `kv_block_size` | 16 | Number of tokens per KV block |
| `overlap_score_credit` | 1.0 | device overlap credit multiplier |
| `prefill_load_scale` | 1.0 | prefill cost scaling |
| `ckf_false_positive_penalty` | 0.3 | penalty coefficient for CKF false positives |

---

## 3. Strategy 2: Model Aware Routing

### 3.1 Goal

Route to a backend that has a compatible model loaded, based on the model required by the request. Considers model version, quantization, and capabilities.

### 3.2 Algorithm

```
For each candidate Backend b:
  1. Query the Model Registry: which models does b have loaded?
  2. Match-degree computation:
     - exact_match (model_name + version + quant): score = 1.0
     - model_match (same name, different version): score = 0.7
     - compatible_match (same architecture, different name, e.g., Qwen2.5-7B vs Qwen2.5-14B): score = 0.3
     - no_match: score = 0.0 (exclude this candidate)
  3. Additional bonuses:
     - Quantization preference: if the request prefers high precision, fp16 > int8 > int4
     - Context length: if the request's token count > b.max_context, exclude
     - Tool-calling capability: if the request requires function_calling, check whether b supports it
  4. cost = 1.0 - match_score
```

### 3.3 Model Compatibility Matrix

```
Compatibility determination (refer to HuggingFace model config):
  - architecture: same transformer architecture (e.g., Qwen2, Llama)
  - vocab_size: compatible tokenizer
  - hidden_size / num_layers: may differ (but KV Cache is not shared)
  - quantization: does not affect compatibility determination, but affects quality scoring
```

---

## 4. Strategy 3: Load Aware Routing

### 4.1 Goal

Perform load balancing based on backends' real-time load (queue depth, GPU utilization, active request count) to avoid hotspots.

### 4.2 Key Metrics

`active_decode_blocks`, `potential_prefill_tokens`, queue policy.

### 4.3 Algorithm

```
For each candidate Backend b:
  1. Query Load Stats: get b's recent metrics
     - active_requests: current number of active requests
     - queue_depth: number of queued requests
     - avg_p50_latency / avg_p99_latency: latency statistics
     - gpu_utilization: GPU utilization (0-1)
     - kv_cache_usage: KV Cache usage (0-1)
     - available_capacity: remaining capacity
  2. Compute load cost:
     load_cost = w_req * active_requests
               + w_queue * queue_depth
               + w_lat * normalize(avg_p99_latency)
               + w_gpu * gpu_utilization
               + w_kv * kv_cache_usage
  3. Capacity check:
     if available_capacity <= 0: exclude this candidate (score = 0)
  4. cost = load_cost
  5. score = 1.0 / (1.0 + load_cost)
```

### 4.4 Sliding-Window Statistics

```
LoadStats maintains a sliding window per Backend:
  - Window size: 60 seconds
  - Sampling interval: 1 second
  - Storage: RingBuffer<Metrics>
  - Computation: p50/p99 uses approximate algorithms (e.g., t-digest) to avoid storing all data

  Metrics update:
    - Request start: active_requests += 1
    - Request end: active_requests -= 1, record latency
    - Periodic collection: pull gpu_utilization / kv_cache_usage from the connector
```

### 4.5 Parameters

| Parameter | Default | Description |
|------|--------|------|
| `w_req` | 1.0 | Active-requests weight |
| `w_queue` | 2.0 | Queue-depth weight |
| `w_lat` | 1.5 | Latency weight |
| `w_gpu` | 0.5 | GPU utilization weight |
| `w_kv` | 0.8 | KV usage weight |
| `stats_window_secs` | 60 | Statistics window |
| `stats_sample_interval` | 1 | Sampling interval (seconds) |

---

## 5. Strategy 4: Topology Aware Routing

### 5.1 Goal

Route preferentially to nearby backends based on the network latency topology, reducing end-to-end latency.

### 5.2 Data Structure

```
TopologyGraph:
  regions: HashMap<RegionId, RegionInfo>
  latency_matrix: HashMap<(RegionId, RegionId), LatencyEstimate>
  
RegionInfo:
  region_id: RegionId
  tier: Cloud | Edge | Device      // tier
  geo: (lat: f64, lon: f64)        // geographic coordinates
  network_zone: String             // network zone

LatencyEstimate:
  rtt_p50: Duration
  rtt_p99: Duration
  bandwidth_mbps: f64
  last_updated: Instant
```

### 5.3 Latency Matrix Construction

```
Latency matrix sources:
  1. Configuration: statically configured latencies between known Regions
  2. Active probing: Gateway instances ping each other to measure RTT
  3. Gossip propagation: probing results are shared via Gossip
  
Updates:
  - Actively probe neighboring Regions every 30 seconds
  - Take the p50 of the most recent 5 probes
  - For unprobed Region pairs, estimate using geographic distance:
    rtt_estimate = distance_km / 200km_per_ms (fiber)
```

### 5.4 Algorithm

```
For each candidate Backend b:
  1. Query the TopologyGraph: get the latency for (self_region, b.region)
  2. network_cost = rtt_p50_ms(b) * w_rtt
                  + bandwidth_penalty(b) * w_bw
  3. Tier preference:
     if self.tier == Device and b.tier == Edge:
       network_cost *= 0.8  // device side prefers edge
     if self.tier == Device and b.tier == Cloud:
       network_cost *= 1.5  // device side avoids cloud (high latency)
  4. cost = network_cost
  5. score = 1.0 / (1.0 + network_cost / 100.0)  // 100ms baseline
```

### 5.5 Parameters

| Parameter | Default | Description |
|------|--------|------|
| `w_rtt` | 1.0 | RTT weight |
| `w_bw` | 0.3 | Bandwidth penalty weight |
| `topology_refresh_secs` | 30 | Topology refresh interval |
| `geo_latency_factor` | 200 | km/ms conversion factor |

---

## 6. Strategy 5: Hybrid Routing (Hybrid Intelligent Routing, Default Strategy)

### 6.1 Goal

Fuse the four strategies — KV / Model / Load / Topology — and make a combined decision via weighted scoring.

### 6.2 Algorithm

```
Hybrid strategy:
  1. Collect the candidate Backend set C = candidates after Model Aware filtering
     (Model Aware acts as a hard filter; non-matching candidates are excluded)
  
  2. For each available strategy S ∈ {KV, Load, Topology}:
     If S.is_available():
       scores_S = S.evaluate(ctx, C, meta)  // one score per backend
  
  3. For each candidate b ∈ C:
     hybrid_score(b) = Σ_S( weight_S * normalize(scores_S[b]) )
     
     where normalize maps each strategy's raw_cost to [0, 1]:
       normalize(cost) = (cost - min_cost_S) / (max_cost_S - min_cost_S)
       
     Dynamic weight adjustment:
       weight_KV = base_kv * kv_confidence
         kv_confidence = 1.0 - (ckf_false_positive_rate)
       weight_Load = base_load * load_freshness
         load_freshness = exp(-(now - last_update).secs / 10)
       weight_Topology = base_topology
  
  4. Select the b with the highest hybrid_score
     If temperature > 0: sample via softmax (router_temperature)
     Otherwise: greedy selection
```

### 6.3 Default Weights

| Strategy | Base weight | Description |
|------|---------|------|
| KV | 0.35 | KV reuse has the largest impact on TTFT |
| Load | 0.30 | Load balancing avoids hotspots |
| Topology | 0.20 | Network latency affects end-to-end |
| Model | 1.0 (filter) | Acts as a hard filter; does not participate in weighting |

### 6.4 Adaptive Weight Adjustment

```
At runtime, weights are automatically adjusted based on degradation state:

if KV Index is unavailable:
    weight_KV = 0; weight_Load += 0.15; weight_Topology += 0.10
elif KV confidence < 0.5:
    weight_KV *= 0.5; weight_Load += 0.10

if Load Stats are stale (last_update > 10s):
    weight_Load *= 0.3; weight_Topology += 0.10

Normalize: ensure the sum of all weights = 1.0
```

### 6.5 Degradation Chain

```
is_available() checks:
  Hybrid:
    - Is the Model Registry available? (must be, otherwise routing is impossible)
    - Is at least one sub-strategy available? (yes → degradation mode)
  
  Degradation order:
    Hybrid (KV+Load+Topo)
      → If KV unavailable: Model+Load+Topo
      → If Load unavailable: Model+Topo
      → If Topo unavailable: Model+Load
      → If only Model is available: Model (degenerates to nearest + random)
      → If Model unavailable: return 503
```

### 6.6 Softmax Sampling

```
When temperature > 0:
  logits = [-hybrid_score(b) / temperature for b in candidates]
  probs = softmax(logits)
  selected = sample(probs)
  
temperature = 0: greedy (pick the highest score)
temperature → ∞: uniform random (degenerates to round robin)
```

---

## 7. Session Affinity

### 7.1 Goal

Route consecutive requests of the same session/conversation preferentially to the same backend to maximize KV Cache reuse.

### 7.2 Implementation

```
RoutingHistory:
  session_id → (backend_id, last_used, kv_overlap_at_route_time)
  TTL: 300 seconds

On routing:
  if session_id in routing_history:
    last_backend, last_time, last_overlap = routing_history[session_id]
    if last_backend still healthy and last_overlap > threshold:
      return last_backend  // reuse directly, skip full evaluation
    else:
      run normal Hybrid evaluation
  
  After evaluation:
    routing_history[session_id] = (selected_backend, now, overlap_score)
```

### 7.3 Cross-Instance Sharing

Routing history updates are broadcast via Gossip (with TTL) so that different Gateway instances can maintain session affinity.

---

## 8. Retry and Failover

```
forward(backend, request):
  try:
    stream = connector.forward(backend, request)
    return stream
  except BackendError:
    1. Mark backend as degraded (degradation stats +1)
    2. Remove this backend from the candidate list
    3. If retry count < max_retries:
       Re-run Hybrid evaluation (excluding failed backends)
       forward(new_backend, request)
    4. else:
       return 503 Service Unavailable

max_retries = 3
```

---

## 9. Strategy 6: KV Capacity Aware Routing

> Attached to Hybrid as a `RoutingPlugin` (the data half lives in [05-kv-estimation.md](../05-kv-estimation.md)).

### 9.1 Goal

Estimate the KV-cache memory footprint of the incoming request and score each candidate backend by its **remaining capacity**, excluding backends that cannot fit the request — the capacity-admission / load-shedding decision. Complementary to `KvAwareStrategy` (which scores by prefix-hit overlap to reduce prefill work): `KvAwareStrategy` decides "how much prefill to skip", `KvCapacityStrategy` decides "whether it fits at all".

### 9.2 Estimate source

The request footprint comes from the standalone leaf crate `hier-kv-gateway-kv-estimate`'s analytical formulas (not a simulation; matches vLLM/SGLang/Mooncake):

```
per_token = f(num_layers, num_kv_heads, head_dim, dtype, attention family)  // MLA uses kv_lora_rank+qk_rope_head_dim
seq_len   = input_tokens + estimated_output_tokens   // output uses client max_tokens as a conservative upper bound
effective = min(seq_len, sliding_window)              // sliding-window cap
blocks    = ceil(effective / block_size) * batch_size
bytes     = per_token * batch_size * (blocks * block_size)   // block-padded
```

### 9.3 Capacity-signal selection & scoring

For each candidate backend:

```
1. Resolve the model the backend actually serves (prefer exact ctx.model_name match, else backend's first model)
2. registry.estimate(model, input) → on None, apply exclude_on_unknown_spec policy
3. Read the backend's resource headroom, pick a capacity signal:
   - KV-block path (exact, preferred): kv_total_blocks>0 and block_size>0
       available_bytes = (kv_total_blocks - kv_used_blocks) * per_block_bytes
   - GPU-memory path (conservative fallback): gpu_memory_total_mb>0
       available_bytes = (gpu_memory_total_mb - gpu_memory_used_mb) * 1e6 * gpu_mem_safety_fraction
   - No signal: neutral (raw_cost=0, score=1), let other sub-strategies decide
4. Admission:
   if available_bytes <= 0 or bytes > available_bytes:
       exclude (raw_cost=∞, score=0)            // load shedding
   else:
       ratio = bytes / available_bytes ∈ [0,1]
       raw_cost = ratio                       // more headroom → lower cost
       score = 1 / (1 + ratio)
```

### 9.4 Key design decisions

1. **output uses `max_tokens` as a conservative upper bound**: the estimated KV growth never underestimates, mirroring `LoadAwareStrategy::w_decode` and `CostAwareStrategy`'s output projection.
2. **`f64::INFINITY` not `f64::MAX`**: exclusion uses `∞` (non-finite), recognized by `HybridStrategy::normalize_costs` via `!is_finite()`. `f64::MAX` is finite and would be misread as "very expensive but valid".
3. **GPU-memory fallback uses a safety fraction**: KV is not the only GPU memory consumer (weights, activations), so only "currently free memory × `gpu_mem_safety_fraction`" is claimable — avoids treating the whole free card as KV budget.
4. **Unknown spec is neutral by default**: with `exclude_on_unknown_spec=false`, an unknown-model backend is left to other sub-strategies, avoiding starving a backend that does have room when we're unsure.
5. **Independent normalization from `KvAwareStrategy`**: the two are independent sub-strategies in Hybrid, each doing its own `normalize_costs` — semantics don't cross. Same relationship as `LoadAwareStrategy` vs `CostAwareStrategy`.

### 9.5 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | When off the strategy is not attached |
| `weight` | `0.20` | Hybrid weight |
| `gpu_mem_safety_fraction` | `0.5` | Claimable fraction in GPU-memory fallback |
| `exclude_on_unknown_spec` | `false` | Exclude (true) / neutral (false) on unknown spec |

### 9.6 Configuration example

```toml
[kv_estimate]
enabled = true
weight = 0.20
gpu_mem_safety_fraction = 0.5
exclude_on_unknown_spec = false

# Optional: register a private model spec (fields map to HuggingFace config.json)
[[kv_estimate.models]]
name = "my-private-model"
num_layers = 20
num_kv_heads = 4
head_dim = 96
dtype = "fp16"
```

### 9.7 End-to-end example

Llama-3-8B (per_token=131_072 B), 4096 prompt, block_size 16:

```
blocks_needed = ceil(4096/16) = 256
backend A: kv_total=1000, kv_used=0   → free=1000, ratio=256/1000=0.256  (admitted)
backend B: kv_total=1000, kv_used=700 → free=300,  ratio=256/300=0.853  (admitted, higher cost)
backend C: kv_total=1000, kv_used=995 → free=5,    256>5                → excluded (raw_cost=∞)
```
