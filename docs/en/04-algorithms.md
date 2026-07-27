# Hier KV Gateway Algorithm Design Document

> English | [中文](../04-algorithms.md)

> This document is the algorithm-specific reference for the system, covering the implementation details of core algorithms including KV block hashing, RadixTree, Cuckoo Filter, CKF Producer/Consumer, the Gossip protocol, routing strategies, and hybrid scoring.
> All constants and code references come from the actual implementation in `crates/hier-kv-gateway-*`.

## 0. Algorithm Overview

| Algorithm | Location | Purpose |
|------|------|------|
| Block Hash | [hier-kv-gateway-core/src/kv_event.rs](../../crates/hier-kv-gateway-core/src/kv_event.rs) | Slice a token sequence into blocks and compute an XXH3 hash for each block |
| RadixTree | [hier-kv-gateway-metadata/src/radix_tree.rs](../../crates/hier-kv-gateway-metadata/src/radix_tree.rs) | Local exact KV block prefix index |
| Cuckoo Filter primitives | [hier-kv-gateway-metadata/src/cuckoo_filter.rs](../../crates/hier-kv-gateway-metadata/src/cuckoo_filter.rs) | Fingerprint addressing and packed bucket operations |
| CKF Producer | [hier-kv-gateway-metadata/src/ckf_producer.rs](../../crates/hier-kv-gateway-metadata/src/ckf_producer.rs) | Local exact ownership + compact projection publication |
| CKF Consumer | [hier-kv-gateway-metadata/src/ckf_consumer.rs](../../crates/hier-kv-gateway-metadata/src/ckf_consumer.rs) | Cross-Region approximate KV index (transposed layout) |
| Gossip | [hier-kv-gateway-cluster/src/gossip.rs](../../crates/hier-kv-gateway-cluster/src/gossip.rs) | Cross-cluster member discovery and metadata propagation |
| Routing strategies | [hier-kv-gateway-routing/src/*.rs](../../crates/hier-kv-gateway-routing) | 5 strategies + Hybrid aggregation |

---

## 1. Block Hash Computation

### 1.1 Goal

Slice a variable-length token sequence into fixed-length blocks, computing a 64-bit hash for each block. This hash is the basic addressing unit for the subsequent RadixTree and CKF.

### 1.2 Algorithm

Implemented in [kv_event.rs](../../crates/hier-kv-gateway-core/src/kv_event.rs) as `compute_block_hashes`:

```
Input: tokens: &[u32], kv_block_size: u32, cache_namespace: Option<&str>, lora_name: Option<&str>

1. seed = compute_seed(cache_namespace, lora_name)
2. Slice tokens into non-overlapping windows of size kv_block_size
   - The last incomplete block is discarded
3. For each window:
   - Write the tokens in the window into a buffer in little-endian byte order
   - hash = xxh3_64_with_seed(bytes, seed)
4. Return hashes: Vec<u64>
```

### 1.3 Seed Derivation

`compute_seed` mixes the namespace and LoRA adapter into the XXH3 seed independently, ensuring that the same token sequence produces different hashes under different tenants/adapters:

```
XXH3_SEED  = 1337
NS_SALT    = 0x4E53_5F4C_4F5F_4C4F
LORA_SALT  = 0x4C52_4F5F_4C4F_5F4C

seed = XXH3_SEED
if cache_namespace is non-empty:
    seed = seed.wrapping_add(xxh3_64_with_seed(ns_bytes, NS_SALT))
    seed ^= NS_SALT
if lora_name is non-empty:
    seed = seed.wrapping_add(xxh3_64_with_seed(lora_bytes, LORA_SALT))
    seed ^= LORA_SALT
```

**Key properties**:
- An empty string is treated as not provided (equivalent to `None`)
- Namespace and LoRA use independent salts; their effects are mutually independent and cannot cancel each other out
- An identical namespace and lora_name (e.g., both "foo") still produce different hashes

### 1.4 Boundaries

- `kv_block_size == 0` → returns an empty vector
- `tokens.len() < kv_block_size` → returns an empty vector (no complete block)

---

## 2. RadixTree (Local Exact KV Index)

### 2.1 Data Structure

Implemented in [radix_tree.rs](../../crates/hier-kv-gateway-metadata/src/radix_tree.rs). Each non-root node represents one block hash in the sequence prefix:

```rust
struct Node {
    hash: u64,                              // block hash of this node (root is 0)
    owners: HashSet<(BackendId, u32)>,      // (backend, rank) pairs that own this block
    children: HashMap<u64, Node>,           // child nodes, indexed by block hash
    ref_count: u32,                         // cache of the owners count
}
```

### 2.2 Concurrency Model: Background Thread + mpsc Channel

All write operations are serialized via a dedicated background thread; reads return results synchronously through `mpsc::Sender<RadixCommand>` + `oneshot`:

```
Caller (async)                      Background thread
   │                                  │
   ├─ RadixCommand::ApplyEvent ──────►│ apply_event()
   │  (oneshot::Sender)               │   ├─ Stored → add_owner
   │                                  │   ├─ Removed → remove_owner
   │◄─ done.send(Result) ─────────────┤   ├─ Clear → clear_backend
   │                                  │   └─ Reset → clear_backend (generation fence)
   │                                  │
   ├─ RadixCommand::FindMatches ─────►│ find_matches()
   │◄─ done.send(u32) ────────────────┤
```

The background thread is named `hier-kv-gateway-radix-tree`, and the channel capacity is 4096. This design:
- Has no internal locks and is simple to implement
- Is safe to call from async contexts (`async fn` interface)
- On Drop, best-effort `try_send(Shutdown)`; if a clone still holds the sender, it is ignored

### 2.3 find_matches Algorithm

Query the prefix overlap length of a given hash sequence for a specified backend:

```
find_matches(hashes, backend):
    current = root
    overlap = 0
    for hash in hashes:
        child = current.children.get(hash)
        if child is None: break           // no matching prefix
        if not child.is_owned_by(backend): break  // this backend does not own this suffix
        overlap += 1
        current = child
    return overlap
```

**Key property**: prefix break — once a backend does not own the k-th block, even if it owns the (k+1)-th block it is not counted (because inference prefill must be contiguous from the front).

### 2.4 find_all_matches

Collect the maximum overlap length for each backend along the prefix path:

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

### 2.5 Event Application

**Stored { block_hashes }**: Treats `block_hashes` as a prefix path starting from the root, adding ownership of `(backend, 0)` (rank defaults to 0) at each node along the path.

**Removed { block_hashes }**: `block_hashes` is a set of **independent** block hashes (not a prefix path). It searches the entire tree for nodes whose hash matches and removes that backend's ownership. This is because content-addressed blocks can be shared across prefixes in the cache.

**Clear { worker }**: Recursively removes that backend's ownership from all nodes.

**Reset { generation }**: A generation fence, semantically equivalent to Clear — clears all ownership of that backend. Typically triggered by a worker restart or generation switch.

### 2.6 Node Reclamation

Empty nodes are reclaimed bottom-up: when `node.ref_count == 0 && node.children.is_empty()`, the node is removed from its parent's children. The root node (hash=0) is never reclaimed, ensuring the tree structure is always preserved.

---

## 3. Cuckoo Filter Primitives

Implemented in [cuckoo_filter.rs](../../crates/hier-kv-gateway-metadata/src/cuckoo_filter.rs). This module only provides stateless bucket operations and addressing functions; the upper-layer Producer/Consumer composes these primitives.

### 3.1 Constants

```
FINGERPRINT_BITS  = 16       // number of fingerprint bits
FP_PER_BUCKET     = 4        // number of fingerprints per bucket
MAX_KICKS         = 500      // maximum number of evictions per insertion
BUCKETS_PER_LANE  = 65536    // number of buckets per lane (must be a power of 2)
BUCKET_MASK       = 0xFFFF   // bucket index mask
ALT_MIX_DOMAIN    = 0x9E37_79B9_7F4A_7C15  // alt_index mixing constant
```

### 3.2 PackedBucket

A `u64` packs four 16-bit fingerprints, with slot 0 in the low bits:

```
| slot 3 | slot 2 | slot 1 | slot 0 |
| 63..48 | 47..32 | 31..16 | 15..0  |
```

`Fp = u16`; `0` is reserved as the "empty slot" sentinel.

### 3.3 Partial-key Cuckoo Hashing

Uses partial-key cuckoo addressing, storing only fingerprints rather than complete keys:

```
probe(hash):
    mixed = xxh3_64_with_seed(hash.to_le_bytes(), 0)
    fp    = (mixed as u16) | 1        // set the lowest bit to 1 to avoid generating 0
    bucket = ((mixed >> 16) as usize) & BUCKET_MASK
    return (fp, bucket)

alt_index(idx, fp):
    mixed = xxh3_64_with_seed(fp.to_le_bytes(), ALT_MIX_DOMAIN)
    delta = (mixed as usize) & BUCKET_MASK
    delta = 1 if delta == 0 else delta   // avoid delta=0 causing the two candidates to coincide
    return (idx ^ delta) & BUCKET_MASK
```

**Key property**: `alt_index` is an involution: `alt_index(alt_index(idx, fp), fp) == idx`. This guarantees that insertion and lookup use the same two candidate buckets.

### 3.4 SIMD-friendly bucket_contains

Instead of comparing slot by slot, a bitwise operation tests all four slots simultaneously:

```
bucket_contains(bucket, fp):
    repeated  = u64(fp) * 0x0001_0001_0001_0001   // copy fp into all 4 slots
    different = bucket ^ repeated
    high_bits = 0x8000_8000_8000_8000              // highest bit of each slot
    return (different.wrapping_sub(0x0001_0001_0001_0001)
            & !different
            & high_bits) != 0
```

This formula uses "borrow-free subtraction + XOR" to determine whether any slot equals fp, and the compiler can vectorize it.

### 3.5 Bucket Operations

```
try_insert(bucket, fp): find an empty slot and write; return false if full
try_delete(bucket, fp): find a matching slot and set it to 0; return false if not found
first_match(bucket, fp): return the first matching slot
first_empty(bucket): return the first empty slot
```

---

## 4. CKF Producer

Implemented in [ckf_producer.rs](../../crates/hier-kv-gateway-metadata/src/ckf_producer.rs). Each pool has one Producer that maintains local exact ownership + a compact CKF projection.

### 4.1 State

```rust
struct CkfProducer {
    buckets: Vec<PackedBucket>,              // all buckets in the lane
    num_items: u64,                           // number of fingerprints inserted
    dirty_buckets: HashSet<usize>,            // buckets changed since last publication
    pub_seq: u64,                             // maximum published sequence number
    hash_refcount: HashMap<u64, HashEntry>,   // hash → (refcount, owners)
    worker_hashes: HashMap<BackendId, HashSet<u64>>,  // backend → set of held hashes
    rng_state: u64,                           // splitmix64 PRNG state
}

struct HashEntry {
    refcount: u32,
    owners: HashSet<BackendId>,
}
```

### 4.2 Ownership 4-Branch Rule

When applying `Stored { block_hashes }`, for each hash:

```
apply_stored(hash, worker):
    entry = hash_refcount.entry(hash).or_default()
    first_owner = entry.owners.is_empty()
    if not entry.owners.insert(worker):    // already held by this worker
        return                             // deduplicate; do not increment refcount or insert fingerprint
    entry.refcount += 1
    worker_hashes[worker].insert(hash)
    if first_owner:
        insert_fingerprint(hash)           // first owner → insert fingerprint
```

When applying `Removed { block_hashes }`, for each hash:

```
apply_removed(hash, worker):
    entry = hash_refcount.get_mut(hash) or return
    if not entry.owners.remove(worker): return
    worker_hashes[worker].remove(hash)
    entry.refcount -= 1
    if entry.refcount == 0:
        delete_fingerprint(hash)           // final owner → delete fingerprint
        hash_refcount.remove(hash)
```

**Summary of the 4 branches**:

| Scenario | Behavior |
|------|------|
| First owner of a hash | Insert fingerprint |
| Another owner of same hash | refcount++ only |
| One of several removes | refcount-- only |
| Final owner removes | Delete fingerprint |

Applying `Clear { worker }`: iterate `worker_hashes[worker]`, applying the "one of several removes / final owner removes" branch for each hash.

Applying `Reset`: clear the entire producer state (generation fence).

### 4.3 Cuckoo Insertion (with Eviction and Rollback)

```
insert_fingerprint(hash):
    (fp, bucket_a) = probe(hash)
    bucket_b = alt_index(bucket_a, fp)
    
    // First try to insert directly into the two candidate buckets
    if try_insert(buckets[bucket_a], fp): mark dirty; num_items++; return true
    if try_insert(buckets[bucket_b], fp): mark dirty; num_items++; return true
    
    // Enter the eviction loop
    touched = []
    current_bucket = bucket_a or bucket_b (random)
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
    
    // Reaching MAX_KICKS without success → roll back all evictions
    for (idx, before) in touched.rev():
        buckets[idx] = before
    return false
```

**Key points**:
- During eviction, the original values of all touched buckets are recorded; on failure they are rolled back in LIFO order
- splitmix64 PRNG is deterministic but well-distributed, avoiding extra dependencies
- On failure (lane full) the upper layer only logs a warning and does not block ingestion

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
    buckets.sort_by_idx()           // sorted for easier consumer application and diagnostics
    dirty_buckets.clear()
    return CkfDelta { sequence: pub_seq, prev_sequence: prev, buckets }
```

**Semantics**:
- A Snapshot is a full absolute image that the consumer can install independently
- A Delta contains only the current absolute values of dirty buckets (not patches); the consumer overwrites directly
- Sequence numbers are monotonically increasing (`wrapping_add`); the consumer can detect out-of-order delivery

---

## 5. CKF Consumer (Transposed Layout)

Implemented in [ckf_consumer.rs](../../crates/hier-kv-gateway-metadata/src/ckf_consumer.rs). Each Gateway instance runs one Consumer that tracks the CKF projections of multiple Regions.

### 5.1 Transposed Layout

```
LANE_COUNT = 16   // tracks up to 16 Regions simultaneously

buckets: Vec<[AtomicU64; 16]>   // bucket-major
                                  // buckets[i][lane] is the packed value of bucket i on a given lane
```

**Why transposed**: Organizing by bucket rather than by lane means that a single prefix query (which accesses multiple consecutive buckets) advances along `buckets[0..k][lane]` on the same lane in a cache-friendly way; at the same time, the same bucket across multiple lanes shares a cache line, enabling concurrent probes.

### 5.2 Lane State Machine

```
LANE_ACTIVE  = 0
LANE_RETIRED = 1

lane_status: [AtomicU8; 16]
```

- `Active`: visible to queries
- `Retired`: invisible to queries (lane is reconnecting or has been retired)

### 5.3 estimate_overlap Algorithm

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
            break                          // prefix break
    return overlap
```

**Key properties**:
- Prefix break: consistent with RadixTree, a miss on the k-th block stops iteration
- No lane-wide lock: each bucket is independently atomic; reads are lock-free
- False positives: CKF may return false positives (a Region appears to have a block it does not); this is corrected by subsequent exact queries or request results

### 5.4 Snapshot Installation (retired → write → active)

```
install_snapshot(lane, snapshot):
    lane_status[lane].store(RETIRED, Release)      // 1. mask reads
    for (i, value) in snapshot.buckets.iter().enumerate():
        buckets[i][lane].store(value, Relaxed)     // 2. write all buckets
    lane_status[lane].store(ACTIVE, Release)       // 3. restore reads
```

The three-step ordering guarantees that readers see either the complete old snapshot or the complete new snapshot, never an intermediate state.

### 5.5 Delta Application (Weak Consistency)

```
apply_delta(lane, delta):
    for (bucket_idx, value) in delta.buckets:
        buckets[bucket_idx][lane].store(value, Release)
```

A Delta is a weakly consistent multi-bucket write; readers may observe a partially applied state. This is a deliberate trade-off in CKF design — false-positive tolerance means no seqlock or retry is needed.

### 5.6 Lane Lifecycle

```
assign_lane(lane, region)   // bind a lane to a Region
activate_lane(lane)         // mark Active
retire_lane(lane)           // mark Retired (excluded from queries)
unassign_lane(lane)         // unbind (Region migrated away)
```

Failure recovery: when a lane disconnects, `retire_lane` is called; upon reconnection, `install_snapshot` installs a new barrier snapshot and then `activate_lane` is called.

---

## 6. Gossip Protocol

Implemented in [hier-kv-gateway-cluster/src/gossip.rs](../../crates/hier-kv-gateway-cluster/src/gossip.rs).

### 6.1 Message Types

| Message | Purpose |
|------|------|
| `PING / PONG` | Heartbeat, carries the sender's metadata digest |
| `MEET` | A new node joins the cluster |
| `SYNC` | Request a full state sync (new node or repair) |
| `CKF_PUBLISH` | Cross-Region KV projection publication (barrier + delta) |
| `METRIC_BROADCAST` | Load/latency metric broadcast |

### 6.2 Gossip Behavior

```
Each Gateway instance maintains:
    members: HashMap<InstanceId, ClusterMember>
        ClusterMember = { instance_id, region, addr, last_pong_unix, status }

Every second:
    1. Randomly select P alive members and send PING
    2. PONG carries the latest metadata digest (MetaDigest)
    3. If PING times out → suspect_count++
    4. N consecutive failures → status = Suspect → confirmed Dead
```

### 6.3 Metadata Version Synchronization

```
MetaDigest = {
    kv_version: u64,
    model_version: u64,
    load_version: u64,
    topology_version: u64,
    members_version: u64,
}

On receiving PONG:
    for each (region, version) in pong.digest:
        if local_version < version:
            send SYNC to request the delta for that region
```

Large state (CKF projections) is not placed in PING; it goes through barrier snapshot + sequenced delta separately.

### 6.4 Member State Machine

```
Alive ──PING timeout──► Suspect ──confirmed──► Dead
   ▲                                       │
   └─────────re-PONG──────────────────────┘
```

New members join via `MEET`: the instance that receives MEET adds it to members and propagates it in subsequent Gossip rounds.

---

## 7. Routing Strategy Algorithms

### 7.1 Unified Scoring Model

All strategies produce a unified `ScoredBackend`:

```rust
struct ScoredBackend {
    backend_id: BackendId,
    score: f64,        // [0, 1], 1.0 = optimal
    raw_cost: f64,    // strategy raw cost (lower is better)
    meta_version: u64,
}
```

### 7.2 KV Aware (KV-aware Routing)

Implemented in [kv_aware.rs](../../crates/hier-kv-gateway-routing/src/kv_aware.rs).

```
For each candidate backend b:
    local_overlap  = RadixTree.find_matches(hashes, b)         // local exact
    remote_overlap = CkfConsumer.estimate_overlap(hashes, b.region)  // cross-domain approximate
    
    effective_remote = remote_overlap * (1 - ckf_false_positive_penalty)
    total_overlap = local_overlap + effective_remote
    
    prefill_blocks = max(len(hashes) - total_overlap, 0)
    decode_blocks  = b.active_decode_blocks   // from LoadStats
    
    cost  = prefill_load_scale * prefill_blocks + decode_blocks
    score = 1.0 / (1.0 + cost) + overlap_score_credit * total_overlap
```

**Default parameters**:
- `overlap_score_credit = 1.0`
- `prefill_load_scale = 1.0`
- `ckf_false_positive_penalty = 0.0` (can be enabled in configuration)
- weight `weight() = 0.35`

**Availability check**: `meta.kv_confidence() > 0.0`; otherwise the strategy is unavailable (triggers degradation).

### 7.3 Model Aware (Model-aware Routing, Hard Filter)

Implemented in [model_aware.rs](../../crates/hier-kv-gateway-routing/src/model_aware.rs). Acts as a hard filter, eliminating non-matching candidates:

```
For each candidate backend b:
    score = match_degree(b, request.model):
        exact_match      (model_name + version + quant)  → 1.0
        model_match      (same name, different version)  → 0.7
        compatible_match (same architecture, different name) → 0.3
        no_match                                         → 0.0 (excluded)
    
    Additional checks:
        - max_context_len >= request.token_count ?
        - supports_tool_calling >= request.requires_tool_calling ?
    
    cost = 1.0 - score
```

Candidates with `score == 0.0` are filtered out by the Hybrid strategy.

### 7.4 Load Aware (Load-aware Routing)

Implemented in [load_aware.rs](../../crates/hier-kv-gateway-routing/src/load_aware.rs).

```
For each candidate backend b:
    m = LoadStats.get_metrics(b)
    
    load_cost = w_req  * m.active_requests
              + w_queue * m.queue_depth
              + w_lat   * (m.p99_latency / 100)
              + w_gpu   * m.gpu_utilization
              + w_kv    * m.kv_cache_usage
    
    Capacity check: if available_capacity <= 0: exclude
    
    cost  = load_cost
    score = 1.0 / (1.0 + load_cost)
```

**Default weights**: `w_req=1.0, w_queue=2.0, w_lat=1.5, w_gpu=0.5, w_kv=0.8`

**Sliding window**: 60-second window, 1-second sampling interval; p50/p99 use approximate algorithms.

### 7.5 Topology Aware (Topology-aware Routing)

Implemented in [topology_aware.rs](../../crates/hier-kv-gateway-routing/src/topology_aware.rs).

```
For each candidate backend b:
    rtt = LatencyMatrix.rtt_ms(self_region, b.region)
    
    network_cost = w_rtt * rtt + w_bw * bandwidth_penalty
    
    Tier preference:
        if self.tier == Device and b.tier == Edge:  network_cost *= 0.8
        if self.tier == Device and b.tier == Cloud: network_cost *= 1.5
    
    cost  = network_cost
    score = 1.0 / (1.0 + network_cost / 100.0)   // 100ms baseline
```

**RTT sources**: configuration + active probing + Gossip propagation + geographic-distance estimation (`rtt ≈ distance_km / 200`, fiber).

---

## 8. Hybrid Routing (Default Strategy)

Implemented in [hybrid.rs](../../crates/hier-kv-gateway-routing/src/hybrid.rs).

### 8.1 Algorithm Flow

```
1. Model Aware performs hard filtering
    filtered = [b for b in candidates if model.evaluate(b).score > 0]
    if filtered.is_empty(): return RoutingFailed

2. Dynamic weight adjustment
    weight_kv  = kv.is_available(meta) ? weights.kv : 0
    load_stale = any(|c| load_freshness(c) > 10s for c in filtered)
    weight_load = load_stale ? weights.load * 0.3 : weights.load
    weight_topo = weights.topology
    
    Normalize: total = weight_kv + weight_load + weight_topo
               if total > 0: each of the three /= total
               else:          all three = 1/3 (fallback uniform)

3. Each sub-strategy scores the filtered set
    kv_scores   = kv.evaluate(...)        if weight_kv  > 0 else []
    load_scores = load.evaluate(...)      if weight_load > 0 else []
    topo_scores = topology.evaluate(...)  if weight_topo > 0 else []

4. Normalize each strategy's raw_cost to [0, 1]
    normalize_costs(scores):
        min = min(s.raw_cost for s in scores if finite)
        max = max(s.raw_cost for s in scores if finite)
        span = max - min
        for s in scores:
            if not finite(s.raw_cost): norm = 0     // constraint not satisfied
            elif span > 0: norm = (s.raw_cost - min) / span
            else: norm = 0
            s.normalized = 1.0 - norm                // lower cost → higher score

5. Weighted sum
    for c in filtered:
        hybrid_score(c) = weight_kv  * kv_norm[c]
                        + weight_load * load_norm[c]
                        + weight_topo * topo_norm[c]
        raw_cost = -hybrid_score   // preserve "raw_cost lower is better" semantics

6. Sort by hybrid_score descending
```

### 8.2 Key Constants

```
STALE_LOAD_THRESHOLD_SECS = 10   // load metric staleness threshold
```

### 8.3 Default Weights

| Strategy | Base weight | Role |
|------|---------|------|
| KV | 0.35 | Weighted |
| Load | 0.30 | Weighted |
| Topology | 0.20 | Weighted |
| Model | 1.0 (filter) | Hard filter, does not participate in weighting |

### 8.4 Softmax Sampling (at the routing engine layer)

```
if temperature > 0:
    logits = [-hybrid_score(b) / temperature for b in candidates]
    probs  = softmax(logits)
    selected = sample(probs)
else:
    selected = argmax(hybrid_score)   // greedy
```

- `temperature = 0`: greedy, picks the highest score
- `temperature → ∞`: uniform random (degenerates to round robin)

### 8.5 Degradation Chain

```
Hybrid (KV + Load + Topo)
  │ KV Index unavailable (kv_confidence == 0)
  ▼
Model + Load + Topo       (weight_kv = 0)
  │ Load Stats all stale (>10s)
  ▼
Model + Topo              (weight_load *= 0.3)
  │ Cross-cluster communication broken
  ▼
Local Load Aware
  │ No available local Backend
  ▼
Return 503
```

---

## 9. Session Affinity

Implemented in [routing_history.rs](../../crates/hier-kv-gateway-metadata/src/routing_history.rs) and the routing engine.

```
On routing:
    if session_id in routing_history:
        (last_backend, last_time, last_overlap) = routing_history[session_id]
        if last_backend still healthy and last_overlap > 0:
            return last_backend          // reuse directly, skip full evaluation
        else:
            run normal Hybrid evaluation

After evaluation:
    routing_history[session_id] = (selected_backend, now, overlap_score)

TTL: 300 seconds
```

Cross-instance sharing: routing history updates are broadcast via Gossip (with TTL) so that different Gateway instances maintain session affinity.

---

## 10. Retry and Failover

```
forward(backend, request):
    try:
        stream = connector.forward(backend, request)
        return stream
    except BackendError:
        1. Mark backend as degraded (degradation stats +1)
        2. Remove this backend from the candidate list
        3. if retry count < max_retries (default 3):
               re-run Hybrid evaluation (excluding failed backends)
               forward(new_backend, request)
        4. else:
               return 503 Service Unavailable
```

---

## 11. Failure Recovery (Narrowest State Boundary)

Failure recovery is designed around the "narrowest state boundary" principle: each failure affects only the minimal subset of state.

| Failure | Recovery boundary | Behavior |
|------|---------|------|
| Backend event gap | That backend's rank state | Recover from the backend's event history, or install the current tree state |
| Backend replacement | All state of that backend | Rebuild from the new source after the completion barrier |
| CKF delivery gap | The affected consumer lane | `retire_lane`; on reconnection, `install_snapshot` installs a new barrier |
| Gateway instance crash | That instance's local state | Other instances detect via Gossip and take over routing; the new instance SYNCs full state |
| Region isolation | That Region's lane | Routing excludes that Region; the lane is reactivated after recovery |

---

## 12. Metadata Cache Tiers

| Tier | Content | Lifecycle |
|------|------|---------|
| L1 Request-Local | block hashes, overlap scores | Single request |
| L2 Hot | RadixTree, LoadStats (TTL 5s), CKF Consumer | Resident in memory, real-time updates |
| L3 Warm | ModelRegistry (TTL 60s), Topology (TTL 30s), Discovery (TTL 15s) | Periodic refresh |
| L4 Cold | RoutingHistory (TTL 300s), DegradationStats (TTL 60s) | On-demand query + periodic cleanup |

### Concurrency-Safety Implementation

| Component | Concurrency strategy |
|------|---------|
| RadixTree | Dedicated background thread + mpsc channel (lock-free) |
| LoadStats | `DashMap<BackendId, ArcSwap<Metrics>>`, lock-free reads |
| CKF Consumer | Bucket-level `AtomicU64`, no lane-wide lock |
| ModelRegistry | `Arc<RwLock<...>>`, read-heavy write-light |

---

## 13. Algorithm Complexity Reference

| Operation | Time complexity | Notes |
|------|----------|------|
| `compute_block_hashes` | O(n / block_size) | n = number of tokens |
| `RadixTree.find_matches` | O(k) | k = number of blocks (prefix break is even better on average) |
| `RadixTree.apply_stored` | O(k) | k = length of block_hashes |
| `RadixTree.apply_removed` | O(N) worst case | N = number of tree nodes (global search for matching hashes) |
| `CkfProducer.insert_fingerprint` | O(1) average, O(MAX_KICKS) worst case | Rollback after 500 evictions |
| `CkfConsumer.estimate_overlap` | O(k) | k = number of blocks; one atomic load per bucket |
| `Hybrid.evaluate` | O(C × S) | C = number of candidates, S = number of sub-strategies (≤3) |
