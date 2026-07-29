# 负载编码 Benchmark 报告

## 1. 背景

Gossip 集群通信中，后端负载数据（`BackendMetrics`）需要跨 Gateway 实例同步。原方案使用 JSON 全量序列化，每个后端约 320 字节。本 benchmark 对比三种编码方案的体积和编解码速度：

| 编码 | 描述 |
|------|------|
| **JSON** | `serde_json::to_vec` — 原方案，字段名 + 引号 + 分隔符 |
| **Postcard** | `postcard::to_allocvec` — 纯二进制 varint 编码，无字段名开销 |
| **LoadPayload** | `LoadPayload::encode_full` — postcard + base64 包装，版本化信封（当前生产方案） |

## 2. 测试环境

- CPU: Apple Silicon (M-series)
- Rust: `cargo bench` release profile (`--opt-level=3`)
- Benchmark 框架: criterion 0.5, sample_size=200
- 数据: 真实 `BackendMetrics` 结构（12 字段: 8×u64 + 2×f64 + 1×i64 + LatencyStats{3×f64+1×u64}）

## 3. 编码体积对比

```
N        JSON (bytes)   Postcard (raw)   LoadPayload (b64)   压缩比
-------------------------------------------------------------------
1        320            61               101                 3.17x
5        1,619          301              421                 3.85x
10       3,294          603              821                 4.01x
20       6,705          1,231            1,661               4.04x
50       17,047         3,109            4,165               4.09x
```

**关键发现**：

- **LoadPayload 在 N=50 时压缩 4.09×**（4,165 vs 17,047 bytes）
- Postcard raw 比 LoadPayload 更小（base64 膨胀 ~33%），但 LoadPayload 需要兼容 JSON 传输层
- 线性扩展：每增加一个 backend，JSON 增 ~340 bytes，LoadPayload 增 ~82 bytes

## 4. 编码速度对比

``
N=1:
  json_serialize:        329 ns
  postcard_serialize:    141 ns   (2.3x faster than JSON)
  loadpayload_encode:    154 ns   (2.1x faster than JSON)

N=10:
  json_serialize:      2,720 ns
  postcard_serialize:    595 ns   (4.6x faster)
  loadpayload_encode:    637 ns   (4.3x faster)

N=50:
  json_serialize:     13,808 ns
  postcard_serialize:  1,528 ns   (9.0x faster)
  loadpayload_encode:  2,387 ns   (5.8x faster)
```

**关键发现**：

- N=50 时 LoadPayload 编码速度 **5.8× 于 JSON**（2.39 µs vs 13.81 µs）
- Postcard raw 比 LoadPayload 快 ~56%，差距来自 base64 编码开销
- 所有编码线性扩展，但 postcard 斜率远低于 JSON

## 5. 解码速度对比

```
N=1:
  json_deserialize:      315 ns
  postcard_deserialize:   56 ns   (5.6x faster)
  loadpayload_decode:     96 ns   (3.3x faster)

N=10:
  json_deserialize:    3,348 ns
  postcard_deserialize:   419 ns   (8.0x faster)
  loadpayload_decode:     567 ns   (5.9x faster)

N=50:
  json_deserialize:   16,328 ns
  postcard_deserialize:  2,219 ns   (7.4x faster)
  loadpayload_decode:    2,943 ns   (5.6x faster)
```

**关键发现**：

- N=50 时 LoadPayload 解码速度 **5.6× 于 JSON**（2.94 µs vs 16.33 µs）
- Postcard raw 解码极快（无字符串解析、无字段名匹配），但 LoadPayload 的 base64 解码增加了 ~33% 开销
- 解码比编码更快（postcard 的 varint 解码效率高于编码）

## 6. 端到端延迟估算

在一次 Gossip 广播周期中（假设 N=20 后端）：

| 路径 | JSON（原方案） | LoadPayload（新方案） | 节省 |
|------|---------------|---------------------|------|
| 编码 | 5.27 µs | 1.11 µs | 4.16 µs |
| 网络传输 | ~6.7 KB | ~1.7 KB | ~5.0 KB |
| 解码 | 6.52 µs | 1.23 µs | 5.29 µs |
| **总 CPU** | **11.79 µs** | **2.34 µs** | **9.45 µs (80%)** |

## 7. 设计决策

### 为什么选 LoadPayload 而非 Postcard raw？

| 维度 | Postcard raw | LoadPayload |
|------|-------------|-------------|
| 体积 | 最小 | +33%（base64 膨胀） |
| 速度 | 最快 | base64 编解码开销 |
| JSON 兼容 | 不兼容 | 兼容（`{"v":1,"data":"..."}`） |
| 版本扩展 | 无 | `v` 字段预留 delta 编码 |
| 跨版本容错 | 无 | 未知版本优雅跳过 |

**结论**：LoadPayload 在体积和速度上已接近 postcard raw 的水平（4× 压缩 + 5.8× 速度），同时保持了 JSON 传输层兼容性和版本扩展能力。

### Delta 编码扩展路径

LoadPayload 的 `v` 字段为未来 delta 编码预留：

```rust
pub const VERSION_FULL: u8 = 1;  // 当前：全量 postcard
// 未来：VERSION_DELTA: u8 = 2  // 增量编码
```

当稳态下负载指标变化缓慢时，delta 编码可将每次广播从 ~1.7 KB 降到 ~100 bytes 级别（仅发送变化的字段），进一步降低带宽消耗。

## 8. 运行方式

```bash
# 编译 benchmark
cargo bench -p hier-kv-gateway-cluster --bench load_encoding --no-run

# 运行 benchmark
cargo bench -p hier-kv-gateway-cluster --bench load_encoding

# HTML 报告位于
# target/criterion/load_encode/index.html
# target/criterion/load_decode/index.html
```

## 9. 文件索引

| 文件 | 说明 |
|------|------|
| [load_encoding.rs](../../crates/hier-kv-gateway-cluster/benches/load_encoding.rs) | Benchmark 源码 |
| [messages.rs](../../crates/hier-kv-gateway-cluster/src/messages.rs) | `LoadPayload` 定义 |
| [cluster_bridge.rs](../../crates/hier-kv-gateway/src/cluster_bridge.rs) | `serialize_load_state` / `apply_load_state` 实现 |
