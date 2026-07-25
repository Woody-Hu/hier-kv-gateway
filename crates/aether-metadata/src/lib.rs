//! Aether LLM Gateway 元数据存储库。
//!
//! 该 crate 集中维护路由所需的所有元数据：KV cache 索引（本地精确 + 跨 Region 近似）、
//! 模型注册表、负载统计、拓扑图与会话亲和历史。所有结构均设计为读写并发安全，
//! 读路径无锁或低争用，写路径通过后台线程或 CAS 更新。
//!
//! 主要模块：
//! - [`radix_tree`]：本地精确 KV block 索引，后台线程串行化所有写操作。
//! - [`cuckoo_filter`]：Cuckoo Filter 基础原语，用于跨 Region 近似 membership。
//! - [`ckf_producer`]：每个 pool 一个的本地 CKF 生产者。
//! - [`ckf_consumer`]：transposed 布局的 CKF 消费者，承载多 Region lane。
//! - [`kv_index`]：统一 KV 索引接口，组合 RadixTree 与 CkfConsumer。
//! - [`model_registry`]：模型注册表。
//! - [`load_stats`]：后端负载统计与滑动窗口。
//! - [`topology_graph`]：Region 拓扑与延迟矩阵。
//! - [`routing_history`]：会话亲和历史，带 TTL 清理。
//! - [`store`]：所有元数据组件的统一入口。

pub mod radix_tree;
pub mod cuckoo_filter;
pub mod ckf_consumer;
pub mod ckf_producer;
pub mod kv_index;
pub mod model_registry;
pub mod load_stats;
pub mod topology_graph;
pub mod routing_history;
pub mod store;
