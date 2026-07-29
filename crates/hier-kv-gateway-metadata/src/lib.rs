//! Hier KV Gateway metadata store.
//!
//! This crate centrally maintains all metadata needed for routing: the KV cache
//! index (local exact + cross-Region approximate), the model registry, load
//! statistics, the topology graph, and session affinity history. All structures
//! are designed to be concurrency-safe for reads and writes — the read path is
//! lock-free or low-contention, and the write path uses background threads or
//! CAS updates.
//!
//! Main modules:
//! - [`radix_tree`]: local exact KV block index; a background thread serializes all writes.
//! - [`cuckoo_filter`]: Cuckoo Filter primitives, used for cross-Region approximate membership.
//! - [`ckf_producer`]: one local CKF producer per pool.
//! - [`ckf_consumer`]: transposed-layout CKF consumer, hosting multiple Region lanes.
//! - [`local_ckf`]: local per-backend CKF projection (transposed), serves cache-friendly batch queries.
//! - [`kv_index`]: unified KV index interface, combining RadixTree, LocalCkf and CkfConsumer.
//! - [`model_registry`]: model registry.
//! - [`load_stats`]: backend load statistics and sliding window.
//! - [`topology_graph`]: Region topology and latency matrix.
//! - [`routing_history`]: session affinity history with TTL cleanup.
//! - [`store`]: unified entry point for all metadata components.

pub mod radix_tree;
pub mod cuckoo_filter;
pub mod ckf_consumer;
pub mod ckf_producer;
pub mod local_ckf;
pub mod kv_index;
pub mod model_registry;
pub mod load_stats;
pub mod topology_graph;
pub mod routing_history;
pub mod store;
