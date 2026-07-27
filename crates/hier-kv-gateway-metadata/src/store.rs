//! Unified metadata store entry point: combines all metadata components and
//! provides a single access point.
//!
//! [`MetadataStore`] holds the KV index, model registry, load statistics,
//! topology graph, routing history, and backend registry, exposing a set of
//! high-level query interfaces to the routing layer.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use hier_kv_gateway_core::backend::{BackendInfo, ModelInstance};
use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::{BackendId, IndexerDomainId, RegionId, SessionId};
use hier_kv_gateway_core::kv_event::KvCacheEvent;
use hier_kv_gateway_core::metrics::BackendMetrics;
use hier_kv_gateway_core::topology::{LatencyEstimate, RegionInfo};
use dashmap::DashMap;
use parking_lot::RwLock;

use crate::ckf_consumer::CkfConsumer;
use crate::kv_index::KvIndex;
use crate::load_stats::LoadStats;
use crate::model_registry::ModelRegistry;
use crate::radix_tree::RadixTree;
use crate::routing_history::{RoutingHistory, SessionAffinity};
use crate::topology_graph::TopologyGraph;

/// Metadata store.
pub struct MetadataStore {
    /// KV index (local exact + cross-Region approximate).
    kv_index: KvIndex,
    /// Model registry.
    models: ModelRegistry,
    /// Load statistics.
    load: LoadStats,
    /// Topology graph.
    topology: TopologyGraph,
    /// Routing history.
    history: RoutingHistory,

    /// backend_id → BackendInfo (used for backends_all / by_region / by_domain).
    backends: DashMap<BackendId, BackendInfo>,
    /// region_id → list of backend_ids (cache, avoids scanning each time).
    by_region: RwLock<HashMap<RegionId, Vec<BackendId>>>,
    /// domain_id → list of backend_ids (cache).
    by_domain: RwLock<HashMap<IndexerDomainId, Vec<BackendId>>>,
}

impl std::fmt::Debug for MetadataStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetadataStore")
            .field("backends", &self.backends.len())
            .field("models", &self.models.len())
            .field("history", &self.history.len())
            .finish()
    }
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataStore {
    /// Create an empty metadata store.
    pub fn new() -> Self {
        Self {
            kv_index: KvIndex::new(),
            models: ModelRegistry::new(),
            load: LoadStats::new(),
            topology: TopologyGraph::new(),
            history: RoutingHistory::new(),
            backends: DashMap::new(),
            by_region: RwLock::new(HashMap::new()),
            by_domain: RwLock::new(HashMap::new()),
        }
    }

    // ===== KV index =====

    /// Local exact query for a backend's prefix overlap on a hash sequence.
    pub async fn kv_find_local_overlap(
        &self,
        hashes: &[u64],
        backend: BackendId,
    ) -> u32 {
        self.kv_index.kv_find_local_overlap(hashes, backend).await
    }

    /// Cross-Region approximate query for a Region's prefix overlap on a hash sequence.
    pub fn kv_find_global_overlap(&self, hashes: &[u64], region: &RegionId) -> u32 {
        self.kv_index.kv_find_global_overlap(hashes, region)
    }

    /// Apply a KV cache event to the local index.
    pub async fn kv_apply_event(
        &self,
        event: KvCacheEvent,
        backend: BackendId,
    ) -> Result<()> {
        self.kv_index.kv_apply_event(event, backend).await
    }

    /// Confidence of the current approximate queries.
    pub fn kv_confidence(&self) -> f64 {
        self.kv_index.kv_confidence()
    }

    // ===== Model registry =====

    /// Compute the match score of a backend for the target model name.
    pub fn model_match_score(&self, backend: &BackendId, model_name: &str) -> f64 {
        self.models.model_match_score(backend, model_name)
    }

    /// Find all backends that can serve the given model name.
    pub fn model_find_backends(&self, model_name: &str) -> Vec<BackendId> {
        self.models.model_find_backends(model_name)
    }

    /// Get the list of model instances for a backend.
    pub fn model_get_instances(&self, backend: &BackendId) -> Vec<ModelInstance> {
        self.models.get_instances(backend)
    }

    // ===== Load statistics =====

    /// Read the latest metrics for a backend.
    pub fn load_get_metrics(&self, backend: &BackendId) -> Option<BackendMetrics> {
        self.load.get(backend)
    }

    /// Update the metrics for a backend.
    pub fn load_update(&self, backend: BackendId, metrics: BackendMetrics) {
        self.load.update(backend, metrics);
    }

    /// Returns how stale the latest metrics are.
    pub fn load_freshness(&self, backend: &BackendId) -> Option<Duration> {
        self.load.freshness(backend)
    }

    // ===== Topology =====

    /// Query the RTT (milliseconds) between two Regions.
    pub fn topo_rtt_ms(&self, from: &RegionId, to: &RegionId) -> f64 {
        self.topology.rtt_ms(from, to)
    }

    /// Get Region information.
    pub fn topo_get_region(&self, region: &RegionId) -> Option<RegionInfo> {
        self.topology.get_region(region)
    }

    /// Update the RTT estimate between two Regions.
    pub fn topo_update_latency(
        &self,
        a: &RegionId,
        b: &RegionId,
        estimate: LatencyEstimate,
    ) {
        self.topology.update_latency(a, b, estimate);
    }

    /// Add a Region.
    pub fn topo_add_region(&self, info: RegionInfo) {
        self.topology.add_region(info);
    }

    // ===== Session affinity =====

    /// Get a session's affinity record.
    pub fn session_get(&self, session: &SessionId) -> Option<SessionAffinity> {
        self.history.get(session)
    }

    /// Write session affinity.
    pub fn session_set(
        &self,
        session: SessionId,
        backend: BackendId,
        kv_overlap_at_route: u32,
    ) {
        self.history.set(session, backend, kv_overlap_at_route);
    }

    /// Evict expired session affinity records.
    pub fn session_evict_expired(&self, ttl: Duration) -> usize {
        self.history.evict_expired(ttl)
    }

    // ===== Backend registry =====

    /// Register a backend (also updates the model registry and region/domain indices).
    pub fn register_backend(&self, info: BackendInfo) {
        let backend_id = info.id.clone();
        let region_id = info.region.clone();
        let domain_id = info.indexer_domain.clone();
        let models = info.models.clone();

        // Write to the primary index
        self.backends.insert(backend_id.clone(), info);

        // Model registry
        self.models.register(backend_id.clone(), models);

        // Update region/domain reverse indices
        let mut by_region = self.by_region.write();
        by_region
            .entry(region_id)
            .or_default()
            .push(backend_id.clone());
        drop(by_region);

        let mut by_domain = self.by_domain.write();
        by_domain
            .entry(domain_id)
            .or_default()
            .push(backend_id);
    }

    /// Unregister a backend (cleans up all related state).
    pub fn unregister_backend(&self, backend_id: &BackendId) {
        // Take out the BackendInfo first to clean up reverse indices
        let info = self.backends.remove(backend_id).map(|(_, v)| v);
        if let Some(info) = info {
            let mut by_region = self.by_region.write();
            if let Some(list) = by_region.get_mut(&info.region) {
                list.retain(|b: &BackendId| b != &info.id);
                if list.is_empty() {
                    by_region.remove(&info.region);
                }
            }
            drop(by_region);

            let mut by_domain = self.by_domain.write();
            if let Some(list) = by_domain.get_mut(&info.indexer_domain) {
                list.retain(|b: &BackendId| b != &info.id);
                if list.is_empty() {
                    by_domain.remove(&info.indexer_domain);
                }
            }
        }

        // Model registry
        self.models.unregister(backend_id);

        // Load statistics
        self.load.remove(backend_id);

        // Ownership in the KV index (async operation via spawn; here we use try_send or ignore)
        // Note: if the caller needs to ensure KV index cleanup is complete, they should additionally call `kv_index.radix().remove_backend(...)`.
        let _ = self.kv_index.radix();
    }

    /// List all registered backends.
    pub fn backends_all(&self) -> Vec<BackendInfo> {
        self.backends
            .iter()
            .map(|r: dashmap::mapref::multiple::RefMulti<'_, BackendId, BackendInfo>| r.value().clone())
            .collect()
    }

    /// List all backends in the specified Region.
    pub fn backends_by_region(&self, region: &RegionId) -> Vec<BackendId> {
        self.by_region
            .read()
            .get(region)
            .cloned()
            .unwrap_or_default()
    }

    /// List all backends in the specified IndexerDomain.
    pub fn backends_by_domain(&self, domain: &IndexerDomainId) -> Vec<BackendId> {
        self.by_domain
            .read()
            .get(domain)
            .cloned()
            .unwrap_or_default()
    }

    /// Get information about a single backend.
    pub fn backend_get(&self, backend: &BackendId) -> Option<BackendInfo> {
        self.backends
            .get(backend)
            .map(|r: dashmap::mapref::one::Ref<'_, BackendId, BackendInfo>| r.value().clone())
    }

    /// Total number of currently registered backends.
    pub fn backends_len(&self) -> usize {
        self.backends.len()
    }

    /// Trigger ownership cleanup for a backend in the KV index (async).
    ///
    /// Since `RadixTree::remove_backend` is async, this returns a future for the
    /// caller to await on the appropriate runtime.
    pub fn kv_remove_backend(
        &self,
        backend: BackendId,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        let radix = self.kv_index.radix().clone();
        async move {
            radix.remove_backend(backend).await;
        }
    }

    /// Shared KV index handle (for fine-grained operations).
    pub fn kv_index(&self) -> &KvIndex {
        &self.kv_index
    }

    /// Shared CkfConsumer handle.
    pub fn ckf_consumer(&self) -> &CkfConsumer {
        self.kv_index.consumer()
    }

    /// Shared RadixTree handle.
    pub fn radix_tree(&self) -> &RadixTree {
        self.kv_index.radix()
    }

    /// Shared model registry.
    pub fn models(&self) -> &ModelRegistry {
        &self.models
    }

    /// Shared load statistics.
    pub fn load(&self) -> &LoadStats {
        &self.load
    }

    /// Shared topology graph.
    pub fn topology(&self) -> &TopologyGraph {
        &self.topology
    }

    /// Shared routing history.
    pub fn history(&self) -> &RoutingHistory {
        &self.history
    }
}

/// Wraps `MetadataStore` in an `Arc` for sharing across tasks.
pub type SharedMetadataStore = Arc<MetadataStore>;

#[allow(dead_code)]
fn _unused_error() -> Result<()> {
    // Only used to ensure HierKvGatewayError is referenced in this module, for future error extensions.
    Err(HierKvGatewayError::Internal("placeholder".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::backend::{
        BackendCapabilities, BackendStatus, BackendType, Endpoint, KvConfig, Protocol,
        Quantization,
    };
    use hier_kv_gateway_core::ids::WorkerWithRank;

    fn backend(n: u8) -> BackendId {
        BackendId::new(format!("r{n}"), format!("i{n}"))
    }

    fn stored(hashes: Vec<u64>) -> KvCacheEvent {
        KvCacheEvent::Stored {
            worker: WorkerWithRank::from_worker_id(1),
            block_hashes: hashes,
            parent_hash: None,
            num_block_tokens: Vec::new(),
        }
    }

    #[tokio::test]
    async fn end_to_end_register_and_query() {
        let store = MetadataStore::new();
        let b = backend(1);

        // Apply a KV event
        store
            .kv_apply_event(stored(vec![1, 2, 3]), b.clone())
            .await
            .unwrap();
        let overlap = store.kv_find_local_overlap(&[1, 2, 3], b).await;
        assert_eq!(overlap, 3);

        // KV confidence
        assert!(store.kv_confidence() > 0.99);
    }

    #[test]
    fn model_match_score_for_registered_backend() {
        let store = MetadataStore::new();
        let b = backend(1);
        let region = RegionId::new("r1");
        let info = BackendInfo {
            id: b.clone(),
            backend_type: BackendType::VllmEngine,
            endpoint: Endpoint {
                url: "http://10.0.0.1:8080".to_string(),
                protocol: Protocol::Http,
            },
            models: vec![ModelInstance {
                model_name: "llama-7b".to_string(),
                model_architecture: "llama".to_string(),
                quantization: Quantization::Fp16,
                max_context_len: 4096,
                supports_tool_calling: false,
                supports_streaming: true,
            }],
            region: region.clone(),
            indexer_domain: IndexerDomainId::new(1),
            capabilities: BackendCapabilities {
                supports_kv_events: true,
                supports_batching: true,
                max_batch_size: 32,
                gpu_count: 1,
                gpu_memory_gb: 24,
            },
            kv_config: KvConfig {
                block_size: 16,
                cache_namespace: "default".to_string(),
                max_kv_blocks: 1024,
            },
            status: BackendStatus::Healthy,
        };
        store.register_backend(info);

        assert_eq!(store.model_match_score(&b, "llama-7b"), 1.0);
        assert!(!store.model_find_backends("llama-7b").is_empty());
        assert!(store.backends_by_region(&region).contains(&b));
    }
}
