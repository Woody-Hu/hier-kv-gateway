//! 元数据存储统一入口：组合所有元数据组件，提供单一访问点。
//!
//! [`MetadataStore`] 持有 KV 索引、模型注册表、负载统计、拓扑图、路由历史
//! 以及 backend 注册表，向 routing 层暴露一组高层查询接口。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use aether_core::backend::{BackendInfo, ModelInstance};
use aether_core::error::{AetherError, Result};
use aether_core::ids::{BackendId, IndexerDomainId, RegionId, SessionId};
use aether_core::kv_event::KvCacheEvent;
use aether_core::metrics::BackendMetrics;
use aether_core::topology::{LatencyEstimate, RegionInfo};
use dashmap::DashMap;
use parking_lot::RwLock;

use crate::ckf_consumer::CkfConsumer;
use crate::kv_index::KvIndex;
use crate::load_stats::LoadStats;
use crate::model_registry::ModelRegistry;
use crate::radix_tree::RadixTree;
use crate::routing_history::{RoutingHistory, SessionAffinity};
use crate::topology_graph::TopologyGraph;

/// 元数据存储。
pub struct MetadataStore {
    /// KV 索引（本地精确 + 跨 Region 近似）。
    kv_index: KvIndex,
    /// 模型注册表。
    models: ModelRegistry,
    /// 负载统计。
    load: LoadStats,
    /// 拓扑图。
    topology: TopologyGraph,
    /// 路由历史。
    history: RoutingHistory,

    /// backend_id → BackendInfo（用于 backends_all / by_region / by_domain）。
    backends: DashMap<BackendId, BackendInfo>,
    /// region_id → backend_id 列表（缓存，避免每次扫描）。
    by_region: RwLock<HashMap<RegionId, Vec<BackendId>>>,
    /// domain_id → backend_id 列表（缓存）。
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
    /// 创建一个空的元数据存储。
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

    // ===== KV 索引 =====

    /// 本地精确查询 backend 对 hash 序列的前缀重叠。
    pub async fn kv_find_local_overlap(
        &self,
        hashes: &[u64],
        backend: BackendId,
    ) -> u32 {
        self.kv_index.kv_find_local_overlap(hashes, backend).await
    }

    /// 跨 Region 近似查询 region 对 hash 序列的前缀重叠。
    pub fn kv_find_global_overlap(&self, hashes: &[u64], region: &RegionId) -> u32 {
        self.kv_index.kv_find_global_overlap(hashes, region)
    }

    /// 应用一个 KV cache 事件到本地索引。
    pub async fn kv_apply_event(
        &self,
        event: KvCacheEvent,
        backend: BackendId,
    ) -> Result<()> {
        self.kv_index.kv_apply_event(event, backend).await
    }

    /// 当前近似查询的可信度。
    pub fn kv_confidence(&self) -> f64 {
        self.kv_index.kv_confidence()
    }

    // ===== 模型注册表 =====

    /// 计算 backend 对目标模型名的匹配分数。
    pub fn model_match_score(&self, backend: &BackendId, model_name: &str) -> f64 {
        self.models.model_match_score(backend, model_name)
    }

    /// 查询所有能服务该模型名的 backend。
    pub fn model_find_backends(&self, model_name: &str) -> Vec<BackendId> {
        self.models.model_find_backends(model_name)
    }

    /// 获取 backend 的模型实例列表。
    pub fn model_get_instances(&self, backend: &BackendId) -> Vec<ModelInstance> {
        self.models.get_instances(backend)
    }

    // ===== 负载统计 =====

    /// 读取 backend 的最新指标。
    pub fn load_get_metrics(&self, backend: &BackendId) -> Option<BackendMetrics> {
        self.load.get(backend)
    }

    /// 更新 backend 的指标。
    pub fn load_update(&self, backend: BackendId, metrics: BackendMetrics) {
        self.load.update(backend, metrics);
    }

    /// 返回最新指标的过期时间。
    pub fn load_freshness(&self, backend: &BackendId) -> Option<Duration> {
        self.load.freshness(backend)
    }

    // ===== 拓扑 =====

    /// 查询两个 Region 之间的 RTT（毫秒）。
    pub fn topo_rtt_ms(&self, from: &RegionId, to: &RegionId) -> f64 {
        self.topology.rtt_ms(from, to)
    }

    /// 获取 Region 信息。
    pub fn topo_get_region(&self, region: &RegionId) -> Option<RegionInfo> {
        self.topology.get_region(region)
    }

    /// 更新两 Region 之间的 RTT 估计。
    pub fn topo_update_latency(
        &self,
        a: &RegionId,
        b: &RegionId,
        estimate: LatencyEstimate,
    ) {
        self.topology.update_latency(a, b, estimate);
    }

    /// 添加 Region。
    pub fn topo_add_region(&self, info: RegionInfo) {
        self.topology.add_region(info);
    }

    // ===== 会话亲和 =====

    /// 获取会话的亲和记录。
    pub fn session_get(&self, session: &SessionId) -> Option<SessionAffinity> {
        self.history.get(session)
    }

    /// 写入会话亲和。
    pub fn session_set(
        &self,
        session: SessionId,
        backend: BackendId,
        kv_overlap_at_route: u32,
    ) {
        self.history.set(session, backend, kv_overlap_at_route);
    }

    /// 清理过期会话亲和。
    pub fn session_evict_expired(&self, ttl: Duration) -> usize {
        self.history.evict_expired(ttl)
    }

    // ===== Backend 注册表 =====

    /// 注册一个 backend（同时更新模型注册表与 region/domain 索引）。
    pub fn register_backend(&self, info: BackendInfo) {
        let backend_id = info.id.clone();
        let region_id = info.region.clone();
        let domain_id = info.indexer_domain.clone();
        let models = info.models.clone();

        // 写入主索引
        self.backends.insert(backend_id.clone(), info);

        // 模型注册表
        self.models.register(backend_id.clone(), models);

        // 更新 region/domain 反向索引
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

    /// 注销一个 backend（清理所有相关状态）。
    pub fn unregister_backend(&self, backend_id: &BackendId) {
        // 先取出 BackendInfo 以便清理反向索引
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

        // 模型注册表
        self.models.unregister(backend_id);

        // 负载统计
        self.load.remove(backend_id);

        // KV 索引中的所有权（异步操作通过 spawn 处理；此处使用 try_send 或忽略）
        // 注：调用方若需要确保 KV 索引清理完成，应额外调用 `kv_index.radix().remove_backend(...)`。
        let _ = self.kv_index.radix();
    }

    /// 列出所有已注册的 backend。
    pub fn backends_all(&self) -> Vec<BackendInfo> {
        self.backends
            .iter()
            .map(|r: dashmap::mapref::multiple::RefMulti<'_, BackendId, BackendInfo>| r.value().clone())
            .collect()
    }

    /// 列出指定 Region 下的所有 backend。
    pub fn backends_by_region(&self, region: &RegionId) -> Vec<BackendId> {
        self.by_region
            .read()
            .get(region)
            .cloned()
            .unwrap_or_default()
    }

    /// 列出指定 IndexerDomain 下的所有 backend。
    pub fn backends_by_domain(&self, domain: &IndexerDomainId) -> Vec<BackendId> {
        self.by_domain
            .read()
            .get(domain)
            .cloned()
            .unwrap_or_default()
    }

    /// 获取单个 backend 的信息。
    pub fn backend_get(&self, backend: &BackendId) -> Option<BackendInfo> {
        self.backends
            .get(backend)
            .map(|r: dashmap::mapref::one::Ref<'_, BackendId, BackendInfo>| r.value().clone())
    }

    /// 当前注册的 backend 总数。
    pub fn backends_len(&self) -> usize {
        self.backends.len()
    }

    /// 触发 KV 索引中某 backend 的所有权清理（异步）。
    ///
    /// 由于 `RadixTree::remove_backend` 是 async，这里返回一个 future，由调用方
    /// 在合适的运行时上 await。
    pub fn kv_remove_backend(
        &self,
        backend: BackendId,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        let radix = self.kv_index.radix().clone();
        async move {
            radix.remove_backend(backend).await;
        }
    }

    /// 共享 KV 索引句柄（用于细粒度操作）。
    pub fn kv_index(&self) -> &KvIndex {
        &self.kv_index
    }

    /// 共享 CkfConsumer 句柄。
    pub fn ckf_consumer(&self) -> &CkfConsumer {
        self.kv_index.consumer()
    }

    /// 共享 RadixTree 句柄。
    pub fn radix_tree(&self) -> &RadixTree {
        self.kv_index.radix()
    }

    /// 共享模型注册表。
    pub fn models(&self) -> &ModelRegistry {
        &self.models
    }

    /// 共享负载统计。
    pub fn load(&self) -> &LoadStats {
        &self.load
    }

    /// 共享拓扑图。
    pub fn topology(&self) -> &TopologyGraph {
        &self.topology
    }

    /// 共享路由历史。
    pub fn history(&self) -> &RoutingHistory {
        &self.history
    }
}

/// 将 `MetadataStore` 包装为 `Arc` 以便跨任务共享。
pub type SharedMetadataStore = Arc<MetadataStore>;

#[allow(dead_code)]
fn _unused_error() -> Result<()> {
    // 仅用于确保 AetherError 在此模块被引用，便于未来错误扩展。
    Err(AetherError::Internal("placeholder".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_core::backend::{
        BackendCapabilities, BackendStatus, BackendType, Endpoint, KvConfig, Protocol,
        Quantization,
    };
    use aether_core::ids::WorkerWithRank;

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

        // 应用一个 KV 事件
        store
            .kv_apply_event(stored(vec![1, 2, 3]), b.clone())
            .await
            .unwrap();
        let overlap = store.kv_find_local_overlap(&[1, 2, 3], b).await;
        assert_eq!(overlap, 3);

        // KV 置信度
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
