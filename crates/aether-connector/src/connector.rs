//! 后端连接器 trait 与通用类型定义。
//!
//! [`BackendConnector`] 抽象了一个推理后端实例集合所共有的能力：
//! 发现、健康检查、流式推理转发、KV 缓存事件订阅、指标采集。
//! 不同引擎（vLLM / llama.cpp / 通用 OpenAI 兼容服务）各自实现该 trait。
//!
//! [`HealthStatus`] 描述一次健康检查的快照结果，被路由层用于排除不健康实例。

use aether_core::backend::{BackendInfo, BackendStatus, BackendType};
use aether_core::error::Result;
use aether_core::ids::BackendId;
use aether_core::kv_event::KvCacheEvent;
use aether_core::metrics::BackendMetrics;
use aether_core::request::{InferenceChunk, InferenceRequest};

use async_trait::async_trait;

/// 后端连接器抽象。
///
/// 一个 `BackendConnector` 通常对应某一类（[`BackendType`]）后端实例的访问入口，
/// 实现该方法的对象以 `Arc<dyn BackendConnector>` 形式注册到 [`crate::registry::ConnectorRegistry`]。
#[async_trait]
pub trait BackendConnector: Send + Sync {
    /// 返回该连接器所代理的后端类型。
    fn backend_type(&self) -> BackendType;

    /// 发现该连接器管理的后端实例列表。
    ///
    /// 通常在握手阶段执行一次，得到一组静态元数据 [`BackendInfo`]，
    /// 并由集群层在成员变更时重新触发。
    async fn discover(&self) -> Result<Vec<BackendInfo>>;

    /// 对指定后端执行健康检查，返回当前快照 [`HealthStatus`]。
    async fn health_check(&self, backend: &BackendId) -> Result<HealthStatus>;

    /// 将推理请求转发至指定后端，并以流的形式返回 [`InferenceChunk`] 序列。
    ///
    /// 流必须以 [`InferenceChunk::Done`] 或 [`InferenceChunk::Error`] 终止，
    /// 调用方据此判断请求结束或失败。
    async fn forward(
        &self,
        backend: &BackendId,
        request: &InferenceRequest,
    ) -> Result<futures::stream::BoxStream<'static, InferenceChunk>>;

    /// 该连接器是否支持订阅 KV 缓存事件。
    ///
    /// OpenAI 兼容服务通常不提供 KV 事件，应返回 `false`。
    fn supports_kv_events(&self) -> bool;

    /// 订阅指定后端的 KV 缓存事件流。
    ///
    /// 当 [`BackendConnector::supports_kv_events`] 返回 `false` 时，
    /// 实现应直接返回 [`aether_core::error::AetherError::ConnectorError`]。
    async fn subscribe_kv_events(
        &self,
        backend: &BackendId,
    ) -> Result<futures::stream::BoxStream<'static, KvCacheEvent>>;

    /// 采集指定后端的实时负载指标 [`BackendMetrics`]。
    ///
    /// 当后端未暴露指标端点时，实现可返回结构化的默认值（零值快照），
    /// 以保证路由层在没有真实指标时仍能继续工作。
    async fn collect_metrics(&self, backend: &BackendId) -> Result<BackendMetrics>;
}

/// 一次健康检查的快照结果。
///
/// 由 [`BackendConnector::health_check`] 返回，记录当前状态、连续健康时间与
/// 累计错误数，供集群层决定是否将实例从路由池中摘除。
#[derive(Clone, Debug)]
pub struct HealthStatus {
    /// 当前后端运行状态。
    pub status: BackendStatus,
    /// 进入当前状态以来的 Unix 时间戳（秒）。
    ///
    /// 当 `status` 为 [`BackendStatus::Healthy`] 时，该字段表示自何时起持续健康；
    /// 其他状态下含义为状态切换时刻。
    pub healthy_since_unix: u64,
    /// 连续探活失败次数，健康状态下应归零。
    pub error_count: u32,
}

impl HealthStatus {
    /// 构造一个健康的快照，`error_count` 为 0。
    pub fn healthy(since_unix: u64) -> Self {
        Self {
            status: BackendStatus::Healthy,
            healthy_since_unix: since_unix,
            error_count: 0,
        }
    }

    /// 构造一个不健康的快照，记录累计错误数。
    pub fn unhealthy(since_unix: u64, error_count: u32) -> Self {
        Self {
            status: BackendStatus::Unhealthy,
            healthy_since_unix: since_unix,
            error_count,
        }
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            status: BackendStatus::Unknown,
            healthy_since_unix: 0,
            error_count: 0,
        }
    }
}
