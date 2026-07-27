//! Aether 系统中跨模块复用的核心标识类型。
//!
//! 这些类型用于唯一确定区域、后端实例、索引域、连接池、请求、会话与 worker。
//! 大多数标识符以 [`Arc<str>`](std::sync::Arc) 形式存储，便于在异步任务之间低成本克隆与共享。

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// 区域标识。
///
/// 内部使用 [`Arc<str>`]，因而克隆与跨任务传递代价低；同时支持基于字符串内容的
/// 比较（`Hash`/`Eq`/`Ord`），不依赖 `Arc` 指针地址。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionId(pub Arc<str>);

impl RegionId {
    /// 以字符串字面量创建 [`RegionId`]。
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    /// 获取内部字符串切片。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for RegionId {
    fn eq(&self, other: &Self) -> bool {
        // 比较字符串内容而非 Arc 指针
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for RegionId {}

impl Hash for RegionId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // 仅哈希字符串内容，保证 Eq / Hash 一致
        self.0.as_ref().hash(state);
    }
}

impl PartialOrd for RegionId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegionId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.as_ref().cmp(other.0.as_ref())
    }
}

impl From<String> for RegionId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for RegionId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for RegionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// 区域层级。
///
/// 决定了区域在 Aether 三层拓扑中的位置，进而影响路由策略与缓存迁移路径。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RegionTier {
    /// 云区域，常作为中心调度节点。
    Cloud,
    /// 边缘区域，靠近用户的接入点。
    Edge,
    /// 设备端区域，部署在终端设备上。
    Device,
}

/// 后端实例标识。
///
/// 后端实例是同一后端服务进程在不同副本下的唯一标识，
/// 通常表现为 hostname 或 Pod 名。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendInstanceId(pub Arc<str>);

impl BackendInstanceId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for BackendInstanceId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for BackendInstanceId {}

impl Hash for BackendInstanceId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}

impl From<String> for BackendInstanceId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for BackendInstanceId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for BackendInstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// 后端标识：由区域与实例两部分组成。
///
/// `region` 描述后端所处的 [`RegionId`]，`instance` 是该区域内的实例标识，
/// 二者组合即可在集群全局唯一确定一个后端进程。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendId {
    pub region: RegionId,
    pub instance: BackendInstanceId,
}

impl BackendId {
    pub fn new(region: impl Into<RegionId>, instance: impl Into<BackendInstanceId>) -> Self {
        Self {
            region: region.into(),
            instance: instance.into(),
        }
    }
}

impl PartialEq for BackendId {
    fn eq(&self, other: &Self) -> bool {
        self.region == other.region && self.instance == other.instance
    }
}

impl Eq for BackendId {}

impl Hash for BackendId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.region.hash(state);
        self.instance.hash(state);
    }
}

impl PartialOrd for BackendId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BackendId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.region
            .cmp(&other.region)
            .then_with(|| self.instance.as_str().cmp(other.instance.as_str()))
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.region, self.instance)
    }
}

/// 索引器域标识。
///
/// 索引器按域划分，多个后端可共享同一索引域，从而共享 KV 缓存索引。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexerDomainId(pub u64);

impl IndexerDomainId {
    pub fn new(v: u64) -> Self {
        Self(v)
    }
}

impl From<u64> for IndexerDomainId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// 连接池标识：由索引域与区域组成。
///
/// 同一连接池内的后端共享索引域，且通常位于同一区域以便就近路由。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolId {
    pub domain: IndexerDomainId,
    pub region: RegionId,
}

impl PartialEq for PoolId {
    fn eq(&self, other: &Self) -> bool {
        self.domain == other.domain && self.region == other.region
    }
}

impl Eq for PoolId {}

impl Hash for PoolId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.domain.hash(state);
        self.region.hash(state);
    }
}

/// 网关实例标识。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceId(pub Arc<str>);

impl InstanceId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for InstanceId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for InstanceId {}

impl Hash for InstanceId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}

impl From<String> for InstanceId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for InstanceId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for InstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// 请求标识。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestId(pub Arc<str>);

impl RequestId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for RequestId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for RequestId {}

impl Hash for RequestId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// 会话标识。
///
/// 同一会话内的多次请求可享有亲和性，命中相同后端的 KV 缓存。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionId(pub Arc<str>);

impl SessionId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for SessionId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for SessionId {}

impl Hash for SessionId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// worker 标识与对应的数据并行 rank。
///
/// `worker_id` 标识 worker 全局唯一身份；`dp_rank` 表示该 worker 在数据并行组中的 rank，
/// `dp_rank = 0` 通常表示 DP 未启用或为首个 rank。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerWithRank {
    pub worker_id: u64,
    pub dp_rank: u32,
}

impl WorkerWithRank {
    pub fn new(worker_id: u64, dp_rank: u32) -> Self {
        Self { worker_id, dp_rank }
    }

    /// 仅以 worker_id 构造，dp_rank 默认为 0。
    pub fn from_worker_id(worker_id: u64) -> Self {
        Self {
            worker_id,
            dp_rank: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_id_eq_ignores_arc_pointer() {
        // 内容相同但 Arc 不共享底层的两个 RegionId 应相等
        let a = RegionId::new("us-east-1");
        let b = RegionId::new("us-east-1");
        // 强制独立分配，避免 Arc::from 复用单例
        let c = RegionId(Arc::from(String::from("us-east-1").as_str()));
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn region_id_hash_consistent_with_eq() {
        let mut set = std::collections::HashSet::new();
        set.insert(RegionId::new("eu-west-1"));
        // 不同 Arc 但内容相同应命中同一 bucket
        assert!(set.contains(&RegionId(Arc::from(
            String::from("eu-west-1").as_str()
        ))));
    }

    #[test]
    fn region_id_ord_lexicographic() {
        let a = RegionId::new("a");
        let b = RegionId::new("b");
        assert!(a < b);
    }

    #[test]
    fn backend_id_round_trip_json() {
        let id = BackendId::new("us-east-1", "worker-0");
        let s = serde_json::to_string(&id).unwrap();
        let back: BackendId = serde_json::from_str(&s).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn backend_id_ord_compound() {
        // 先按 region 后按 instance 比较
        let a = BackendId::new("r1", "i1");
        let b = BackendId::new("r1", "i2");
        let c = BackendId::new("r2", "i1");
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn worker_with_rank_order() {
        let a = WorkerWithRank::new(1, 0);
        let b = WorkerWithRank::new(1, 1);
        let c = WorkerWithRank::new(2, 0);
        assert!(a < b);
        assert!(b < c);
    }
}
