//! 集群成员管理。
//!
//! 参考 Redis Cluster 的成员状态机设计：
//! - [`MemberStatus::Alive`]：最近 Pong 正常。
//! - [`MemberStatus::Suspect`]：Pong 超时，进入疑似下线。
//! - [`MemberStatus::Dead`]：Suspect 后再次超时，确认下线。
//!
//! [`MemberList`] 使用 [`DashMap`] 实现并发安全的成员表，提供：
//! - 心跳更新（`update_pong`）
//! - 状态转移（`mark_suspect` / `mark_dead`）
//! - 活跃成员过滤（`alive_members`）
//! - 随机选取（`random_members`，供 Gossip 周期性 PING 用）

use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use aether_core::ids::{InstanceId, RegionId};

use crate::messages::MetaDigest;

/// 成员状态机。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberStatus {
    /// 在线：最近一次 Pong 在预期时间内。
    Alive,
    /// 疑似下线：Pong 超时，等待进一步确认。
    Suspect,
    /// 已下线：Suspect 后再次超时，确认下线。
    Dead,
}

/// 集群中的一个成员实例。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterMember {
    /// 实例标识。
    pub instance_id: InstanceId,
    /// 所在区域。
    pub region: RegionId,
    /// 对外可达地址（host:port）。
    pub addr: String,
    /// 最近一次 Pong 时间戳（Unix 毫秒）。
    pub last_pong_unix: u64,
    /// 当前成员状态。
    pub status: MemberStatus,
    /// 该成员最新的元数据摘要。
    pub meta_digest: MetaDigest,
}

impl ClusterMember {
    /// 构造一个新成员，状态默认为 [`MemberStatus::Alive`]。
    pub fn new(
        instance_id: InstanceId,
        region: RegionId,
        addr: String,
        meta_digest: MetaDigest,
    ) -> Self {
        Self {
            instance_id,
            region,
            addr,
            last_pong_unix: now_unix_millis(),
            status: MemberStatus::Alive,
            meta_digest,
        }
    }
}

/// 取当前 Unix 时间戳（毫秒）。
///
/// 在系统时钟异常时退化为 0，避免 panic 影响集群协议可用性。
pub fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// 并发安全的集群成员列表。
///
/// 内部使用 [`DashMap`]，所有读写均无粗粒度锁；适合 Gossip 引擎的高频读写。
pub struct MemberList {
    members: DashMap<InstanceId, ClusterMember>,
}

impl Default for MemberList {
    fn default() -> Self {
        Self::new()
    }
}

impl MemberList {
    /// 创建空成员列表。
    pub fn new() -> Self {
        Self {
            members: DashMap::new(),
        }
    }

    /// 插入或覆盖一个成员。
    pub fn upsert(&self, member: ClusterMember) {
        self.members.insert(member.instance_id.clone(), member);
    }

    /// 更新成员最近一次 Pong 时间戳与元数据摘要，并将状态置为 [`MemberStatus::Alive`]。
    ///
    /// 若该成员不存在，则忽略更新。
    pub fn update_pong(&self, instance_id: &InstanceId, digest: MetaDigest) {
        if let Some(mut entry) = self.members.get_mut(instance_id) {
            entry.last_pong_unix = now_unix_millis();
            entry.status = MemberStatus::Alive;
            entry.meta_digest = digest;
        }
    }

    /// 将成员标记为 [`MemberStatus::Suspect`]。
    pub fn mark_suspect(&self, instance_id: &InstanceId) {
        if let Some(mut entry) = self.members.get_mut(instance_id) {
            entry.status = MemberStatus::Suspect;
        }
    }

    /// 将成员标记为 [`MemberStatus::Dead`]。
    pub fn mark_dead(&self, instance_id: &InstanceId) {
        if let Some(mut entry) = self.members.get_mut(instance_id) {
            entry.status = MemberStatus::Dead;
        }
    }

    /// 获取某个成员的克隆。
    pub fn get(&self, instance_id: &InstanceId) -> Option<ClusterMember> {
        self.members.get(instance_id).map(|r| r.clone())
    }

    /// 返回所有状态为 [`MemberStatus::Alive`] 的成员。
    pub fn alive_members(&self) -> Vec<ClusterMember> {
        self.members
            .iter()
            .filter(|r| r.status == MemberStatus::Alive)
            .map(|r| r.clone())
            .collect()
    }

    /// 返回所有成员（含 Suspect / Dead）。
    pub fn all_members(&self) -> Vec<ClusterMember> {
        self.members.iter().map(|r| r.clone()).collect()
    }

    /// 从 alive 成员中随机选取至多 `n` 个。
    ///
    /// 使用 Fisher-Yates 打乱后取前 n 个，保证不同成员被选中的概率均匀。
    pub fn random_members(&self, n: usize) -> Vec<ClusterMember> {
        let mut alive = self.alive_members();
        if alive.is_empty() || n == 0 {
            return Vec::new();
        }
        let mut rng = rand::rng();
        alive.shuffle(&mut rng);
        alive.into_iter().take(n).collect()
    }

    /// 当前成员总数。
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// 移除指定成员。
    pub fn remove(&self, instance_id: &InstanceId) -> Option<ClusterMember> {
        self.members.remove(instance_id).map(|(_, v)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(name: &str, region: &str) -> ClusterMember {
        ClusterMember::new(
            InstanceId::new(name),
            RegionId::new(region),
            format!("127.0.0.1:{}", 7000 + name.len()),
            MetaDigest::default(),
        )
    }

    #[test]
    fn upsert_and_get() {
        let list = MemberList::new();
        let m = member("g1", "r1");
        list.upsert(m.clone());
        assert_eq!(list.len(), 1);
        let got = list.get(&InstanceId::new("g1")).unwrap();
        assert_eq!(got.instance_id.as_str(), "g1");
        assert_eq!(got.status, MemberStatus::Alive);
    }

    #[test]
    fn update_pong_refreshes_status_and_digest() {
        let list = MemberList::new();
        let m = member("g1", "r1");
        list.upsert(m.clone());
        list.mark_suspect(&InstanceId::new("g1"));

        let digest = MetaDigest {
            kv_version: 9,
            ..MetaDigest::default()
        };
        list.update_pong(&InstanceId::new("g1"), digest.clone());

        let got = list.get(&InstanceId::new("g1")).unwrap();
        assert_eq!(got.status, MemberStatus::Alive);
        assert_eq!(got.meta_digest.kv_version, 9);
    }

    #[test]
    fn mark_suspect_and_dead_transitions() {
        let list = MemberList::new();
        list.upsert(member("g1", "r1"));
        list.mark_suspect(&InstanceId::new("g1"));
        assert_eq!(
            list.get(&InstanceId::new("g1")).unwrap().status,
            MemberStatus::Suspect
        );
        list.mark_dead(&InstanceId::new("g1"));
        assert_eq!(
            list.get(&InstanceId::new("g1")).unwrap().status,
            MemberStatus::Dead
        );
    }

    #[test]
    fn alive_members_excludes_suspect_and_dead() {
        let list = MemberList::new();
        list.upsert(member("g1", "r1"));
        list.upsert(member("g2", "r1"));
        list.upsert(member("g3", "r1"));
        list.mark_suspect(&InstanceId::new("g2"));
        list.mark_dead(&InstanceId::new("g3"));

        let alive = list.alive_members();
        assert_eq!(alive.len(), 1);
        assert_eq!(alive[0].instance_id.as_str(), "g1");
    }

    #[test]
    fn random_members_returns_at_most_n() {
        let list = MemberList::new();
        for i in 0..10 {
            list.upsert(member(&format!("g{i}"), "r1"));
        }
        let picked = list.random_members(3);
        assert!(picked.len() <= 3);
        assert!(!picked.is_empty());

        // 取 0 时返回空
        assert!(list.random_members(0).is_empty());

        // 取超过 alive 总数时只返回 alive 总数
        let picked_all = list.random_members(100);
        assert_eq!(picked_all.len(), 10);
    }

    #[test]
    fn random_members_empty_list_returns_empty() {
        let list = MemberList::new();
        assert!(list.random_members(3).is_empty());
    }

    #[test]
    fn remove_drops_member() {
        let list = MemberList::new();
        list.upsert(member("g1", "r1"));
        let removed = list.remove(&InstanceId::new("g1"));
        assert!(removed.is_some());
        assert!(list.is_empty());
        assert!(list.get(&InstanceId::new("g1")).is_none());
    }
}
