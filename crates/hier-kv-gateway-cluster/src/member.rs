//! Cluster member management.
//!
//! Member state machine design:
//! - [`MemberStatus::Alive`]: the most recent Pong was normal.
//! - [`MemberStatus::Suspect`]: Pong timed out, entered suspected-offline.
//! - [`MemberStatus::Dead`]: timed out again after Suspect, confirmed offline.
//!
//! [`MemberList`] uses [`DashMap`] to implement a concurrency-safe member table, providing:
//! - heartbeat update (`update_pong`)
//! - state transition (`mark_suspect` / `mark_dead`)
//! - alive member filter (`alive_members`)
//! - random selection (`random_members`, for Gossip periodic PING)

use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use hier_kv_gateway_core::ids::{InstanceId, RegionId};

use crate::messages::MetaDigest;

/// Member state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberStatus {
    /// Online: the most recent Pong was within the expected time.
    Alive,
    /// Suspected offline: Pong timed out, awaiting further confirmation.
    Suspect,
    /// Offline: timed out again after Suspect, confirmed offline.
    Dead,
}

/// A member instance in the cluster.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterMember {
    /// Instance identifier.
    pub instance_id: InstanceId,
    /// Region where the instance resides.
    pub region: RegionId,
    /// Externally reachable address (host:port).
    pub addr: String,
    /// Timestamp of the most recent Pong (Unix milliseconds).
    pub last_pong_unix: u64,
    /// Current member status.
    pub status: MemberStatus,
    /// The latest metadata digest of this member.
    pub meta_digest: MetaDigest,
}

impl ClusterMember {
    /// Construct a new member, with status defaulting to [`MemberStatus::Alive`].
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

/// Returns the current Unix timestamp (milliseconds).
///
/// Degrades to 0 when the system clock is abnormal, to avoid panics that affect cluster protocol availability.
pub fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Concurrency-safe cluster member list.
///
/// Internally uses [`DashMap`]; all reads and writes have no coarse-grained lock,
/// suitable for the high-frequency reads and writes of the Gossip engine.
pub struct MemberList {
    members: DashMap<InstanceId, ClusterMember>,
}

impl Default for MemberList {
    fn default() -> Self {
        Self::new()
    }
}

impl MemberList {
    /// Create an empty member list.
    pub fn new() -> Self {
        Self {
            members: DashMap::new(),
        }
    }

    /// Insert or overwrite a member.
    pub fn upsert(&self, member: ClusterMember) {
        self.members.insert(member.instance_id.clone(), member);
    }

    /// Update the member's most recent Pong timestamp and metadata digest, and set the status to [`MemberStatus::Alive`].
    ///
    /// If the member does not exist, the update is ignored.
    pub fn update_pong(&self, instance_id: &InstanceId, digest: MetaDigest) {
        if let Some(mut entry) = self.members.get_mut(instance_id) {
            entry.last_pong_unix = now_unix_millis();
            entry.status = MemberStatus::Alive;
            entry.meta_digest = digest;
        }
    }

    /// Mark a member as [`MemberStatus::Suspect`].
    pub fn mark_suspect(&self, instance_id: &InstanceId) {
        if let Some(mut entry) = self.members.get_mut(instance_id) {
            entry.status = MemberStatus::Suspect;
        }
    }

    /// Mark a member as [`MemberStatus::Dead`].
    pub fn mark_dead(&self, instance_id: &InstanceId) {
        if let Some(mut entry) = self.members.get_mut(instance_id) {
            entry.status = MemberStatus::Dead;
        }
    }

    /// Get a clone of a member.
    pub fn get(&self, instance_id: &InstanceId) -> Option<ClusterMember> {
        self.members.get(instance_id).map(|r| r.clone())
    }

    /// Returns all members whose status is [`MemberStatus::Alive`].
    pub fn alive_members(&self) -> Vec<ClusterMember> {
        self.members
            .iter()
            .filter(|r| r.status == MemberStatus::Alive)
            .map(|r| r.clone())
            .collect()
    }

    /// Returns all members (including Suspect / Dead).
    pub fn all_members(&self) -> Vec<ClusterMember> {
        self.members.iter().map(|r| r.clone()).collect()
    }

    /// Randomly select at most `n` members from alive members.
    ///
    /// Uses Fisher-Yates shuffle and takes the first n, ensuring a uniform probability
    /// of selection across different members.
    pub fn random_members(&self, n: usize) -> Vec<ClusterMember> {
        let mut alive = self.alive_members();
        if alive.is_empty() || n == 0 {
            return Vec::new();
        }
        let mut rng = rand::rng();
        alive.shuffle(&mut rng);
        alive.into_iter().take(n).collect()
    }

    /// Current total number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Remove a member.
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

        // Returns empty when n is 0
        assert!(list.random_members(0).is_empty());

        // When n exceeds the total alive count, returns only the total alive count
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
