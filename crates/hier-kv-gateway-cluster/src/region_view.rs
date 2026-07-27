//! Region-grouped view over [`MemberList`].
//!
//! The underlying [`MemberList`] is a flat `DashMap<InstanceId, ClusterMember>`
//! with no Region awareness. In a cloud-edge deployment the membership is
//! typically spread across multiple Regions (cloud / edge / device), and the
//! gossip/routing layers frequently need to answer questions like:
//!
//! - "Which alive members are in *my* Region?" (local peers for cheap Ping)
//! - "Which alive members are in *foreign* Regions?" (cross-Region fanout)
//! - "How many Regions are currently represented in the cluster?"
//!
//! [`RegionMemberView`] answers these by wrapping a shared `MemberList` handle
//! and providing Region-filtered queries. It is **stateless** — every query
//! scans the current `MemberList` snapshot — so it never goes stale and never
//! needs a `rebuild()` call. The trade-off is O(N) per query, which is fine
//! for the cluster sizes this gateway targets (tens to low-hundreds of
//! members). If a future deployment reaches thousands of members, a cached
//! index with explicit invalidation can be layered on top without changing
//! the public API.

use std::collections::HashMap;
use std::sync::Arc;

use hier_kv_gateway_core::ids::RegionId;

use crate::member::{ClusterMember, MemberList};

/// Region-grouped view over a shared [`MemberList`].
///
/// Cheap to construct (just an `Arc` clone); all queries are read-only and
/// lock-free. See the module docs for the design trade-offs.
pub struct RegionMemberView {
    members: Arc<MemberList>,
}

impl Clone for RegionMemberView {
    fn clone(&self) -> Self {
        Self {
            members: self.members.clone(),
        }
    }
}

impl RegionMemberView {
    /// Wrap a shared `MemberList`.
    pub fn new(members: Arc<MemberList>) -> Self {
        Self { members }
    }

    /// All alive members whose Region equals `region`.
    ///
    /// This is the primary "local peers" query: in a given Region the gossip
    /// loop can use it to prioritise same-Region Pings (lower RTT, cheaper
    /// bandwidth).
    pub fn alive_in_region(&self, region: &RegionId) -> Vec<ClusterMember> {
        self.members
            .alive_members()
            .into_iter()
            .filter(|m| &m.region == region)
            .collect()
    }

    /// All alive members whose Region is *not* `self_region`.
    ///
    /// Use this to drive cross-Region fanout: when broadcasting a CKF snapshot
    /// or a metrics digest, one peer per foreign Region is enough (the
    /// intra-Region gossip loop spreads it within the Region afterwards).
    pub fn foreign_alive(&self, self_region: &RegionId) -> Vec<ClusterMember> {
        self.members
            .alive_members()
            .into_iter()
            .filter(|m| m.region != *self_region)
            .collect()
    }

    /// All alive members whose Region equals `self_region`.
    ///
    /// Convenience alias for [`alive_in_region`](Self::alive_in_region) with
    /// `self_region` as the argument.
    pub fn local_alive(&self, self_region: &RegionId) -> Vec<ClusterMember> {
        self.alive_in_region(self_region)
    }

    /// Returns at most `n` alive members per foreign Region.
    ///
    /// Used by cross-Region gossip fanout: pick a small number of
    /// representatives from each foreign Region rather than fanning out to
    /// every member of every foreign Region.
    pub fn foreign_alive_per_region(
        &self,
        self_region: &RegionId,
        n_per_region: usize,
    ) -> HashMap<RegionId, Vec<ClusterMember>> {
        let mut by_region: HashMap<RegionId, Vec<ClusterMember>> = HashMap::new();
        for m in self.members.alive_members() {
            if m.region == *self_region {
                continue;
            }
            let bucket = by_region.entry(m.region.clone()).or_default();
            if bucket.len() < n_per_region {
                bucket.push(m);
            }
        }
        by_region
    }

    /// Distinct Regions currently represented in the alive membership,
    /// sorted for deterministic output.
    pub fn known_regions(&self) -> Vec<RegionId> {
        let mut regions: Vec<RegionId> = self
            .members
            .alive_members()
            .into_iter()
            .map(|m| m.region)
            .collect();
        regions.sort();
        regions.dedup();
        regions
    }

    /// Distinct foreign Regions (i.e. excluding `self_region`), sorted.
    pub fn foreign_regions(&self, self_region: &RegionId) -> Vec<RegionId> {
        let mut regions: Vec<RegionId> = self
            .members
            .alive_members()
            .into_iter()
            .map(|m| m.region)
            .filter(|r| r != self_region)
            .collect();
        regions.sort();
        regions.dedup();
        regions
    }

    /// Count of alive members per Region.
    ///
    /// Useful for diagnostics (`GET /admin/cluster` style endpoints) and for
    /// topology-aware routing heuristics that weight Regions by capacity.
    pub fn region_counts(&self) -> HashMap<RegionId, usize> {
        let mut counts: HashMap<RegionId, usize> = HashMap::new();
        for m in self.members.alive_members() {
            *counts.entry(m.region).or_insert(0) += 1;
        }
        counts
    }

    /// Total alive member count (delegates to [`MemberList::alive_members`]).
    pub fn alive_count(&self) -> usize {
        self.members.alive_members().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::MetaDigest;
    use hier_kv_gateway_core::ids::InstanceId;

    fn member(name: &str, region: &str) -> ClusterMember {
        ClusterMember::new(
            InstanceId::new(name),
            RegionId::new(region),
            format!("127.0.0.1:{}", 7000 + name.len()),
            MetaDigest::default(),
        )
    }

    fn view_with_members(members: &[ClusterMember]) -> RegionMemberView {
        let list = Arc::new(MemberList::new());
        for m in members {
            list.upsert(m.clone());
        }
        RegionMemberView::new(list)
    }

    #[test]
    fn alive_in_region_filters_correctly() {
        let v = view_with_members(&[
            member("g1", "cloud"),
            member("g2", "cloud"),
            member("g3", "edge"),
            member("g4", "edge"),
            member("g5", "device"),
        ]);
        let cloud = v.alive_in_region(&RegionId::new("cloud"));
        assert_eq!(cloud.len(), 2);
        assert!(cloud.iter().all(|m| m.region.as_str() == "cloud"));

        let edge = v.alive_in_region(&RegionId::new("edge"));
        assert_eq!(edge.len(), 2);

        let none = v.alive_in_region(&RegionId::new("nonexistent"));
        assert!(none.is_empty());
    }

    #[test]
    fn foreign_alive_excludes_self_region() {
        let v = view_with_members(&[
            member("g1", "cloud"),
            member("g2", "cloud"),
            member("g3", "edge"),
            member("g4", "device"),
        ]);
        let foreign = v.foreign_alive(&RegionId::new("cloud"));
        assert_eq!(foreign.len(), 2);
        assert!(foreign.iter().all(|m| m.region.as_str() != "cloud"));
    }

    #[test]
    fn local_alive_returns_self_region_only() {
        let v = view_with_members(&[
            member("g1", "cloud"),
            member("g2", "edge"),
        ]);
        let local = v.local_alive(&RegionId::new("cloud"));
        assert_eq!(local.len(), 1);
        assert_eq!(local[0].instance_id.as_str(), "g1");
    }

    #[test]
    fn foreign_alive_per_region_caps_per_region() {
        let v = view_with_members(&[
            member("g1", "cloud"),
            member("g2", "edge"),
            member("g3", "edge"),
            member("g4", "edge"),
            member("g5", "device"),
            member("g6", "device"),
        ]);
        let per_region = v.foreign_alive_per_region(&RegionId::new("cloud"), 2);
        // cloud is self, should not appear
        assert!(!per_region.contains_key(&RegionId::new("cloud")));
        // edge has 3 members, should be capped at 2
        assert_eq!(per_region.get(&RegionId::new("edge")).unwrap().len(), 2);
        // device has 2 members, within cap
        assert_eq!(per_region.get(&RegionId::new("device")).unwrap().len(), 2);
    }

    #[test]
    fn known_regions_are_sorted_and_deduped() {
        let v = view_with_members(&[
            member("g1", "device"),
            member("g2", "cloud"),
            member("g3", "edge"),
            member("g4", "cloud"),
        ]);
        let regions = v.known_regions();
        assert_eq!(regions.len(), 3);
        // Sorted alphabetically
        assert_eq!(regions[0].as_str(), "cloud");
        assert_eq!(regions[1].as_str(), "device");
        assert_eq!(regions[2].as_str(), "edge");
    }

    #[test]
    fn foreign_regions_excludes_self() {
        let v = view_with_members(&[
            member("g1", "cloud"),
            member("g2", "edge"),
            member("g3", "device"),
        ]);
        let foreign = v.foreign_regions(&RegionId::new("cloud"));
        assert_eq!(foreign.len(), 2);
        assert!(!foreign.contains(&RegionId::new("cloud")));
    }

    #[test]
    fn region_counts_aggregates_correctly() {
        let v = view_with_members(&[
            member("g1", "cloud"),
            member("g2", "cloud"),
            member("g3", "edge"),
        ]);
        let counts = v.region_counts();
        assert_eq!(counts.get(&RegionId::new("cloud")), Some(&2));
        assert_eq!(counts.get(&RegionId::new("edge")), Some(&1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn dead_members_are_excluded_from_views() {
        let list = Arc::new(MemberList::new());
        list.upsert(member("g1", "cloud"));
        list.upsert(member("g2", "cloud"));
        list.mark_dead(&InstanceId::new("g2"));

        let v = RegionMemberView::new(list);
        let cloud = v.alive_in_region(&RegionId::new("cloud"));
        assert_eq!(cloud.len(), 1);
        assert_eq!(cloud[0].instance_id.as_str(), "g1");
    }

    #[test]
    fn empty_member_list_yields_empty_views() {
        let v = view_with_members(&[]);
        assert!(v.alive_in_region(&RegionId::new("cloud")).is_empty());
        assert!(v.foreign_alive(&RegionId::new("cloud")).is_empty());
        assert!(v.known_regions().is_empty());
        assert_eq!(v.alive_count(), 0);
    }
}
