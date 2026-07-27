//! Real Gossip member management integration test.
//!
//! This test directly manipulates `hier_kv_gateway_cluster::member::MemberList` (the
//! concurrency-safe member table shared with the Gossip engine) to verify member join,
//! state transition, and random selection semantics. All assertions are based on the
//! real state evolution of the `DashMap`; no mocks are introduced.

use hier_kv_gateway_cluster::member::{ClusterMember, MemberList, MemberStatus};
use hier_kv_gateway_cluster::messages::MetaDigest;
use hier_kv_gateway_core::ids::{InstanceId, RegionId};

/// Construct a test member.
fn make_member(name: &str, region: &str) -> ClusterMember {
    ClusterMember::new(
        InstanceId::new(name),
        RegionId::new(region),
        format!("127.0.0.1:{}", 7000 + name.len() as u16),
        MetaDigest::default(),
    )
}

#[test]
fn member_list_add_and_query_alive() {
    let list = MemberList::new();

    // Add 3 alive members
    list.upsert(make_member("g1", "r1"));
    list.upsert(make_member("g2", "r1"));
    list.upsert(make_member("g3", "r2"));

    assert_eq!(list.len(), 3, "total member count should be 3");
    let alive = list.alive_members();
    assert_eq!(
        alive.len(),
        3,
        "alive_members should return 3, actual: {}",
        alive.len()
    );

    // All member states should be Alive
    for m in &alive {
        assert_eq!(
            m.status,
            MemberStatus::Alive,
            "newly joined member state should be Alive"
        );
    }
}

#[test]
fn member_list_mark_suspect_then_dead_transitions() {
    let list = MemberList::new();
    list.upsert(make_member("g1", "r1"));
    list.upsert(make_member("g2", "r1"));
    list.upsert(make_member("g3", "r1"));

    let target = InstanceId::new("g2");

    // 1) Mark g2 as Suspect
    list.mark_suspect(&target);
    let m = list.get(&target).expect("g2 should exist");
    assert_eq!(
        m.status,
        MemberStatus::Suspect,
        "after mark_suspect, the state should be Suspect"
    );

    // alive_members should exclude Suspect
    let alive = list.alive_members();
    assert_eq!(
        alive.len(),
        2,
        "after Suspect, alive_members should be 2, actual: {}",
        alive.len()
    );
    assert!(
        !alive.iter().any(|m| m.instance_id.as_str() == "g2"),
        "g2 should not appear in the alive list"
    );

    // 2) Continue marking the same member as Dead
    list.mark_dead(&target);
    let m = list.get(&target).expect("g2 should still exist");
    assert_eq!(
        m.status,
        MemberStatus::Dead,
        "after mark_dead, the state should be Dead"
    );

    // alive_members should still exclude Dead
    let alive = list.alive_members();
    assert_eq!(
        alive.len(),
        2,
        "after Dead, alive_members should remain 2"
    );
    assert!(
        !alive.iter().any(|m| m.instance_id.as_str() == "g2"),
        "Dead members should not appear in the alive list"
    );

    // 3) all_members should contain all 3 members (including Dead)
    let all = list.all_members();
    assert_eq!(
        all.len(),
        3,
        "all_members should contain all 3 members (including Dead), actual: {}",
        all.len()
    );
}

#[test]
fn member_list_random_members_returns_at_most_n_alive() {
    let list = MemberList::new();

    // Add 3 alive members + 1 Suspect + 1 Dead
    list.upsert(make_member("a1", "r1"));
    list.upsert(make_member("a2", "r1"));
    list.upsert(make_member("a3", "r2"));
    list.upsert(make_member("s1", "r1"));
    list.upsert(make_member("d1", "r2"));
    list.mark_suspect(&InstanceId::new("s1"));
    list.mark_dead(&InstanceId::new("d1"));

    // Request 2 alive members: should return exactly 2
    let picked = list.random_members(2);
    assert_eq!(
        picked.len(),
        2,
        "random_members(2) should return 2 alive members, actual: {}",
        picked.len()
    );
    for m in &picked {
        assert_eq!(
            m.status,
            MemberStatus::Alive,
            "randomly returned members must be Alive"
        );
    }

    // Request more than the total alive count (3): should only return 3 alive members
    let picked_all = list.random_members(10);
    assert_eq!(
        picked_all.len(),
        3,
        "when requesting more than the alive count, only the alive count (3) is returned"
    );

    // Request 0: return empty
    let picked_zero = list.random_members(0);
    assert!(picked_zero.is_empty(), "random_members(0) should return empty");

    // Multiple random selections should cover all alive members (statistically)
    let mut seen = std::collections::HashSet::new();
    for _ in 0..50 {
        for m in list.random_members(3) {
            seen.insert(m.instance_id.as_str().to_string());
        }
    }
    assert_eq!(
        seen.len(),
        3,
        "multiple random selections should cover all 3 alive members, actual coverage: {:?}",
        seen
    );
}

#[test]
fn member_list_update_pong_revives_suspect_member() {
    // This case verifies: a Suspect member should return to Alive after receiving a new Pong.
    let list = MemberList::new();
    list.upsert(make_member("g1", "r1"));
    list.mark_suspect(&InstanceId::new("g1"));

    let digest = MetaDigest {
        kv_version: 42,
        ..MetaDigest::default()
    };
    list.update_pong(&InstanceId::new("g1"), digest.clone());

    let m = list.get(&InstanceId::new("g1")).expect("g1 should exist");
    assert_eq!(m.status, MemberStatus::Alive, "after update_pong, should return to Alive");
    assert_eq!(
        m.meta_digest.kv_version, 42,
        "update_pong should also sync the meta_digest"
    );
    assert!(list.alive_members().iter().any(|m| m.instance_id.as_str() == "g1"));
}

#[test]
fn member_list_remove_drops_member() {
    let list = MemberList::new();
    list.upsert(make_member("g1", "r1"));
    list.upsert(make_member("g2", "r1"));

    let removed = list.remove(&InstanceId::new("g1"));
    assert!(removed.is_some(), "remove should return the removed member");
    assert_eq!(list.len(), 1, "after remove, member count should be 1");
    assert!(
        list.get(&InstanceId::new("g1")).is_none(),
        "the removed member should no longer be queryable"
    );
}
