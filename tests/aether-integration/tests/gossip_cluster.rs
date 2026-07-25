//! 真实 Gossip 成员管理集成测试。
//!
//! 该测试直接操作 `aether_cluster::member::MemberList`（与 Gossip 引擎共享的
//! 并发安全成员表），验证成员加入、状态转移与随机选取语义。所有断言基于
//! `DashMap` 内的真实状态演化，不引入任何 mock。

use aether_cluster::member::{ClusterMember, MemberList, MemberStatus};
use aether_cluster::messages::MetaDigest;
use aether_core::ids::{InstanceId, RegionId};

/// 构造一个测试用成员。
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

    // 添加 3 个 alive 成员
    list.upsert(make_member("g1", "r1"));
    list.upsert(make_member("g2", "r1"));
    list.upsert(make_member("g3", "r2"));

    assert_eq!(list.len(), 3, "成员总数应为 3");
    let alive = list.alive_members();
    assert_eq!(
        alive.len(),
        3,
        "alive_members 应返回 3 个，实际: {}",
        alive.len()
    );

    // 成员状态全部应为 Alive
    for m in &alive {
        assert_eq!(
            m.status,
            MemberStatus::Alive,
            "新加入成员状态应为 Alive"
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

    // 1) 把 g2 标记为 Suspect
    list.mark_suspect(&target);
    let m = list.get(&target).expect("g2 应存在");
    assert_eq!(
        m.status,
        MemberStatus::Suspect,
        "mark_suspect 后状态应为 Suspect"
    );

    // alive_members 应排除 Suspect
    let alive = list.alive_members();
    assert_eq!(
        alive.len(),
        2,
        "Suspect 后 alive_members 应为 2，实际: {}",
        alive.len()
    );
    assert!(
        !alive.iter().any(|m| m.instance_id.as_str() == "g2"),
        "g2 不应出现在 alive 列表中"
    );

    // 2) 同一成员继续标记为 Dead
    list.mark_dead(&target);
    let m = list.get(&target).expect("g2 仍应存在");
    assert_eq!(
        m.status,
        MemberStatus::Dead,
        "mark_dead 后状态应为 Dead"
    );

    // alive_members 仍应排除 Dead
    let alive = list.alive_members();
    assert_eq!(
        alive.len(),
        2,
        "Dead 后 alive_members 应保持为 2"
    );
    assert!(
        !alive.iter().any(|m| m.instance_id.as_str() == "g2"),
        "Dead 成员不应出现在 alive 列表中"
    );

    // 3) all_members 应包含全部 3 个成员（含 Dead）
    let all = list.all_members();
    assert_eq!(
        all.len(),
        3,
        "all_members 应包含全部 3 个成员（含 Dead），实际: {}",
        all.len()
    );
}

#[test]
fn member_list_random_members_returns_at_most_n_alive() {
    let list = MemberList::new();

    // 添加 3 个 alive 成员 + 1 个 Suspect + 1 个 Dead
    list.upsert(make_member("a1", "r1"));
    list.upsert(make_member("a2", "r1"));
    list.upsert(make_member("a3", "r2"));
    list.upsert(make_member("s1", "r1"));
    list.upsert(make_member("d1", "r2"));
    list.mark_suspect(&InstanceId::new("s1"));
    list.mark_dead(&InstanceId::new("d1"));

    // 请求 2 个 alive 成员：应返回恰好 2 个
    let picked = list.random_members(2);
    assert_eq!(
        picked.len(),
        2,
        "random_members(2) 应返回 2 个 alive 成员，实际: {}",
        picked.len()
    );
    for m in &picked {
        assert_eq!(
            m.status,
            MemberStatus::Alive,
            "随机返回的成员必须为 Alive"
        );
    }

    // 请求超过 alive 总数（3）：应只返回 3 个 alive 成员
    let picked_all = list.random_members(10);
    assert_eq!(
        picked_all.len(),
        3,
        "请求超过 alive 总数时只返回 alive 总数 (3)"
    );

    // 请求 0：返回空
    let picked_zero = list.random_members(0);
    assert!(picked_zero.is_empty(), "random_members(0) 应返回空");

    // 多次随机选取应覆盖所有 alive 成员（统计意义上）
    let mut seen = std::collections::HashSet::new();
    for _ in 0..50 {
        for m in list.random_members(3) {
            seen.insert(m.instance_id.as_str().to_string());
        }
    }
    assert_eq!(
        seen.len(),
        3,
        "多次随机选取应覆盖所有 3 个 alive 成员，实际覆盖: {:?}",
        seen
    );
}

#[test]
fn member_list_update_pong_revives_suspect_member() {
    // 该用例验证：Suspect 成员收到新的 Pong 后应回到 Alive 状态。
    let list = MemberList::new();
    list.upsert(make_member("g1", "r1"));
    list.mark_suspect(&InstanceId::new("g1"));

    let digest = MetaDigest {
        kv_version: 42,
        ..MetaDigest::default()
    };
    list.update_pong(&InstanceId::new("g1"), digest.clone());

    let m = list.get(&InstanceId::new("g1")).expect("g1 应存在");
    assert_eq!(m.status, MemberStatus::Alive, "update_pong 后应回到 Alive");
    assert_eq!(
        m.meta_digest.kv_version, 42,
        "update_pong 应同步更新 meta_digest"
    );
    assert!(list.alive_members().iter().any(|m| m.instance_id.as_str() == "g1"));
}

#[test]
fn member_list_remove_drops_member() {
    let list = MemberList::new();
    list.upsert(make_member("g1", "r1"));
    list.upsert(make_member("g2", "r1"));

    let removed = list.remove(&InstanceId::new("g1"));
    assert!(removed.is_some(), "remove 应返回被移除的成员");
    assert_eq!(list.len(), 1, "remove 后成员数应为 1");
    assert!(
        list.get(&InstanceId::new("g1")).is_none(),
        "已移除的成员不应再可查询"
    );
}
