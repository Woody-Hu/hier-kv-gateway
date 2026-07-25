//! 真实 CKF Producer / Consumer 跨 Region KV 投影集成测试。
//!
//! 该测试驱动 `CkfProducer` 应用多个真实 KV 事件，生成全量 `CkfSnapshot`
//! 并通过 `CkfConsumer::install_snapshot` 安装；随后验证 `estimate_overlap`
//! 对已存储 hash 的命中、未存储 hash 的中断行为，以及通过 `CkfDelta`
//! 增量发布新 hash 后 consumer 立即可查的真实数据流。

use aether_core::ids::{BackendId, RegionId, WorkerWithRank};
use aether_core::kv_event::KvCacheEvent;
use aether_metadata::ckf_consumer::CkfConsumer;
use aether_metadata::ckf_producer::CkfProducer;

/// 构造一个 `Stored` 事件。
fn stored(hashes: Vec<u64>) -> KvCacheEvent {
    KvCacheEvent::Stored {
        worker: WorkerWithRank::from_worker_id(1),
        block_hashes: hashes,
        parent_hash: None,
        num_block_tokens: Vec::new(),
    }
}

/// 构造测试用 backend 标识。
fn backend(name: &str) -> BackendId {
    BackendId::new("r1", name)
}

#[tokio::test]
async fn ckf_snapshot_install_then_estimate_overlap() {
    // 1) Producer 端：应用多个 KV 事件以产生 fingerprints
    let mut producer = CkfProducer::with_seed(42);
    let b1 = backend("worker-1");
    let b2 = backend("worker-2");

    // b1 存入 5 个块哈希
    producer.apply_event(&stored(vec![101, 102, 103, 104, 105]), &b1);
    // b2 与 b1 共享 101，并新增 106；101 应去重（同 backend 不重复）
    producer.apply_event(&stored(vec![101, 106]), &b2);

    // 已跟踪 distinct hash 数量应为 6：{101, 102, 103, 104, 105, 106}
    assert_eq!(
        producer.tracked_hashes(),
        6,
        "应跟踪 6 个 distinct hash"
    );
    // fingerprint 数量应为 6（每个 distinct hash 插一次）
    assert_eq!(
        producer.num_items(),
        6,
        "应插入 6 个 fingerprint"
    );

    // 2) 生成全量快照
    let snapshot = producer.snapshot();
    assert_eq!(snapshot.sequence, 1, "首次 snapshot 序列号应为 1");
    assert_eq!(
        snapshot.buckets.len(),
        aether_metadata::cuckoo_filter::BUCKETS_PER_LANE,
        "快照应包含完整 lane 的 bucket 数"
    );

    // 3) Consumer 端：分配 lane + 安装快照
    let consumer = CkfConsumer::new();
    let region = RegionId::new("r1");
    consumer.assign_lane(0, region.clone());
    consumer.install_snapshot(0, &snapshot);
    assert!(
        consumer.is_lane_active(0),
        "安装快照后 lane 0 应处于 Active 状态"
    );

    // 4) estimate_overlap 对已存储的 hash 序列应能完整命中
    let overlap_known = consumer.estimate_overlap(&[101, 102, 103, 104, 105, 106], &region);
    assert_eq!(
        overlap_known, 6,
        "已存储的 6 个 hash 应全部命中，实际: {}",
        overlap_known
    );

    // 5) 未存储的 hash 应在中断处停止；hash 999 不存在，第 1 个就中断
    let overlap_unknown = consumer.estimate_overlap(&[999], &region);
    assert_eq!(
        overlap_unknown, 0,
        "未存储的 hash 不应命中，实际: {}",
        overlap_unknown
    );

    // 6) 已存储 + 未存储的混合序列应在前缀处中断
    let overlap_mixed = consumer.estimate_overlap(&[101, 102, 999], &region);
    assert_eq!(
        overlap_mixed, 2,
        "前缀 [101, 102] 命中后，第 3 个 hash 999 中断 → 2"
    );

    // 7) 未分配 lane 的 Region 查询返回 0
    let other_region = RegionId::new("other-region");
    let overlap_no_lane = consumer.estimate_overlap(&[101], &other_region);
    assert_eq!(
        overlap_no_lane, 0,
        "未分配 lane 的 Region 查询应返回 0"
    );
}

#[tokio::test]
async fn ckf_delta_propagation_makes_new_hash_queryable() {
    // 该用例验证 delta 增量发布：先安装 snapshot，再通过 delta 同步新增 hash。
    let mut producer = CkfProducer::with_seed(7);
    let b = backend("worker-1");

    // 初始：存入 [1, 2, 3] 并发布 snapshot
    producer.apply_event(&stored(vec![1, 2, 3]), &b);
    let snapshot = producer.snapshot();

    let consumer = CkfConsumer::new();
    let region = RegionId::new("r2");
    consumer.assign_lane(2, region.clone());
    consumer.install_snapshot(2, &snapshot);

    // 验证初始状态下 [1, 2, 3] 可查，[4] 不可查
    assert_eq!(
        consumer.estimate_overlap(&[1, 2, 3], &region),
        3,
        "初始 snapshot 应使 [1, 2, 3] 全部命中"
    );
    assert_eq!(
        consumer.estimate_overlap(&[4], &region),
        0,
        "hash 4 尚未发布，不应命中"
    );

    // 应用新事件：存入 [4]，然后生成 delta
    producer.apply_event(&stored(vec![4]), &b);
    let delta = producer
        .delta()
        .expect("应用新事件后应能生成非空 delta");
    assert_eq!(
        delta.prev_sequence, 1,
        "delta 的 prev_sequence 应为 snapshot 的 sequence=1"
    );
    assert_eq!(delta.sequence, 2, "delta 的 sequence 应递增为 2");
    assert!(
        !delta.buckets.is_empty(),
        "delta 应至少包含一个变动的 bucket"
    );

    // consumer 应用 delta
    consumer.apply_delta(2, &delta);

    // 验证新 hash 4 现在可查
    let overlap_after_delta = consumer.estimate_overlap(&[1, 2, 3, 4], &region);
    assert_eq!(
        overlap_after_delta, 4,
        "应用 delta 后，[1, 2, 3, 4] 应全部命中，实际: {}",
        overlap_after_delta
    );
}

#[tokio::test]
async fn ckf_false_positive_rate_is_low_for_unstored_hashes() {
    // 该用例验证 CKF 对未存储 hash 的假阳性率保持在低水平。
    // 我们插入少量真实 hash，再随机探测大量未存储 hash，假阳性数应接近 0。
    let mut producer = CkfProducer::with_seed(123);
    let b = backend("worker-1");

    // 插入 50 个真实 hash
    let stored_hashes: Vec<u64> = (0..50u64).map(|i| 10_000 + i * 7).collect();
    producer.apply_event(&stored(stored_hashes.clone()), &b);
    let snapshot = producer.snapshot();

    let consumer = CkfConsumer::new();
    let region = RegionId::new("r3");
    consumer.assign_lane(5, region.clone());
    consumer.install_snapshot(5, &snapshot);

    // 真实 hash 全部命中
    let overlap_real = consumer.estimate_overlap(&stored_hashes, &region);
    assert_eq!(
        overlap_real, 50,
        "50 个真实 hash 应全部命中，实际: {}",
        overlap_real
    );

    // 探测 200 个未存储 hash，假阳性总数应远小于 1%
    // 注意：CKF 的理论 FPR 约 8.5e-5，200 个 hash 的期望假阳性数 << 1。
    // 由于 estimate_overlap 在首个未命中处中断，若假阳性恰好出现在第 1 位，
    // overlap 会是 1，否则为 0。
    let mut false_positive_count = 0usize;
    for i in 0..200u64 {
        let probe = 1_000_000 + i * 13; // 与 stored_hashes 不重叠
        let overlap = consumer.estimate_overlap(&[probe], &region);
        if overlap > 0 {
            false_positive_count += 1;
        }
    }
    assert!(
        false_positive_count <= 2,
        "假阳性数应 <= 2，实际: {}",
        false_positive_count
    );
}
