//! KV 感知路由策略。
//!
//! 参考 Dynamo 的成本函数：候选后端对本请求已分词序列的 KV 命中越多，则 prefill
//! 阶段需要补齐的块越少，整体成本越低。本地命中由 RadixTree 精确给出，跨 Region
//! 命中由 Cuckoo Filter 消费者近似给出，二者相加作为总命中重叠。

use async_trait::async_trait;

use aether_core::error::Result;
use aether_core::ids::BackendId;
use aether_core::request::{RoutingContext, ScoredBackend};
use aether_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// KV 感知路由策略。
pub struct KvAwareStrategy {
    /// 命中重叠时给予候选后端的额外评分信用（用于在分数上做正向偏置）。
    pub overlap_score_credit: f64,
    /// prefill 阶段负载缩放系数，用于调节 prefill_blocks 在成本中的权重。
    pub prefill_load_scale: f64,
    /// CKF 假阳性惩罚因子，用于在远程命中分量上施加折扣。
    pub ckf_false_positive_penalty: f64,
}

impl Default for KvAwareStrategy {
    fn default() -> Self {
        Self {
            overlap_score_credit: 0.0,
            prefill_load_scale: 1.0,
            ckf_false_positive_penalty: 0.0,
        }
    }
}

#[async_trait]
impl RoutingStrategy for KvAwareStrategy {
    fn name(&self) -> &'static str {
        "kv_aware"
    }

    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        let hashes: &[u64] = ctx.block_hashes.as_slice();
        let hash_count = hashes.len() as i64;

        let mut out = Vec::with_capacity(candidates.len());
        for cand in candidates {
            // 本地精确重叠（RadixTree）
            let local_overlap = meta.kv_find_local_overlap(hashes, cand.clone()).await;
            // 跨 Region 近似重叠（Cuckoo Filter 消费者）
            let remote_overlap = meta.kv_find_global_overlap(hashes, &cand.region);

            // 对远程重叠施加假阳性惩罚：按比例折减
            let effective_remote = if self.ckf_false_positive_penalty > 0.0 {
                remote_overlap as f64 * (1.0 - self.ckf_false_positive_penalty.clamp(0.0, 1.0))
            } else {
                remote_overlap as f64
            };

            let total_overlap = local_overlap as f64 + effective_remote;

            // 需要补齐的 prefill 块数：不能小于 0
            let prefill_blocks = (hash_count as f64 - total_overlap).max(0.0);

            // decode 阶段当前活跃块数（无指标时视为 0）
            let decode_blocks = meta
                .load_get_metrics(cand)
                .map(|m| m.active_decode_blocks as f64)
                .unwrap_or(0.0);

            // 综合 prefill + decode 成本
            let cost = self.prefill_load_scale * prefill_blocks + decode_blocks;

            // 评分：成本越低分数越高；命中重叠通过信用做正向偏置
            let score = 1.0 / (1.0 + cost) + self.overlap_score_credit * total_overlap;

            out.push(ScoredBackend {
                backend_id: cand.clone(),
                score,
                raw_cost: cost,
                meta_version: 0,
            });
        }
        Ok(out)
    }

    fn is_available(&self, meta: &MetadataStore) -> bool {
        // KV 索引可信度 > 0 才启用
        meta.kv_confidence() > 0.0
    }

    fn weight(&self) -> f64 {
        0.35
    }
}
