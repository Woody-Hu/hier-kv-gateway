//! KV-aware routing strategy.
//!
//! The more KV hits a candidate backend has on the request's already-tokenized
//! sequence, the fewer blocks need to be filled in during the prefill phase, and
//! the lower the overall cost. Local hits are given exactly by the RadixTree, and
//! cross-Region hits are given approximately by the Cuckoo Filter consumer; the two
//! are summed as the total hit overlap.

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// KV-aware routing strategy.
pub struct KvAwareStrategy {
    /// Extra score credit given to a candidate backend on hit overlap (used to apply a positive bias on the score).
    pub overlap_score_credit: f64,
    /// Load scaling factor for the prefill phase, used to adjust the weight of prefill_blocks in the cost.
    pub prefill_load_scale: f64,
    /// CKF false-positive penalty factor, used to discount the remote hit component.
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
            // Local exact overlap (RadixTree)
            let local_overlap = meta.kv_find_local_overlap(hashes, cand.clone()).await;
            // Cross-Region approximate overlap (Cuckoo Filter consumer)
            let remote_overlap = meta.kv_find_global_overlap(hashes, &cand.region);

            // Apply a false-positive penalty to the remote overlap: discount proportionally
            let effective_remote = if self.ckf_false_positive_penalty > 0.0 {
                remote_overlap as f64 * (1.0 - self.ckf_false_positive_penalty.clamp(0.0, 1.0))
            } else {
                remote_overlap as f64
            };

            let total_overlap = local_overlap as f64 + effective_remote;

            // Number of prefill blocks that need to be filled in: cannot be less than 0
            let prefill_blocks = (hash_count as f64 - total_overlap).max(0.0);

            // Number of active blocks in the decode phase (treated as 0 when no metrics are available)
            let decode_blocks = meta
                .load_get_metrics(cand)
                .map(|m| m.active_decode_blocks as f64)
                .unwrap_or(0.0);

            // Combined prefill + decode cost
            let cost = self.prefill_load_scale * prefill_blocks + decode_blocks;

            // Score: the lower the cost, the higher the score; hit overlap applies a positive bias via the credit
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
        // Enabled only when KV index confidence > 0
        meta.kv_confidence() > 0.0
    }

    fn weight(&self) -> f64 {
        0.35
    }
}
