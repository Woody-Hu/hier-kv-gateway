//! Model-aware routing strategy.
//!
//! Acts as a hard filter: backends with non-matching model names, insufficient
//! context length, or no support for tool calls all receive a score of 0 and
//! should be removed from the candidate set by the hybrid strategy.

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// Model-aware routing strategy.
#[derive(Default)]
pub struct ModelAwareStrategy;

#[async_trait]
impl RoutingStrategy for ModelAwareStrategy {
    fn name(&self) -> &'static str {
        "model_aware"
    }

    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        let model_name = ctx.model_name.as_deref().unwrap_or("");
        let token_count = ctx.token_ids.len() as u32;
        let requires_tool = ctx.requires_tool_calling;

        let mut out = Vec::with_capacity(candidates.len());
        for cand in candidates {
            let match_score = meta.model_match_score(cand, model_name);
            if match_score <= 0.0 {
                // Model mismatch: exclude directly
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 0.0,
                    raw_cost: f64::MAX,
                    meta_version: 0,
                });
                continue;
            }

            // Further check context length and tool-calling support
            let instances = meta.model_get_instances(cand);
            let mut passes_ctx = false;
            let mut passes_tool = false;
            for inst in &instances {
                // Only check capabilities for instances matching this model name
                let name_match = inst.model_name == model_name
                    || (!inst.model_architecture.is_empty()
                        && inst.model_architecture == model_name);
                if !name_match {
                    continue;
                }
                let ctx_ok = token_count == 0 || inst.max_context_len >= token_count;
                let tool_ok = !requires_tool || inst.supports_tool_calling;
                if ctx_ok {
                    passes_ctx = true;
                }
                if tool_ok {
                    passes_tool = true;
                }
            }

            if !passes_ctx || !passes_tool {
                // Does not satisfy context or tool-calling constraints: exclude
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 0.0,
                    raw_cost: f64::MAX,
                    meta_version: 0,
                });
                continue;
            }

            // Passes the filter: cost is 1 - match_score; the more exact the match, the lower the cost
            let cost = 1.0 - match_score;
            let score = 1.0 / (1.0 + cost);
            out.push(ScoredBackend {
                backend_id: cand.clone(),
                score,
                raw_cost: cost,
                meta_version: 0,
            });
        }
        Ok(out)
    }

    fn is_available(&self, _meta: &MetadataStore) -> bool {
        // The model-aware strategy is always available
        true
    }

    fn weight(&self) -> f64 {
        // As a hard filter, it has the highest weight
        1.0
    }
}
