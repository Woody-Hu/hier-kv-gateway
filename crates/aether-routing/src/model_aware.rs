//! 模型感知路由策略。
//!
//! 作为硬性过滤器：模型名不匹配、上下文长度不足或不支持工具调用的后端
//! 一律得到 0 分，应被混合策略从候选集中剔除。

use async_trait::async_trait;

use aether_core::error::Result;
use aether_core::ids::BackendId;
use aether_core::request::{RoutingContext, ScoredBackend};
use aether_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// 模型感知路由策略。
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
                // 模型不匹配：直接排除
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 0.0,
                    raw_cost: f64::MAX,
                    meta_version: 0,
                });
                continue;
            }

            // 进一步检查上下文长度与工具调用支持
            let instances = meta.model_get_instances(cand);
            let mut passes_ctx = false;
            let mut passes_tool = false;
            for inst in &instances {
                // 仅对匹配该模型名的实例做能力检查
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
                // 不满足上下文或工具调用约束：排除
                out.push(ScoredBackend {
                    backend_id: cand.clone(),
                    score: 0.0,
                    raw_cost: f64::MAX,
                    meta_version: 0,
                });
                continue;
            }

            // 通过过滤：成本取 1 - 匹配分，匹配越精确成本越低
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
        // 模型感知策略始终可用
        true
    }

    fn weight(&self) -> f64 {
        // 作为硬性过滤器，权重最大
        1.0
    }
}
