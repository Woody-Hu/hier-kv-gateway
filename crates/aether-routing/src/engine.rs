//! 路由引擎：整合会话亲和与混合策略，产出最终路由决策。
//!
//! [`RoutingEngine`] 持有一个 [`HybridStrategy`] 与若干运行时参数，对外暴露
//! [`RoutingEngine::route`]：先做会话亲和检查（命中且后端仍在线则复用），
//! 否则用混合策略评估候选集，再用 softmax/贪心从评分中选出最终后端，
//! 并把会话亲和写回元数据存储。

use std::time::Duration;

use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

use aether_core::error::{AetherError, Result};
use aether_core::ids::{BackendId, RegionId};
use aether_core::request::{RoutingContext, ScoredBackend};
use aether_metadata::store::MetadataStore;

use crate::hybrid::HybridStrategy;
use crate::strategy::RoutingStrategy;

/// 路由决策结果。
#[derive(Clone, Debug)]
pub struct RouteDecision {
    /// 最终选中的后端标识。
    pub backend: BackendId,
    /// 触发决策的策略名（如 `hybrid` 或 `session_affinity`）。
    pub strategy: String,
    /// 选中后端与本请求的 KV 重叠长度。
    pub kv_overlap: u32,
    /// 各子策略对选中后端的子分数，便于追踪与日志。
    pub scores: Vec<(String, f64)>,
}

/// 路由引擎。
pub struct RoutingEngine {
    /// 内嵌的混合策略。
    pub hybrid: HybridStrategy,
    /// 会话亲和 TTL。
    pub session_affinity_ttl: Duration,
    /// 最大重试次数。
    pub max_retries: u32,
    /// 本网关所在区域。
    pub self_region: RegionId,
}

impl RoutingEngine {
    /// 创建一个新路由引擎。
    pub fn new(
        hybrid: HybridStrategy,
        session_affinity_ttl: Duration,
        max_retries: u32,
        self_region: RegionId,
    ) -> Self {
        Self {
            hybrid,
            session_affinity_ttl,
            max_retries,
            self_region,
        }
    }

    /// 执行路由决策。
    ///
    /// 流程：
    /// 1. 若请求带 `session_id`，先查亲和记录；命中且后端仍在线则复用。
    /// 2. 否则收集所有候选 backend，调用混合策略评分。
    /// 3. 用 softmax/贪心从评分中选出最终后端。
    /// 4. 把会话亲和写回元数据存储。
    pub async fn route(
        &self,
        ctx: &RoutingContext,
        meta: &MetadataStore,
    ) -> Result<RouteDecision> {
        // 1. 会话亲和检查
        if let Some(session_id) = ctx.session_id.as_ref() {
            if let Some(affinity) = meta.session_get(session_id) {
                // 校验后端是否仍在线
                if meta.backend_get(&affinity.backend).is_some() {
                    // 计算当前 KV 重叠
                    let kv_overlap = meta
                        .kv_find_local_overlap(
                            ctx.block_hashes.as_slice(),
                            affinity.backend.clone(),
                        )
                        .await;
                    // 更新会话亲和的时间戳
                    meta.session_set(
                        session_id.clone(),
                        affinity.backend.clone(),
                        kv_overlap,
                    );
                    return Ok(RouteDecision {
                        backend: affinity.backend,
                        strategy: "session_affinity".to_string(),
                        kv_overlap,
                        scores: Vec::new(),
                    });
                }
            }
        }

        // 2. 候选集：优先按模型名预筛，否则取全部
        let candidates: Vec<BackendId> = match ctx.model_name.as_deref() {
            Some(name) if !name.is_empty() => {
                let by_model = meta.model_find_backends(name);
                if by_model.is_empty() {
                    meta.backends_all()
                        .into_iter()
                        .map(|b| b.id)
                        .collect()
                } else {
                    by_model
                }
            }
            _ => meta
                .backends_all()
                .into_iter()
                .map(|b| b.id)
                .collect(),
        };

        if candidates.is_empty() {
            return Err(AetherError::BackendUnavailable);
        }

        // 3. 混合策略评分
        let scored = self.hybrid.evaluate(ctx, &candidates, meta).await?;
        if scored.is_empty() {
            return Err(AetherError::RoutingFailed(
                "混合策略未产出任何候选评分".to_string(),
            ));
        }

        // 4. 选中后端
        let selected = select_with_temperature(&scored, self.hybrid.temperature)
            .ok_or_else(|| AetherError::RoutingFailed("无法从评分中选出后端".to_string()))?;

        // 5. 查询选中后端与本请求的 KV 重叠
        let kv_overlap = meta
            .kv_find_local_overlap(
                ctx.block_hashes.as_slice(),
                selected.backend_id.clone(),
            )
            .await;

        // 6. 写回会话亲和
        if let Some(session_id) = ctx.session_id.as_ref() {
            meta.session_set(
                session_id.clone(),
                selected.backend_id.clone(),
                kv_overlap,
            );
        }

        // 7. 收集各子策略对选中后端的子分数（仅用于追踪）
        let mut scores: Vec<(String, f64)> = Vec::new();
        let kv_scores = self.hybrid.kv.evaluate(ctx, &candidates, meta).await?;
        let load_scores = self.hybrid.load.evaluate(ctx, &candidates, meta).await?;
        let topo_scores = self
            .hybrid
            .topology
            .evaluate(ctx, &candidates, meta)
            .await?;
        for s in &kv_scores {
            if s.backend_id == selected.backend_id {
                scores.push((self.hybrid.kv.name().to_string(), s.score));
                break;
            }
        }
        for s in &load_scores {
            if s.backend_id == selected.backend_id {
                scores.push((self.hybrid.load.name().to_string(), s.score));
                break;
            }
        }
        for s in &topo_scores {
            if s.backend_id == selected.backend_id {
                scores.push((self.hybrid.topology.name().to_string(), s.score));
                break;
            }
        }
        scores.push((self.hybrid.name().to_string(), selected.score));

        Ok(RouteDecision {
            backend: selected.backend_id.clone(),
            strategy: self.hybrid.name().to_string(),
            kv_overlap,
            scores,
        })
    }
}

/// 根据 `temperature` 从评分列表中选出一个后端。
///
/// - `temperature <= 0`：贪心选最高分。
/// - `temperature > 0`：以 `softmax(score / temperature)` 作为概率分布进行采样；
///   评分列表为空时返回 `None`。
pub fn select_with_temperature(scores: &[ScoredBackend], temperature: f64) -> Option<ScoredBackend> {
    if scores.is_empty() {
        return None;
    }
    if temperature <= 0.0 {
        // 贪心：返回最高分
        return scores
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
    }
    // softmax 采样
    let max_score = scores
        .iter()
        .map(|s| s.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores
        .iter()
        .map(|s| ((s.score - max_score) / temperature).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        // 数值异常退化为贪心
        return scores
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
    }
    // 用 thread-local RNG 做采样
    let mut rng = rand::rng();
    let weights: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    let dist = WeightedIndex::new(&weights).ok()?;
    let idx = dist.sample(&mut rng);
    Some(scores[idx].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_core::ids::BackendId;

    fn scored(backend: &str, score: f64) -> ScoredBackend {
        ScoredBackend {
            backend_id: BackendId::new("r1", backend),
            score,
            raw_cost: -score,
            meta_version: 0,
        }
    }

    #[test]
    fn greedy_picks_highest() {
        let s = vec![scored("a", 0.1), scored("b", 0.9), scored("c", 0.5)];
        let chosen = select_with_temperature(&s, 0.0).unwrap();
        assert_eq!(chosen.backend_id.instance.as_str(), "b");
    }

    #[test]
    fn softmax_returns_some_on_non_empty() {
        let s = vec![scored("a", 0.1), scored("b", 0.9)];
        let chosen = select_with_temperature(&s, 0.5).unwrap();
        // 不强求具体被选中的，但应在集合内
        assert!(s.iter().any(|x| x.backend_id == chosen.backend_id));
    }

    #[test]
    fn empty_returns_none() {
        let s: Vec<ScoredBackend> = Vec::new();
        assert!(select_with_temperature(&s, 0.0).is_none());
        assert!(select_with_temperature(&s, 1.0).is_none());
    }
}
