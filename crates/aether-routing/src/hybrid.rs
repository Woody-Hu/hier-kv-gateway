//! 混合智能路由策略（默认策略）。
//!
//! 融合 KV 感知、负载感知与拓扑感知三类子策略的评分：
//!
//! 1. 先用模型感知策略对候选集做硬性过滤（剔除 `score == 0`）。
//! 2. 对每个可用子策略调用 `evaluate`，得到各自的 `ScoredBackend` 列表。
//! 3. 动态调整权重：KV 不可用时权重归零；任一候选指标过期 10s 以上时降低负载权重。
//! 4. 对每个候选计算 `hybrid_score = Σ(weight_s * normalize(score_s))`，
//!    其中 `normalize` 把代价归一化到 `[0, 1]` 区间，0 表示该策略下成本最低。
//! 5. `temperature > 0` 时由调用方按 softmax 采样，否则贪心选最高分。
//!
//! 输出按 `hybrid_score` 降序排列的 `ScoredBackend` 列表，`score` 字段即混合分。

use std::collections::HashMap;

use async_trait::async_trait;

use aether_core::config::StrategyWeights;
use aether_core::error::{AetherError, Result};
use aether_core::ids::BackendId;
use aether_core::request::{RoutingContext, ScoredBackend};
use aether_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// 负载指标过期阈值：超过该时长则对负载权重做折扣。
const STALE_LOAD_THRESHOLD_SECS: u64 = 10;

/// 混合智能路由策略。
pub struct HybridStrategy {
    /// KV 感知子策略。
    pub kv: Box<dyn RoutingStrategy>,
    /// 模型感知子策略（硬性过滤器）。
    pub model: Box<dyn RoutingStrategy>,
    /// 负载感知子策略。
    pub load: Box<dyn RoutingStrategy>,
    /// 拓扑感知子策略。
    pub topology: Box<dyn RoutingStrategy>,
    /// 三类子策略的静态权重。
    pub weights: StrategyWeights,
    /// 路由温度参数：> 0 时调用方按 softmax 采样，== 0 时贪心选最高分。
    pub temperature: f64,
}

impl HybridStrategy {
    /// 用给定的子策略构造混合策略。
    pub fn new(
        kv: Box<dyn RoutingStrategy>,
        model: Box<dyn RoutingStrategy>,
        load: Box<dyn RoutingStrategy>,
        topology: Box<dyn RoutingStrategy>,
        weights: StrategyWeights,
        temperature: f64,
    ) -> Self {
        Self {
            kv,
            model,
            load,
            topology,
            weights,
            temperature,
        }
    }

    /// 计算给定候选集在某个子策略下的代价归一化分（`1 - normalize(cost)`）。
    ///
    /// 即：成本最低的候选得 1.0，成本最高的得 0.0；所有候选成本相同时返回 1.0。
    fn normalize_costs(scores: &[ScoredBackend]) -> HashMap<BackendId, f64> {
        let mut min_cost = f64::INFINITY;
        let mut max_cost = f64::NEG_INFINITY;
        for s in scores {
            if !s.raw_cost.is_finite() {
                // 排除用 f64::MAX 标记的候选：跳过统计但仍参与输出（取 0）
                continue;
            }
            if s.raw_cost < min_cost {
                min_cost = s.raw_cost;
            }
            if s.raw_cost > max_cost {
                max_cost = s.raw_cost;
            }
        }
        let mut out = HashMap::new();
        let span = max_cost - min_cost;
        for s in scores {
            if !s.raw_cost.is_finite() {
                // 不满足约束的候选：归一化分为 0
                out.insert(s.backend_id.clone(), 0.0);
                continue;
            }
            let normalized = if span > 0.0 {
                (s.raw_cost - min_cost) / span
            } else {
                0.0
            };
            // 转换为"越大越好"：成本越低分越高
            out.insert(s.backend_id.clone(), 1.0 - normalized);
        }
        out
    }
}

#[async_trait]
impl RoutingStrategy for HybridStrategy {
    fn name(&self) -> &'static str {
        "hybrid"
    }

    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 模型感知做硬性过滤
        let model_scores = self.model.evaluate(ctx, candidates, meta).await?;
        let filtered: Vec<BackendId> = model_scores
            .iter()
            .filter(|s| s.score > 0.0)
            .map(|s| s.backend_id.clone())
            .collect();
        if filtered.is_empty() {
            return Err(AetherError::RoutingFailed(
                "没有任何候选后端通过模型感知过滤".to_string(),
            ));
        }

        // 2. 动态权重调整
        let kv_available = self.kv.is_available(meta);
        let mut weight_kv = if kv_available { self.weights.kv } else { 0.0 };

        // 任一候选的负载指标过期阈值以上则对负载权重打折扣
        let mut load_stale = false;
        for cand in &filtered {
            if let Some(freshness) = meta.load_freshness(cand) {
                if freshness.as_secs() > STALE_LOAD_THRESHOLD_SECS {
                    load_stale = true;
                    break;
                }
            }
        }
        let mut weight_load = if load_stale {
            self.weights.load * 0.3
        } else {
            self.weights.load
        };
        let weight_topo = self.weights.topology;

        // 归一化权重，使总和为 1.0
        let total = weight_kv + weight_load + weight_topo;
        if total > 0.0 {
            weight_kv /= total;
            weight_load /= total;
        } else {
            // 全部为 0：兜底为均匀权重
            weight_kv = 1.0 / 3.0;
            weight_load = 1.0 / 3.0;
        }
        let weight_topo = if total > 0.0 {
            weight_topo / total
        } else {
            1.0 / 3.0
        };

        // 3. 各子策略对过滤后的候选集评分
        let kv_scores = if weight_kv > 0.0 {
            self.kv.evaluate(ctx, &filtered, meta).await?
        } else {
            Vec::new()
        };
        let load_scores = if weight_load > 0.0 {
            self.load.evaluate(ctx, &filtered, meta).await?
        } else {
            Vec::new()
        };
        let topo_scores = if weight_topo > 0.0 {
            self.topology.evaluate(ctx, &filtered, meta).await?
        } else {
            Vec::new()
        };

        // 4. 归一化每个子策略的代价到 [0, 1]
        let kv_norm = Self::normalize_costs(&kv_scores);
        let load_norm = Self::normalize_costs(&load_scores);
        let topo_norm = Self::normalize_costs(&topo_scores);

        // 5. 加权求和得到 hybrid_score
        let mut hybrid: Vec<ScoredBackend> = Vec::with_capacity(filtered.len());
        for cand in &filtered {
            let kv_s = kv_norm.get(cand).copied().unwrap_or(0.0);
            let load_s = load_norm.get(cand).copied().unwrap_or(0.0);
            let topo_s = topo_norm.get(cand).copied().unwrap_or(0.0);

            let hybrid_score = weight_kv * kv_s + weight_load * load_s + weight_topo * topo_s;

            // raw_cost 取负的 hybrid_score，便于在 SortBackend 上游保持
            // "raw_cost 越低越好" 的语义一致性（hybrid_score 越高越好）。
            hybrid.push(ScoredBackend {
                backend_id: cand.clone(),
                score: hybrid_score,
                raw_cost: -hybrid_score,
                meta_version: 0,
            });
        }

        // 6. 按 hybrid_score 降序排序
        hybrid.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(hybrid)
    }

    fn is_available(&self, meta: &MetadataStore) -> bool {
        // 模型感知是硬性过滤器，其可用性决定整体可用性
        self.model.is_available(meta)
    }

    fn weight(&self) -> f64 {
        // 作为顶层组合策略，权重为 1.0
        1.0
    }
}
