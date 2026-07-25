//! 模型注册表：维护每个 backend 当前提供的模型列表，并支持按模型名匹配。
//!
//! 匹配评分规则：
//! - 精确匹配模型名 → 1.0
//! - 同 architecture 不同名 → 0.7
//! - 无匹配 → 0.0
//!
//! 当一个 backend 提供多个模型时，取最高分。

use dashmap::DashMap;
use aether_core::backend::ModelInstance;
use aether_core::ids::BackendId;

/// 模型注册表。
#[derive(Default)]
pub struct ModelRegistry {
    backends: DashMap<BackendId, Vec<ModelInstance>>,
}

impl ModelRegistry {
    /// 创建一个空的注册表。
    pub fn new() -> Self {
        Self {
            backends: DashMap::new(),
        }
    }

    /// 注册 backend 及其提供的模型列表（覆盖原有）。
    pub fn register(&self, backend: BackendId, models: Vec<ModelInstance>) {
        self.backends.insert(backend, models);
    }

    /// 注销 backend。
    pub fn unregister(&self, backend: &BackendId) {
        self.backends.remove(backend);
    }

    /// 计算 backend 对目标模型名的匹配分数（取最高分）。
    pub fn model_match_score(&self, backend: &BackendId, model_name: &str) -> f64 {
        let Some(models) = self.backends.get(backend) else {
            return 0.0;
        };
        let mut best = 0.0f64;
        for m in models.value().iter() {
            let score = match_score(m, model_name);
            if score > best {
                best = score;
            }
        }
        best
    }

    /// 查询所有能服务该模型名的 backend（score > 0）。
    pub fn model_find_backends(&self, model_name: &str) -> Vec<BackendId> {
        self.backends
            .iter()
            .filter_map(|entry| {
                let score = entry
                    .value()
                    .iter()
                    .map(|m| match_score(m, model_name))
                    .fold(0.0f64, f64::max);
                if score > 0.0 {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// 获取 backend 的模型实例列表（用于 routing 上下文）。
    pub fn get_instances(&self, backend: &BackendId) -> Vec<ModelInstance> {
        self.backends
            .get(backend)
            .map(|r| r.value().clone())
            .unwrap_or_default()
    }

    /// 当前注册的 backend 数量。
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

/// 单个模型实例的匹配分数。
///
/// - 精确匹配 `model.model_name == model_name` → 1.0
/// - 同 architecture（且非精确匹配）→ 0.7
/// - 否则 → 0.0
fn match_score(m: &ModelInstance, model_name: &str) -> f64 {
    if m.model_name == model_name {
        return 1.0;
    }
    if !m.model_architecture.is_empty() && m.model_architecture == model_name {
        // 调用方传入的是 architecture 名而非具体模型名时，给予部分分数。
        return 0.7;
    }
    0.0
}

impl std::fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRegistry")
            .field("backends", &self.backends.len())
            .finish()
    }
}
