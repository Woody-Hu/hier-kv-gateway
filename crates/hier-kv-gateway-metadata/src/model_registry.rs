//! Model registry: maintains the list of models currently provided by each
//! backend and supports matching by model name.
//!
//! Match scoring rules:
//! - Exact match on model name → 1.0
//! - Same architecture, different name → 0.7
//! - No match → 0.0
//!
//! When a backend provides multiple models, the highest score is taken.

use dashmap::DashMap;
use hier_kv_gateway_core::backend::ModelInstance;
use hier_kv_gateway_core::ids::BackendId;

/// Model registry.
#[derive(Default)]
pub struct ModelRegistry {
    backends: DashMap<BackendId, Vec<ModelInstance>>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            backends: DashMap::new(),
        }
    }

    /// Register a backend and its list of models (overwrites any existing entry).
    pub fn register(&self, backend: BackendId, models: Vec<ModelInstance>) {
        self.backends.insert(backend, models);
    }

    /// Unregister a backend.
    pub fn unregister(&self, backend: &BackendId) {
        self.backends.remove(backend);
    }

    /// Compute the match score of a backend for the target model name (takes the highest score).
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

    /// Find all backends that can serve the given model name (score > 0).
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

    /// Get the list of model instances for a backend (used for routing context).
    pub fn get_instances(&self, backend: &BackendId) -> Vec<ModelInstance> {
        self.backends
            .get(backend)
            .map(|r| r.value().clone())
            .unwrap_or_default()
    }

    /// Number of currently registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

/// Match score for a single model instance.
///
/// - Exact match `model.model_name == model_name` → 1.0
/// - Same architecture (and not an exact match) → 0.7
/// - Otherwise → 0.0
fn match_score(m: &ModelInstance, model_name: &str) -> f64 {
    if m.model_name == model_name {
        return 1.0;
    }
    if !m.model_architecture.is_empty() && m.model_architecture == model_name {
        // When the caller passes an architecture name rather than a specific model name, give a partial score.
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
