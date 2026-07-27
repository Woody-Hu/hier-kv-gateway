//! 连接器注册表。
//!
//! 按 [`BackendType`] 注册并查找对应的 [`BackendConnector`] 实现，
//! 使路由层和网关层能通过统一的接口访问不同类型的推理后端。

use std::sync::Arc;

use aether_core::backend::{BackendType, Endpoint};
use aether_core::config::BackendConfig;
use aether_core::ids::RegionId;

use dashmap::DashMap;

use crate::connector::BackendConnector;
use crate::openai_compat::OpenAICompatConnector;

/// 连接器注册表，按后端类型索引。
pub struct ConnectorRegistry {
    /// BackendType → 连接器实例
    connectors: DashMap<BackendType, Arc<dyn BackendConnector>>,
}

impl ConnectorRegistry {
    /// 创建空注册表。
    pub fn new() -> Self {
        Self {
            connectors: DashMap::new(),
        }
    }

    /// 注册一个连接器。
    pub fn register(&self, connector: Arc<dyn BackendConnector>) {
        let bt = connector.backend_type();
        self.connectors.insert(bt, connector);
    }

    /// 按后端类型获取连接器。
    pub fn get(&self, backend_type: &BackendType) -> Option<Arc<dyn BackendConnector>> {
        self.connectors.get(backend_type).map(|e| e.clone())
    }

    /// 列出所有已注册的后端类型。
    pub fn registered_types(&self) -> Vec<BackendType> {
        self.connectors.iter().map(|e| e.key().clone()).collect()
    }

    /// 从后端配置列表创建注册表。
    ///
    /// 对每个 [`BackendConfig`] 创建对应的连接器实例：
    /// - `VllmEngine` / `LlamaCppEngine` / `GenericOpenAI` → [`OpenAICompatConnector`]
    /// - `DynamoCluster` / `LlmDCluster` → 同样使用 OpenAI 兼容连接器（后续可替换为专用连接器）
    pub fn from_configs(configs: &[BackendConfig], region: &RegionId) -> Self {
        let registry = Self::new();

        for cfg in configs {
            let connector: Arc<dyn BackendConnector> = match cfg.backend_type {
                BackendType::VllmEngine
                | BackendType::LlamaCppEngine
                | BackendType::GenericOpenAI
                | BackendType::DynamoCluster
                | BackendType::LlmDCluster => Arc::new(OpenAICompatConnector::from_endpoint(
                    &cfg.endpoint,
                    cfg.backend_type.clone(),
                    &cfg.region,
                    cfg.models.clone(),
                    cfg.kv_block_size,
                )),
            };
            registry.register(connector);
        }

        registry
    }
}

impl Default for ConnectorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_get_after_register() {
        let registry = ConnectorRegistry::new();
        let connector = Arc::new(OpenAICompatConnector::new(
            "http://localhost:8080",
            BackendType::VllmEngine,
            RegionId::new("test"),
            "instance-1",
            vec!["test-model".to_string()],
            16,
        ));
        registry.register(connector);
        assert!(registry.get(&BackendType::VllmEngine).is_some());
        assert!(registry.get(&BackendType::LlamaCppEngine).is_none());
    }

    #[test]
    fn from_configs_creates_connectors() {
        let configs = vec![BackendConfig {
            backend_type: BackendType::VllmEngine,
            endpoint: Endpoint {
                url: "http://localhost:8000".to_string(),
                protocol: aether_core::backend::Protocol::Http,
            },
            models: vec!["qwen2.5-7b".to_string()],
            region: RegionId::new("edge-1"),
            kv_block_size: 16,
            quantization: None,
        }];

        let registry = ConnectorRegistry::from_configs(&configs, &RegionId::new("edge-1"));
        assert!(registry.get(&BackendType::VllmEngine).is_some());
        assert_eq!(registry.registered_types().len(), 1);
    }
}
