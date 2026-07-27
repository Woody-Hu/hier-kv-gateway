//! Connector registry.
//!
//! Registers and looks up the corresponding [`BackendConnector`] implementation by
//! [`BackendType`], allowing the routing and gateway layers to access different types of
//! inference backends through a unified interface.

use std::sync::Arc;

use hier_kv_gateway_core::backend::BackendType;
use hier_kv_gateway_core::config::BackendConfig;
use hier_kv_gateway_core::ids::RegionId;

use dashmap::DashMap;

use crate::connector::BackendConnector;
use crate::dynamo::{DynamoConnector, DynamoConnectorConfig};
use crate::openai_compat::OpenAICompatConnector;

/// Connector registry, indexed by backend type.
pub struct ConnectorRegistry {
    /// BackendType -> connector instance
    connectors: DashMap<BackendType, Arc<dyn BackendConnector>>,
}

impl ConnectorRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            connectors: DashMap::new(),
        }
    }

    /// Register a connector.
    pub fn register(&self, connector: Arc<dyn BackendConnector>) {
        let bt = connector.backend_type();
        self.connectors.insert(bt, connector);
    }

    /// Get a connector by backend type.
    pub fn get(&self, backend_type: &BackendType) -> Option<Arc<dyn BackendConnector>> {
        self.connectors.get(backend_type).map(|e| e.clone())
    }

    /// List all registered backend types.
    pub fn registered_types(&self) -> Vec<BackendType> {
        self.connectors.iter().map(|e| e.key().clone()).collect()
    }

    /// Create a registry from a list of backend configs.
    ///
    /// For each [`BackendConfig`] a corresponding connector instance is created:
    /// - `VllmEngine` / `LlamaCppEngine` / `GenericOpenAI` / `LlmDCluster` ->
    ///   [`OpenAICompatConnector`]
    /// - `DynamoEngine` -> [`DynamoConnector`] (uses the endpoint URL as the
    ///   NATS URL; falls back to HTTP when the `dynamo` feature is disabled)
    pub fn from_configs(configs: &[BackendConfig], _region: &RegionId) -> Self {
        let registry = Self::new();

        for cfg in configs {
            let connector: Arc<dyn BackendConnector> = match cfg.backend_type {
                BackendType::VllmEngine
                | BackendType::LlamaCppEngine
                | BackendType::GenericOpenAI
                | BackendType::LlmDCluster => Arc::new(OpenAICompatConnector::from_endpoint(
                    &cfg.endpoint,
                    cfg.backend_type.clone(),
                    &cfg.region,
                    cfg.models.clone(),
                    cfg.kv_block_size,
                )),
                BackendType::DynamoEngine => {
                    // Derive a stable instance id from the endpoint URL
                    // (host:port) when the config does not provide one.
                    let instance_id = cfg
                        .endpoint
                        .url
                        .replace("nats://", "")
                        .replace("http://", "")
                        .replace("https://", "");
                    let dynamo_cfg = DynamoConnectorConfig::from_endpoint(
                        &cfg.endpoint,
                        &cfg.region,
                        instance_id,
                        cfg.models.clone(),
                        cfg.kv_block_size,
                    );
                    Arc::new(DynamoConnector::new(dynamo_cfg)) as Arc<dyn BackendConnector>
                }
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
    use hier_kv_gateway_core::backend::Endpoint;

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
                protocol: hier_kv_gateway_core::backend::Protocol::Http,
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

    #[test]
    fn from_configs_creates_dynamo_connector() {
        let configs = vec![BackendConfig {
            backend_type: BackendType::DynamoEngine,
            endpoint: Endpoint {
                url: "nats://localhost:4222".to_string(),
                protocol: hier_kv_gateway_core::backend::Protocol::Nats,
            },
            models: vec!["llama-3.1-8b".to_string()],
            region: RegionId::new("cloud-1"),
            kv_block_size: 16,
            quantization: None,
        }];
        let registry = ConnectorRegistry::from_configs(&configs, &RegionId::new("cloud-1"));
        assert!(registry.get(&BackendType::DynamoEngine).is_some());
        assert_eq!(registry.registered_types().len(), 1);
    }
}
