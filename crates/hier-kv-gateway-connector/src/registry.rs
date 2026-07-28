//! Connector registry.
//!
//! Registers [`BackendConnector`] implementations keyed by their anchored
//! [`BackendId`] (see [`BackendConnector::backend_id`]), allowing the routing
//! and gateway layers to address *individual backend instances* through a
//! unified interface — a prerequisite for per-backend retry/failover, since
//! same-type backends (e.g. two vLLM replicas) must not share one endpoint.
//!
//! A connector may additionally serve backends beyond its anchor (e.g. a
//! Dynamo cluster front-end discovering several workers); those are attached
//! via [`ConnectorRegistry::register_alias`] after discovery.

use std::sync::Arc;

use hier_kv_gateway_core::backend::BackendType;
use hier_kv_gateway_core::config::{BackendConfig, ForwardingConfig};
use hier_kv_gateway_core::ids::{BackendId, RegionId};

use dashmap::DashMap;

use crate::connector::BackendConnector;
use crate::dynamo::{DynamoConnector, DynamoConnectorConfig};
use crate::openai_compat::OpenAICompatConnector;
use crate::sglang::SglangConnector;

/// Connector registry, indexed by backend identifier.
pub struct ConnectorRegistry {
    /// BackendId -> connector instance.
    connectors: DashMap<BackendId, Arc<dyn BackendConnector>>,
}

impl ConnectorRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            connectors: DashMap::new(),
        }
    }

    /// Register a connector under its anchored [`BackendId`].
    ///
    /// Re-registering the same id replaces the previous connector.
    pub fn register(&self, connector: Arc<dyn BackendConnector>) {
        let id = connector.backend_id();
        self.connectors.insert(id, connector);
    }

    /// Attach an additional [`BackendId`] to an already-registered connector.
    ///
    /// Used after `discover()` reports backends beyond the connector's anchor
    /// (e.g. a Dynamo cluster front-end), so the routing layer can address
    /// each discovered instance individually.
    pub fn register_alias(&self, id: BackendId, connector: &Arc<dyn BackendConnector>) {
        self.connectors.entry(id).or_insert_with(|| connector.clone());
    }

    /// Get the connector serving the given backend instance.
    pub fn get(&self, backend: &BackendId) -> Option<Arc<dyn BackendConnector>> {
        self.connectors.get(backend).map(|e| e.clone())
    }

    /// List all registered connectors (de-duplicated by anchor id).
    ///
    /// Used by the startup discovery loop: each connector is asked for its
    /// backend inventory exactly once.
    pub fn all(&self) -> Vec<Arc<dyn BackendConnector>> {
        self.connectors.iter().map(|e| e.value().clone()).collect()
    }

    /// List the backend types covered by at least one registered connector.
    pub fn registered_types(&self) -> Vec<BackendType> {
        let mut types: Vec<BackendType> = self
            .connectors
            .iter()
            .map(|e| e.value().backend_type())
            .collect();
        types.sort_by_key(|b| format!("{:?}", b));
        types.dedup();
        types
    }

    /// Create a registry from a list of backend configs.
    ///
    /// For each [`BackendConfig`] a corresponding connector instance is created:
    /// - `VllmEngine` / `LlamaCppEngine` / `GenericOpenAI` / `LlmDCluster` ->
    ///   [`OpenAICompatConnector`]
    /// - `SglangEngine` -> [`SglangConnector`] (OpenAI chat path by default;
    ///   native `/generate` when token-id forwarding is enabled)
    /// - `DynamoEngine` -> [`DynamoConnector`] (uses the endpoint URL as the
    ///   NATS URL; falls back to HTTP when the `dynamo` feature is disabled)
    ///
    /// `forwarding` carries the gateway-wide downstream forwarding behavior
    /// (currently: whether tokenized requests are emitted as `prompt_token_ids`).
    pub fn from_configs(
        configs: &[BackendConfig],
        _region: &RegionId,
        forwarding: &ForwardingConfig,
    ) -> Self {
        let registry = Self::new();

        for cfg in configs {
            let connector: Arc<dyn BackendConnector> = match cfg.backend_type {
                BackendType::VllmEngine
                | BackendType::LlamaCppEngine
                | BackendType::GenericOpenAI
                | BackendType::LlmDCluster => Arc::new(
                    OpenAICompatConnector::from_endpoint(
                        &cfg.endpoint,
                        cfg.backend_type.clone(),
                        &cfg.region,
                        cfg.models.clone(),
                        cfg.kv_block_size,
                    )
                    .with_emit_token_ids(forwarding.emit_token_ids),
                ),
                BackendType::SglangEngine => Arc::new(
                    SglangConnector::from_endpoint(
                        &cfg.endpoint,
                        &cfg.region,
                        cfg.models.clone(),
                        cfg.kv_block_size,
                    )
                    .with_emit_token_ids(forwarding.emit_token_ids),
                ),
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

    fn test_connector(url: &str, instance: &str) -> Arc<dyn BackendConnector> {
        Arc::new(OpenAICompatConnector::new(
            url,
            BackendType::VllmEngine,
            RegionId::new("test"),
            instance,
            vec!["test-model".to_string()],
            16,
        ))
    }

    #[test]
    fn registry_get_after_register() {
        let registry = ConnectorRegistry::new();
        let connector = test_connector("http://localhost:8080", "instance-1");
        let id = connector.backend_id();
        registry.register(connector);
        assert!(registry.get(&id).is_some());
        assert!(registry
            .get(&BackendId::new("test", "instance-2"))
            .is_none());
    }

    #[test]
    fn registry_keeps_same_type_backends_distinct() {
        // Two vLLM replicas must not overwrite each other.
        let registry = ConnectorRegistry::new();
        let a = test_connector("http://localhost:8080", "instance-a");
        let b = test_connector("http://localhost:8081", "instance-b");
        let id_a = a.backend_id();
        let id_b = b.backend_id();
        registry.register(a);
        registry.register(b);
        assert!(registry.get(&id_a).is_some());
        assert!(registry.get(&id_b).is_some());
        assert_eq!(registry.registered_types().len(), 1);
    }

    #[test]
    fn alias_makes_discovered_backend_addressable() {
        let registry = ConnectorRegistry::new();
        let connector = test_connector("http://localhost:8080", "front-end");
        registry.register(connector.clone());
        let alias = BackendId::new("test", "worker-behind-front-end");
        assert!(registry.get(&alias).is_none());
        registry.register_alias(alias.clone(), &connector);
        assert!(registry.get(&alias).is_some());
        // Aliases do not replace an existing registration.
        let other = test_connector("http://localhost:8081", "worker-behind-front-end");
        registry.register(other);
        registry.register_alias(alias.clone(), &connector);
        let resolved = registry.get(&alias).unwrap();
        assert_eq!(resolved.backend_id().instance.as_str(), "worker-behind-front-end");
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

        let registry = ConnectorRegistry::from_configs(
            &configs,
            &RegionId::new("edge-1"),
            &ForwardingConfig::default(),
        );
        assert!(registry
            .get(&BackendId::new("edge-1", "localhost:8000"))
            .is_some());
        assert_eq!(registry.registered_types().len(), 1);
    }

    #[test]
    fn from_configs_keeps_same_type_backends_distinct() {
        let mk = |url: &str| BackendConfig {
            backend_type: BackendType::VllmEngine,
            endpoint: Endpoint {
                url: url.to_string(),
                protocol: hier_kv_gateway_core::backend::Protocol::Http,
            },
            models: vec!["qwen2.5-7b".to_string()],
            region: RegionId::new("edge-1"),
            kv_block_size: 16,
            quantization: None,
        };
        let configs = vec![mk("http://10.0.0.1:8000"), mk("http://10.0.0.2:8000")];
        let registry = ConnectorRegistry::from_configs(
            &configs,
            &RegionId::new("edge-1"),
            &ForwardingConfig::default(),
        );
        assert!(registry
            .get(&BackendId::new("edge-1", "10.0.0.1:8000"))
            .is_some());
        assert!(registry
            .get(&BackendId::new("edge-1", "10.0.0.2:8000"))
            .is_some());
    }

    #[test]
    fn from_configs_creates_sglang_connector() {
        let configs = vec![BackendConfig {
            backend_type: BackendType::SglangEngine,
            endpoint: Endpoint {
                url: "http://localhost:30000".to_string(),
                protocol: hier_kv_gateway_core::backend::Protocol::Http,
            },
            models: vec!["qwen2.5-7b".to_string()],
            region: RegionId::new("edge-1"),
            kv_block_size: 16,
            quantization: None,
        }];
        let registry = ConnectorRegistry::from_configs(
            &configs,
            &RegionId::new("edge-1"),
            &ForwardingConfig::default(),
        );
        let connector = registry
            .get(&BackendId::new("edge-1", "localhost:30000"))
            .expect("sglang connector registered");
        assert_eq!(connector.backend_type(), BackendType::SglangEngine);
        assert!(!connector.supports_kv_events());
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
        let registry = ConnectorRegistry::from_configs(
            &configs,
            &RegionId::new("cloud-1"),
            &ForwardingConfig::default(),
        );
        assert!(registry
            .get(&BackendId::new("cloud-1", "localhost:4222"))
            .is_some());
        assert_eq!(registry.registered_types().len(), 1);
    }
}
