//! Aether LLM Gateway 系统的核心类型库。
//!
//! 该 crate 提供了 Aether 网关在路由、集群、连接器与 API 层之间共享的基础数据结构，
//! 包括：标识类型、后端信息模型、负载指标、KV Cache 事件、拓扑信息、配置模型、
//! 请求/响应类型以及统一的错误类型。

pub mod backend;
pub mod config;
pub mod error;
pub mod ids;
pub mod kv_event;
pub mod metrics;
pub mod request;
pub mod topology;
