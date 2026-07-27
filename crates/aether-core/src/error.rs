//! Aether 网关统一的错误类型定义。
//!
//! 使用 [`thiserror`] 派生实现 [`std::error::Error`] 与 [`Display`](std::fmt::Display)，
//! 便于在跨 crate 调用链中传递与上下文化错误。

use thiserror::Error;

/// Aether 网关运行过程中可能出现的所有错误种类。
#[derive(Debug, Error)]
pub enum AetherError {
    /// 所有已知后端均不可用，无法选出可用目标。
    #[error("没有可用的后端实例")]
    BackendUnavailable,

    /// 路由决策失败，例如无任何后端满足约束或评分全部失效。
    #[error("路由失败: {0}")]
    RoutingFailed(String),

    /// 连接器在与后端通信时发生错误，例如连接被拒绝或协议解析失败。
    #[error("连接器错误: {0}")]
    ConnectorError(String),

    /// 指标采集或计算出错。
    #[error("指标错误: {0}")]
    MetricsError(String),

    /// 配置加载或校验出错。
    #[error("配置错误: {0}")]
    ConfigError(String),

    /// 集群成员协议（gossip/探活）相关错误。
    #[error("集群错误: {0}")]
    ClusterError(String),

    /// 请求的资源（后端、模型、配置项等）不存在。
    #[error("未找到: {0}")]
    NotFound(String),

    /// 触发限流，需要调用方退避重试。
    #[error("被限流")]
    RateLimited,

    /// 其他内部错误，用于未分类的失败场景。
    #[error("内部错误: {0}")]
    Internal(String),
}

/// Aether crate 通用 Result 别名。
pub type Result<T> = std::result::Result<T, AetherError>;
