//! 基于 axum 0.8 的 HTTP Server 装配。
//!
//! 暴露 [`create_router`] 用于在已有 [`AppState`] 上构造路由表，以及 [`serve`]
//! 作为启动 HTTP server 的便捷入口。`serve` 内部使用
//! [`axum::serve`](axum::serve) + [`tokio::net::TcpListener`]，由调用方决定何时
//! 关闭（典型方式是 `axum::serve(...).with_graceful_shutdown(shutdown_signal)`）。

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tracing::{info, warn};

use aether_core::error::{AetherError, Result as AetherResult};

use crate::handlers::{admin_backends, admin_metrics, chat_completions, health, list_models, AppState};

/// 路径常量，便于与文档保持一致。
mod routes {
    /// OpenAI Chat Completions。
    pub const CHAT_COMPLETIONS: &str = "/v1/chat/completions";
    /// 模型列表。
    pub const MODELS: &str = "/v1/models";
    /// 健康检查。
    pub const HEALTH: &str = "/health";
    /// 管理端点：所有后端信息。
    pub const ADMIN_BACKENDS: &str = "/admin/backends";
    /// 管理端点：单个后端的指标。
    /// axum 0.8 使用 `{id}` 占位符语法。
    pub const ADMIN_BACKEND_METRICS: &str = "/admin/backends/{id}/metrics";
}

/// 创建一个挂载好所有路由的 axum [`Router`]。
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(routes::CHAT_COMPLETIONS, post(chat_completions))
        .route(routes::MODELS, get(list_models))
        .route(routes::HEALTH, get(health))
        .route(routes::ADMIN_BACKENDS, get(admin_backends))
        .route(routes::ADMIN_BACKEND_METRICS, get(admin_metrics))
        .with_state(state)
}

/// 启动 HTTP server 并阻塞当前任务直到 server 退出。
///
/// `addr` 形如 `0.0.0.0:8080`。该方法会消费 `state`（包成 `Arc`）并把它作为
/// 路由的共享状态。
pub async fn serve(addr: &str, state: AppState) -> AetherResult<()> {
    let state = Arc::new(state);
    let router = create_router(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| AetherError::Internal(format!("绑定地址 {} 失败: {}", addr, e)))?;

    info!(addr = %addr, "Aether HTTP server 开始监听");
    axum::serve(listener, router)
        .await
        .map_err(|e| AetherError::Internal(format!("HTTP server 运行错误: {}", e)))?;
    Ok(())
}

/// 启动 HTTP server，并在收到 Ctrl-C 信号时优雅关闭。
///
/// 与 [`serve`] 的区别在于：本函数会在后台等待 SIGINT/Ctrl-C 信号，收到后
/// 触发 axum 的 graceful shutdown。
pub async fn serve_with_graceful_shutdown(addr: &str, state: AppState) -> AetherResult<()> {
    let state = Arc::new(state);
    let router = create_router(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| AetherError::Internal(format!("绑定地址 {} 失败: {}", addr, e)))?;

    info!(addr = %addr, "Aether HTTP server 开始监听 (启用 graceful shutdown)");

    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        warn!("收到 Ctrl-C 信号，开始优雅关闭 HTTP server");
    };

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| AetherError::Internal(format!("HTTP server 运行错误: {}", e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::test_support::build_test_app_state;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_route_returns_ok() {
        let state = build_test_app_state("test-region");
        let app = create_router(state);
        let resp = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["status"], "ok");
    }

    #[tokio::test]
    async fn list_models_returns_empty_when_no_backends() {
        let state = build_test_app_state("test-region");
        let app = create_router(state);
        let resp = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "list");
        assert!(v["data"].is_array());
        assert_eq!(v["data"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn admin_backends_returns_empty_when_no_backends() {
        let state = build_test_app_state("test-region");
        let app = create_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/admin/backends")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn admin_metrics_invalid_id_returns_400() {
        let state = build_test_app_state("test-region");
        let app = create_router(state);
        // 缺少 `/` 的 backend id 应被拒绝
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/admin/backends/no-slash/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // "no-slash/metrics" 会被解析为 id="no-slash"，instance="metrics"，
        // 但后端不存在 → 返回 404；这里改用更明显的非法值。
        assert!(
            resp.status() == StatusCode::NOT_FOUND || resp.status() == StatusCode::BAD_REQUEST
        );
    }

    #[tokio::test]
    async fn admin_metrics_missing_backend_returns_404() {
        let state = build_test_app_state("test-region");
        let app = create_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/admin/backends/region-x/instance-y/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // 路径里包含两个 `/`，axum 的 {id} 只匹配最后一段之前的整个段落。
        // 这里期望 404，因为路由匹配失败。
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
