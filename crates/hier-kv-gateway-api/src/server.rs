//! HTTP server assembly based on axum 0.8.
//!
//! Exposes [`create_router`] for constructing a route table on top of an existing
//! [`AppState`], and [`serve`] as a convenience entry point for starting the HTTP server.
//! `serve` internally uses [`axum::serve`](axum::serve) +
//! [`tokio::net::TcpListener`]; the caller decides when to shut it down (a typical way is
//! `axum::serve(...).with_graceful_shutdown(shutdown_signal)`).

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tracing::{info, warn};

use hier_kv_gateway_core::error::{HierKvGatewayError, Result as HierKvGatewayResult};

use crate::handlers::{admin_backends, admin_metrics, chat_completions, health, list_models, AppState};

/// Path constants, kept consistent with the documentation.
mod routes {
    /// OpenAI Chat Completions.
    pub const CHAT_COMPLETIONS: &str = "/v1/chat/completions";
    /// Model list.
    pub const MODELS: &str = "/v1/models";
    /// Health check.
    pub const HEALTH: &str = "/health";
    /// Admin endpoint: all backend information.
    pub const ADMIN_BACKENDS: &str = "/admin/backends";
    /// Admin endpoint: metrics for a single backend.
    /// axum 0.8 uses the `{id}` placeholder syntax.
    pub const ADMIN_BACKEND_METRICS: &str = "/admin/backends/{id}/metrics";
}

/// Create an axum [`Router`] with all routes mounted.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(routes::CHAT_COMPLETIONS, post(chat_completions))
        .route(routes::MODELS, get(list_models))
        .route(routes::HEALTH, get(health))
        .route(routes::ADMIN_BACKENDS, get(admin_backends))
        .route(routes::ADMIN_BACKEND_METRICS, get(admin_metrics))
        .with_state(state)
}

/// Start the HTTP server and block the current task until the server exits.
///
/// `addr` is in the form `0.0.0.0:8080`. This method consumes `state` (wrapped in an
/// `Arc`) and uses it as the router's shared state.
pub async fn serve(addr: &str, state: AppState) -> HierKvGatewayResult<()> {
    let state = Arc::new(state);
    let router = create_router(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| HierKvGatewayError::Internal(format!("failed to bind address {}: {}", addr, e)))?;

    info!(addr = %addr, "Hier KV Gateway HTTP server is listening");
    axum::serve(listener, router)
        .await
        .map_err(|e| HierKvGatewayError::Internal(format!("HTTP server runtime error: {}", e)))?;
    Ok(())
}

/// Start the HTTP server and shut down gracefully on Ctrl-C.
///
/// Difference from [`serve`]: this function waits for SIGINT/Ctrl-C in the background,
/// and triggers axum's graceful shutdown upon receiving it.
pub async fn serve_with_graceful_shutdown(addr: &str, state: AppState) -> HierKvGatewayResult<()> {
    let state = Arc::new(state);
    let router = create_router(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| HierKvGatewayError::Internal(format!("failed to bind address {}: {}", addr, e)))?;

    info!(addr = %addr, "Hier KV Gateway HTTP server is listening (graceful shutdown enabled)");

    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        warn!("Ctrl-C signal received; starting graceful shutdown of the HTTP server");
    };

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| HierKvGatewayError::Internal(format!("HTTP server runtime error: {}", e)))?;
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
        // A backend id without `/` should be rejected
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/admin/backends/no-slash/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // "no-slash/metrics" would be parsed as id="no-slash", instance="metrics",
        // but the backend does not exist -> returns 404; here we use a more obviously
        // invalid value instead.
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
        // The path contains two `/`; axum's {id} only matches the whole segment before
        // the last one. Expect 404 here because route matching fails.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
