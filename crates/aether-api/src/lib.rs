//! Aether LLM Gateway 的 HTTP API 层。
//!
//! 该 crate 基于 axum 0.8 实现 OpenAI 兼容的 HTTP API：
//!
//! - [`openai_types`]：定义 OpenAI Chat Completions 请求/响应、模型列表等 JSON 类型，
//!   并提供与 [`aether_core::request::InferenceRequest`] 之间的转换。
//! - [`handlers`]：HTTP 路由处理函数，封装路由决策、连接器转发、SSE 流式响应等。
//! - [`server`]：基于 axum 的 HTTP server，组装路由表与监听地址。
//!
//! API 路由表：
//! - `POST /v1/chat/completions`：OpenAI Chat Completions（流式 / 非流式）。
//! - `GET /v1/models`：列出所有后端承载的模型。
//! - `GET /health`：健康检查。
//! - `GET /admin/backends`：管理端点，列出所有后端信息。
//! - `GET /admin/backends/:id/metrics`：管理端点，查询指定后端指标。

pub mod server;
pub mod handlers;
pub mod openai_types;
