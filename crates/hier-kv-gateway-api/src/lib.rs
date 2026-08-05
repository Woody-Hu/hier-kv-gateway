//! HTTP API layer of Hier KV Gateway.
//!
//! This crate implements an OpenAI-compatible HTTP API based on axum 0.8:
//!
//! - [`openai_types`]: Defines OpenAI Chat Completions request/response, model list, and
//!   other JSON types, and provides conversions to/from
//!   [`hier_kv_gateway_core::request::InferenceRequest`].
//! - [`handlers`]: HTTP route handler functions, encapsulating routing decisions,
//!   connector forwarding, and SSE streaming responses.
//! - [`server`]: axum-based HTTP server, assembling the route table and listen address.
//! - [`telemetry`]: decision-event sinks (ring buffer / tracing / NDJSON file)
//!   and the assembly helper driven by `TelemetryConfig`.
//!
//! API route table:
//! - `POST /v1/chat/completions`: OpenAI Chat Completions (streaming / non-streaming).
//! - `GET /v1/models`: Lists all models served by backends.
//! - `GET /health`: Health check.
//! - `GET /admin/backends`: Admin endpoint, lists all backend information.
//! - `GET /admin/backends/:id/metrics`: Admin endpoint, queries metrics for a specified
//!   backend.
//! - `GET /admin/decision_events`: Admin endpoint, reads the in-memory ring
//!   buffer of recent routing decision events.

pub mod server;
pub mod handlers;
pub mod openai_types;
pub mod telemetry;
pub mod coalescer;
