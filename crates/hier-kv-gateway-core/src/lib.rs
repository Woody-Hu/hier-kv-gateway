//! Core type library for the Hier KV Gateway system.
//!
//! This crate provides the foundational data structures shared across the routing,
//! cluster, connector, and API layers of the Hier KV Gateway, including: identifier
//! types, backend info models, load metrics, KV cache events, topology info,
//! configuration models, request/response types, and a unified error type.

pub mod backend;
pub mod config;
pub mod decision_event;
pub mod error;
pub mod ids;
pub mod kv_event;
pub mod metrics;
pub mod request;
pub mod topology;
