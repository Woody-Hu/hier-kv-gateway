//! Hier KV Gateway backend connector layer.
//!
//! This crate abstracts protocol differences between inference engines (vLLM, llama.cpp,
//! generic OpenAI-compatible services, etc.) and provides a unified [`BackendConnector`]
//! trait abstraction to the upper gateway layer, responsible for backend discovery,
//! health checks, inference request forwarding, KV cache event subscription, and metrics
//! collection.
//!
//! Module organization:
//! - [`connector`]: Defines the backend connector trait and common types;
//! - [`openai_compat`]: HTTP/SSE-based OpenAI-compatible connector implementation (vLLM / llama.cpp / GenericOpenAI);
//! - [`registry`]: Registry for registering and looking up connectors by [`BackendType`].

pub mod connector;
pub mod openai_compat;
pub mod registry;
