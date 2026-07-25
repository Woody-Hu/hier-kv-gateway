//! Aether LLM Gateway 后端连接器层。
//!
//! 该 crate 屏蔽不同推理引擎（vLLM、llama.cpp、通用 OpenAI 兼容服务等）
//! 的协议差异，向网关上层提供统一的 [`BackendConnector`] trait 抽象，
//! 负责后端发现、健康检查、推理请求转发、KV 缓存事件订阅与指标采集。
//!
//! 模块组织：
//! - [`connector`]：定义后端连接器 trait 与通用类型；
//! - [`openai_compat`]：基于 HTTP/SSE 的 OpenAI 兼容连接器实现（vLLM / llama.cpp / GenericOpenAI）；
//! - [`registry`]：按 [`BackendType`] 注册并查找连接器的注册表。

pub mod connector;
pub mod openai_compat;
pub mod registry;
