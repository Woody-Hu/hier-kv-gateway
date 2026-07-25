//! Aether LLM Gateway 集成测试 crate。
//!
//! 该 crate 不产出任何业务代码，仅作为承载 `tests/` 目录下集成测试的容器。
//! 各集成测试文件直接调用真实组件（不 mock、不短路），验证 Aether 各层
//! 之间的数据流转与路由逻辑。
