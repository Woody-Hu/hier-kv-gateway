"""Third-party system integrations for the Hier KV Gateway Python SDK.

This subpackage contains optional adapters that bridge the Hier KV Gateway
with popular LLM serving frameworks and client libraries. Each module is
imported lazily — the corresponding third-party dependency is only required
when the integration is actually used.

Available integrations:

* :mod:`hier_kv_gateway.integrations.vllm` — launch a vLLM OpenAI-compatible
  server with KV-block alignment matching the gateway.
* :mod:`hier_kv_gateway.integrations.llm_d` — discover and configure an
  LLM-D cluster as a gateway backend.
* :mod:`hier_kv_gateway.integrations.dynamo` — NVIDIA Dynamo worker
  registration / NATS bridge.
* :mod:`hier_kv_gateway.integrations.openai_sdk` — adapt the official
  ``openai`` Python SDK to talk to the gateway.
* :mod:`hier_kv_gateway.integrations.langchain` — use the gateway as a
  LangChain ``ChatModel``.

All adapters share two design principles:

1. The third-party dependency is imported inside the module body. If the
   dependency is missing, a helpful :class:`ImportError` is raised on first
   use, not at package import time.
2. Each adapter exposes at least one *factory function* (``make_*``) and one
   *example entry point* (``main``) suitable for direct execution from the
   ``python/examples/`` directory.
"""

from __future__ import annotations

__all__ = [
    "vllm",
    "llm_d",
    "dynamo",
    "openai_sdk",
    "langchain",
]
