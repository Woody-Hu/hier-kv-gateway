"""Hier KV Gateway Python SDK.

A Python glue layer for the Hier KV Gateway — an OpenAI-compatible LLM
gateway written in Rust. The SDK provides both synchronous (``requests``)
and asynchronous (``aiohttp``) clients, typed pydantic v2 models, and a
subprocess launcher for spinning up a local gateway instance.

Quick start::

    from hier_kv_gateway import HierKvGatewayClient, ChatCompletionRequest, Message

    with HierKvGatewayClient("http://localhost:8080") as client:
        resp = client.chat_completions(
            ChatCompletionRequest(
                model="qwen2.5-7b",
                messages=[Message(role="user", content="hello")],
            )
        )
        print(resp.choices[0].message.content)
        print(resp.route_trace)  # which backend / strategy / KV overlap
"""

from __future__ import annotations

from .async_client import AsyncHierKvGatewayClient
from .client import HierKvGatewayClient
from .exceptions import (
    GatewayConnectionError,
    GatewayError,
    GatewayTimeoutError,
)
from .launcher import GatewayLauncher
from .models import (
    BackendIdRef,
    BackendInfo,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    Delta,
    Endpoint,
    HealthStatus,
    Message,
    ModelInfo,
    ModelInstance,
    ModelsListResponse,
    RouteTrace,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    # Clients
    "HierKvGatewayClient",
    "AsyncHierKvGatewayClient",
    "GatewayLauncher",
    # Exceptions
    "GatewayError",
    "GatewayConnectionError",
    "GatewayTimeoutError",
    # Models
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "Choice",
    "ChunkChoice",
    "Delta",
    "Message",
    "Usage",
    "ModelInfo",
    "ModelsListResponse",
    "BackendInfo",
    "BackendIdRef",
    "Endpoint",
    "ModelInstance",
    "HealthStatus",
    "RouteTrace",
    # Metadata
    "__version__",
]
