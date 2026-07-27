"""Pydantic v2 data models mirroring the Hier KV Gateway OpenAI-compatible API.

These models are 1:1 with the JSON types defined in
``crates/hier-kv-gateway-api/src/openai_types.rs`` plus the admin types in
``crates/hier-kv-gateway-core/src/backend.rs``.

The gateway follows the OpenAI Chat Completions protocol, so most fields will
be familiar. A few fields are Hier KV Gateway extensions (``session_id`` /
``session``) used for session-affinity routing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Chat completions request / response
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single chat message in the OpenAI ``role`` / ``content`` style."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Request body for ``POST /v1/chat/completions``.

    Mirrors :class:`OpenAIChatRequest` in the Rust API crate. The Python
    attribute ``session_id`` is serialized as the JSON field ``session`` to
    match the gateway's wire format (the gateway field is named ``session``).
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[Message]
    temperature: float = 1.0
    max_tokens: int = 1024
    stream: bool = False
    # Hier KV Gateway extension: session affinity routing key.
    session_id: Optional[str] = Field(
        default=None,
        serialization_alias="session",
        description="Optional session id used for session-affinity routing.",
    )
    # Standard OpenAI optional sampling knobs. The gateway currently ignores
    # these but accepts them for OpenAI compatibility.
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None


class Usage(BaseModel):
    """Token usage statistics."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    """A single candidate in a non-streaming chat completion response."""

    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Response body for a non-streaming ``POST /v1/chat/completions`` call.

    The SDK attaches a :attr:`route_trace` attribute (not part of the JSON
    payload) populated from the ``X-Hier-KV-Gateway-*`` response headers so
    callers can inspect which backend / strategy / KV overlap was selected.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)

    # Not part of the wire format; populated by the client from response
    # headers. Defaults to None so the model is usable on its own.
    route_trace: Optional["RouteTrace"] = None


# ---------------------------------------------------------------------------
# Streaming (SSE) chunks
# ---------------------------------------------------------------------------


class Delta(BaseModel):
    """Incremental content carried by a streaming chunk."""

    model_config = ConfigDict(extra="allow")

    role: Optional[str] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    """A single candidate in a streaming chat completion chunk."""

    model_config = ConfigDict(extra="allow")

    index: int = 0
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk for streaming ``POST /v1/chat/completions``.

    The gateway emits a stream of these (``object == "chat.completion.chunk"``)
    followed by a final ``data: [DONE]`` sentinel. The SDK populates
    :attr:`route_trace` on the first yielded chunk (from the response headers)
    and leaves it ``None`` on subsequent chunks.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChunkChoice] = Field(default_factory=list)

    # Not part of the wire format; populated by the client on the first chunk.
    route_trace: Optional["RouteTrace"] = None


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """A single model entry returned by ``GET /v1/models``."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = None


class ModelsListResponse(BaseModel):
    """Response body for ``GET /v1/models``."""

    model_config = ConfigDict(extra="allow")

    object: str = "list"
    data: List[ModelInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Admin / health types
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    """Response body for ``GET /health``.

    The gateway currently only returns ``{"status": "ok"}``; ``instance_id``
    and ``region`` are accepted for forward compatibility when richer health
    payloads are added.
    """

    model_config = ConfigDict(extra="allow")

    status: str
    instance_id: Optional[str] = None
    region: Optional[str] = None


class BackendIdRef(BaseModel):
    """The structured form of a :class:`BackendId` over the wire.

    The gateway serializes ``BackendId`` as ``{"region": ..., "instance": ...}``.
    Its ``Display`` form (``region/instance``) is used in the
    ``X-Hier-KV-Gateway-Backend`` header and in admin path parameters.
    """

    model_config = ConfigDict(extra="allow")

    region: str
    instance: str

    def as_path(self) -> str:
        """Return the ``region/instance`` form used in admin URL paths."""
        return f"{self.region}/{self.instance}"


class Endpoint(BaseModel):
    """A backend connection endpoint."""

    model_config = ConfigDict(extra="allow")

    url: str
    protocol: Optional[str] = None


class ModelInstance(BaseModel):
    """Metadata for a single model served by a backend."""

    model_config = ConfigDict(extra="allow")

    model_name: str
    model_architecture: Optional[str] = None
    quantization: Optional[str] = None
    max_context_len: Optional[int] = None
    supports_tool_calling: Optional[bool] = None
    supports_streaming: Optional[bool] = None


class BackendInfo(BaseModel):
    """Static metadata for a registered inference backend.

    Mirrors :class:`BackendInfo` in ``crates/hier-kv-gateway-core/src/backend.rs``.
    The ``id`` field is exposed both as a structured object (:attr:`id`) and,
    for convenience, as the ``region/instance`` string via :meth:`id_str`.
    """

    model_config = ConfigDict(extra="allow")

    id: Union[BackendIdRef, str]
    backend_type: str
    endpoint: Optional[Endpoint] = None
    models: List[ModelInstance] = Field(default_factory=list)
    region: Optional[str] = None
    status: Optional[str] = None

    def id_str(self) -> str:
        """Return the ``region/instance`` string form of the backend id.

        Falls back to the raw string when the gateway returns ``id`` as a
        plain string rather than an object.
        """
        if isinstance(self.id, BackendIdRef):
            return self.id.as_path()
        return str(self.id)


# ---------------------------------------------------------------------------
# Route trace (from X-Hier-KV-Gateway-* response headers)
# ---------------------------------------------------------------------------


class RouteTrace(BaseModel):
    """Routing metadata extracted from ``X-Hier-KV-Gateway-*`` headers.

    Populated by the client after every ``chat_completions`` call. The
    ``region`` field is captured for forward compatibility; the current
    gateway only emits ``Backend`` / ``Strategy`` / ``KV-Overlap`` headers.
    """

    backend: Optional[str] = None
    region: Optional[str] = None
    strategy: Optional[str] = None
    kv_overlap: Optional[int] = None

    @classmethod
    def from_headers(
        cls, headers: "Any"
    ) -> "RouteTrace":
        """Build a :class:`RouteTrace` from a case-insensitive header mapping.

        Accepts both ``requests.Response.headers`` and
        ``aiohttp.ClientResponse.headers`` (both support case-insensitive
        lookup via ``headers["Name"]``).
        """
        backend = _get_header(headers, "X-Hier-KV-Gateway-Backend")
        region = _get_header(headers, "X-Hier-KV-Gateway-Region")
        strategy = _get_header(headers, "X-Hier-KV-Gateway-Strategy")
        overlap_raw = _get_header(headers, "X-Hier-KV-Gateway-KV-Overlap")
        kv_overlap: Optional[int] = None
        if overlap_raw is not None:
            try:
                kv_overlap = int(overlap_raw)
            except (TypeError, ValueError):
                kv_overlap = None
        return cls(
            backend=backend,
            region=region,
            strategy=strategy,
            kv_overlap=kv_overlap,
        )


def _get_header(headers: Any, name: str) -> Optional[str]:
    """Case-insensitive header lookup that works for requests & aiohttp."""
    if headers is None:
        return None
    # Both requests.structures.CaseInsensitiveDict and aiohttp's CIMultiDict
    # support ``headers[name]`` lookup; missing keys raise KeyError.
    try:
        value = headers[name]
    except KeyError:
        return None
    if value is None:
        return None
    # aiohttp may return a list-like; coerce to str.
    return str(value)


# Resolve forward references for route_trace fields.
ChatCompletionResponse.model_rebuild()
ChatCompletionChunk.model_rebuild()
