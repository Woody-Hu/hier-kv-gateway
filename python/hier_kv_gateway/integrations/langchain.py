"""LangChain integration: use the Hier KV Gateway as a LangChain ``ChatModel``.

LangChain (https://python.langchain.com) is a popular framework for
building LLM-powered applications. Because the gateway is OpenAI-compatible,
the simplest way to use it from LangChain is via :class:`langchain_openai.ChatOpenAI`.
This adapter provides a thin wrapper around that pattern, plus a
factory that pre-configures session affinity and KV-aware routing.

Two entry points are exposed:

1. :func:`make_chat_model` — build a configured :class:`ChatOpenAI` instance
   pointed at the gateway.
2. :class:`HierKvGatewayChatModel` — a minimal subclass that adds a
   ``session_id`` convenience parameter and a ``route_trace`` accessor
   populated after each call.

The ``langchain_openai`` and ``openai`` packages are imported lazily.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..models import RouteTrace


# Default API key (the gateway does not validate API keys today).
DEFAULT_GATEWAY_API_KEY = "hier-kv-gateway"


def make_chat_model(
    *,
    base_url: str = "http://localhost:8080/v1",
    model: str = "qwen2.5-7b",
    api_key: str = DEFAULT_GATEWAY_API_KEY,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    session_id: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
    streaming: bool = False,
    **extra: Any,
) -> Any:
    """Build a :class:`langchain_openai.ChatOpenAI` pointed at the gateway.

    Args:
        base_url: Gateway base URL **including** ``/v1``.
        model: Model id.
        api_key: API key (any non-empty string).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout: Per-request timeout in seconds.
        session_id: If provided, every call will carry this session id for
            affinity routing. Override per-call via ``model.with_config(...)``.
        default_headers: Extra HTTP headers attached to every request.
        streaming: Enable LangChain streaming by default.
        **extra: Forwarded to ``ChatOpenAI`` (e.g. ``model_kwargs={"top_p": 0.9}``).

    Returns:
        A :class:`langchain_openai.ChatOpenAI` instance.

    Raises:
        ImportError: if ``langchain_openai`` (and ``openai``) are not installed.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — depends on env
        raise ImportError(
            "the 'langchain-openai' package is required for this adapter. "
            "Install it with: pip install langchain-openai"
        ) from exc

    headers: Dict[str, str] = {
        "User-Agent": "hier-kv-gateway-langchain-adapter/0.1",
    }
    if default_headers:
        headers.update(default_headers)

    # Use ``model_kwargs`` for gateway-specific fields that ChatOpenAI does
    # not know about natively. The OpenAI SDK forwards ``extra_body`` to
    # the request body, and ``model_kwargs`` is LangChain's passthrough for
    # arbitrary request fields.
    model_kwargs: Dict[str, Any] = dict(extra.pop("model_kwargs", {}) or {})
    if session_id is not None:
        model_kwargs.setdefault("session", session_id)

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        streaming=streaming,
        default_headers=headers,
        model_kwargs=model_kwargs,
        **extra,
    )


class HierKvGatewayChatModel:
    """Convenience wrapper around :class:`langchain_openai.ChatOpenAI`.

    The wrapper:

    * Pre-points a ``ChatOpenAI`` at the gateway.
    * Exposes :meth:`set_session_id` to swap the session-affinity key at any time.
    * Captures the last :class:`RouteTrace` from response headers into
      :attr:`last_route_trace` after each ``invoke`` / ``stream`` call.

    The wrapper deliberately does **not** inherit from ``ChatOpenAI`` (which
    is a Pydantic v1 model in older versions and a v2 model in newer ones);
    instead it delegates via ``__getattr__`` so it works across versions.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080/v1",
        model: str = "qwen2.5-7b",
        api_key: str = DEFAULT_GATEWAY_API_KEY,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        session_id: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        streaming: bool = False,
        **extra: Any,
    ) -> None:
        self._session_id = session_id
        self._model = make_chat_model(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            session_id=session_id,
            default_headers=default_headers,
            streaming=streaming,
            **extra,
        )
        self.last_route_trace: Optional[RouteTrace] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Update the session-affinity key on the underlying model.

        Passing ``None`` removes the session field from subsequent requests.
        """
        self._session_id = session_id
        # ``model_kwargs`` is a dict on ChatOpenAI; mutating it in place
        # propagates to subsequent calls.
        model_kwargs = self._model.model_kwargs or {}
        if session_id is None:
            model_kwargs.pop("session", None)
        else:
            model_kwargs["session"] = session_id
        # Some LangChain versions freeze ``model_kwargs``; reassign to be safe.
        try:
            self._model.model_kwargs = model_kwargs
        except Exception:
            pass

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def underlying(self) -> Any:
        """Access the wrapped :class:`ChatOpenAI` directly."""
        return self._model

    # ------------------------------------------------------------------
    # LangChain-style invocation
    # ------------------------------------------------------------------

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped model and capture the route trace from the response.

        ``messages`` may be a string, a list of ``BaseMessage``, or a
        ``PromptValue`` — anything LangChain's ``ChatOpenAI.invoke`` accepts.
        """
        result = self._model.invoke(messages, *args, **kwargs)
        self.last_route_trace = _extract_route_trace_from_lc_result(result)
        return result

    def stream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Stream the wrapped model, capturing the route trace from the final chunk."""
        last_chunk = None
        for chunk in self._model.stream(messages, *args, **kwargs):
            last_chunk = chunk
            yield chunk
        if last_chunk is not None:
            self.last_route_trace = _extract_route_trace_from_lc_result(last_chunk)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only called when the attribute is not found on the wrapper itself.
        # Forward to the underlying ChatOpenAI.
        return getattr(self._model, name)


def _extract_route_trace_from_lc_result(result: Any) -> RouteTrace:
    """Best-effort extraction of a :class:`RouteTrace` from a LangChain result.

    LangChain's ``AIMessage`` / ``ChatGenerationChunk`` objects do not
    surface HTTP headers directly. The route trace is therefore only
    available when the underlying OpenAI SDK attached ``response_metadata``
    with a ``headers`` key — which varies by ``langchain-openai`` version.

    When no headers are found, an empty :class:`RouteTrace` is returned.
    """
    # ``AIMessage.response_metadata`` is the canonical place for response
    # info in modern langchain-openai; some versions put it under
    # ``additional_kwargs``.
    metadata = getattr(result, "response_metadata", None) or {}
    if not isinstance(metadata, dict):
        return RouteTrace()
    headers = metadata.get("headers")
    if headers is None:
        # Try ``additional_kwargs`` next.
        extra = getattr(result, "additional_kwargs", None) or {}
        if isinstance(extra, dict):
            headers = extra.get("headers")
    if headers is None:
        return RouteTrace()
    return RouteTrace.from_headers(headers)


def main() -> int:
    """Entry point: run a chat completion through the LangChain adapter.

    Run via::

        python -m hier_kv_gateway.integrations.langchain \\
            --base-url http://localhost:8080/v1 \\
            --model qwen2.5-7b
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Use LangChain's ChatOpenAI against the Hier KV Gateway.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--session-id", default="langchain-demo")
    parser.add_argument("--prompt", default="Reply with the single word: pong")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stream", action="store_true", default=False)
    args = parser.parse_args()

    try:
        from langchain_core.messages import HumanMessage  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"[langchain] {exc}", file=sys.stderr)
        return 2

    try:
        model = HierKvGatewayChatModel(
            base_url=args.base_url,
            model=args.model,
            session_id=args.session_id,
            max_tokens=args.max_tokens,
            streaming=args.stream,
        )
    except ImportError as exc:
        print(f"[langchain] {exc}", file=sys.stderr)
        return 2

    print(f"[langchain] sending prompt to {args.base_url} (model={args.model})")
    try:
        messages = [HumanMessage(content=args.prompt)]
        if args.stream:
            collected = []
            for chunk in model.stream(messages):
                text = getattr(chunk, "content", "") or ""
                if text:
                    collected.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
            print()
            print(f"[langchain] collected {len(collected)} chunk(s)")
        else:
            result = model.invoke(messages)
            content = getattr(result, "content", "") or ""
            print(f"[langchain] response: {content!r}")
        rt = model.last_route_trace
        if rt is not None:
            print(
                f"[langchain][route] backend={rt.backend} "
                f"strategy={rt.strategy} kv_overlap={rt.kv_overlap}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[langchain] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
