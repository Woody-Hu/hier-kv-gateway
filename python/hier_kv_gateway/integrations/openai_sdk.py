"""OpenAI Python SDK adapter: use the official ``openai`` package against the gateway.

The official ``openai`` Python SDK (https://github.com/openai/openai-python)
is a natural fit for the Hier KV Gateway because the gateway is fully
OpenAI-compatible. The only friction is that the SDK has no concept of
the gateway's session-affinity extension (the ``session`` field), so users
who want session affinity need to inject it manually.

This adapter provides:

1. :func:`make_openai_client` — build an :class:`openai.OpenAI` (or
   :class:`openai.AsyncOpenAI`) client pre-pointed at the gateway and
   pre-configured with the right headers.
2. :func:`with_session` — attach a session id to a request payload for
   session-affinity routing.
3. :func:`extract_route_trace` — pull the gateway's ``X-Hier-KV-Gateway-*``
   headers off an SDK response object into a :class:`~hier_kv_gateway.models.RouteTrace`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..models import RouteTrace

# Default API key used by the SDK. The gateway does not validate API keys
# today, so any non-empty string works.
DEFAULT_GATEWAY_API_KEY = "hier-kv-gateway"

# HTTP response header names emitted by the gateway.
_HEADER_BACKEND = "X-Hier-KV-Gateway-Backend"
_HEADER_REGION = "X-Hier-KV-Gateway-Region"
_HEADER_STRATEGY = "X-Hier-KV-Gateway-Strategy"
_HEADER_KV_OVERLAP = "X-Hier-KV-Gateway-KV-Overlap"


def make_openai_client(
    base_url: str = "http://localhost:8080/v1",
    *,
    api_key: str = DEFAULT_GATEWAY_API_KEY,
    timeout: float = 60.0,
    default_headers: Optional[Dict[str, str]] = None,
    async_client: bool = False,
    **extra: Any,
) -> Any:
    """Build an ``openai.OpenAI`` / ``openai.AsyncOpenAI`` client for the gateway.

    Args:
        base_url: Gateway base URL **including** the ``/v1`` prefix. The
            OpenAI SDK appends ``/chat/completions`` etc. to this.
        api_key: API key. The gateway accepts any non-empty string.
        timeout: Per-request timeout in seconds.
        default_headers: Optional extra headers sent on every request.
        async_client: Return :class:`openai.AsyncOpenAI` instead of
            :class:`openai.OpenAI`.
        **extra: Forwarded to the SDK constructor (e.g.
            ``http_client=httpx.Client(...)``).

    Returns:
        An :class:`openai.OpenAI` or :class:`openai.AsyncOpenAI` instance.

    Raises:
        ImportError: if the ``openai`` package is not installed.
    """
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — depends on env
        raise ImportError(
            "the 'openai' package is required for this adapter. "
            "Install it with: pip install openai"
        ) from exc

    headers: Dict[str, str] = {
        "User-Agent": "hier-kv-gateway-openai-adapter/0.1",
    }
    if default_headers:
        headers.update(default_headers)

    cls = openai.AsyncOpenAI if async_client else openai.OpenAI
    return cls(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        default_headers=headers,
        **extra,
    )


def with_session(
    payload: Dict[str, Any],
    session_id: str,
) -> Dict[str, Any]:
    """Attach a gateway ``session`` field to an OpenAI request payload.

    The OpenAI SDK does not natively know about the gateway's session
    extension. Callers using the SDK's low-level ``client.chat.completions.create``
    can pass ``extra_body={"session": "..."}`` instead; this helper is for
    code paths that build the payload manually (e.g. when bypassing the SDK
    or using ``client.post``).

    Args:
        payload: The OpenAI Chat Completions request body (mutated & returned).
        session_id: Session id used for session-affinity routing.

    Returns:
        The same dict, with ``session`` added.
    """
    payload["session"] = session_id
    return payload


def extract_route_trace(response: Any) -> RouteTrace:
    """Extract a :class:`RouteTrace` from an OpenAI SDK response object.

    The OpenAI SDK v1 stores HTTP headers on ``response.headers`` (a
    ``httpx.Headers`` instance). This helper normalizes the lookup and
    returns a :class:`RouteTrace` populated from the gateway's
    ``X-Hier-KV-Gateway-*`` headers.

    Args:
        response: Any object with a ``headers`` attribute. If the attribute
            is missing or no gateway headers are present, an empty
            :class:`RouteTrace` (all-``None`` fields) is returned.
    """
    headers = getattr(response, "headers", None)
    if headers is None:
        return RouteTrace()
    return RouteTrace.from_headers(headers)


def chat_completion_with_session(
    client: Any,
    *,
    model: str,
    messages: list,
    session_id: str,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper around ``client.chat.completions.create`` that
    injects the gateway ``session`` field via ``extra_body``.

    Args:
        client: An :class:`openai.OpenAI` or :class:`openai.AsyncOpenAI`
            instance built by :func:`make_openai_client`.
        model: Model id.
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        session_id: Session id for affinity routing.
        stream: Whether to stream the response.
        **kwargs: Forwarded to ``create``.

    Returns:
        The SDK's response object (``ChatCompletion`` or
        ``AsyncStream[ChatCompletionChunk]``).
    """
    extra_body = kwargs.pop("extra_body", {}) or {}
    extra_body.setdefault("session", session_id)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        extra_body=extra_body,
        **kwargs,
    )


def main() -> int:
    """Entry point: run a chat completion through the OpenAI SDK adapter.

    Run via::

        python -m hier_kv_gateway.integrations.openai_sdk \\
            --base-url http://localhost:8080/v1 \\
            --model qwen2.5-7b
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Use the official openai SDK against the Hier KV Gateway.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--session-id", default="openai-sdk-demo")
    parser.add_argument("--prompt", default="Reply with the single word: pong")
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    try:
        client = make_openai_client(base_url=args.base_url)
    except ImportError as exc:
        print(f"[openai] {exc}", file=sys.stderr)
        return 2

    print(f"[openai] sending chat completion to {args.base_url} (model={args.model})")
    try:
        if args.stream:
            stream = chat_completion_with_session(
                client,
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                session_id=args.session_id,
                stream=True,
                max_tokens=args.max_tokens,
            )
            collected = []
            last_chunk = None
            for chunk in stream:
                last_chunk = chunk
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    collected.append(delta.content)
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()
            print()
            rt = extract_route_trace(last_chunk)
            print(f"[openai][route] backend={rt.backend} strategy={rt.strategy} kv_overlap={rt.kv_overlap}")
            print(f"[openai] collected {len(collected)} delta(s)")
        else:
            resp = chat_completion_with_session(
                client,
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                session_id=args.session_id,
                max_tokens=args.max_tokens,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            print(f"[openai] response: {content!r}")
            rt = extract_route_trace(resp)
            print(f"[openai][route] backend={rt.backend} strategy={rt.strategy} kv_overlap={rt.kv_overlap}")
    except Exception as exc:  # noqa: BLE001
        print(f"[openai] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
