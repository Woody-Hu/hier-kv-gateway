"""Asynchronous client for the Hier KV Gateway HTTP API.

Built on top of :mod:`aiohttp`. Mirrors :class:`hier_kv_gateway.client.HierKvGatewayClient`
but exposes coroutine methods and an async iterator for streaming.

Example::

    import asyncio
    from hier_kv_gateway import AsyncHierKvGatewayClient, ChatCompletionRequest, Message

    async def main():
        async with AsyncHierKvGatewayClient("http://localhost:8080") as client:
            models = await client.list_models()
            resp = await client.chat_completions(
                ChatCompletionRequest(
                    model="qwen2.5-7b",
                    messages=[Message(role="user", content="hello")],
                )
            )
            print(resp.choices[0].message.content)
            print(resp.route_trace)

    asyncio.run(main())
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp

from .exceptions import GatewayConnectionError, GatewayError, GatewayTimeoutError
from .models import (
    BackendInfo,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthStatus,
    ModelsListResponse,
    RouteTrace,
)

_DONE_SENTINEL = "[DONE]"


class AsyncHierKvGatewayClient:
    """Asyncio-compatible client for the Hier KV Gateway.

    A single :class:`aiohttp.ClientSession` is created lazily on first use
    (or eagerly via :meth:`start`) and reused for all subsequent requests.
    Always close the client (``await client.close()`` or use ``async with``)
    to avoid leaking the session.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "hier-kv-gateway-python/0.1",
        }
        if default_headers:
            self.default_headers.update(default_headers)
        self._session: Optional[aiohttp.ClientSession] = None
        self._closed = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise GatewayError("client has been closed")
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                timeout=timeout,
                headers=self.default_headers,
            )
        return self._session

    async def start(self) -> None:
        """Eagerly create the underlying :class:`aiohttp.ClientSession`."""
        await self._get_session()

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Any] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> aiohttp.ClientResponse:
        session = await self._get_session()
        headers: Dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        try:
            response = await session.request(
                method=method,
                url=path,
                json=json_body,
                headers=headers or None,
            )
        except aiohttp.ServerTimeoutError as exc:
            raise GatewayTimeoutError(
                f"request to {method} {path} timed out after {self.timeout}s"
            ) from exc
        except aiohttp.ClientConnectorError as exc:
            raise GatewayConnectionError(
                f"cannot connect to {self.base_url} ({method} {path}): {exc}"
            ) from exc
        except aiohttp.ClientError as exc:
            raise GatewayError(
                f"request to {method} {path} failed: {exc}"
            ) from exc

        if not (200 <= response.status < 300):
            await self._raise_for_status(response)
        return response

    @staticmethod
    async def _raise_for_status(response: aiohttp.ClientResponse) -> None:
        message = f"gateway returned HTTP {response.status}"
        try:
            payload = await response.json(content_type=None)
        except (aiohttp.ContentTypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and "message" in err:
                message = str(err["message"])
            elif "message" in payload:
                message = str(payload["message"])
        else:
            try:
                text = await response.text()
            except Exception:
                text = ""
            if text:
                message = text.strip() or message
        raise GatewayError(message, status_code=response.status)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request.

        Populates :attr:`ChatCompletionResponse.route_trace` from the
        ``X-Hier-KV-Gateway-*`` response headers.
        """
        payload = json.loads(
            request.model_dump_json(by_alias=True, exclude_none=True)
        )
        payload["stream"] = False
        response = await self._request(
            "POST",
            "/v1/chat/completions",
            json_body=payload,
        )
        try:
            body = await response.json()
        finally:
            response.release()
        result = ChatCompletionResponse.model_validate(body)
        result.route_trace = RouteTrace.from_headers(response.headers)
        return result

    async def chat_completions_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Send a streaming chat completion request.

        Returns an async iterator yielding :class:`ChatCompletionChunk`
        objects. The first chunk carries the :attr:`route_trace` populated
        from the response headers. Iteration stops after ``data: [DONE]``.
        """
        payload = json.loads(
            request.model_dump_json(by_alias=True, exclude_none=True)
        )
        payload["stream"] = True
        response = await self._request(
            "POST",
            "/v1/chat/completions",
            json_body=payload,
            extra_headers={"Accept": "text/event-stream"},
        )
        route_trace = RouteTrace.from_headers(response.headers)
        first = True
        try:
            async for raw_line in response.content:
                if raw_line is None:
                    continue
                if isinstance(raw_line, (bytes, bytearray)):
                    line = raw_line.decode("utf-8", errors="replace")
                else:
                    line = raw_line
                line = line.strip()
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                else:
                    continue
                if data == _DONE_SENTINEL:
                    break
                if not data:
                    continue
                try:
                    chunk_body = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunk = ChatCompletionChunk.model_validate(chunk_body)
                if first:
                    chunk.route_trace = route_trace
                    first = False
                else:
                    chunk.route_trace = None
                yield chunk
        finally:
            response.release()

    async def list_models(self) -> ModelsListResponse:
        """List models served by all registered backends (``GET /v1/models``)."""
        response = await self._request("GET", "/v1/models")
        try:
            body = await response.json()
        finally:
            response.release()
        return ModelsListResponse.model_validate(body)

    async def health(self) -> HealthStatus:
        """Query gateway health (``GET /health``)."""
        response = await self._request("GET", "/health")
        try:
            body = await response.json()
        finally:
            response.release()
        return HealthStatus.model_validate(body)

    async def list_backends(self) -> list:
        """List all registered backends (``GET /admin/backends``)."""
        response = await self._request("GET", "/admin/backends")
        try:
            data = await response.json()
        finally:
            response.release()
        if not isinstance(data, list):
            raise GatewayError(
                "unexpected /admin/backends payload: expected a JSON array",
                status_code=response.status,
            )
        return [BackendInfo.model_validate(item) for item in data]

    async def get_topology(self) -> dict:
        """Fetch the cluster topology matrix (``GET /admin/topology``).

        Returns the raw JSON document as a dict. Note: this endpoint is
        documented but not yet implemented by the gateway; calls will raise
        :class:`GatewayError` with a 404 status until it is.
        """
        response = await self._request("GET", "/admin/topology")
        try:
            body = await response.json(content_type=None)
        finally:
            response.release()
        return body

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._closed:
            return
        self._closed = True
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "AsyncHierKvGatewayClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
