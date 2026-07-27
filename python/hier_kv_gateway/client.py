"""Synchronous client for the Hier KV Gateway HTTP API.

Built on top of :mod:`requests`. For an asyncio-compatible client see
:mod:`hier_kv_gateway.async_client`.

Example::

    from hier_kv_gateway import HierKvGatewayClient, ChatCompletionRequest, Message

    with HierKvGatewayClient("http://localhost:8080") as client:
        models = client.list_models()
        resp = client.chat_completions(
            ChatCompletionRequest(
                model="qwen2.5-7b",
                messages=[Message(role="user", content="hello")],
            )
        )
        print(resp.choices[0].message.content)
        print(resp.route_trace)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

import requests

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

# SSE sentinel marking the end of a streaming response.
_DONE_SENTINEL = "[DONE]"


class HierKvGatewayClient:
    """Synchronous client for the Hier KV Gateway.

    The client is safe to share across threads only if callers do not invoke
    methods concurrently that mutate shared state (the underlying
    :class:`requests.Session` is thread-safe for concurrent reads, but
    :meth:`close` is not). For typical request/response use, prefer using the
    client as a context manager.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        # Normalize trailing slash so urljoin-style concatenation is predictable.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "hier-kv-gateway-python/0.1",
        }
        if default_headers:
            self.default_headers.update(default_headers)
        self._session = requests.Session()
        self._session.headers.update(self.default_headers)
        self._closed = False

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

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Any] = None,
        stream: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        if self._closed:
            raise GatewayError("client has been closed")
        headers: Optional[Dict[str, str]] = extra_headers
        try:
            response = self._session.request(
                method=method,
                url=self._url(path),
                json=json_body,
                stream=stream,
                timeout=self.timeout,
                headers=headers,
            )
        except requests.exceptions.Timeout as exc:
            raise GatewayTimeoutError(
                f"request to {method} {path} timed out after {self.timeout}s"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise GatewayConnectionError(
                f"cannot connect to {self.base_url} ({method} {path}): {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise GatewayError(
                f"request to {method} {path} failed: {exc}"
            ) from exc

        if not (200 <= response.status_code < 300):
            self._raise_for_status(response)
        return response

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """Translate a non-2xx response into a :class:`GatewayError`."""
        message = f"gateway returned HTTP {response.status_code}"
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and "message" in err:
                message = str(err["message"])
            elif "message" in payload:
                message = str(payload["message"])
        elif response.text:
            message = response.text.strip() or message
        raise GatewayError(message, status_code=response.status_code)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request.

        The returned :class:`ChatCompletionResponse` carries a
        :attr:`route_trace` attribute populated from the
        ``X-Hier-KV-Gateway-*`` response headers describing which backend /
        strategy / KV overlap the gateway selected.
        """
        # Always force stream=False on the wire for the non-streaming path,
        # regardless of what the caller set on the request object.
        payload = json.loads(request.model_dump_json(by_alias=True, exclude_none=True))
        payload["stream"] = False
        response = self._request(
            "POST",
            "/v1/chat/completions",
            json_body=payload,
        )
        body = response.json()
        result = ChatCompletionResponse.model_validate(body)
        result.route_trace = RouteTrace.from_headers(response.headers)
        return result

    def chat_completions_stream(
        self, request: ChatCompletionRequest
    ) -> Iterator[ChatCompletionChunk]:
        """Send a streaming chat completion request.

        Yields :class:`ChatCompletionChunk` objects parsed from the SSE
        ``data:`` lines. The generator stops after consuming the
        ``data: [DONE]`` sentinel. Route-trace headers are emitted on the
        first yielded chunk's ``route_trace`` attribute (and ``None`` on
        subsequent chunks).
        """
        payload = json.loads(request.model_dump_json(by_alias=True, exclude_none=True))
        payload["stream"] = True
        response = self._request(
            "POST",
            "/v1/chat/completions",
            json_body=payload,
            stream=True,
            extra_headers={"Accept": "text/event-stream"},
        )
        route_trace = RouteTrace.from_headers(response.headers)
        first = True
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", errors="replace")
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(":"):
                    # SSE comment / keep-alive.
                    continue
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                elif line.startswith("data: "):
                    data = line[len("data: "):].strip()
                else:
                    # Ignore non-data SSE fields (event:, id:, retry:).
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
            # Release the underlying connection back to the pool.
            response.close()

    def list_models(self) -> ModelsListResponse:
        """List models served by all registered backends (``GET /v1/models``)."""
        response = self._request("GET", "/v1/models")
        return ModelsListResponse.model_validate(response.json())

    def health(self) -> HealthStatus:
        """Query gateway health (``GET /health``)."""
        response = self._request("GET", "/health")
        return HealthStatus.model_validate(response.json())

    def list_backends(self) -> list:
        """List all registered backends (``GET /admin/backends``)."""
        response = self._request("GET", "/admin/backends")
        data = response.json()
        if not isinstance(data, list):
            raise GatewayError(
                "unexpected /admin/backends payload: expected a JSON array",
                status_code=response.status_code,
            )
        return [BackendInfo.model_validate(item) for item in data]

    def get_topology(self) -> dict:
        """Fetch the cluster topology matrix (``GET /admin/topology``).

        Returns the raw JSON document as a dict. Note: this endpoint is
        documented but not yet implemented by the gateway; calls will raise
        :class:`GatewayError` with a 404 status until it is.
        """
        response = self._request("GET", "/admin/topology")
        return response.json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session and release connections."""
        if self._closed:
            return
        self._closed = True
        self._session.close()

    def __enter__(self) -> "HierKvGatewayClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
