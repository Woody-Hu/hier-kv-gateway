"""Exception hierarchy for the Hier KV Gateway Python SDK.

All errors raised by the SDK derive from :class:`GatewayError` so callers can
catch gateway-related failures with a single ``except`` clause while still
distinguishing connection / timeout / HTTP failures when useful.
"""

from __future__ import annotations

from typing import Optional


class GatewayError(Exception):
    """Base exception for all errors raised by the Hier KV Gateway SDK.

    Attributes:
        message: Human-readable description of the failure.
        status_code: HTTP status code returned by the gateway, when the error
            originated from an HTTP response. ``None`` for transport-level
            failures (connection refused, DNS, timeout, etc.).
    """

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        if self.status_code is not None:
            return f"[{self.status_code}] {self.message}"
        return self.message


class GatewayConnectionError(GatewayError):
    """Raised when the SDK cannot establish or maintain a network connection.

    Typical causes: the gateway is not running, the wrong ``base_url`` was
    supplied, DNS resolution failed, or the connection was reset.
    """


class GatewayTimeoutError(GatewayError):
    """Raised when a request to the gateway exceeds the configured timeout.

    The :attr:`status_code` attribute is ``None`` because timeouts are
    client-side; the gateway never had a chance to respond.
    """
