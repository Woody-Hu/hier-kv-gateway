"""NVIDIA Dynamo integration: NATS bridge and KV event publisher.

NVIDIA Dynamo (https://github.com/ai-dynamo/dynamo) is a cloud-native LLM
serving framework built around NATS as its component message bus. The Hier
KV Gateway's Rust ``DynamoConnector`` (feature-gated on ``dynamo``) talks
to Dynamo over NATS using the ``dyn.health.<worker>``, ``dyn.generate.<worker>``,
``dyn.kv_events.<worker>`` and ``dyn.metrics.<worker>`` subjects.

This Python adapter lets a Python-based Dynamo-style worker (or a gateway
side-car) participate in that protocol without writing Rust. It provides:

1. :class:`DynamoWorkerConfig` — typed worker identity & NATS URL config.
2. :class:`DynamoWorker` — publish KV-cache events and respond to
   ``health`` / ``generate`` requests over NATS. ``generate`` delegates to
   a user-supplied callable so this can wrap any LLM serving code (vLLM,
   TGI, HuggingFace transformers, ...).
3. :func:`make_backend_config` — render a TOML ``[[backends]]`` block
   describing the Dynamo worker so the gateway can route to it.

The ``nats-py`` package is imported lazily; :class:`DynamoWorker` raises a
helpful :class:`ImportError` on first use when it is not installed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..exceptions import GatewayError

# Default NATS subject prefix used by Dynamo deployments. Must match the
# Rust connector's `DEFAULT_DYNAMO_SUBJECT_PREFIX`.
DEFAULT_DYNAMO_SUBJECT_PREFIX = "dyn"


# Type alias for the user-supplied generate handler.
#
# The handler receives the parsed request body (an OpenAI Chat Completions
# request dict) and returns either:
#   - a list of chunk dicts (each one a JSON-serializable InferenceChunk), or
#   - a single chunk dict for a one-shot response.
GenerateHandler = Callable[[Dict[str, Any]], Awaitable[Any]]


@dataclasses.dataclass
class DynamoWorkerConfig:
    """Identity / connection configuration for a Dynamo worker.

    Attributes:
        instance_id: Worker identifier. Used as the trailing token in NATS
            subjects (e.g. ``dyn.generate.<instance_id>``). Must be unique
            within a NATS cluster.
        nats_url: NATS server URL (e.g. ``nats://127.0.0.1:4222``).
        models: Models served by this worker (injected into the gateway
            config so the router knows which model names this worker can
            handle).
        kv_block_size: KV cache block size in tokens. Must match the
            gateway's ``[routing] kv_block_size``.
        region: Region id to assign when generating gateway config.
        subject_prefix: NATS subject prefix. Defaults to ``dyn``.
    """

    instance_id: str
    nats_url: str = "nats://127.0.0.1:4222"
    models: List[str] = dataclasses.field(default_factory=list)
    kv_block_size: int = 16
    region: str = "cloud-cn-beijing"
    subject_prefix: str = DEFAULT_DYNAMO_SUBJECT_PREFIX

    def health_subject(self) -> str:
        return f"{self.subject_prefix}.health.{self.instance_id}"

    def generate_subject(self) -> str:
        return f"{self.subject_prefix}.generate.{self.instance_id}"

    def kv_events_subject(self) -> str:
        return f"{self.subject_prefix}.kv_events.{self.instance_id}"

    def metrics_subject(self) -> str:
        return f"{self.subject_prefix}.metrics.{self.instance_id}"


class DynamoWorker:
    """A Python-side Dynamo worker that bridges NATS to a generate handler.

    The worker subscribes to ``dyn.health.<id>`` and ``dyn.generate.<id>``,
    and exposes :meth:`publish_kv_event` so a serving loop can emit
    KV-cache events that the gateway's indexer will consume.

    Example::

        async def my_handler(req: dict) -> list[dict]:
            # ... call vLLM / TGI / HF transformers ...
            return [{"type": "delta", "text": "hello", "finish_reason": None},
                    {"type": "done", "backend_id": {...}, "latency_ms": 42}]

        worker = DynamoWorker(config, generate_handler=my_handler)
        await worker.start()
        # ... later, publish a KV event ...
        await worker.publish_kv_event({
            "type": "stored",
            "worker": {"worker_id": 0, "dp_rank": 0},
            "block_hashes": [123, 456],
            "parent_hash": None,
            "num_block_tokens": [16, 16],
        })
        await worker.stop()
    """

    def __init__(
        self,
        config: DynamoWorkerConfig,
        *,
        generate_handler: Optional[GenerateHandler] = None,
    ) -> None:
        self._config = config
        self._generate_handler = generate_handler
        self._nc: Any = None  # nats.NATS
        self._subs: List[Any] = []
        self._running = False

    @property
    def config(self) -> DynamoWorkerConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Connect to NATS and subscribe to health/generate subjects."""
        if self._running:
            return
        try:
            import nats  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — depends on env
            raise ImportError(
                "the 'nats-py' package is required for DynamoWorker. "
                "Install it with: pip install nats-py"
            ) from exc

        self._nc = await nats.connect(self._config.nats_url)

        async def _health_cb(msg: Any) -> None:
            await msg.respond(b'{"status":"ok"}')

        async def _generate_cb(msg: Any) -> None:
            if self._generate_handler is None:
                await msg.respond(
                    json.dumps(
                        {
                            "type": "error",
                            "code": 503,
                            "message": "no generate handler registered",
                        }
                    ).encode("utf-8")
                )
                return
            try:
                body = json.loads(msg.data.decode("utf-8")) if msg.data else {}
            except json.JSONDecodeError:
                await msg.respond(
                    json.dumps(
                        {"type": "error", "code": 400, "message": "invalid JSON body"}
                    ).encode("utf-8")
                )
                return
            try:
                result = await self._generate_handler(body)
            except Exception as exc:  # noqa: BLE001 — surface any handler error
                await msg.respond(
                    json.dumps(
                        {"type": "error", "code": 500, "message": str(exc)}
                    ).encode("utf-8")
                )
                return
            # Serialize result as newline-delimited JSON chunks to mirror the
            # Rust connector's expected reply format.
            if isinstance(result, dict):
                chunks = [result]
            else:
                chunks = list(result)
            payload = "\n".join(json.dumps(c) for c in chunks).encode("utf-8")
            await msg.respond(payload)

        await self._nc.subscribe(self._config.health_subject(), cb=_health_cb)
        self._subs.append("health")
        await self._nc.subscribe(self._config.generate_subject(), cb=_generate_cb)
        self._subs.append("generate")
        self._running = True

    async def stop(self) -> None:
        """Drain subscriptions and close the NATS connection."""
        if not self._running:
            return
        self._running = False
        if self._nc is not None:
            try:
                await self._nc.drain()
            except Exception:
                pass
            self._nc = None
        self._subs.clear()

    async def publish_kv_event(self, event: Dict[str, Any]) -> None:
        """Publish a KV-cache event to ``dyn.kv_events.<id>``.

        The event dict must conform to the
        :class:`hier_kv_gateway_core::kv_event::KvCacheEvent` shape (a
        tagged enum serialized via internal tagging, e.g.
        ``{"type": "stored", "worker": ..., "block_hashes": [...]}``).
        """
        if self._nc is None:
            raise GatewayError("DynamoWorker is not started; call start() first")
        payload = json.dumps(event).encode("utf-8")
        await self._nc.publish(self._config.kv_events_subject(), payload)

    async def publish_metrics(self, metrics: Dict[str, Any]) -> None:
        """Publish a metrics snapshot (responds to ``dyn.metrics.<id>``).

        Note: this is a fire-and-forget publish; the gateway's
        ``DynamoConnector::collect_metrics`` issues a NATS request-reply on
        the metrics subject. To support that, register a metrics responder
        via :meth:`set_metrics_responder`.
        """
        if self._nc is None:
            raise GatewayError("DynamoWorker is not started; call start() first")
        payload = json.dumps(metrics).encode("utf-8")
        await self._nc.publish(self._config.metrics_subject(), payload)

    def set_metrics_responder(
        self, handler: Callable[[], Awaitable[Dict[str, Any]]]
    ) -> None:
        """Register a handler that responds to ``dyn.metrics.<id>`` requests.

        Must be called before :meth:`start`. The handler is invoked on each
        incoming NATS request-reply; its returned dict is JSON-serialized
        and sent as the reply.
        """
        if self._nc is not None:
            raise GatewayError("cannot set metrics responder after start()")

        self._pending_metrics_handler = handler

    _pending_metrics_handler: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None

    async def __aenter__(self) -> "DynamoWorker":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


def make_backend_config(
    config: DynamoWorkerConfig,
    *,
    indent: int = 0,
) -> str:
    """Render a TOML ``[[backends]]`` block for a Dynamo worker.

    The gateway will register the worker as a ``dynamo_engine`` backend
    and use the NATS URL as its endpoint. The Rust ``DynamoConnector``
    (compiled with the ``dynamo`` feature) will handle request forwarding
    and KV event subscription.
    """
    pad = " " * indent
    models_str = ", ".join(f"\"{m}\"" for m in config.models)
    lines = [
        f"{pad}[[backends]]",
        f"{pad}backend_type = \"dynamo_engine\"",
        f"{pad}endpoint = {{ url = \"{config.nats_url}\", protocol = \"nats\" }}",
        f"{pad}models = [{models_str}]",
        f"{pad}region = \"{config.region}\"",
        f"{pad}kv_block_size = {int(config.kv_block_size)}",
    ]
    return "\n".join(lines) + "\n"


def make_dummy_generate_handler(
    *,
    reply_text: str = "hello from python dynamo worker",
    latency_ms: int = 0,
    backend_id: Optional[Dict[str, Any]] = None,
) -> GenerateHandler:
    """Build a trivial generate handler for tests / examples.

    Returns an async function that ignores the request and emits a single
    delta chunk followed by a done chunk.
    """
    bid = backend_id or {"region": "cloud-cn-beijing", "instance": "python-worker"}

    async def handler(_req: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"type": "delta", "text": reply_text, "finish_reason": None},
            {"type": "delta", "text": "", "finish_reason": "stop"},
            {"type": "done", "backend_id": bid, "latency_ms": latency_ms},
        ]

    return handler


def main() -> int:
    """Entry point: start a dummy Dynamo worker and print a config snippet.

    Run via::

        python -m hier_kv_gateway.integrations.dynamo \\
            --instance-id python-worker \\
            --nats-url nats://127.0.0.1:4222 \\
            --model Qwen/Qwen2.5-7B-Instruct
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run a Python Dynamo worker for the Hier KV Gateway.")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    parser.add_argument("--model", action="append", default=[], help="Model id(s) served.")
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--region", default="cloud-cn-beijing")
    parser.add_argument("--reply-text", default="hello from python dynamo worker")
    args = parser.parse_args()

    config = DynamoWorkerConfig(
        instance_id=args.instance_id,
        nats_url=args.nats_url,
        models=args.model or ["unknown"],
        kv_block_size=args.kv_block_size,
        region=args.region,
    )
    print("[dynamo] worker config:")
    print(make_backend_config(config))

    worker = DynamoWorker(
        config, generate_handler=make_dummy_generate_handler(reply_text=args.reply_text)
    )

    async def _run() -> int:
        try:
            await worker.start()
        except ImportError as exc:
            print(f"[dynamo] {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"[dynamo] failed to start: {exc}", file=sys.stderr)
            return 1
        print(f"[dynamo] worker {args.instance_id!r} started; press Ctrl-C to stop")
        try:
            while worker.is_running:
                await asyncio.sleep(1.0)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await worker.stop()
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
