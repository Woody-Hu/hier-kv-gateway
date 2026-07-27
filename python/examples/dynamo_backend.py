"""Example: run a Python Dynamo worker and emit gateway backend config.

This example starts a Python-side Dynamo worker that listens on NATS
subjects ``dyn.health.<id>`` and ``dyn.generate.<id>``, replies to health
pings, and forwards generate requests to a user-supplied handler. It also
prints the TOML ``[[backends]]`` snippet to splice into the gateway config.

The default handler is a stub that replies with a fixed string. To plug in
a real model, subclass :class:`DynamoWorker` or pass a custom
``generate_handler`` to :class:`DynamoWorker` directly — see the body of
this example for the pattern.

Prerequisites:

* A NATS server reachable at ``--nats-url`` (e.g. ``nats://127.0.0.1:4222``).
  Run one locally with ``docker run -p 4222:4222 nats:latest``.
* The ``nats-py`` package: ``pip install nats-py``.
* A Hier KV Gateway binary built with the ``dynamo`` connector feature:
  ``cargo build --release --features hier-kv-gateway-connector/dynamo``.

Run from the repository root::

    python3 python/examples/dynamo_backend.py \\
        --instance-id python-worker \\
        --nats-url nats://127.0.0.1:4222 \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow running this example directly from a source checkout without
# installing the package: prepend python/ to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from hier_kv_gateway.integrations.dynamo import (  # noqa: E402
    DynamoWorker,
    DynamoWorkerConfig,
    make_backend_config,
    make_dummy_generate_handler,
)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a Python Dynamo worker for the Hier KV Gateway.")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        required=True,
        help="Model id(s) served by this worker. Can be repeated.",
    )
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--region", default="cloud-cn-beijing")
    parser.add_argument(
        "--reply-text",
        default="hello from python dynamo worker",
        help="Text the stub handler returns. Ignored if --real-handler is set.",
    )
    args = parser.parse_args()

    config = DynamoWorkerConfig(
        instance_id=args.instance_id,
        nats_url=args.nats_url,
        models=args.model,
        kv_block_size=args.kv_block_size,
        region=args.region,
    )

    print("[dynamo] gateway backend config:")
    print(make_backend_config(config))

    worker = DynamoWorker(
        config,
        generate_handler=make_dummy_generate_handler(
            reply_text=args.reply_text,
            backend_id={
                "region": args.region,
                "instance": args.instance_id,
            },
        ),
    )

    async def _publish_demo_kv_event() -> None:
        """Publish a demo KV-cache event every 10 seconds.

        In a real worker, you'd publish events whenever the underlying
        engine stored / removed KV-cache blocks. Here we emit a synthetic
        "stored" event so the gateway's indexer can be observed picking
        it up.
        """
        event = {
            "type": "stored",
            "worker": {"worker_id": 0, "dp_rank": 0},
            "block_hashes": [1234567890],
            "parent_hash": None,
            "num_block_tokens": [args.kv_block_size],
        }
        while worker.is_running:
            try:
                await worker.publish_kv_event(event)
                print(f"[dynamo] published KV event: {event}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[dynamo] KV event publish failed: {exc}", file=sys.stderr, flush=True)
            await asyncio.sleep(10.0)

    async def _run() -> int:
        try:
            await worker.start()
        except ImportError as exc:
            print(f"[dynamo] {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"[dynamo] failed to start: {exc}", file=sys.stderr)
            return 1
        print(f"[dynamo] worker {args.instance_id!r} listening on {args.nats_url}")
        print("[dynamo] press Ctrl-C to stop")

        kv_task = asyncio.create_task(_publish_demo_kv_event())
        try:
            while worker.is_running:
                await asyncio.sleep(1.0)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            kv_task.cancel()
            await worker.stop()
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
