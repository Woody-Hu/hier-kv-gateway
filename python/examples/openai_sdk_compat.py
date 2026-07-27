"""Example: use the official ``openai`` Python SDK against the gateway.

This example shows how to use the OpenAI SDK adapter
(:mod:`hier_kv_gateway.integrations.openai_sdk`) to send chat completions
to the gateway with session-affinity routing enabled. It mirrors the basic
SDK usage but adds the gateway-specific ``session`` field via
``extra_body``.

Prerequisites:

* The ``openai`` package: ``pip install openai``.
* A running Hier KV Gateway at ``--base-url`` (default ``http://localhost:8080/v1``).
* At least one backend registered with the gateway (see other examples).

Run from the repository root::

    python3 python/examples/openai_sdk_compat.py \\
        --base-url http://localhost:8080/v1 \\
        --model qwen2.5-7b \\
        --session-id openai-sdk-demo
"""

from __future__ import annotations

import os
import sys

# Allow running this example directly from a source checkout without
# installing the package: prepend python/ to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from hier_kv_gateway.integrations.openai_sdk import (  # noqa: E402
    chat_completion_with_session,
    extract_route_trace,
    make_openai_client,
)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Use the openai SDK against the Hier KV Gateway.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--session-id", default="openai-sdk-demo")
    parser.add_argument("--prompt", default="Reply with the single word: pong")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true", default=False)
    args = parser.parse_args()

    try:
        client = make_openai_client(base_url=args.base_url)
    except ImportError as exc:
        print(f"[openai] {exc}", file=sys.stderr)
        return 2

    print(f"[openai] base_url={args.base_url} model={args.model} session={args.session_id}")
    print(f"[openai] prompt: {args.prompt!r}")

    try:
        if args.stream:
            print("[openai] streaming response:")
            stream = chat_completion_with_session(
                client,
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                session_id=args.session_id,
                stream=True,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
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
            print(f"[openai] collected {len(collected)} delta(s)")
            if last_chunk is not None:
                rt = extract_route_trace(last_chunk)
                print(
                    f"[route] backend={rt.backend} strategy={rt.strategy} "
                    f"kv_overlap={rt.kv_overlap}"
                )
        else:
            resp = chat_completion_with_session(
                client,
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                session_id=args.session_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            print(f"[openai] response: {content!r}")
            rt = extract_route_trace(resp)
            print(
                f"[route] backend={rt.backend} strategy={rt.strategy} "
                f"kv_overlap={rt.kv_overlap}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[openai] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
