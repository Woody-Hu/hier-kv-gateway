"""Basic usage example for the Hier KV Gateway Python SDK.

Demonstrates:
  * Creating a synchronous client
  * Listing models
  * Sending a non-streaming chat completion request
  * Sending a streaming chat completion request
  * Inspecting the route trace (which backend / strategy / KV overlap)

Prerequisite: a Hier KV Gateway instance must be running and reachable at
``GATEWAY_URL`` (defaults to http://localhost:8080).

    # Build the Rust binary first:
    cargo build --release
    # Run it with the example config:
    ./target/release/hier-kv-gateway --config examples/hier-kv-gateway.toml
    # Then run this example from the repository root:
    python3 python/examples/basic_usage.py
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

from hier_kv_gateway import (  # noqa: E402  (import after sys.path tweak)
    ChatCompletionRequest,
    GatewayError,
    HierKvGatewayClient,
    Message,
)


def main() -> int:
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8080")
    model_name = os.environ.get("GATEWAY_MODEL", "qwen2.5-7b")

    with HierKvGatewayClient(base_url=gateway_url, timeout=60.0) as client:
        # 1) Health check
        try:
            health = client.health()
            print(f"[health] status={health.status}")
        except GatewayError as exc:
            print(f"[health] gateway unreachable: {exc}", file=sys.stderr)
            return 1

        # 2) List models
        try:
            models = client.list_models()
            print(f"[models] count={len(models.data)}")
            for m in models.data[:5]:
                owned = m.owned_by or "-"
                print(f"  - {m.id}  (owned_by={owned})")
        except GatewayError as exc:
            print(f"[models] failed: {exc}", file=sys.stderr)

        # 3) Non-streaming chat completion
        request = ChatCompletionRequest(
            model=model_name,
            messages=[
                Message(role="system", content="You are a concise assistant."),
                Message(role="user", content="Say hello in one short sentence."),
            ],
            temperature=0.3,
            max_tokens=64,
            # Hier KV Gateway extension: pin the session for affinity routing.
            session_id="demo-session-1",
        )

        try:
            resp = client.chat_completions(request)
        except GatewayError as exc:
            print(f"[chat] request failed: {exc}", file=sys.stderr)
            return 1

        content = resp.choices[0].message.content if resp.choices else ""
        print(f"[chat] id={resp.id} model={resp.model}")
        print(f"[chat] content={content!r}")
        print(f"[chat] usage={resp.usage}")

        # Route trace: which backend was selected, and the KV-cache overlap
        # that influenced the decision. Populated from X-Hier-KV-Gateway-*
        # response headers.
        if resp.route_trace is not None:
            rt = resp.route_trace
            print(
                f"[route] backend={rt.backend} strategy={rt.strategy} "
                f"kv_overlap={rt.kv_overlap} region={rt.region}"
            )

        # 4) Streaming chat completion
        print("[stream] starting...")
        stream_request = request.model_copy(update={"stream": True, "max_tokens": 48})
        collected = []
        try:
            for chunk in client.chat_completions_stream(stream_request):
                for choice in chunk.choices:
                    if choice.delta.content:
                        collected.append(choice.delta.content)
                        # Echo tokens as they arrive.
                        sys.stdout.write(choice.delta.content)
                        sys.stdout.flush()
                    if choice.finish_reason:
                        print(f"\n[stream] finish_reason={choice.finish_reason}")
                if chunk.route_trace is not None:
                    print(
                        f"[stream][route] backend={chunk.route_trace.backend} "
                        f"strategy={chunk.route_trace.strategy} "
                        f"kv_overlap={chunk.route_trace.kv_overlap}"
                    )
        except GatewayError as exc:
            print(f"\n[stream] failed: {exc}", file=sys.stderr)
            return 1

        print(f"[stream] collected {len(collected)} delta(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
