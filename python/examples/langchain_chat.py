"""Example: use the gateway as a LangChain ``ChatModel``.

This example wraps the LangChain adapter
(:mod:`hier_kv_gateway.integrations.langchain`) and runs a multi-turn
conversation against the gateway. It demonstrates:

* Building a :class:`HierKvGatewayChatModel` with session-affinity enabled.
* Mixing system / human / AI messages via LangChain primitives.
* Inspecting the route trace after each call.
* (Optionally) streaming the response token-by-token.

Prerequisites:

* The ``langchain-openai`` (and ``langchain-core``) packages:
  ``pip install langchain-openai``.
* A running Hier KV Gateway at ``--base-url`` (default ``http://localhost:8080/v1``).
* At least one backend registered with the gateway (see other examples).

Run from the repository root::

    python3 python/examples/langchain_chat.py \\
        --base-url http://localhost:8080/v1 \\
        --model qwen2.5-7b \\
        --session-id langchain-demo
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

from hier_kv_gateway.integrations.langchain import (  # noqa: E402
    HierKvGatewayChatModel,
)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Use LangChain against the Hier KV Gateway.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--session-id", default="langchain-demo")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--stream", action="store_true", default=False)
    args = parser.parse_args()

    try:
        from langchain_core.messages import (  # type: ignore[import-not-found]  # noqa: E402
            AIMessage,
            HumanMessage,
            SystemMessage,
        )
    except ImportError as exc:
        print(f"[langchain] {exc}", file=sys.stderr)
        return 2

    try:
        model = HierKvGatewayChatModel(
            base_url=args.base_url,
            model=args.model,
            session_id=args.session_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            streaming=args.stream,
        )
    except ImportError as exc:
        print(f"[langchain] {exc}", file=sys.stderr)
        return 2

    print(f"[langchain] base_url={args.base_url} model={args.model} session={args.session_id}")

    # A tiny two-turn conversation.
    turns = [
        [
            SystemMessage(content="You are a concise assistant."),
            HumanMessage(content="Reply with the single word: pong"),
        ],
        [
            AIMessage(content="pong"),
            HumanMessage(content="Now reply with: ping"),
        ],
    ]

    for i, messages in enumerate(turns, start=1):
        print(f"\n[langchain] turn {i}:")
        for m in messages:
            print(f"  {m.type}: {m.content}")
        try:
            if args.stream:
                sys.stdout.write("[langchain] AI (streamed): ")
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
                print(f"[langchain] AI: {content!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"[langchain] error: {exc}", file=sys.stderr)
            return 1

        rt = model.last_route_trace
        if rt is not None:
            print(
                f"[route] backend={rt.backend} strategy={rt.strategy} "
                f"kv_overlap={rt.kv_overlap}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
