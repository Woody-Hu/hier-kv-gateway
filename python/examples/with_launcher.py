"""Example: launch the Rust gateway from Python, use it, then stop it.

Demonstrates the :class:`GatewayLauncher` helper which spawns the
``hier-kv-gateway`` binary as a subprocess, polls ``/health`` until it is
ready, and tears it down cleanly on exit.

Prerequisite: build the Rust binary first::

    cargo build --release

Then run this example from the repository root::

    python3 python/examples/with_launcher.py
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

from hier_kv_gateway import (  # noqa: E402
    ChatCompletionRequest,
    GatewayError,
    GatewayLauncher,
    HierKvGatewayClient,
    Message,
)


def main() -> int:
    repo_root = os.path.dirname(_PKG_ROOT)

    # Default binary path assumes `cargo build --release` has been run.
    binary_path = os.environ.get(
        "GATEWAY_BINARY",
        os.path.join(repo_root, "target", "release", "hier-kv-gateway"),
    )
    config_path = os.environ.get(
        "GATEWAY_CONFIG",
        os.path.join(repo_root, "examples", "hier-kv-gateway.toml"),
    )
    listen_port = int(os.environ.get("GATEWAY_PORT", "8080"))
    base_url = f"http://localhost:{listen_port}"

    launcher = GatewayLauncher(
        binary_path=binary_path,
        config_path=config_path,
        base_url=base_url,
        # Pass the listen port through the env if needed by the config;
        # the example config already binds 0.0.0.0:8080.
        env=None,
    )

    try:
        launcher.start()
    except GatewayError as exc:
        print(f"[launcher] failed to start gateway: {exc}", file=sys.stderr)
        print(
            "[launcher] hint: run `cargo build --release` first, or set "
            "GATEWAY_BINARY to the absolute path of the binary.",
            file=sys.stderr,
        )
        return 1

    # Poll /health until the gateway is ready (or the timeout elapses).
    if not launcher.wait_for_ready(timeout=15.0):
        print("[launcher] gateway did not become ready in time", file=sys.stderr)
        launcher.stop()
        return 1
    print("[launcher] gateway is ready")

    try:
        with HierKvGatewayClient(base_url=base_url, timeout=60.0) as client:
            health = client.health()
            print(f"[health] {health.status}")

            models = client.list_models()
            print(f"[models] {len(models.data)} model(s) available")

            request = ChatCompletionRequest(
                model=os.environ.get("GATEWAY_MODEL", "qwen2.5-7b"),
                messages=[
                    Message(role="user", content="Reply with the single word: pong"),
                ],
                temperature=0.0,
                max_tokens=16,
                session_id="launcher-demo",
            )
            resp = client.chat_completions(request)
            content = resp.choices[0].message.content if resp.choices else ""
            print(f"[chat] response: {content!r}")
            if resp.route_trace is not None:
                print(
                    f"[route] backend={resp.route_trace.backend} "
                    f"strategy={resp.route_trace.strategy} "
                    f"kv_overlap={resp.route_trace.kv_overlap}"
                )
    except GatewayError as exc:
        print(f"[client] error: {exc}", file=sys.stderr)
        return 1
    finally:
        launcher.stop()
        print("[launcher] gateway stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
