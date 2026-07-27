"""Example: launch a vLLM backend, then route a request through the gateway.

This example demonstrates the full vLLM integration workflow:

1. Build a :class:`VllmLaunchSpec` with a KV-block size matching the gateway.
2. Spawn the vLLM OpenAI-compatible server via :func:`launch_vllm`.
3. Print the TOML ``[[backends]]`` snippet to splice into the gateway config.
4. (Optional) Use the gateway's Python SDK to send a chat completion that
   gets routed to the freshly-launched vLLM backend.

Prerequisites:

* A vLLM installation reachable from the current Python interpreter
  (``pip install vllm``), OR set ``VLLM_PYTHON`` to a virtualenv that has it.
* A built ``hier-kv-gateway`` binary (``cargo build --release``).
* A GPU available to vLLM.

Run from the repository root::

    python3 python/examples/vllm_backend.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --port 8000 \\
        --kv-block-size 16
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
from hier_kv_gateway.integrations.vllm import (  # noqa: E402
    VllmLaunchSpec,
    launch_vllm,
    make_backend_config,
)


def main() -> int:
    import argparse

    repo_root = os.path.dirname(_PKG_ROOT)

    parser = argparse.ArgumentParser(description="Launch a vLLM backend for the Hier KV Gateway.")
    parser.add_argument(
        "--model",
        default=os.environ.get("GATEWAY_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="HuggingFace model id or local path.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument("--region", default="cloud-cn-beijing")
    parser.add_argument(
        "--skip-gateway",
        action="store_true",
        help="Only launch vLLM and print the config; do not start the gateway.",
    )
    args, _vllm_extra = parser.parse_known_args()

    spec = VllmLaunchSpec(
        model=args.model,
        kv_block_size=args.kv_block_size,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    print(f"[vllm] launching {args.model!r} on port {args.port} (kv_block_size={args.kv_block_size})")
    with launch_vllm(spec) as vllm_proc:
        print(f"[vllm] pid={vllm_proc.pid}; waiting for /health")
        if not vllm_proc.wait_for_ready(timeout=args.ready_timeout):
            print("[vllm] server did not become healthy in time", file=sys.stderr)
            return 1
        print(f"[vllm] ready at {vllm_proc.base_url}")

        backend_config = make_backend_config(spec, region=args.region)
        print("\n[vllm] append this snippet to your gateway TOML:")
        print(backend_config)

        if args.skip_gateway:
            print("\n[vllm] --skip-gateway set; keeping vLLM alive. Press Ctrl-C to stop.")
            try:
                import time
                while vllm_proc.is_running():
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
            return 0

        # --- Optionally run the gateway and route a request to vLLM -------
        binary_path = os.environ.get(
            "GATEWAY_BINARY",
            os.path.join(repo_root, "target", "release", "hier-kv-gateway"),
        )
        config_path = os.environ.get(
            "GATEWAY_CONFIG",
            os.path.join(repo_root, "examples", "hier-kv-gateway.toml"),
        )
        gateway_port = int(os.environ.get("GATEWAY_PORT", "8080"))
        gateway_url = f"http://localhost:{gateway_port}"

        print(f"\n[gateway] starting gateway binary={binary_path!r}")
        print(f"[gateway] config={config_path!r}")
        print("[gateway] (make sure the config's [[backends]] block matches the vLLM URL above)")
        launcher = GatewayLauncher(
            binary_path=binary_path,
            config_path=config_path,
            base_url=gateway_url,
        )
        try:
            launcher.start()
        except GatewayError as exc:
            print(f"[gateway] failed to start: {exc}", file=sys.stderr)
            return 1

        if not launcher.wait_for_ready(timeout=15.0):
            print("[gateway] did not become ready in time", file=sys.stderr)
            launcher.stop()
            return 1
        print("[gateway] ready")

        try:
            with HierKvGatewayClient(base_url=gateway_url, timeout=120.0) as client:
                req = ChatCompletionRequest(
                    model=args.model,
                    messages=[
                        Message(role="user", content="Reply with the single word: pong"),
                    ],
                    temperature=0.0,
                    max_tokens=16,
                    session_id="vllm-backend-demo",
                )
                resp = client.chat_completions(req)
                content = resp.choices[0].message.content if resp.choices else ""
                print(f"[chat] response: {content!r}")
                if resp.route_trace is not None:
                    print(
                        f"[route] backend={resp.route_trace.backend} "
                        f"strategy={resp.route_trace.strategy} "
                        f"kv_overlap={resp.route_trace.kv_overlap}"
                    )
        except GatewayError as exc:
            print(f"[chat] error: {exc}", file=sys.stderr)
            return 1
        finally:
            launcher.stop()
            print("[gateway] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
