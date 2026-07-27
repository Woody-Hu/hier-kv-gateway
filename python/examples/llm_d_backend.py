"""Example: discover LLM-D services and emit gateway backend config.

This example wraps the :mod:`hier_kv_gateway.integrations.llm_d` adapter
in a friendly CLI. Two modes are supported:

1. **Kubernetes discovery** (default): queries the cluster for services
   annotated with ``llm-d.ai/role=openai-api`` and prints a ``[[backends]]``
   block for each.
2. **Manual URL list** (``--from-url``): for testing without a K8s cluster.
   Pass one or more ``--from-url`` flags plus matching ``--model`` flags.

Examples::

    # Discover from the current K8s context:
    python3 python/examples/llm_d_backend.py \\
        --namespace llm-d --region cloud-cn-beijing

    # Bypass K8s and emit a manual block:
    python3 python/examples/llm_d_backend.py \\
        --from-url http://10.0.0.1:8080 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --kv-block-size 16

    # Save the snippet directly into a config fragment:
    python3 python/examples/llm_d_backend.py --from-url http://10.0.0.1:8080 \\
        --model Qwen/Qwen2.5-7B-Instruct > llm-d-backends.toml
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

from hier_kv_gateway.integrations.llm_d import (  # noqa: E402
    LlmDEndpoint,
    discover_from_kubernetes,
    render_gateway_config,
)
from hier_kv_gateway.exceptions import GatewayError  # noqa: E402


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Discover LLM-D services and emit gateway backend config.")
    parser.add_argument("--namespace", default=None, help="Kubernetes namespace.")
    parser.add_argument("--label-selector", default=None)
    parser.add_argument("--kubeconfig", default=None)
    parser.add_argument("--region", default="cloud-cn-beijing")
    parser.add_argument("--default-kv-block-size", type=int, default=16)
    parser.add_argument(
        "--from-url",
        action="append",
        default=[],
        help="Manually add an LLM-D endpoint URL. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model id(s) for --from-url endpoints (one per --from-url).",
    )
    parser.add_argument("--no-health-probe", action="store_true", help="Skip the /health probe.")
    args = parser.parse_args()

    if args.from_url:
        endpoints = []
        for i, url in enumerate(args.from_url):
            model = args.model[i] if i < len(args.model) else "unknown"
            endpoints.append(
                LlmDEndpoint(
                    name=f"manual-{i}",
                    base_url=url,
                    models=[model],
                    kv_block_size=args.default_kv_block_size,
                    region=args.region,
                )
            )
    else:
        try:
            endpoints = discover_from_kubernetes(
                namespace=args.namespace,
                kubeconfig=args.kubeconfig,
                label_selector=args.label_selector,
                region=args.region,
                default_kv_block_size=args.default_kv_block_size,
            )
        except ImportError as exc:
            print(f"[llm-d] {exc}", file=sys.stderr)
            return 2
        except GatewayError as exc:
            print(f"[llm-d] discovery failed: {exc}", file=sys.stderr)
            return 1

    if not endpoints:
        print("[llm-d] no LLM-D endpoints found", file=sys.stderr)
        return 1

    print(f"[llm-d] found {len(endpoints)} endpoint(s):")
    for ep in endpoints:
        if args.no_health_probe:
            healthy = "skipped"
        else:
            healthy = "OK" if ep.is_healthy() else "unreachable"
        print(f"  - {ep.name} @ {ep.base_url}  models={ep.models}  kv={ep.kv_block_size}  [{healthy}]")

    print("\n[llm-d] gateway backend config:")
    print(render_gateway_config(endpoints))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
