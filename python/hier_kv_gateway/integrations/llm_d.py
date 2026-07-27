"""LLM-D integration: register an LLM-D cluster as a gateway backend.

LLM-D (https://github.com/llm-d/llm-d) is a Kubernetes-native, highly
distributed LLM inference framework. Each LLM-D deployment exposes an
OpenAI-compatible HTTP endpoint through its prefill/decode disaggregated
workers, behind a Kubernetes Service.

Because LLM-D already speaks the OpenAI API, the gateway treats it as a
``llm_d_cluster`` backend type — see
:class:`hier_kv_gateway_core::backend::BackendType::LlmDCluster`. The
gateway's OpenAI-compatible connector handles request forwarding; this
Python adapter focuses on:

1. :class:`LlmDEndpoint` — typed description of an LLM-D Service endpoint
   (URL, models, KV block size, optional K8s context).
2. :func:`discover_from_kubernetes` — auto-discover LLM-D Services in a
   cluster by looking for services annotated with
   ``llm-d.ai/role=openai-api``.
3. :func:`make_backend_config` — render a TOML ``[[backends]]`` block for
   each discovered endpoint.
4. :func:`render_gateway_config` — render a complete multi-backend
   ``[[backends]]`` section suitable for splicing into a gateway TOML.

The adapter imports ``kubernetes`` lazily; if the package is not installed,
:func:`discover_from_kubernetes` raises a clear :class:`ImportError`.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, List, Optional, Sequence

import requests

from ..exceptions import GatewayError

# Kubernetes Service annotation marking an LLM-D OpenAI-compatible endpoint.
LLM_D_ROLE_ANNOTATION = "llm-d.ai/role"
LLM_D_ROLE_OPENAI_API = "openai-api"

# Service annotation exposing the served model list (JSON array of strings).
LLM_D_MODELS_ANNOTATION = "llm-d.ai/models"

# Service annotation exposing the KV cache block size in tokens.
LLM_D_KV_BLOCK_SIZE_ANNOTATION = "llm-d.ai/kv-block-size"


@dataclasses.dataclass
class LlmDEndpoint:
    """A discovered LLM-D OpenAI-compatible endpoint.

    Attributes:
        name: Kubernetes Service name (or arbitrary identifier when not
            discovered from K8s).
        base_url: HTTP(S) URL of the OpenAI-compatible API.
        models: List of model ids served by this endpoint.
        kv_block_size: KV cache block size in tokens. Defaults to 16 when
            not declared by annotations.
        namespace: Kubernetes namespace, when discovered via K8s.
        region: Region id to assign when generating gateway config.
    """

    name: str
    base_url: str
    models: List[str]
    kv_block_size: int = 16
    namespace: Optional[str] = None
    region: str = "cloud-cn-beijing"

    def health_url(self) -> str:
        """Default ``/health`` URL for the endpoint."""
        return f"{self.base_url.rstrip('/')}/health"

    def is_healthy(self, timeout: float = 3.0) -> bool:
        """Probe ``/health`` once. Returns ``False`` on any error."""
        try:
            resp = requests.get(self.health_url(), timeout=timeout)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False


def discover_from_kubernetes(
    *,
    namespace: Optional[str] = None,
    kubeconfig: Optional[str] = None,
    label_selector: Optional[str] = None,
    region: str = "cloud-cn-beijing",
    default_kv_block_size: int = 16,
) -> List[LlmDEndpoint]:
    """Discover LLM-D OpenAI-compatible Services in a Kubernetes cluster.

    Args:
        namespace: Limit discovery to a single namespace. ``None`` searches
            across all namespaces (requires cluster-wide Service read
            permission).
        kubeconfig: Optional path to a kubeconfig file. ``None`` uses the
            in-cluster config or ``~/.kube/config``.
        label_selector: Optional Kubernetes label selector to narrow the
            search (e.g. ``"app=llm-d"``).
        region: Region id to assign to discovered endpoints.
        default_kv_block_size: Fallback KV block size when the Service does
            not declare one via annotations.

    Returns:
        A list of :class:`LlmDEndpoint` objects. The list may be empty if
        no matching services are found.

    Raises:
        ImportError: if the ``kubernetes`` Python package is not installed.
        GatewayError: if the Kubernetes API cannot be reached.
    """
    try:
        from kubernetes import client, config  # type: ignore[import-not-found]
        from kubernetes.client.rest import ApiException  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — depends on env
        raise ImportError(
            "the 'kubernetes' package is required for LLM-D discovery. "
            "Install it with: pip install kubernetes"
        ) from exc

    try:
        if kubeconfig is not None:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as exc:
        raise GatewayError(f"failed to load Kubernetes config: {exc}") from exc

    api = client.CoreV1Api()
    try:
        if namespace is None:
            services = api.list_service_for_all_namespaces(
                label_selector=label_selector
            )
        else:
            services = api.list_namespaced_service(
                namespace=namespace, label_selector=label_selector
            )
    except ApiException as exc:
        raise GatewayError(
            f"Kubernetes API error listing services: {exc.status} {exc.reason}"
        ) from exc

    endpoints: List[LlmDEndpoint] = []
    for svc in services.items:
        annotations = svc.metadata.annotations or {}
        role = annotations.get(LLM_D_ROLE_ANNOTATION)
        if role != LLM_D_ROLE_OPENAI_API:
            continue

        name = svc.metadata.name
        ns = svc.metadata.namespace

        # Determine the URL: prefer an explicit annotation, then a LoadBalancer
        # ingress, then fall back to ClusterIP:port.
        url = annotations.get("llm-d.ai/base-url")
        if not url:
            port = _pick_openai_port(svc.spec.ports)
            if svc.spec.type == "LoadBalancer" and svc.status.loadBalancer and \
               svc.status.loadBalancer.ingress:
                ingress = svc.status.loadBalancer.ingress[0]
                host = ingress.hostname or ingress.ip
            else:
                host = svc.spec.cluster_ip
            if not host:
                continue
            url = f"http://{host}:{port}"

        models = _parse_models_annotation(
            annotations.get(LLM_D_MODELS_ANNOTATION), fallback=name
        )
        kv_block_size = _parse_int_annotation(
            annotations.get(LLM_D_KV_BLOCK_SIZE_ANNOTATION),
            default=default_kv_block_size,
        )

        endpoints.append(
            LlmDEndpoint(
                name=name,
                base_url=url,
                models=models,
                kv_block_size=kv_block_size,
                namespace=ns,
                region=region,
            )
        )
    return endpoints


def _pick_openai_port(ports: Optional[List[Any]]) -> int:
    """Pick the port exposing the OpenAI API from a Service's port list."""
    if not ports:
        return 80
    # Prefer a port named "http" / "openai" / "api"; else the first.
    for p in ports:
        name = (p.name or "").lower()
        if name in ("http", "openai", "api"):
            return int(p.port)
    return int(ports[0].port)


def _parse_models_annotation(raw: Optional[str], *, fallback: str) -> List[str]:
    """Parse the ``llm-d.ai/models`` annotation (JSON array of strings)."""
    if not raw:
        return [fallback]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [fallback]
    if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
        return parsed
    return [fallback]


def _parse_int_annotation(raw: Optional[str], *, default: int) -> int:
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def make_backend_config(
    endpoint: LlmDEndpoint,
    *,
    indent: int = 0,
) -> str:
    """Render a TOML ``[[backends]]`` block for an LLM-D endpoint.

    The gateway will register the endpoint as a ``llm_d_cluster`` backend
    and route to it via the OpenAI-compatible connector.
    """
    pad = " " * indent
    models_str = ", ".join(f"\"{m}\"" for m in endpoint.models)
    lines = [
        f"{pad}[[backends]]",
        f"{pad}backend_type = \"llm_d_cluster\"",
        f"{pad}endpoint = {{ url = \"{endpoint.base_url.rstrip('/')}\", protocol = \"http\" }}",
        f"{pad}models = [{models_str}]",
        f"{pad}region = \"{endpoint.region}\"",
        f"{pad}kv_block_size = {int(endpoint.kv_block_size)}",
    ]
    return "\n".join(lines) + "\n"


def render_gateway_config(
    endpoints: Sequence[LlmDEndpoint],
    *,
    region: Optional[str] = None,
) -> str:
    """Render a multi-backend ``[[backends]]`` section.

    Args:
        endpoints: Discovered LLM-D endpoints.
        region: If provided, override each endpoint's region with this value.
    """
    if not endpoints:
        return "# no LLM-D endpoints discovered\n"
    blocks: List[str] = []
    for ep in endpoints:
        if region is not None:
            ep = dataclasses.replace(ep, region=region)
        blocks.append(make_backend_config(ep))
    return "\n".join(blocks)


def main() -> int:
    """Entry point: discover LLM-D services and print gateway config.

    Run via::

        python -m hier_kv_gateway.integrations.llm_d \\
            [--namespace NAMESPACE] [--label-selector SEL] [--region REGION]
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Discover LLM-D services and emit gateway backend config.")
    parser.add_argument("--namespace", default=None, help="Kubernetes namespace to search.")
    parser.add_argument("--label-selector", default=None, help="K8s label selector.")
    parser.add_argument("--kubeconfig", default=None, help="Path to a kubeconfig file.")
    parser.add_argument("--region", default="cloud-cn-beijing")
    parser.add_argument("--default-kv-block-size", type=int, default=16)
    parser.add_argument(
        "--from-url",
        action="append",
        default=[],
        help="Manually add an LLM-D endpoint URL (bypasses K8s discovery). Can be repeated.",
    )
    parser.add_argument("--model", action="append", default=[], help="Model id for --from-url endpoints.")
    args = parser.parse_args()

    endpoints: List[LlmDEndpoint] = []

    if args.from_url:
        for url in args.from_url:
            endpoints.append(
                LlmDEndpoint(
                    name=f"manual-{len(endpoints)}",
                    base_url=url,
                    models=args.model or ["unknown"],
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

    print(f"[llm-d] discovered {len(endpoints)} endpoint(s):")
    for ep in endpoints:
        healthy = "OK" if ep.is_healthy() else "unreachable"
        print(f"  - {ep.name} @ {ep.base_url} models={ep.models} kv={ep.kv_block_size} [{healthy}]")
    print("\n[llm-d] gateway backend config:")
    print(render_gateway_config(endpoints))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
