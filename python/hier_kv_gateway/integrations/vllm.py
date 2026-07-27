"""vLLM integration: launch a vLLM OpenAI-compatible server aligned with the gateway.

vLLM (https://docs.vllm.ai) is a high-throughput LLM serving engine that
exposes an OpenAI-compatible HTTP API. The Hier KV Gateway can route
requests to one or more vLLM backends via the ``vllm_engine`` backend type.

This adapter provides three things:

1. :class:`VllmLaunchSpec` — a typed spec for launching a vLLM server with
   KV-block alignment matching the gateway's configured ``kv_block_size``.
2. :func:`launch_vllm` — spawn a vLLM server as a subprocess and return a
   handle suitable for use with :class:`~hier_kv_gateway.launcher.GatewayLauncher`.
3. :func:`make_backend_config` — build a TOML ``[[backends]]`` snippet
   describing the vLLM backend so the gateway can route to it.

The adapter does **not** depend on the ``vllm`` Python package at import
time — it spawns the ``vllm`` CLI (``python -m vllm.entrypoints.openai.api_server``),
which is the recommended way to run vLLM in production. This means the
adapter works even when vLLM is installed in a different virtualenv or
container.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import requests

from ..exceptions import GatewayError


# Default vLLM OpenAI-compatible server port.
DEFAULT_VLLM_PORT = 8000

# Default health-check path exposed by vLLM's OpenAI server.
DEFAULT_VLLM_HEALTH_PATH = "/health"


@dataclasses.dataclass
class VllmLaunchSpec:
    """Specification for launching a vLLM OpenAI-compatible server.

    Attributes:
        model: HuggingFace model id (e.g. ``Qwen/Qwen2.5-7B-Instruct``) or a
            local path to a model directory.
        kv_block_size: KV cache block size **in tokens**. Must match the
            gateway's ``[routing] kv_block_size`` for prefix-overlap scoring
            to be accurate. vLLM calls this ``block_size``.
        host: Bind host. Defaults to ``0.0.0.0``.
        port: Bind port. Defaults to 8000.
        tensor_parallel_size: Number of GPUs to shard the model across.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        max_model_len: Maximum context length. ``None`` uses the model's
            default.
        extra_args: Additional raw CLI flags forwarded to vLLM verbatim
            (e.g. ``["--enable-lora", "--max-loras", "4"]``).
        python_executable: Python interpreter used to launch vLLM. Defaults
            to :data:`sys.executable`. Override to point at a virtualenv
            where ``vllm`` is installed.
        env: Optional environment overrides merged over ``os.environ``.
    """

    model: str
    kv_block_size: int = 16
    host: str = "0.0.0.0"
    port: int = DEFAULT_VLLM_PORT
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    extra_args: Sequence[str] = ()
    python_executable: Optional[str] = None
    env: Optional[Dict[str, str]] = None

    def build_cli_args(self) -> List[str]:
        """Build the CLI argument list for ``python -m vllm.entrypoints.openai.api_server``."""
        args: List[str] = [
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--block-size",
            str(self.kv_block_size),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if self.max_model_len is not None:
            args.extend(["--max-model-len", str(self.max_model_len)])
        args.extend(self.extra_args)
        return args

    def base_url(self) -> str:
        """HTTP base URL of the vLLM server."""
        return f"http://{self._connect_host()}:{self.port}"

    def health_url(self) -> str:
        """Full URL of the vLLM ``/health`` endpoint."""
        return f"{self.base_url()}{DEFAULT_VLLM_HEALTH_PATH}"

    def _connect_host(self) -> str:
        # When binding 0.0.0.0, connect to 127.0.0.1.
        return "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host


class VllmProcess:
    """Handle to a launched vLLM subprocess.

    Wraps :class:`subprocess.Popen` with a ``stop()`` method that escalates
    from SIGTERM to SIGKILL, plus a :meth:`wait_for_ready` polling helper.
    """

    def __init__(self, process: "subprocess.Popen[bytes]", spec: VllmLaunchSpec):
        self._process = process
        self._spec = spec

    @property
    def base_url(self) -> str:
        return self._spec.base_url()

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid

    def is_running(self) -> bool:
        return self._process.poll() is None

    def wait_for_ready(self, timeout: float = 120.0, poll_interval: float = 1.0) -> bool:
        """Poll vLLM's ``/health`` until it returns 200 or ``timeout`` elapses.

        vLLM startup is slow (model loading can take minutes), so the default
        timeout is 120 s.
        """
        deadline = time.monotonic() + timeout
        url = self._spec.health_url()
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return False
            try:
                resp = requests.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(poll_interval)
        return False

    def stop(self) -> None:
        """Terminate the vLLM subprocess, escalating to SIGKILL if needed."""
        proc = self._process
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc.pid)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def __enter__(self) -> "VllmProcess":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def launch_vllm(spec: VllmLaunchSpec) -> VllmProcess:
    """Spawn a vLLM OpenAI-compatible server as a subprocess.

    Args:
        spec: Launch specification (model, port, KV block size, ...).

    Returns:
        A :class:`VllmProcess` handle. Use :meth:`VllmProcess.wait_for_ready`
        to block until the server is healthy, then point a Hier KV Gateway
        instance at ``spec.base_url()`` via the ``vllm_engine`` backend
        type.

    Raises:
        GatewayError: if the Python interpreter cannot be found or the
            subprocess fails to spawn.
    """
    python = spec.python_executable or sys.executable
    if not python or not shutil.which(python):
        raise GatewayError(f"python interpreter not found: {python!r}")

    cmd = [python] + spec.build_cli_args()
    env = os.environ.copy()
    if spec.env:
        env.update({k: str(v) for k, v in spec.env.items()})

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise GatewayError(f"failed to spawn vLLM: {exc}") from exc

    return VllmProcess(process, spec)


def make_backend_config(
    spec: VllmLaunchSpec,
    *,
    region: str = "cloud-cn-beijing",
    indent: int = 0,
) -> str:
    """Render a TOML ``[[backends]]`` block for a vLLM backend.

    The returned string is suitable for splicing into a gateway config file
    or appending to an existing one.

    Args:
        spec: The vLLM launch spec describing the running server.
        region: Region id to assign to the backend.
        indent: Number of spaces to indent each line by (useful when
            emitting nested fragments).
    """
    pad = " " * indent
    lines = [
        f"{pad}[[backends]]",
        f"{pad}backend_type = \"vllm_engine\"",
        f"{pad}endpoint = {{ url = \"{spec.base_url()}\", protocol = \"http\" }}",
        f"{pad}models = [\"{spec.model}\"]",
        f"{pad}region = \"{region}\"",
        f"{pad}kv_block_size = {int(spec.kv_block_size)}",
    ]
    return "\n".join(lines) + "\n"


def _kill_process_tree(pid: int) -> None:
    """Best-effort SIGKILL of a process group."""
    import signal

    try:
        if hasattr(signal, "SIGKILL"):
            import os

            os.killpg(os.getpgid(pid), signal.SIGKILL)
        else:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
    except (ProcessLookupError, PermissionError, OSError):
        pass


def main() -> int:
    """Entry point: launch a small vLLM server and print a gateway config snippet.

    Run via::

        python -m hier_kv_gateway.integrations.vllm \\
            --model Qwen/Qwen2.5-7B-Instruct \\
            --port 8000
    """
    import argparse

    parser = argparse.ArgumentParser(description="Launch a vLLM server aligned with the Hier KV Gateway.")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path.")
    parser.add_argument("--port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--region", default="cloud-cn-beijing")
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for vLLM to become healthy before giving up.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the vLLM command but do not spawn it.")
    args, extra = parser.parse_known_args()

    spec = VllmLaunchSpec(
        model=args.model,
        kv_block_size=args.kv_block_size,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        extra_args=extra,
    )

    if args.dry_run:
        cmd_str = shlex.join([sys.executable] + spec.build_cli_args())
        print(f"[vllm] would run: {cmd_str}")
        print("\n[vllm] gateway backend config:")
        print(make_backend_config(spec, region=args.region))
        return 0

    print(f"[vllm] launching model={args.model!r} on http://127.0.0.1:{args.port}")
    with launch_vllm(spec) as proc:
        print(f"[vllm] pid={proc.pid}; waiting for /health (timeout={args.ready_timeout}s)")
        if not proc.wait_for_ready(timeout=args.ready_timeout):
            print("[vllm] server did not become healthy in time", file=sys.stderr)
            return 1
        print(f"[vllm] ready at {proc.base_url}")
        print("\n[vllm] gateway backend config:")
        print(make_backend_config(spec, region=args.region))
        print("\n[vllm] press Ctrl-C to stop the server...")
        try:
            while proc.is_running():
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[vllm] shutting down...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
