"""Launch the Rust Hier KV Gateway binary as a subprocess from Python.

Useful for integration tests and local development workflows where a Python
process needs to spin up its own gateway instance.

Example::

    from hier_kv_gateway import GatewayLauncher, HierKvGatewayClient

    with GatewayLauncher(binary_path="./target/release/hier-kv-gateway",
                         config_path="examples/hier-kv-gateway.toml") as launcher:
        launcher.wait_for_ready()
        with HierKvGatewayClient() as client:
            print(client.health())
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import time
from typing import List, Optional

import requests

from .exceptions import GatewayError


class GatewayLauncher:
    """Manage the lifecycle of a Hier KV Gateway subprocess.

    The launcher is intentionally lightweight: it spawns the binary, polls
    ``/health`` until the gateway is ready (or the timeout elapses), and
    terminates the process on shutdown. It does not parse the gateway's
    stdout beyond what the OS provides.

    Note:
        The launcher performs blocking I/O (``requests.get`` for health
        polling, ``subprocess`` waits). It is not suitable for use inside an
        asyncio event loop without offloading to a thread executor.
    """

    def __init__(
        self,
        binary_path: str = "hier-kv-gateway",
        config_path: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        base_url: str = "http://localhost:8080",
        env: Optional[dict] = None,
    ) -> None:
        self.binary_path = binary_path
        self.config_path = config_path
        self.extra_args: List[str] = list(extra_args) if extra_args else []
        self.base_url = base_url.rstrip("/")
        self.env = env
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the gateway subprocess.

        Raises :class:`GatewayError` if the binary cannot be found, or if a
        process is already running for this launcher instance.
        """
        if self._process is not None and self._process.poll() is None:
            raise GatewayError("gateway process is already running")

        if shutil.which(self.binary_path) is None and not _looks_like_path(
            self.binary_path
        ):
            raise GatewayError(
                f"gateway binary not found on PATH: {self.binary_path!r}"
            )

        cmd: List[str] = [self.binary_path]
        if self.config_path is not None:
            cmd.extend(["--config", self.config_path])
        cmd.extend(self.extra_args)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            # Put the subprocess in its own process group so we can kill the
            # whole tree on stop(); this avoids orphaned child processes.
            start_new_session=True,
        )

    def stop(self) -> None:
        """Terminate the gateway subprocess.

        Sends SIGTERM (graceful) and waits up to a few seconds; if the
        process is still alive, escalates to SIGKILL. Safe to call multiple
        times or when no process is running.
        """
        proc = self._process
        if proc is None:
            return
        if proc.poll() is None:
            # Process still running: try graceful shutdown first.
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc.pid)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Best-effort; nothing more we can do.
                    pass
        # Close any inherited pipe fds.
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        self._process = None

    def is_running(self) -> bool:
        """Return ``True`` if the subprocess is currently alive."""
        proc = self._process
        return proc is not None and proc.poll() is None

    def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Poll ``/health`` until it responds 200 or ``timeout`` elapses.

        Returns ``True`` if the gateway became ready within the timeout,
        ``False`` otherwise. Polls every 100 ms.
        """
        deadline = time.monotonic() + timeout
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                # Subprocess exited before becoming ready.
                return False
            try:
                resp = requests.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)
        return False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GatewayLauncher":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_path(candidate: str) -> bool:
    """Heuristic: treat strings containing a path separator or a dot/slash
    prefix as explicit paths (which ``shutil.which`` may not find but which
    may still be executable, e.g. ``./target/release/hier-kv-gateway``)."""
    return "/" in candidate or candidate.startswith(".")


def _kill_process_tree(pid: int) -> None:
    """Best-effort kill of a process and its children.

    On POSIX we signal the whole process group (created via
    ``start_new_session=True``). On Windows we fall back to ``taskkill``.
    """
    try:
        if hasattr(signal, "SIGKILL"):
            # kill the process group leader's group id (== pid here)
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
