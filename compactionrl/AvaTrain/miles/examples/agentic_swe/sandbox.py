"""Inspire sandboxes: boot one, run commands in it, reach it from the trainer.

Platform code only -- replace this module to run the same rollout elsewhere.
"""

from __future__ import annotations

import asyncio
import contextlib
import select
import logging
import os
import random
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass

# The SDK is installed at a cluster path; the run script puts it on PYTHONPATH.
from inspire_sandbox import AsyncSandbox, CommandExitException, SandboxException, TimeoutException

SANDBOX_TTL = 3 * 60 * 60  # platform lifetime of each sandbox
SANDBOX_CREATE_MAX_ATTEMPTS = int(os.environ.get("SANDBOX_CREATE_MAX_ATTEMPTS", "8"))
SANDBOX_CREATE_RETRY_BASE_SEC = float(os.environ.get("SANDBOX_CREATE_RETRY_BASE_SEC", "2"))
SANDBOX_CREATE_RETRY_MAX_SEC = float(os.environ.get("SANDBOX_CREATE_RETRY_MAX_SEC", "30"))
TUNNEL_FAILURE_THRESHOLD = int(os.environ.get("TUNNEL_SUPERVISOR_FAILURE_THRESHOLD", "3"))
TUNNEL_FAILURE_WINDOW_SEC = float(os.environ.get("TUNNEL_SUPERVISOR_FAILURE_WINDOW_SEC", "30"))
TUNNEL_MONITOR_POLL_SEC = float(os.environ.get("TUNNEL_SUPERVISOR_POLL_SEC", "0.25"))
# During sandbox startup the reverse endpoint can legitimately return 408 while
# the in-sandbox wstunnel server is still being provisioned. Do not treat
# those startup handshakes as a persistent tunnel failure.
TUNNEL_STARTUP_GRACE_SEC = float(os.environ.get("TUNNEL_SUPERVISOR_STARTUP_GRACE_SEC", "60"))

if TUNNEL_FAILURE_THRESHOLD < 1:
    raise ValueError("TUNNEL_SUPERVISOR_FAILURE_THRESHOLD must be positive")
if TUNNEL_FAILURE_WINDOW_SEC <= 0 or TUNNEL_MONITOR_POLL_SEC <= 0 or TUNNEL_STARTUP_GRACE_SEC < 0:
    raise ValueError("tunnel supervisor timeouts must be positive")

_TUNNEL_FAILURE_MARKERS = (
    "failed to do websocket handshake",
    "invalid status code: 408",
    "invalid status code: 502",
    "cannot connect to remote server",
)


def _retryable_create_failure(exc: SandboxException) -> bool:
    """Return whether a create response is plausibly transient.

    The SDK exposes the HTTP status only in the exception text.  Restricting
    this to server/rate-limit statuses avoids retrying authentication,
    template, or argument errors forever while handling the platform's
    ``500: Failed to create sandbox`` response.
    """
    message = str(exc).lower()
    match = re.match(r"\s*(\d{3})\s*:", message)
    if match and int(match.group(1)) in {429, 500, 502, 503, 504}:
        return True
    return any(
        marker in message
        for marker in ("no available resources", "cpu quota exceeded", "rate limit exceeded")
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()


async def create_sandbox(
    *,
    template: str,
    envs: Mapping[str, str],
    project: str = "primary",
) -> AsyncSandbox:
    project = project.strip().lower()
    if project == "primary":
        api_key = os.environ.get("SBX_API_KEY_PRIMARY") or os.environ.get("SBX_API_KEY")
        api_url = os.environ.get("SBX_API_URL_PRIMARY") or os.environ.get("SBX_API_URL")
    elif project == "secondary":
        api_key = os.environ.get("SBX_API_KEY_SECONDARY")
        api_url = os.environ.get("SBX_API_URL_SECONDARY") or os.environ.get("SBX_API_URL")
    else:
        raise ValueError(f"unsupported sandbox project: {project!r}")
    if not api_key or not api_url:
        raise RuntimeError(f"credentials are not configured for sandbox project {project!r}")

    for attempt in range(1, SANDBOX_CREATE_MAX_ATTEMPTS + 1):
        try:
            return await AsyncSandbox.create(
                template=template,
                timeout=SANDBOX_TTL,
                envs=dict(envs),
                network={"allow_public_traffic": True},
                api_key=api_key,
                api_url=api_url,
            )
        except SandboxException as exc:
            if not _retryable_create_failure(exc) or attempt == SANDBOX_CREATE_MAX_ATTEMPTS:
                raise

            delay = min(
                SANDBOX_CREATE_RETRY_BASE_SEC * (2 ** (attempt - 1)),
                SANDBOX_CREATE_RETRY_MAX_SEC,
            )
            delay *= random.uniform(0.8, 1.2)
            logger.warning(
                "Transient sandbox create failure (%s); retrying in %.1fs (attempt %d/%d)",
                str(exc),
                delay,
                attempt + 1,
                SANDBOX_CREATE_MAX_ATTEMPTS,
            )
            await asyncio.sleep(delay)

    raise AssertionError("unreachable")


async def run(
    sandbox: AsyncSandbox,
    script: str,
    *,
    timeout: int,
    user: str | None = None,
    cwd: str | None = None,
) -> ExecResult:
    """Run a shell script and always return a result.

    Non-zero is normal here, so the SDK's exception is turned back into data.
    """
    try:
        result = await sandbox.commands.run(
            f"/bin/bash -c {shlex.quote(script)}",
            timeout=timeout,
            request_timeout=timeout + 60,
            user=user,
            cwd=cwd,
        )
        return ExecResult(0, result.stdout or "", result.stderr or "")
    except CommandExitException as exc:
        return ExecResult(int(exc.exit_code), exc.stdout or "", exc.stderr or "")


async def _supervise_tunnel(
    process: subprocess.Popen,
    sandbox: AsyncSandbox,
    stop_event: asyncio.Event,
    state: dict[str, str],
) -> None:
    """Stop a tunnel and its sandbox after repeated gateway failures.

    wstunnel keeps retrying handshake failures while its process remains alive,
    so checking ``poll()`` alone cannot detect a dead route.  The combined
    stdout/stderr pipe is polled with a bounded timeout so task cancellation
    never leaves a blocked reader thread behind.
    """
    stream = process.stdout
    failures: list[float] = []
    pending = ""
    loop = asyncio.get_running_loop()
    started_at = loop.time()

    while not stop_event.is_set():
        if process.poll() is not None:
            state.setdefault("reason", f"host wstunnel exited with code {process.returncode}")
            return
        if stream is None:
            await asyncio.sleep(TUNNEL_MONITOR_POLL_SEC)
            continue

        try:
            readable, _, _ = await asyncio.to_thread(
                select.select, [stream.fileno()], [], [], TUNNEL_MONITOR_POLL_SEC
            )
        except (OSError, ValueError):
            if not stop_event.is_set():
                state.setdefault("reason", "host wstunnel stderr became unreadable")
            return
        if not readable:
            continue

        try:
            chunk = await asyncio.to_thread(os.read, stream.fileno(), 4096)
        except OSError:
            if not stop_event.is_set():
                state.setdefault("reason", "host wstunnel stderr became unreadable")
            return
        if not chunk:
            if process.poll() is not None and not stop_event.is_set():
                state.setdefault("reason", f"host wstunnel exited with code {process.returncode}")
                return
            continue
        pending += chunk.decode("utf-8", errors="replace")
        complete_lines = pending.split("\n")
        pending = complete_lines.pop()
        for raw_line in complete_lines:
            text = raw_line.strip()
            if text:
                logger.warning("host wstunnel: %s", text)
            lowered = text.lower()
            if not any(marker in lowered for marker in _TUNNEL_FAILURE_MARKERS):
                continue

            now = loop.time()
            if now - started_at < TUNNEL_STARTUP_GRACE_SEC:
                # The readiness probe owns this initial window. Handshake
                # failures here are expected while the sandbox-side listener
                # comes up and must not trigger sandbox.kill().
                continue
            failures[:] = [
                timestamp
                for timestamp in failures
                if now - timestamp <= TUNNEL_FAILURE_WINDOW_SEC
            ]
            failures.append(now)
            if len(failures) < TUNNEL_FAILURE_THRESHOLD:
                continue

            reason = (
                f"host wstunnel had {len(failures)} handshake failures within "
                f"{TUNNEL_FAILURE_WINDOW_SEC:.0f}s; last={text!r}"
            )
            state.setdefault("reason", reason)
            logger.error("Tunnel supervisor tripped: %s", reason)
            stop_event.set()
            if process.poll() is None:
                with contextlib.suppress(Exception):
                    process.terminate()
            # Killing the sandbox also terminates the qwen-code command that may
            # be blocked inside sandbox.run, instead of waiting for its 360s API timeout.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sandbox.kill(), timeout=10)
            return


@contextlib.asynccontextmanager
async def reverse_tunnel(
    sandbox: AsyncSandbox,
    *,
    proxy_port: int,
    sandbox_port: int,
    server_port: int,
    wstunnel_bin: str,
    user: str | None = None,
):
    """Expose the trainer-side proxy at ``sandbox_port`` inside the sandbox."""
    handle = await sandbox.commands.run(
        f"{shlex.quote(wstunnel_bin)} server ws://0.0.0.0:{server_port}",
        background=True,
        timeout=0,
        request_timeout=120,
        user=user,
    )
    supervisor_stop = asyncio.Event()
    supervisor_state: dict[str, str] = {}
    supervisor_task: asyncio.Task[None] | None = None
    try:
        # Capture both streams for the supervisor; wstunnel's tracing output
        # location differs across protocol-bundle images.
        process = subprocess.Popen(
            [
                "wstunnel",
                "client",
                # Keep the gateway WebSocket alive between model requests.
                "--websocket-ping-frequency",
                "30s",
                "--connection-min-idle",
                "2",
                "-R",
                f"tcp://127.0.0.1:{sandbox_port}:127.0.0.1:{proxy_port}",
                f"wss://{sandbox.get_host(server_port)}",
            ],
            # wstunnel's tracing subscriber can write to either stream
            # depending on the image.  Feed both into the supervisor so 408/
            # 502 handshake loops cannot hide on stdout.
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        supervisor_task = asyncio.create_task(
            _supervise_tunnel(process, sandbox, supervisor_stop, supervisor_state)
        )
        try:
            await asyncio.sleep(3)
            if process.poll() is not None:
                raise RuntimeError("host wstunnel client exited before the agent started")
            yield
        finally:
            body_exception = sys.exc_info()[1]
            supervisor_stop.set()
            if supervisor_task is not None:
                if not supervisor_task.done():
                    supervisor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await supervisor_task
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(process.wait, timeout=10)
                if process.poll() is None:
                    process.kill()
            if supervisor_state.get("reason") and body_exception is None:
                raise RuntimeError(f"reverse tunnel failed: {supervisor_state['reason']}")
    finally:
        with contextlib.suppress(Exception):
            await handle.kill()
