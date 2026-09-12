"""Inspire sandboxes: boot one, run commands in it, reach it from the trainer.

Platform code only -- replace this module to run the same rollout elsewhere.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

# The SDK is installed at a cluster path; the run script puts it on PYTHONPATH.
from inspire_sandbox import AsyncSandbox, CommandExitException, SandboxException, TimeoutException

SANDBOX_TTL = 3 * 60 * 60  # platform lifetime of each sandbox
SANDBOX_CREATE_MAX_ATTEMPTS = int(os.environ.get("SANDBOX_CREATE_MAX_ATTEMPTS", "8"))
SANDBOX_CREATE_RETRY_BASE_SEC = float(os.environ.get("SANDBOX_CREATE_RETRY_BASE_SEC", "2"))
SANDBOX_CREATE_RETRY_MAX_SEC = float(os.environ.get("SANDBOX_CREATE_RETRY_MAX_SEC", "30"))

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
            message = str(exc).lower()
            resource_exhausted = any(
                marker in message
                for marker in (
                    "no available resources",
                    "cpu quota exceeded",
                    "rate limit exceeded",
                    # The control plane sometimes collapses allocator errors
                    # into this generic 500 response. Treat it as transient,
                    # but keep the existing bounded retry budget.
                    "failed to create sandbox",
                )
            )
            if not resource_exhausted or attempt == SANDBOX_CREATE_MAX_ATTEMPTS:
                raise

            delay = min(
                SANDBOX_CREATE_RETRY_BASE_SEC * (2 ** (attempt - 1)),
                SANDBOX_CREATE_RETRY_MAX_SEC,
            )
            delay *= random.uniform(0.8, 1.2)
            logger.warning(
                "Sandbox capacity is exhausted; retrying create in %.1fs (attempt %d/%d)",
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
    try:
        # Inherit stdout/stderr: a dying tunnel has to say why.
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
        )
        try:
            await asyncio.sleep(3)
            if process.poll() is not None:
                raise RuntimeError("host wstunnel client exited before the agent started")
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(process.wait, timeout=10)
                if process.poll() is None:
                    process.kill()
    finally:
        with contextlib.suppress(Exception):
            await handle.kill()
