from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

if "sandbox" not in sys.modules:
    sandbox_stub = ModuleType("sandbox")

    async def _unused_run(*args, **kwargs):
        raise AssertionError("sandbox.run must be replaced by the test")

    sandbox_stub.run = _unused_run
    sys.modules["sandbox"] = sandbox_stub
if "swe" not in sys.modules:
    swe_stub = ModuleType("swe")
    swe_stub.SweTask = object
    sys.modules["swe"] = swe_stub

from compaction_swe import generate


@pytest.mark.asyncio
async def test_tunnel_readiness_retries_until_sandbox_can_reach_proxy(monkeypatch):
    results = iter(
        [
            SimpleNamespace(exit_code=1, stdout="", output="connection refused"),
            SimpleNamespace(
                exit_code=0,
                stdout="compactionrl-tunnel-ready\n",
                output="compactionrl-tunnel-ready",
            ),
        ]
    )
    calls = 0

    async def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return next(results)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(generate.sandbox, "run", run)
    monkeypatch.setattr(generate.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(generate, "TUNNEL_READY_TIMEOUT", 10)
    monkeypatch.setattr(generate, "TUNNEL_READY_PROBE_TIMEOUT", 1)

    await generate.wait_for_reverse_tunnel(object(), user="root")

    assert calls == 2


@pytest.mark.asyncio
async def test_tunnel_readiness_fails_before_agent_start(monkeypatch):
    async def run(*args, **kwargs):
        return SimpleNamespace(exit_code=1, stdout="", output="gateway unavailable")

    times = iter([0.0, 0.0, 2.0])
    loop = SimpleNamespace(time=lambda: next(times))

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(generate.sandbox, "run", run)
    monkeypatch.setattr(generate.asyncio, "get_running_loop", lambda: loop)
    monkeypatch.setattr(generate.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(generate, "TUNNEL_READY_TIMEOUT", 1)
    monkeypatch.setattr(generate, "TUNNEL_READY_PROBE_TIMEOUT", 1)

    with pytest.raises(RuntimeError, match="did not become reachable"):
        await generate.wait_for_reverse_tunnel(object(), user="root")
