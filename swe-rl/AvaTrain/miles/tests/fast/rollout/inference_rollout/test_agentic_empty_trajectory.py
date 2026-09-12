import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

AGENTIC_SWE_DIR = Path(__file__).resolve().parents[4] / "examples" / "agentic_swe"
sys.path.insert(0, str(AGENTIC_SWE_DIR))

import generate as generate_module  # noqa: E402


@pytest.mark.asyncio
async def test_empty_trajectory_skips_grader_and_releases_agent_sandbox(monkeypatch):
    killed = False
    proxy_closed = False

    class EmptyTrajectory:
        def has_trainable_tokens(self):
            return False

    class FakeProxy:
        port = 12345
        trajectory = EmptyTrajectory()

        def __init__(self, **kwargs):
            pass

        def start(self):
            return self

        def close(self):
            nonlocal proxy_closed
            proxy_closed = True

    class FakeRuntime:
        async def kill(self):
            nonlocal killed
            killed = True

    class FakeTask:
        template = "template"
        env = {}
        sandbox_project = "secondary"
        sandbox_user = "root"
        workdir = "/testbed"

        async def setup(self, runtime):
            pass

        async def collect_patch(self, runtime):
            return ""

        async def grade_patch(self, patch):
            raise AssertionError("empty trajectories must not create a grader sandbox")

    @contextlib.asynccontextmanager
    async def reverse_tunnel(*args, **kwargs):
        yield

    async def create_sandbox(**kwargs):
        return FakeRuntime()

    async def run(*args, **kwargs):
        return SimpleNamespace(exit_code=1, stdout="", stderr="request failed")

    monkeypatch.setattr(generate_module, "ModelProxy", FakeProxy)
    monkeypatch.setattr(generate_module, "get_model_url", lambda *args: "http://unused")
    monkeypatch.setattr(generate_module.sandbox, "create_sandbox", create_sandbox)
    monkeypatch.setattr(generate_module.sandbox, "reverse_tunnel", reverse_tunnel)
    monkeypatch.setattr(generate_module.sandbox, "run", run)

    input = SimpleNamespace(
        args=SimpleNamespace(),
        sample=SimpleNamespace(prompt=[{"content": "fix it"}], label="repo__issue-1"),
        state=SimpleNamespace(tokenizer=object()),
        sampling_params={"max_new_tokens": 16384},
    )

    with pytest.raises(RuntimeError, match="no trainable trajectory"):
        await generate_module.generate_episode(input, FakeTask())

    assert killed
    assert proxy_closed
