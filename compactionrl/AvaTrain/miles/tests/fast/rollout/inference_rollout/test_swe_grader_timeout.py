import sys
from pathlib import Path

import pytest

AGENTIC_SWE_DIR = Path(__file__).resolve().parents[4] / "examples" / "agentic_swe"
sys.path.insert(0, str(AGENTIC_SWE_DIR))

import swe  # noqa: E402


@pytest.mark.asyncio
async def test_grader_timeout_returns_zero_and_kills_sandbox(monkeypatch):
    killed = False

    class FakeFiles:
        async def write(self, path, content, user=None):
            raise AssertionError("empty patches must not be uploaded")

    class FakeSandbox:
        files = FakeFiles()

        async def kill(self):
            nonlocal killed
            killed = True

    async def create_sandbox(**kwargs):
        return FakeSandbox()

    async def run(*args, **kwargs):
        raise swe.sandbox.TimeoutException("grader timed out")

    monkeypatch.setattr(swe.sandbox, "create_sandbox", create_sandbox)
    monkeypatch.setattr(swe.sandbox, "run", run)
    task = swe.SweTask(
        instance_id="repo__project-1",
        sandbox_project="secondary",
        repo="repo/project",
        workdir="/testbed",
        template="template-id",
        sandbox_user="root",
        env={},
        base_commit="deadbeef",
        test_patch="",
        test_command="pytest",
        fail_to_pass=(),
        pass_to_pass=(),
        log_parser="pytest",
    )

    reward = await task.grade_patch("")

    assert reward == 0.0
    assert killed
