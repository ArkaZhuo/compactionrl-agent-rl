from __future__ import annotations

import contextlib
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

AGENTIC_SWE_DIR = Path(__file__).resolve().parents[4] / "examples" / "agentic_swe"
sys.path.insert(0, str(AGENTIC_SWE_DIR))

import generate as generate_module  # noqa: E402


def test_agent_command_disables_qwen_code_background_model_calls(monkeypatch):
    monkeypatch.setattr(generate_module, "QWEN_MAX_RETRIES", 0)

    command = generate_module.agent_command("/testbed", "fix it")
    settings_line = next(line for line in command.splitlines() if line.startswith("printf "))
    settings = json.loads(shlex.split(settings_line)[2])

    assert settings["model"]["generationConfig"] == {
        "maxRetries": 0,
        "contextWindowSize": 1_000_000,
    }
    assert settings["model"]["skipNextSpeakerCheck"] is True
    assert settings["memory"] == {
        "enableManagedAutoMemory": False,
        "enableManagedAutoDream": False,
        "enableAutoSkill": False,
    }


@pytest.mark.parametrize(
    ("exit_code", "stderr", "expected"),
    [
        (53, "Reached max session turns for this session.", True),
        (53, "an unrelated qwen-code failure", False),
        (1, "Reached max session turns for this session.", False),
        (0, "", False),
        (53, None, False),
    ],
)
def test_max_session_turns_exit_is_classified_strictly(exit_code, stderr, expected):
    result = SimpleNamespace(exit_code=exit_code, stderr=stderr)

    assert generate_module.reached_max_session_turns(result) is expected


@pytest.mark.asyncio
async def test_max_session_turns_episode_is_drained_and_graded(monkeypatch):
    events: list[str] = []

    class Trajectory:
        def has_trainable_tokens(self):
            return True

        def to_sample(self, sample, *, reward, max_response_length):
            assert max_response_length == 16384
            return SimpleNamespace(
                reward=reward,
                response_length=123,
                status=SimpleNamespace(value="completed"),
            )

    class FakeProxy:
        port = 12345
        trajectory = Trajectory()
        failure = None

        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("proxy_start")
            return self

        def close(self):
            events.append("proxy_close")

        def lifecycle_metadata(self):
            return {
                "active_completions": 0,
                "inflight_futures": 0,
                "pending_generations": 0,
            }

    class Runtime:
        async def kill(self):
            events.append("runtime_kill")

    class Task:
        template = "template"
        env = {}
        sandbox_project = "secondary"
        sandbox_user = "root"
        workdir = "/testbed"

        async def setup(self, runtime):
            events.append("setup")

        async def collect_patch(self, runtime):
            events.append("collect_patch")
            return "diff --git a/a.py b/a.py"

        async def grade_patch(self, patch):
            events.append("grade_patch")
            return 1.0

    @contextlib.asynccontextmanager
    async def reverse_tunnel(*args, **kwargs):
        yield

    async def create_sandbox(**kwargs):
        return Runtime()

    async def run(*args, **kwargs):
        return SimpleNamespace(
            exit_code=53,
            stdout="",
            stderr="Reached max session turns for this session.",
        )

    async def tunnel_ready(*args, **kwargs):
        return None

    monkeypatch.setattr(generate_module, "ModelProxy", FakeProxy)
    monkeypatch.setattr(generate_module, "get_model_url", lambda *args: "http://unused")
    monkeypatch.setattr(generate_module.sandbox, "create_sandbox", create_sandbox)
    monkeypatch.setattr(generate_module.sandbox, "reverse_tunnel", reverse_tunnel)
    monkeypatch.setattr(generate_module.sandbox, "run", run)
    monkeypatch.setattr(generate_module, "wait_for_reverse_tunnel", tunnel_ready)

    input = SimpleNamespace(
        args=SimpleNamespace(),
        sample=SimpleNamespace(prompt=[{"content": "fix it"}], label="repo__issue-1"),
        state=SimpleNamespace(tokenizer=object()),
        sampling_params={"max_new_tokens": 16384},
    )

    output = await generate_module.generate_episode(input, Task())

    assert output.samples.reward == 1.0
    assert events.index("proxy_close") < events.index("collect_patch")
    assert events.index("runtime_kill") < events.index("grade_patch")


@pytest.mark.asyncio
async def test_proxy_failure_discovered_during_close_skips_patch_and_grade(monkeypatch):
    events: list[str] = []

    class Trajectory:
        def has_trainable_tokens(self):
            return True

    class FakeProxy:
        port = 12345
        trajectory = Trajectory()
        failure = None

        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("proxy_start")
            return self

        def close(self):
            events.append("proxy_close")
            self.failure = RuntimeError("backend failed")

        def lifecycle_metadata(self):
            return {
                "active_completions": 0,
                "inflight_futures": 0,
                "pending_generations": 0,
            }

    class Runtime:
        async def kill(self):
            events.append("runtime_kill")

    class Task:
        template = "template"
        env = {}
        sandbox_project = "secondary"
        sandbox_user = "root"
        workdir = "/testbed"

        async def setup(self, runtime):
            events.append("setup")

        async def collect_patch(self, runtime):
            raise AssertionError("a partial trajectory must not be collected")

        async def grade_patch(self, patch):
            raise AssertionError("a partial trajectory must not be graded")

    @contextlib.asynccontextmanager
    async def reverse_tunnel(*args, **kwargs):
        yield

    async def create_sandbox(**kwargs):
        return Runtime()

    async def run(*args, **kwargs):
        return SimpleNamespace(exit_code=0, stdout="done", stderr="")

    async def tunnel_ready(*args, **kwargs):
        return None

    monkeypatch.setattr(generate_module, "ModelProxy", FakeProxy)
    monkeypatch.setattr(generate_module, "get_model_url", lambda *args: "http://unused")
    monkeypatch.setattr(generate_module.sandbox, "create_sandbox", create_sandbox)
    monkeypatch.setattr(generate_module.sandbox, "reverse_tunnel", reverse_tunnel)
    monkeypatch.setattr(generate_module.sandbox, "run", run)
    monkeypatch.setattr(generate_module, "wait_for_reverse_tunnel", tunnel_ready)

    input = SimpleNamespace(
        args=SimpleNamespace(),
        sample=SimpleNamespace(prompt=[{"content": "fix it"}], label="repo__issue-1"),
        state=SimpleNamespace(tokenizer=object()),
        sampling_params={"max_new_tokens": 16384},
    )

    with pytest.raises(RuntimeError, match="partial trajectory"):
        await generate_module.generate_episode(input, Task())

    assert events.count("proxy_close") >= 1
    assert "runtime_kill" in events
