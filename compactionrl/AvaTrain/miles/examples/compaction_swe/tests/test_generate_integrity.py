from __future__ import annotations

import contextlib
import json
import shlex
import sys
from types import ModuleType, SimpleNamespace

import pytest

if "sandbox" not in sys.modules:
    sys.modules["sandbox"] = ModuleType("sandbox")
if "swe" not in sys.modules:
    swe_stub = ModuleType("swe")
    swe_stub.SweTask = object
    sys.modules["swe"] = swe_stub

from compaction_swe import generate


def test_agent_command_disables_qwen_code_background_model_calls(monkeypatch):
    monkeypatch.setattr(generate, "QWEN_MAX_RETRIES", 0)

    command = generate.agent_command("/testbed", "fix it")
    settings_line = next(line for line in command.splitlines() if line.startswith("printf "))
    settings = json.loads(shlex.split(settings_line)[2])

    assert settings["model"]["generationConfig"]["maxRetries"] == 0
    assert settings["model"]["generationConfig"]["contextWindowSize"] == 1_000_000
    assert settings["model"]["skipNextSpeakerCheck"] is True
    assert settings["memory"] == {
        "enableManagedAutoMemory": False,
        "enableManagedAutoDream": False,
        "enableAutoSkill": False,
    }


def _routing(*, failed_requests: int = 0) -> dict[str, int]:
    return {
        "routing_requests": 1,
        "routing_main_requests": 1,
        "routing_summary_requests": 0,
        "routing_auxiliary_requests": 0,
        "routing_replay_rewrite_requests": 0,
        "routing_main_prompt_tokens": 2,
        "routing_summary_prompt_tokens": 0,
        "routing_auxiliary_prompt_tokens": 0,
        "routing_replay_rewrite_prompt_tokens": 0,
        "routing_main_generated_tokens": 1,
        "routing_summary_generated_tokens": 0,
        "routing_auxiliary_generated_tokens": 0,
        "routing_replay_rewrite_generated_tokens": 0,
        "routing_trainable_generated_tokens": 1,
        "routing_untracked_requests": 0,
        "routing_untracked_generated_tokens": 0,
        "routing_failed_requests": failed_requests,
        "routing_concurrent_request_failures": 0,
        "routing_max_active_requests": 1,
        "routing_accounted_generated_tokens": 1,
        "routing_unexplained_generated_tokens": 0,
    }


class _Ledger:
    def has_trainable_tokens(self) -> bool:
        return True


class _FakeProxy:
    port = 12345
    ledger = _Ledger()
    compaction_count = 0
    generated_tokens = 1
    episode_tokens = 1
    truncated_observations = 0
    retained_recent_steps: tuple[int, ...] = ()
    rebase_count = 0

    def __init__(
        self,
        events: list[str],
        *,
        failure_after_close: BaseException | None = None,
        failed_requests: int = 0,
        lifecycle: dict[str, int] | None = None,
    ) -> None:
        self.events = events
        self.failure = None
        self._failure_after_close = failure_after_close
        self._routing = _routing(failed_requests=failed_requests)
        self._lifecycle = lifecycle or {
            "active_completions": 0,
            "inflight_futures": 0,
            "pending_generations": 0,
        }

    def start(self):
        self.events.append("proxy_start")
        return self

    def close(self) -> None:
        self.events.append("proxy_close")
        if self._failure_after_close is not None:
            self.failure = self._failure_after_close

    def lifecycle_metadata(self) -> dict[str, int]:
        return dict(self._lifecycle)

    def routing_metadata(self) -> dict[str, int]:
        return dict(self._routing)

    def to_samples(self, _base_sample, reward: float):
        return [
            SimpleNamespace(
                reward=reward,
                loss_mask=[1],
                train_metadata={"optimized_tokens": 1},
            )
        ]


class _Runtime:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def kill(self) -> None:
        self.events.append("runtime_kill")


class _Task:
    template = "template"
    env = {}
    sandbox_project = "secondary"
    sandbox_user = "root"
    workdir = "/testbed"

    def __init__(self, events: list[str], reward: float = 0.0) -> None:
        self.events = events
        self.reward = reward

    async def setup(self, _runtime) -> None:
        self.events.append("task_setup")

    async def collect_patch(self, _runtime) -> str:
        self.events.append("collect_patch")
        return "diff --git a/a.py b/a.py"

    async def grade_patch(self, _patch: str) -> float:
        self.events.append("grade_patch")
        return self.reward


def _input() -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(sglang_tool_call_parser=None, sglang_reasoning_parser=None),
        sample=SimpleNamespace(prompt=[{"content": "fix it"}], label="repo__issue-1"),
        state=SimpleNamespace(tokenizer=object()),
        sampling_params={"max_new_tokens": 16},
    )


def _install_episode_fakes(monkeypatch, proxy: _FakeProxy, events: list[str], *, exit_code: int) -> None:
    @contextlib.asynccontextmanager
    async def reverse_tunnel(*_args, **_kwargs):
        events.append("tunnel_enter")
        try:
            yield
        finally:
            events.append("tunnel_exit")

    async def create_sandbox(**_kwargs):
        events.append("sandbox_create")
        return _Runtime(events)

    async def run(*_args, **_kwargs):
        events.append("agent_run")
        return SimpleNamespace(exit_code=exit_code, stdout="stdout", stderr="stderr")

    async def tunnel_ready(*_args, **_kwargs):
        events.append("tunnel_ready")

    monkeypatch.setattr(generate, "CompactionModelProxy", lambda **_kwargs: proxy)
    monkeypatch.setattr(generate, "get_model_url", lambda *_args: "http://unused/generate")
    monkeypatch.setattr(
        generate.CompactionConfig,
        "from_env",
        classmethod(lambda _cls: SimpleNamespace(auxiliary_mode="reject")),
    )
    monkeypatch.setattr(generate, "wait_for_reverse_tunnel", tunnel_ready)
    monkeypatch.setattr(generate.sandbox, "create_sandbox", create_sandbox, raising=False)
    monkeypatch.setattr(generate.sandbox, "reverse_tunnel", reverse_tunnel, raising=False)
    monkeypatch.setattr(generate.sandbox, "run", run, raising=False)


@pytest.mark.asyncio
async def test_nonzero_agent_exit_rejects_before_patch_and_grade(monkeypatch):
    events: list[str] = []
    proxy = _FakeProxy(events)
    _install_episode_fakes(monkeypatch, proxy, events, exit_code=7)

    with pytest.raises(RuntimeError, match="exit_code=7"):
        await generate.generate_episode(_input(), _Task(events))

    assert "collect_patch" not in events
    assert "grade_patch" not in events
    assert "proxy_close" in events
    assert "runtime_kill" in events


@pytest.mark.asyncio
async def test_failure_discovered_during_close_rejects_before_patch_and_grade(monkeypatch):
    events: list[str] = []
    proxy = _FakeProxy(events, failure_after_close=RuntimeError("backend failed"))
    _install_episode_fakes(monkeypatch, proxy, events, exit_code=0)

    with pytest.raises(RuntimeError, match="partial trajectory"):
        await generate.generate_episode(_input(), _Task(events))

    assert "collect_patch" not in events
    assert "grade_patch" not in events
    assert events.index("proxy_close") < events.index("tunnel_exit")


@pytest.mark.asyncio
async def test_routing_failure_rejects_before_patch_and_grade(monkeypatch):
    events: list[str] = []
    proxy = _FakeProxy(events, failed_requests=1)
    _install_episode_fakes(monkeypatch, proxy, events, exit_code=0)

    with pytest.raises(RuntimeError, match="failed_requests=1"):
        await generate.generate_episode(_input(), _Task(events))

    assert "collect_patch" not in events
    assert "grade_patch" not in events


@pytest.mark.asyncio
async def test_nonzero_lifecycle_state_rejects_before_patch_and_grade(monkeypatch):
    events: list[str] = []
    proxy = _FakeProxy(
        events,
        lifecycle={"active_completions": 0, "inflight_futures": 1, "pending_generations": 0},
    )
    _install_episode_fakes(monkeypatch, proxy, events, exit_code=0)

    with pytest.raises(RuntimeError, match="did not drain completely"):
        await generate.generate_episode(_input(), _Task(events))

    assert "collect_patch" not in events
    assert "grade_patch" not in events


@pytest.mark.asyncio
async def test_complete_zero_reward_episode_is_kept(monkeypatch):
    events: list[str] = []
    proxy = _FakeProxy(events)
    _install_episode_fakes(monkeypatch, proxy, events, exit_code=0)

    output = await generate.generate_episode(_input(), _Task(events, reward=0.0))

    assert len(output.samples) == 1
    assert output.samples[0].reward == 0.0
    assert output.samples[0].train_metadata["routing_failed_requests"] == 0
    assert events.index("proxy_close") < events.index("collect_patch")
    assert events.index("collect_patch") < events.index("grade_patch")
