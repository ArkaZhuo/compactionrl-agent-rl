from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import threading
from types import SimpleNamespace

import pytest

from compaction_swe.batching import align_compaction_trajectories, validate_compaction_trajectories
from compaction_swe.config import CompactionConfig
from compaction_swe.proxy import CompactionModelProxy
from miles.utils.types import Sample


def _sample(trajectory_id: str, segment_index: int, optimized: int, future: int, reward: float = 1.0):
    metadata = {
        "compaction": True,
        "trajectory_id": trajectory_id,
        "segment_index": segment_index,
        "segment_type": "execution",
        "optimized_tokens": optimized,
        "future_optimized_tokens": future,
    }
    return SimpleNamespace(train_metadata=metadata, loss_mask=[1] * optimized, reward=reward)


def test_dp_alignment_removes_only_complete_trajectory():
    samples = [
        _sample("a", 0, 2, 1),
        _sample("a", 1, 1, 0),
        _sample("b", 0, 3, 0),
        _sample("c", 0, 1, 1),
        _sample("c", 1, 1, 0),
    ]

    result = align_compaction_trajectories(samples, divisor=2)

    assert len(result.samples) == 4
    assert result.removed_trajectory_ids == ("b",)
    assert {sample.train_metadata["trajectory_id"] for sample in result.samples} == {"a", "c"}


def test_dp_alignment_handles_multiple_trajectory_lengths_for_divisor_four():
    samples = [
        _sample("a", 0, 1, 2),
        _sample("a", 1, 1, 1),
        _sample("a", 2, 1, 0),
        _sample("b", 0, 1, 0),
        _sample("c", 0, 1, 1),
        _sample("c", 1, 1, 0),
        _sample("d", 0, 1, 0),
    ]

    result = align_compaction_trajectories(samples, divisor=4)

    assert len(result.samples) == 4
    assert result.removed_trajectory_ids == ("a",)
    assert [sample.train_metadata["trajectory_id"] for sample in result.samples] == ["b", "c", "c", "d"]


def test_dp_alignment_rejects_dropping_every_trajectory():
    with pytest.raises(ValueError, match="without dropping every trajectory"):
        align_compaction_trajectories([_sample("a", 0, 1, 0)], divisor=2)


def test_validation_rejects_mixed_ordinary_and_compaction_samples():
    ordinary = SimpleNamespace(train_metadata=None, loss_mask=[1], reward=1.0)
    with pytest.raises(ValueError, match="cannot share one training batch"):
        validate_compaction_trajectories([_sample("a", 0, 1, 0), ordinary])


def test_validation_rejects_non_contiguous_segments():
    samples = [_sample("a", 0, 1, 1), _sample("a", 2, 1, 0)]
    with pytest.raises(ValueError, match="non-contiguous segment indices"):
        validate_compaction_trajectories(samples)


def test_validation_rejects_inconsistent_rewards():
    samples = [_sample("a", 0, 1, 1, reward=1.0), _sample("a", 1, 1, 0, reward=0.0)]
    with pytest.raises(ValueError, match="inconsistent rewards"):
        validate_compaction_trajectories(samples)


def test_validation_rejects_stale_future_token_count():
    samples = [_sample("a", 0, 2, 7), _sample("a", 1, 1, 0)]
    with pytest.raises(ValueError, match="future_optimized_tokens=7, expected=1"):
        validate_compaction_trajectories(samples)


def test_validation_accepts_execution_rebase_after_execution():
    samples = [_sample("a", 0, 2, 1), _sample("a", 1, 1, 0)]
    samples[0].train_metadata["segment_type"] = "execution"
    samples[1].train_metadata["segment_type"] = "execution_rebase"

    groups = validate_compaction_trajectories(samples)

    assert list(groups) == ["a"]


def test_validation_rejects_trajectory_starting_with_execution_rebase():
    sample = _sample("a", 0, 1, 0)
    sample.train_metadata["segment_type"] = "execution_rebase"
    with pytest.raises(ValueError, match="cannot start with execution_rebase"):
        validate_compaction_trajectories([sample])


class _LengthTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return str(sum(len(str(message.get("content") or "")) for message in messages))

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(int(text)))}


def _config() -> CompactionConfig:
    return CompactionConfig(
        context_budget=100,
        model_sequence_limit=120,
        trigger_tokens=20,
        per_turn_tokens=10,
        summary_tokens=20,
        max_compactions=3,
        recent_steps=2,
    )


def test_resume_context_reduces_recent_steps_until_it_fits():
    proxy = CompactionModelProxy(
        tokenizer=_LengthTokenizer(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=CompactionConfig(
            context_budget=320,
            model_sequence_limit=400,
            trigger_tokens=20,
            per_turn_tokens=10,
            summary_tokens=20,
            max_compactions=3,
            recent_steps=2,
        ),
    )
    proxy.working_messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "x" * 45},
        {"role": "assistant", "content": "b"},
        {"role": "tool", "content": "y" * 45},
    ]

    messages = proxy._resume_context("summary", tools=[])

    assert proxy.retained_recent_steps == (1,)
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    proxy.loop.close()


def test_summary_prompt_truncates_only_a_temporary_copy():
    proxy = CompactionModelProxy(
        tokenizer=_LengthTokenizer(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=CompactionConfig(
            context_budget=700,
            model_sequence_limit=800,
            trigger_tokens=100,
            per_turn_tokens=20,
            summary_tokens=20,
            max_compactions=3,
            recent_steps=2,
        ),
    )
    proxy.working_messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "goal"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 2000},
    ]
    original = copy.deepcopy(proxy.working_messages)

    messages, prompt_ids = proxy._summary_prompt(tools=[])

    assert len(prompt_ids) <= proxy.config.model_sequence_limit - proxy.config.summary_tokens
    assert proxy.truncated_observations > 0
    assert proxy.working_messages == original
    assert messages[2]["tool_calls"] == original[2]["tool_calls"]
    assert messages[3]["content"] != original[3]["content"]
    proxy.loop.close()


def test_summary_prompt_fails_if_only_non_tool_content_is_too_long():
    proxy = CompactionModelProxy(
        tokenizer=_LengthTokenizer(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=CompactionConfig(
            context_budget=700,
            model_sequence_limit=800,
            trigger_tokens=100,
            per_turn_tokens=20,
            summary_tokens=20,
            max_compactions=3,
            recent_steps=2,
        ),
    )
    proxy.working_messages = [{"role": "user", "content": "x" * 2000}]

    with pytest.raises(RuntimeError, match="exhausting tool-observation truncation"):
        proxy._summary_prompt(tools=[])
    proxy.loop.close()


def test_final_working_window_returns_stop_and_never_generates_again(monkeypatch):
    class _QueuedLengthTokenizer:
        def __init__(self):
            self.calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            # With three prior compactions, the proxy renders both the raw CLI
            # request and its rebuilt working context. Keep the first request
            # below C, then let the next observation fill the fourth window.
            return "2" if self.calls <= 2 else "100"

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(int(text)))}

    proxy = CompactionModelProxy(
        tokenizer=_QueuedLengthTokenizer(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=CompactionConfig(
            context_budget=100,
            model_sequence_limit=120,
            trigger_tokens=20,
            per_turn_tokens=2,
            summary_tokens=20,
            max_compactions=3,
            recent_steps=2,
        ),
    )
    generations = 0

    def generate(prompt_ids, max_new_tokens):
        nonlocal generations
        generations += 1
        return {
            "text": "first",
            "token_ids": [2, 3],
            "log_probs": [-1.0, -1.0],
            "finish_reason": "stop",
        }

    monkeypatch.setattr(proxy, "_run_generation", generate)
    proxy._compaction_count = proxy.config.max_compactions
    initial = [{"role": "user", "content": "goal"}]
    first = proxy.complete({"model": "default", "messages": initial})
    replay = [
        *initial,
        first["choices"][0]["message"],
        {"role": "user", "content": "tool observation"},
    ]
    terminal = proxy.complete({"model": "default", "messages": replay})
    replay_again = [*replay, terminal["choices"][0]["message"], {"role": "user", "content": "next"}]
    terminal_again = proxy.complete({"model": "default", "messages": replay_again})

    assert generations == 1
    assert proxy._budget_exhausted is True
    assert terminal["choices"][0]["finish_reason"] == "stop"
    assert terminal_again["choices"][0]["finish_reason"] == "stop"
    assert proxy.episode_tokens == 98
    assert proxy.ledger.current is not None
    assert sum(proxy.ledger.current.loss_mask) == 2
    assert len(proxy.ledger.current.token_ids) == 100
    proxy.loop.close()


def test_auxiliary_qwen_request_is_served_but_not_trained(monkeypatch):
    proxy = CompactionModelProxy(
        tokenizer=object(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=_config(),
    )
    rendered = iter([[1, 2], [9, 10]])
    generated = iter(
        [
            {"text": "main", "token_ids": [3], "log_probs": [-1.0], "finish_reason": "stop"},
            {"text": "auxiliary", "token_ids": [4], "log_probs": [-2.0], "finish_reason": "stop"},
        ]
    )
    monkeypatch.setattr("compaction_swe.proxy.render_prompt_ids", lambda *args: next(rendered))
    monkeypatch.setattr(proxy, "_run_generation", lambda *args: next(generated))

    proxy.complete({"model": "default", "messages": [{"role": "user", "content": "main"}]})
    before_tokens = list(proxy.ledger.current.token_ids)
    before_routing = list(proxy._routing_token_ids)
    response = proxy.complete({"model": "default", "messages": [{"role": "user", "content": "side"}]})

    assert response["choices"][0]["message"]["content"] == "auxiliary"
    assert before_tokens == [1, 2, 3]
    assert proxy.ledger.current.token_ids == before_tokens
    assert proxy._routing_token_ids == before_routing
    assert proxy.episode_tokens == 1
    assert proxy.generated_tokens == 2
    routing = proxy.routing_metadata()
    assert routing["routing_main_requests"] == 1
    assert routing["routing_main_generated_tokens"] == 1
    assert routing["routing_auxiliary_requests"] == 1
    assert routing["routing_auxiliary_generated_tokens"] == 1
    assert routing["routing_untracked_requests"] == 1
    assert routing["routing_untracked_generated_tokens"] == 1
    assert routing["routing_failed_requests"] == 0
    assert routing["routing_unexplained_generated_tokens"] == 0
    assert routing["routing_trainable_generated_tokens"] == 1
    assert proxy.failure is None
    proxy.loop.close()


def test_auxiliary_after_compaction_does_not_enter_current_segment(monkeypatch):
    proxy = _proxy(
        monkeypatch,
        token_sequences=[[1, 2, 3]],
        generations=[_completion([4], "auxiliary")],
    )
    proxy._compaction_count = 1
    proxy._raw_prefix = [{"role": "user", "content": "canonical"}]
    proxy._routing_token_ids = [10, 11, 12]
    execution = proxy.ledger.begin("execution")
    execution.append_turn([10], [11], [-1.0], "main", "tool_calls")
    before_tokens = list(execution.token_ids)

    response = proxy.complete(
        {"model": "default", "messages": [{"role": "user", "content": "helper"}]}
    )

    assert response["choices"][0]["message"]["content"] == "auxiliary"
    assert execution.token_ids == before_tokens
    routing = proxy.routing_metadata()
    assert routing["routing_auxiliary_requests"] == 1
    assert routing["routing_trainable_generated_tokens"] == 0
    assert proxy.failure is None
    proxy.loop.close()


def test_semantic_replay_with_changed_token_prefix_becomes_trainable_rebase(monkeypatch):
    proxy = CompactionModelProxy(
        tokenizer=object(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=_config(),
    )
    rendered = iter([[1, 2], [1, 2, 99]])
    generated = iter(
        [
            {"text": "main", "token_ids": [3], "log_probs": [-1.0], "finish_reason": "stop"},
            {"text": "rewrite", "token_ids": [4], "log_probs": [-2.0], "finish_reason": "stop"},
        ]
    )
    monkeypatch.setattr("compaction_swe.proxy.render_prompt_ids", lambda *args: next(rendered))
    monkeypatch.setattr(proxy, "_run_generation", lambda *args: next(generated))

    initial = [{"role": "user", "content": "goal"}]
    first = proxy.complete({"model": "default", "messages": initial})
    replay = [
        *initial,
        first["choices"][0]["message"],
        {"role": "user", "content": "observation"},
    ]
    proxy.complete({"model": "default", "messages": replay})

    assert [segment.segment_type for segment in proxy.ledger.segments] == ["execution"]
    assert proxy.ledger.segments[0].token_ids == [1, 2, 3]
    assert proxy.ledger.current is not None
    assert proxy.ledger.current.segment_type == "execution_rebase"
    assert proxy.ledger.current.token_ids == [1, 2, 99, 4]
    assert proxy.ledger.current.loss_mask == [0, 0, 0, 1]
    assert proxy.episode_tokens == 2
    assert proxy.rebase_count == 1
    routing = proxy.routing_metadata()
    assert routing["routing_main_requests"] == 1
    assert routing["routing_replay_rewrite_requests"] == 1
    assert routing["routing_replay_rewrite_generated_tokens"] == 1
    assert routing["routing_auxiliary_requests"] == 0
    assert routing["routing_trainable_generated_tokens"] == 2
    assert routing["routing_untracked_requests"] == 0
    assert routing["routing_untracked_generated_tokens"] == 0
    assert routing["routing_unexplained_generated_tokens"] == 0
    samples = proxy.to_samples(
        Sample(index=0, prompt="goal", metadata={"instance_id": "x"}),
        reward=1.0,
    )
    assert [sample.train_metadata["segment_type"] for sample in samples] == [
        "execution",
        "execution_rebase",
    ]
    assert [sample.train_metadata["optimized_tokens"] for sample in samples] == [1, 1]
    assert [sample.train_metadata["future_optimized_tokens"] for sample in samples] == [1, 0]
    assert sum(sum(sample.loss_mask) for sample in samples) == routing["routing_trainable_generated_tokens"]
    validate_compaction_trajectories(samples)
    proxy.loop.close()


def test_main_continuation_after_replay_rebase_stays_trainable(monkeypatch):
    proxy = CompactionModelProxy(
        tokenizer=object(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=_config(),
    )
    rendered = iter([[1, 2], [1, 2, 99], [1, 2, 99, 4, 5]])
    generated = iter(
        [
            {"text": "first", "token_ids": [3], "log_probs": [-1.0], "finish_reason": "stop"},
            {"text": "rewrite", "token_ids": [4], "log_probs": [-2.0], "finish_reason": "stop"},
            {"text": "third", "token_ids": [6], "log_probs": [-3.0], "finish_reason": "stop"},
        ]
    )
    monkeypatch.setattr("compaction_swe.proxy.render_prompt_ids", lambda *args: next(rendered))
    monkeypatch.setattr(proxy, "_run_generation", lambda *args: next(generated))

    initial = [{"role": "user", "content": "goal"}]
    first = proxy.complete({"model": "default", "messages": initial})
    replay = [*initial, first["choices"][0]["message"], {"role": "user", "content": "observation"}]
    second = proxy.complete({"model": "default", "messages": replay})
    continuation = [
        *replay,
        second["choices"][0]["message"],
        {"role": "user", "content": "next"},
    ]
    proxy.complete({"model": "default", "messages": continuation})

    assert proxy.ledger.current is not None
    assert proxy.ledger.current.segment_type == "execution_rebase"
    assert proxy.ledger.current.token_ids == [1, 2, 99, 4, 5, 6]
    assert proxy.ledger.current.loss_mask == [0, 0, 0, 1, 0, 1]
    routing = proxy.routing_metadata()
    assert routing["routing_main_requests"] == 2
    assert routing["routing_replay_rewrite_requests"] == 1
    assert routing["routing_trainable_generated_tokens"] == 3
    samples = proxy.to_samples(
        Sample(index=0, prompt="goal", metadata={"instance_id": "x"}),
        reward=1.0,
    )
    assert [sample.train_metadata["optimized_tokens"] for sample in samples] == [1, 2]
    assert [sample.train_metadata["future_optimized_tokens"] for sample in samples] == [2, 0]
    assert sum(sum(sample.loss_mask) for sample in samples) == routing["routing_trainable_generated_tokens"]
    validate_compaction_trajectories(samples)
    proxy.loop.close()


def test_proxy_exception_marks_whole_episode_failed(monkeypatch):
    proxy = CompactionModelProxy(
        tokenizer=object(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=_config(),
    )
    monkeypatch.setattr("compaction_swe.proxy.render_prompt_ids", lambda *args: [1, 2])
    monkeypatch.setattr(
        proxy,
        "_run_generation",
        lambda *args: (_ for _ in ()).throw(RuntimeError("backend failed")),
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        proxy.complete({"model": "default", "messages": [{"role": "user", "content": "main"}]})

    assert isinstance(proxy.failure, RuntimeError)
    assert str(proxy.failure) == "backend failed"
    assert not proxy.ledger.has_trainable_tokens()
    routing = proxy.routing_metadata()
    assert routing["routing_failed_requests"] == 1
    assert routing["routing_unexplained_generated_tokens"] == 0
    proxy.loop.close()


def test_overlapping_requests_fail_episode_before_ambiguous_generation(monkeypatch):
    proxy = CompactionModelProxy(
        tokenizer=object(),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=_config(),
    )
    started = threading.Event()
    release = threading.Event()

    def blocked_complete(_body):
        started.set()
        assert release.wait(timeout=2)
        return {"ok": True}

    monkeypatch.setattr(proxy, "_complete", blocked_complete)
    first_errors = []
    first = threading.Thread(
        target=lambda: _capture_error(first_errors, proxy.complete, {"model": "default", "messages": []})
    )
    first.start()
    assert started.wait(timeout=2)
    try:
        with pytest.raises(RuntimeError, match="causal order is undefined"):
            proxy.complete({"model": "default", "messages": []})
    finally:
        release.set()
        first.join(timeout=2)

    assert not first.is_alive()
    assert not first_errors
    assert isinstance(proxy.failure, RuntimeError)
    assert proxy.routing_metadata()["routing_concurrent_request_failures"] == 1
    proxy.loop.close()


def test_proxy_close_cancels_and_drains_generation(monkeypatch):
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    generation_started = threading.Event()
    generation_cancelled = threading.Event()

    async def blocked_generate(prompt_ids, *, max_new_tokens):
        generation_started.set()
        try:
            await asyncio.Future()
        finally:
            generation_cancelled.set()

    proxy = CompactionModelProxy(
        tokenizer=object(),
        loop=loop,
        model_url="unused",
        sampling_params={},
        config=_config(),
    ).start()
    monkeypatch.setattr(proxy, "_generate", blocked_generate)
    errors = []

    request = threading.Thread(target=lambda: _capture_error(errors, proxy._run_generation, [1, 2], 3))
    request.start()
    assert generation_started.wait(timeout=2)
    try:
        proxy.close(timeout_seconds=2)
        request.join(timeout=2)
        assert not request.is_alive()
        assert generation_cancelled.wait(timeout=2)
        assert len(errors) == 1
        assert isinstance(errors[0], concurrent.futures.CancelledError)
        assert not proxy._inflight
        assert proxy._pending_generations == 0
        assert proxy.lifecycle_metadata() == {
            "active_completions": 0,
            "inflight_futures": 0,
            "pending_generations": 0,
        }
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)
        loop.close()


@pytest.mark.asyncio
async def test_generation_uses_scoped_timeout_and_single_attempt(monkeypatch):
    calls = []

    async def fake_post(url, payload, *, max_retries, timeout):
        calls.append(
            {
                "url": url,
                "payload": payload,
                "max_retries": max_retries,
                "timeout": timeout,
            }
        )
        return {
            "text": "ok",
            "meta_info": {
                "output_token_logprobs": [[-0.5, 7]],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr("compaction_swe.proxy.post", fake_post)
    monkeypatch.setattr("compaction_swe.proxy.PROXY_HTTP_TIMEOUT_SECONDS", 3)
    monkeypatch.setattr("compaction_swe.proxy.PROXY_HTTP_MAX_ATTEMPTS", 1)
    proxy = CompactionModelProxy(
        tokenizer=object(),
        loop=asyncio.get_running_loop(),
        model_url="http://router/generate",
        sampling_params={"temperature": 0.7},
        config=_config(),
    )

    output = await proxy._generate([1, 2], max_new_tokens=4)

    assert output["token_ids"] == [7]
    assert len(calls) == 1
    assert calls[0]["max_retries"] == 1
    assert calls[0]["timeout"] == 3
    assert calls[0]["payload"]["sampling_params"]["max_new_tokens"] == 4


@pytest.mark.asyncio
async def test_generation_wall_clock_timeout_cancels_blocked_post(monkeypatch):
    cancelled = asyncio.Event()

    async def blocked_post(*_args, **_kwargs):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr("compaction_swe.proxy.post", blocked_post)
    monkeypatch.setattr("compaction_swe.proxy.PROXY_HTTP_TIMEOUT_SECONDS", 0.01)
    proxy = CompactionModelProxy(
        tokenizer=object(),
        loop=asyncio.get_running_loop(),
        model_url="http://router/generate",
        sampling_params={},
        config=_config(),
    )

    with pytest.raises(TimeoutError):
        await proxy._generate([1, 2], max_new_tokens=4)

    assert cancelled.is_set()


def _capture_error(errors, function, *args):
    try:
        function(*args)
    except BaseException as exc:
        errors.append(exc)
