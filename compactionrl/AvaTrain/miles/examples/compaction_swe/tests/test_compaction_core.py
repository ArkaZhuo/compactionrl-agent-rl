from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from compaction_swe.advantages import _chunked_variable_lambda_gae
from compaction_swe.config import CompactionConfig
from compaction_swe.prompts import resume_messages, split_atomic_steps
from compaction_swe.proxy import CompactionModelProxy, _align_tool_call_ids, _canonical_message
from compaction_swe.trajectory import SegmentLedger
from miles.utils.types import Sample


class _QueuedTokenizer:
    def __init__(self, token_sequences: list[list[int]]):
        self._token_sequences = iter(token_sequences)

    def apply_chat_template(self, messages, **kwargs):
        return "rendered"

    def __call__(self, text, **kwargs):
        return {"input_ids": next(self._token_sequences)}


def _config(*, context_budget: int = 10, model_sequence_limit: int = 14) -> CompactionConfig:
    return CompactionConfig(
        context_budget=context_budget,
        model_sequence_limit=model_sequence_limit,
        trigger_tokens=2,
        per_turn_tokens=3,
        summary_tokens=2,
        max_compactions=3,
        recent_steps=1,
    )


def _completion(token_ids: list[int], text: str) -> dict:
    return {
        "token_ids": token_ids,
        "log_probs": [-1.0] * len(token_ids),
        "text": text,
        "finish_reason": "stop",
    }


def _proxy(monkeypatch, token_sequences: list[list[int]], generations: list[dict], config=None):
    proxy = CompactionModelProxy(
        tokenizer=_QueuedTokenizer(token_sequences),
        loop=asyncio.new_event_loop(),
        model_url="unused",
        sampling_params={},
        config=config or _config(),
    )
    queued_generations = iter(generations)
    monkeypatch.setattr(proxy, "_run_generation", lambda prompt_ids, max_new_tokens: next(queued_generations))
    return proxy


def test_atomic_steps_keep_tool_observations_together():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "act-1", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "obs-1"},
        {"role": "assistant", "content": "act-2", "tool_calls": [{"id": "b"}]},
        {"role": "tool", "tool_call_id": "b", "content": "obs-2"},
        {"role": "assistant", "content": "done"},
    ]
    prefix, steps = split_atomic_steps(messages)
    assert [m["role"] for m in prefix] == ["system", "user"]
    assert len(steps) == 3
    assert [m["role"] for m in steps[0]] == ["assistant", "tool"]
    assert steps[0][1]["content"] == "obs-1"


def test_resume_keeps_system_summary_and_last_atomic_steps():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "old"},
        {"role": "tool", "content": "old-observation"},
        {"role": "assistant", "content": "recent"},
        {"role": "tool", "content": "recent-observation"},
    ]
    resumed = resume_messages(messages, "summary", recent_steps=1)
    assert [m["role"] for m in resumed] == ["system", "user", "assistant", "tool"]
    assert resumed[1]["content"].endswith("summary")
    assert resumed[-1]["content"] == "recent-observation"
    assert all(m.get("content") != "goal" for m in resumed[1:])


def test_segment_metadata_and_future_token_counts():
    base = Sample(index=7, prompt="goal", metadata={"instance_id": "x"})
    ledger = SegmentLedger(trajectory_id="trace")
    first = ledger.begin("execution")
    first.append_turn([10, 11], [12, 13], [-1.0, -2.0], "a", "tool_calls")
    ledger.finish_current()
    second = ledger.begin("summary")
    second.append_turn([20], [21], [-3.0], "s", "stop")
    samples = ledger.to_samples(base, reward=1.0, max_sequence_length=20, compaction_count=1)
    assert len(samples) == 2
    assert samples[0].train_metadata["future_optimized_tokens"] == 1
    assert samples[1].train_metadata["future_optimized_tokens"] == 0
    assert samples[0].loss_mask == [1, 1]
    assert samples[1].loss_mask == [1]
    assert all(sample.reward == 1.0 for sample in samples)


def test_variable_lambda_gae_matches_two_token_hand_calculation():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
    values = torch.zeros_like(rewards)
    lengths = torch.tensor([2, 2])
    lambdas = torch.tensor([0.5, 0.25])
    result = _chunked_variable_lambda_gae(rewards, values, lengths, lambdas, gamma=1.0)
    assert torch.allclose(result, torch.tensor([[0.5, 1.0], [0.5, 2.0]]))


def test_empty_tail_after_compaction_is_not_emitted():
    base = Sample(index=8, prompt="goal", metadata={"instance_id": "x"})
    ledger = SegmentLedger(trajectory_id="trace")
    execution = ledger.begin("execution")
    execution.append_turn([1], [2], [-1.0], "a", "tool_calls")
    ledger.finish_current()
    summary = ledger.begin("summary")
    summary.append_turn([3], [4], [-2.0], "s", "stop")
    ledger.finish_current()
    ledger.begin("execution")
    samples = ledger.to_samples(base, reward=0.5, max_sequence_length=8, compaction_count=1)
    assert [sample.train_metadata["segment_type"] for sample in samples] == ["execution", "summary"]


def test_replayed_tool_ids_are_aligned_to_proxy_history():
    previous = [
        {"role": "assistant", "tool_calls": [{"id": "proxy-id", "function": {"name": "read"}}]},
    ]
    replayed = [
        {"role": "assistant", "tool_calls": [{"id": "client-id", "function": {"name": "read"}}]},
    ]
    tail = [{"role": "tool", "tool_call_id": "client-id", "content": "ok"}]
    normalized = _align_tool_call_ids(previous, replayed, tail)
    assert normalized[0]["tool_call_id"] == "proxy-id"
    assert _canonical_message(
        {"role": "assistant", "tool_calls": [{"id": "a", "function": {"arguments": '{"path":"x"}'}}]}
    ) == _canonical_message(
        {"role": "assistant", "tool_calls": [{"id": "b", "function": {"arguments": {"path": "x"}}}]}
    )


def test_first_request_above_trigger_generates_before_compaction(monkeypatch):
    proxy = _proxy(
        monkeypatch,
        token_sequences=[list(range(8))],
        generations=[_completion([8, 9, 10], "first")],
    )

    proxy.complete({"model": "default", "messages": [{"role": "user", "content": "goal"}]})

    assert proxy.compaction_count == 0
    assert proxy.ledger.current is not None
    assert proxy.ledger.current.has_trainable_tokens()
    assert proxy.generated_tokens == 3
    assert proxy.routing_metadata()["routing_unexplained_generated_tokens"] == 0
    assert proxy.episode_tokens == 3


def test_realistic_20k_initial_prompt_fits_64k_working_window(monkeypatch):
    config = CompactionConfig(
        context_budget=65536,
        model_sequence_limit=262144,
        trigger_tokens=10240,
        per_turn_tokens=2048,
        summary_tokens=2048,
        max_compactions=3,
        recent_steps=2,
    )
    requested = []
    proxy = _proxy(
        monkeypatch,
        token_sequences=[list(range(20827))],
        generations=[_completion([20827, 20828], "first")],
        config=config,
    )
    monkeypatch.setattr(
        proxy,
        "_run_generation",
        lambda prompt_ids, max_new_tokens: (requested.append(max_new_tokens) or _completion([20827, 20828], "first")),
    )

    proxy.complete({"model": "default", "messages": [{"role": "user", "content": "goal"}]})

    assert requested == [2048]
    assert proxy.compaction_count == 0
    assert proxy.ledger.current is not None
    assert proxy.ledger.current.has_trainable_tokens()
    assert proxy.episode_tokens == 2


def test_next_request_compacts_nonempty_execution_and_generates_new_turn(monkeypatch):
    proxy = _proxy(
        monkeypatch,
        token_sequences=[
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [20, 21],
            [30, 31],
            [30, 31],
        ],
        generations=[
            _completion([6, 7], "first"),
            _completion([22], "summary"),
            _completion([32], "after-summary"),
        ],
    )
    initial = [{"role": "user", "content": "goal"}]
    proxy.complete({"model": "default", "messages": initial})
    replay = [
        *initial,
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
    ]

    proxy.complete({"model": "default", "messages": replay})

    assert proxy.compaction_count == 1
    assert [segment.segment_type for segment in proxy.ledger.segments] == ["execution", "summary"]
    assert proxy.ledger.current is not None
    assert proxy.ledger.current.segment_type == "execution"
    assert proxy.ledger.current.has_trainable_tokens()
    # 2 first-turn tokens + 1 new observation + 1 summary + 1 resumed action.
    assert proxy.episode_tokens == 5
    routing = proxy.routing_metadata()
    assert routing["routing_main_requests"] == 2
    assert routing["routing_main_generated_tokens"] == 3
    assert routing["routing_summary_requests"] == 1
    assert routing["routing_summary_generated_tokens"] == 1
    assert routing["routing_unexplained_generated_tokens"] == 0


def test_initial_prompt_is_excluded_but_later_observation_consumes_budget(monkeypatch):
    proxy = _proxy(
        monkeypatch,
        token_sequences=[
            list(range(20)),
            list(range(23)),
        ],
        generations=[
            _completion([20, 21], "first"),
            _completion([23], "second"),
        ],
        config=_config(context_budget=100, model_sequence_limit=128),
    )
    initial = [{"role": "user", "content": "goal"}]
    proxy.complete({"model": "default", "messages": initial})
    replay = [
        *initial,
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "observation"},
    ]
    proxy.complete({"model": "default", "messages": replay})

    assert proxy.episode_tokens == 4  # 2 generated + 1 observation + 1 generated


def test_segments_are_not_truncated_by_a_global_trajectory_limit():
    base = Sample(index=9, prompt="goal", metadata={"instance_id": "x"})
    ledger = SegmentLedger(trajectory_id="trace")
    first = ledger.begin("execution")
    first.append_turn([1, 2], [3, 4], [-1.0, -1.0], "first", "stop")
    first.append_context([1, 2, 3, 4, 5])
    ledger.finish_current()
    second = ledger.begin("summary")
    second.append_turn([10], [11, 12], [-1.0, -1.0], "summary", "stop")

    samples = ledger.to_samples(base, reward=1.0, max_sequence_length=20, compaction_count=1)

    assert [sample.response_length for sample in samples] == [3, 2]
    assert sum(sample.response_length for sample in samples) == 5
    assert all(sample.status == Sample.Status.COMPLETED for sample in samples)
    assert samples[0].train_metadata["future_optimized_tokens"] == 2


def test_empty_execution_at_sequence_limit_fails_without_generation(monkeypatch):
    proxy = _proxy(
        monkeypatch,
        token_sequences=[list(range(14))],
        generations=[],
    )

    with pytest.raises(RuntimeError, match="cannot start an execution segment"):
        proxy.complete({"model": "default", "messages": [{"role": "user", "content": "goal"}]})

    assert proxy.compaction_count == 0
    assert proxy.ledger.current is not None
    assert not proxy.ledger.current.has_trainable_tokens()
