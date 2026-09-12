from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data


def _args(**overrides):
    values = {
        "snr_filter_keep_ratio": None,
        "disable_rollout_trim_samples": False,
        "global_batch_size": 2,
        "use_dynamic_global_batch_size": False,
        "num_steps_per_rollout": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ordinary_sample(index: int):
    return SimpleNamespace(index=index, reward=1.0, train_metadata=None)


def _compaction_sample(trajectory_id: str, segment_index: int, future: int):
    metadata = {
        "compaction": True,
        "trajectory_id": trajectory_id,
        "segment_index": segment_index,
        "segment_type": "execution",
        "optimized_tokens": 1,
        "future_optimized_tokens": future,
    }
    return SimpleNamespace(train_metadata=metadata, loss_mask=[1], reward=1.0)


def test_ordinary_rollout_keeps_original_flatten_and_trim_behavior(monkeypatch):
    monkeypatch.setattr(
        "miles.ray.rollout.rollout_data_conversion.group_reward_variance",
        lambda args, group: 0.0,
    )
    monkeypatch.setattr(
        "miles.ray.rollout.rollout_data_conversion.variance_metrics",
        lambda variances: {"variance_count": len(variances)},
    )
    data = [[_ordinary_sample(0), _ordinary_sample(1)], [_ordinary_sample(2)]]

    processed, metadata = postprocess_rollout_data(_args(), data, {"dp_size": 2})

    assert [sample.index for sample in processed] == [0, 1]
    assert metadata == {"metrics": {"variance_count": 2}}


def test_compaction_rollout_aligns_before_dynamic_batch_size(monkeypatch):
    monkeypatch.setattr(
        "miles.ray.rollout.rollout_data_conversion.group_reward_variance",
        lambda args, group: 0.0,
    )
    monkeypatch.setattr(
        "miles.ray.rollout.rollout_data_conversion.variance_metrics",
        lambda variances: {},
    )
    data = [
        [_compaction_sample("a", 0, 1), _compaction_sample("a", 1, 0)],
        [_compaction_sample("b", 0, 0)],
        [_compaction_sample("c", 0, 1), _compaction_sample("c", 1, 0)],
    ]

    processed, metadata = postprocess_rollout_data(_args(use_dynamic_global_batch_size=True), data, {"dp_size": 2})

    assert len(processed) == 4
    assert {sample.train_metadata["trajectory_id"] for sample in processed} == {"a", "c"}
    assert metadata["dynamic_global_batch_size"] == 4


def test_mixed_rollout_is_rejected_in_postprocessing(monkeypatch):
    monkeypatch.setattr(
        "miles.ray.rollout.rollout_data_conversion.group_reward_variance",
        lambda args, group: 0.0,
    )
    monkeypatch.setattr(
        "miles.ray.rollout.rollout_data_conversion.variance_metrics",
        lambda variances: {},
    )
    data = [[_compaction_sample("a", 0, 0)], [_ordinary_sample(1)]]

    with pytest.raises(ValueError, match="cannot share one training batch"):
        postprocess_rollout_data(_args(), data, {"dp_size": 2})
