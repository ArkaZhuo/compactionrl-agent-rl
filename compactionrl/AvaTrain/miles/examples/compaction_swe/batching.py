"""Trajectory-aware validation and DP alignment for compacted rollouts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlignmentResult:
    samples: list[Any]
    removed_trajectory_ids: tuple[str, ...]
    removed_segments: int


def is_compaction_sample(sample: Any) -> bool:
    metadata = getattr(sample, "train_metadata", None)
    return isinstance(metadata, dict) and metadata.get("compaction") is True


def _trajectory_groups(samples: list[Any]) -> OrderedDict[str, list[Any]]:
    groups: OrderedDict[str, list[Any]] = OrderedDict()
    for sample in samples:
        metadata = sample.train_metadata
        trajectory_id = metadata.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("CompactionRL sample is missing a non-empty trajectory_id")
        groups.setdefault(trajectory_id, []).append(sample)
    return groups


def validate_compaction_trajectories(samples: list[Any]) -> OrderedDict[str, list[Any]]:
    """Validate the segment chain and cross-segment token distances."""
    if not samples:
        raise ValueError("CompactionRL rollout contains no samples")
    flags = [is_compaction_sample(sample) for sample in samples]
    if any(flags) and not all(flags):
        raise ValueError("CompactionRL and ordinary samples cannot share one training batch")
    if not all(flags):
        return OrderedDict()

    groups = _trajectory_groups(samples)
    for trajectory_id, trajectory in groups.items():
        metadata = [sample.train_metadata for sample in trajectory]
        indices = [item.get("segment_index") for item in metadata]
        if indices != list(range(len(trajectory))):
            raise ValueError(f"CompactionRL trajectory {trajectory_id} has non-contiguous segment indices: {indices}")

        rewards = [float(sample.reward) for sample in trajectory]
        if any(reward != rewards[0] for reward in rewards[1:]):
            raise ValueError(f"CompactionRL trajectory {trajectory_id} has inconsistent rewards: {rewards}")

        segment_types = [item.get("segment_type") for item in metadata]
        allowed_types = {"execution", "execution_rebase", "summary"}
        if any(segment_type not in allowed_types for segment_type in segment_types):
            raise ValueError(f"CompactionRL trajectory {trajectory_id} has invalid segment types: {segment_types}")
        if segment_types[0] == "execution_rebase":
            raise ValueError(f"CompactionRL trajectory {trajectory_id} cannot start with execution_rebase")

        optimized = []
        for sample, item in zip(trajectory, metadata, strict=True):
            mask_tokens = int(sum(sample.loss_mask or []))
            declared_tokens = item.get("optimized_tokens")
            if declared_tokens != mask_tokens or mask_tokens <= 0:
                raise ValueError(
                    f"CompactionRL trajectory {trajectory_id} segment {item.get('segment_index')} "
                    f"optimized_tokens={declared_tokens}, actual={mask_tokens}"
                )
            optimized.append(mask_tokens)

        future = 0
        for item, token_count in zip(reversed(metadata), reversed(optimized), strict=True):
            declared_future = item.get("future_optimized_tokens")
            if declared_future != future:
                raise ValueError(
                    f"CompactionRL trajectory {trajectory_id} segment {item.get('segment_index')} "
                    f"future_optimized_tokens={declared_future}, expected={future}"
                )
            future += token_count

    return groups


def align_compaction_trajectories(samples: list[Any], divisor: int) -> AlignmentResult:
    """Drop the smallest complete-trajectory set needed for DP divisibility."""
    if divisor <= 0:
        raise ValueError(f"alignment divisor must be positive, got {divisor}")
    groups = validate_compaction_trajectories(samples)
    if not groups or len(samples) % divisor == 0:
        return AlignmentResult(list(samples), (), 0)

    target = len(samples) % divisor
    # residue -> (removed_segment_count, removed_trajectory_ids). Keeping only
    # one best candidate per residue is sufficient because future transitions
    # depend only on that residue and additive removal cost.
    states: dict[int, tuple[int, tuple[str, ...]]] = {0: (0, ())}
    for trajectory_id, trajectory in groups.items():
        segment_count = len(trajectory)
        updated = dict(states)
        for residue, (removed, ids) in states.items():
            next_residue = (residue + segment_count) % divisor
            candidate = (removed + segment_count, (*ids, trajectory_id))
            current = updated.get(next_residue)
            if current is None or (candidate[0], len(candidate[1])) < (current[0], len(current[1])):
                updated[next_residue] = candidate
        states = updated

    candidate = states.get(target)
    if candidate is None or candidate[0] >= len(samples):
        raise ValueError(
            f"cannot align {len(samples)} CompactionRL segments to divisor {divisor} "
            "without dropping every trajectory"
        )

    removed_segments, removed_ids = candidate
    removed_set = set(removed_ids)
    aligned = [sample for sample in samples if sample.train_metadata["trajectory_id"] not in removed_set]
    if not aligned or len(aligned) % divisor:
        raise AssertionError("trajectory-aware alignment produced an invalid sample count")
    validate_compaction_trajectories(aligned)
    return AlignmentResult(aligned, removed_ids, removed_segments)
