from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import compaction_swe.advantages as compaction_advantages
from compaction_swe.advantages import compute_compaction_advantages


def _parallel_state(cp_size: int, cp_rank: int = 0):
    return SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=cp_rank, group=None))


def _rollout(values: torch.Tensor) -> dict:
    return {
        "values": [values],
        "loss_masks": [torch.tensor([0, 1, 1, 0, 1, 1, 1, 1])],
        "rewards": [1.25],
        "metadata": [
            {
                "compaction": True,
                "trajectory_id": "trajectory-0",
                "segment_index": 0,
                "optimized_tokens": 6,
                "future_optimized_tokens": 3,
            }
        ],
        "total_lengths": [12],
        "response_lengths": [8],
    }


def _args():
    return SimpleNamespace(gamma=0.9, kl_coef=0.2, compaction_gae_alpha=1.5, qkv_format="thd")


def _args_without_kl():
    return SimpleNamespace(gamma=0.9, kl_coef=0.0, compaction_gae_alpha=1.5, qkv_format="thd")


def test_advantage_whitening_includes_context_parallel_ranks(monkeypatch):
    import miles.backends.training_utils.loss_hub.advantages as base_advantages

    captured = {}

    def fake_whiten(values, mask, *, process_group, shift_mean):
        captured["group"] = process_group
        captured["mask"] = mask
        return values

    monkeypatch.setattr(base_advantages, "distributed_masked_whiten", fake_whiten)
    monkeypatch.setattr(
        base_advantages,
        "get_logits_and_tokens_offset_with_cp",
        lambda *args: (2, None, None, ((0, 0), (7, 8))),
    )
    monkeypatch.setattr(
        base_advantages,
        "get_parallel_state",
        lambda: SimpleNamespace(
            cp=SimpleNamespace(size=2, rank=0),
            intra_dp=SimpleNamespace(group="dp-only"),
            intra_dp_cp=SimpleNamespace(group="dp-and-cp"),
        ),
    )

    # CP rank 0 owns only the last response token for this short layout.
    advantages = [torch.tensor([4.0])]
    masks = [torch.ones(4, dtype=torch.int)]
    result = base_advantages.normalize_advantages(
        SimpleNamespace(qkv_format="thd"),
        advantages,
        masks,
        total_lengths=[8],
        response_lengths=[4],
    )

    assert captured["group"] == "dp-and-cp"
    assert torch.equal(captured["mask"], torch.ones(1, dtype=torch.int))
    assert torch.equal(result[0], advantages[0])


def _distributed_cp_gather_worker(rank: int, init_file: str, output_dir: str) -> None:
    import miles.backends.training_utils.cp_utils as cp_utils
    import miles.backends.training_utils.parallel as parallel

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        state = SimpleNamespace(cp=SimpleNamespace(size=2, rank=rank, group=dist.group.WORLD))
        parallel.get_parallel_state = lambda: state
        cp_utils.get_parallel_state = lambda: state
        total_lengths = [12, 13]
        response_lengths = [8, 7]
        full_values = [
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]),
        ]
        local_values = [
            cp_utils.slice_log_prob_with_cp(value, total_length, response_length)
            for value, total_length, response_length in zip(
                full_values,
                total_lengths,
                response_lengths,
                strict=True,
            )
        ]
        gathered = compaction_advantages._gather_full_responses_with_cp(
            local_values,
            total_lengths,
            response_lengths,
            "thd",
            None,
            [("trajectory-0", 0), ("trajectory-1", 2)],
        )
        torch.save(gathered, Path(output_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def _distributed_empty_cp_gather_worker(rank: int, init_file: str, output_dir: str) -> None:
    import miles.backends.training_utils.cp_utils as cp_utils
    import miles.backends.training_utils.parallel as parallel

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        state = SimpleNamespace(cp=SimpleNamespace(size=2, rank=rank, group=dist.group.WORLD))
        parallel.get_parallel_state = lambda: state
        cp_utils.get_parallel_state = lambda: state
        total_lengths = [12, 16]
        response_lengths = [2, 3]
        full_values = [torch.tensor([0.1, 0.2]), torch.tensor([1.1, 1.2, 1.3])]
        if rank == 0:
            local_values = [
                cp_utils.slice_log_prob_with_cp(value, total_length, response_length)
                for value, total_length, response_length in zip(
                    full_values,
                    total_lengths,
                    response_lengths,
                    strict=True,
                )
            ]
        else:
            # This is how Megatron represents a CP rank whose forward pass
            # produced no response-aligned non-loss tensors.
            local_values = []
        gathered = compaction_advantages._gather_full_responses_with_cp(
            local_values,
            total_lengths,
            response_lengths,
            "thd",
            None,
            [("trajectory-0", 0), ("trajectory-1", 0)],
            torch.zeros(1),
        )
        torch.save(gathered, Path(output_dir) / f"empty-rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_compaction_advantages_cp1_preserves_response_layout(monkeypatch):
    import miles.backends.training_utils.parallel as parallel

    monkeypatch.setattr(parallel, "get_parallel_state", lambda: _parallel_state(1))
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    kl = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

    advantages, returns = compute_compaction_advantages(_args(), _rollout(values), [kl])

    assert advantages[0].shape == values.shape
    assert returns[0].shape == values.shape
    assert advantages[0][[0, 3]].tolist() == [0.0, 0.0]
    assert torch.equal(returns[0][[0, 3]], values[[0, 3]])


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_compaction_advantages_cp2_matches_cp1_after_zigzag_reslice(monkeypatch, cp_rank):
    import miles.backends.training_utils.cp_utils as cp_utils
    import miles.backends.training_utils.parallel as parallel

    full_values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    full_kl = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

    monkeypatch.setattr(parallel, "get_parallel_state", lambda: _parallel_state(1))
    baseline_advantages, baseline_returns = compute_compaction_advantages(
        _args(), _rollout(full_values), [full_kl]
    )

    state = _parallel_state(2, cp_rank)
    monkeypatch.setattr(parallel, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    local_values = cp_utils.slice_log_prob_with_cp(full_values, 12, 8)
    local_kl = cp_utils.slice_log_prob_with_cp(full_kl, 12, 8)

    gathered = iter([full_values, full_kl])
    monkeypatch.setattr(
        compaction_advantages,
        "_gather_full_responses_with_cp",
        lambda *args, **kwargs: [next(gathered)],
    )
    advantages, returns = compute_compaction_advantages(_args(), _rollout(local_values), [local_kl])

    expected_advantages = cp_utils.slice_log_prob_with_cp(baseline_advantages[0], 12, 8)
    expected_returns = cp_utils.slice_log_prob_with_cp(baseline_returns[0], 12, 8)
    assert torch.allclose(advantages[0], expected_advantages)
    assert torch.allclose(returns[0], expected_returns)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_compaction_advantages_cp2_skips_zero_kl_gather(monkeypatch, cp_rank):
    import miles.backends.training_utils.cp_utils as cp_utils
    import miles.backends.training_utils.parallel as parallel

    full_values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    full_kl = torch.zeros_like(full_values)

    monkeypatch.setattr(parallel, "get_parallel_state", lambda: _parallel_state(1))
    baseline_advantages, baseline_returns = compute_compaction_advantages(
        _args_without_kl(), _rollout(full_values), [full_kl]
    )

    state = _parallel_state(2, cp_rank)
    monkeypatch.setattr(parallel, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    local_values = cp_utils.slice_log_prob_with_cp(full_values, 12, 8)
    local_kl = cp_utils.slice_log_prob_with_cp(full_kl, 12, 8)
    gather_calls = []

    def gather_values_only(tensors, *args, **kwargs):
        gather_calls.append(tensors)
        return [full_values]

    monkeypatch.setattr(compaction_advantages, "_gather_full_responses_with_cp", gather_values_only)
    advantages, returns = compute_compaction_advantages(
        _args_without_kl(), _rollout(local_values), [local_kl]
    )

    assert gather_calls == [[local_values]]
    expected_advantages = cp_utils.slice_log_prob_with_cp(baseline_advantages[0], 12, 8)
    expected_returns = cp_utils.slice_log_prob_with_cp(baseline_returns[0], 12, 8)
    assert torch.allclose(advantages[0], expected_advantages)
    assert torch.allclose(returns[0], expected_returns)


def test_compaction_advantages_rejects_response_length_mismatch(monkeypatch):
    import miles.backends.training_utils.parallel as parallel

    monkeypatch.setattr(parallel, "get_parallel_state", lambda: _parallel_state(1))
    rollout = _rollout(torch.zeros(7))
    with pytest.raises(RuntimeError, match="value, KL, and loss-mask lengths differ"):
        compute_compaction_advantages(_args(), rollout, [torch.zeros(7)])


def test_cp2_variable_response_batch_reconstructs_cp1_values(monkeypatch):
    import miles.backends.training_utils.cp_utils as cp_utils

    total_lengths = [12, 13]
    response_lengths = [8, 7]
    full_values = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]),
    ]

    contributions = []
    for cp_rank in (0, 1):
        state = _parallel_state(2, cp_rank)
        monkeypatch.setattr(cp_utils, "get_parallel_state", lambda state=state: state)
        local_values = [
            cp_utils.slice_log_prob_with_cp(value, total_length, response_length)
            for value, total_length, response_length in zip(
                full_values,
                total_lengths,
                response_lengths,
                strict=True,
            )
        ]
        contributions.append(
            compaction_advantages._build_local_cp_response_batch(
                local_values,
                total_lengths,
                response_lengths,
                "thd",
                None,
            )
        )

    reconstructed = contributions[0] + contributions[1]
    assert torch.equal(reconstructed[0, : response_lengths[0]], full_values[0])
    assert torch.equal(reconstructed[1, : response_lengths[1]], full_values[1])
    assert torch.equal(
        reconstructed[1, response_lengths[1] :],
        torch.zeros_like(reconstructed[1, response_lengths[1] :]),
    )


def test_cp2_fixed_shape_collectives_reconstruct_variable_responses(tmp_path):
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _distributed_cp_gather_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    expected = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]),
    ]
    for rank in (0, 1):
        gathered = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        assert len(gathered) == len(expected)
        assert all(torch.equal(actual, wanted) for actual, wanted in zip(gathered, expected, strict=True))


def test_cp2_rank_with_empty_local_forward_still_joins_collectives(tmp_path):
    init_file = tmp_path / "empty-gloo-init"
    mp.spawn(
        _distributed_empty_cp_gather_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    expected = [torch.tensor([0.1, 0.2]), torch.tensor([1.1, 1.2, 1.3])]
    for rank in (0, 1):
        gathered = torch.load(tmp_path / f"empty-rank-{rank}.pt", weights_only=True)
        assert len(gathered) == len(expected)
        assert all(torch.equal(actual, wanted) for actual, wanted in zip(gathered, expected, strict=True))


def test_cuda_collective_is_completed_before_return(monkeypatch):
    synchronized = []
    fake_tensor = SimpleNamespace(device=torch.device("cuda", 3))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: synchronized.append(device))

    compaction_advantages._synchronize_cuda_collective(fake_tensor)

    assert synchronized == [torch.device("cuda", 3)]


def test_cpu_collective_does_not_call_cuda_synchronize(monkeypatch):
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: synchronized.append(device))

    compaction_advantages._synchronize_cuda_collective(torch.zeros(1))

    assert synchronized == []


def test_custom_advantage_runs_on_pp_last_cp_rank_without_local_outputs(monkeypatch):
    import miles.backends.training_utils.loss as training_loss

    called = []

    def custom_advantage(args, rollout_data, kl):
        called.append((rollout_data["log_probs"], rollout_data["values"], kl))
        return [], []

    monkeypatch.setattr(
        training_loss,
        "get_parallel_state",
        lambda: SimpleNamespace(is_pp_last_stage=True),
    )
    monkeypatch.setattr(training_loss, "load_function", lambda path: custom_advantage)
    args = SimpleNamespace(
        use_rollout_logprobs=False,
        custom_advantage_function_path="compaction_swe.advantages.compute_compaction_advantages",
        kl_coef=0.0,
        normalize_advantages=False,
    )
    rollout_data = {
        "rewards": [],
        "response_lengths": [],
        "loss_masks": [],
        "total_lengths": [],
    }

    training_loss.compute_advantages_and_returns(args, rollout_data)

    assert called == [([], [], [])]
    assert rollout_data["advantages"] == []
    assert rollout_data["returns"] == []


def test_custom_advantage_still_skips_pp_intermediate_rank(monkeypatch):
    import miles.backends.training_utils.loss as training_loss

    monkeypatch.setattr(
        training_loss,
        "get_parallel_state",
        lambda: SimpleNamespace(is_pp_last_stage=False),
    )
    monkeypatch.setattr(
        training_loss,
        "load_function",
        lambda path: pytest.fail("custom advantage must not run on an intermediate PP stage"),
    )
    args = SimpleNamespace(
        use_rollout_logprobs=False,
        custom_advantage_function_path="compaction_swe.advantages.compute_compaction_advantages",
    )
    rollout_data = {
        "rewards": [],
        "response_lengths": [],
        "loss_masks": [],
        "total_lengths": [],
    }

    training_loss.compute_advantages_and_returns(args, rollout_data)

    assert "advantages" not in rollout_data
