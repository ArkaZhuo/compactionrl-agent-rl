from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _actor_critic_dtype_broadcast_worker(rank: int, init_file: str, output_dir: str) -> None:
    from miles.backends.training_utils.data import sync_actor_critic_data

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        args = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0.0, use_kl_loss=False)
        if rank == 0:
            rollout_data = {
                "log_probs": [
                    torch.empty(0, dtype=torch.bfloat16),
                    torch.tensor([-0.2, -0.3], dtype=torch.float32),
                ]
            }
        else:
            rollout_data = {
                "values": [
                    torch.empty(0, dtype=torch.float32),
                    torch.tensor([1.25, -0.5], dtype=torch.float32),
                ]
            }

        sync_actor_critic_data(args, rollout_data, group=dist.group.WORLD)
        values = rollout_data["values"]
        torch.save(
            {
                "dtypes": [str(tensor.dtype) for tensor in values],
                "shapes": [tuple(tensor.shape) for tensor in values],
                "nonempty": values[1],
            },
            Path(output_dir) / f"actor-critic-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_empty_standard_log_probs_follow_fp32_nonempty_contract():
    from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy

    logits = torch.empty(0, 8, dtype=torch.bfloat16)
    tokens = torch.empty(0, dtype=torch.long)

    log_probs, entropy = calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=True,
    )

    assert log_probs.shape == (0,)
    assert entropy.shape == (0,)
    assert log_probs.dtype == torch.float32
    assert entropy.dtype == torch.float32


def test_actor_allocates_fp32_value_receivers_for_empty_cp_responses(monkeypatch):
    import miles.backends.training_utils.data as data

    broadcasts = []

    class _Handle:
        def wait(self):
            return None

    def fake_broadcast(tensor, *, src, group, async_op):
        broadcasts.append((tensor, src, group, async_op))
        return _Handle()

    monkeypatch.setattr(data.dist, "broadcast", fake_broadcast)
    rollout_data = {
        "log_probs": [
            torch.empty(0, dtype=torch.bfloat16),
            torch.tensor([-0.2, -0.3], dtype=torch.float32),
        ]
    }
    args = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0.0, use_kl_loss=False)

    data.sync_actor_critic_data(args, rollout_data, group="actor-critic")

    values = rollout_data["values"]
    assert [tensor.shape for tensor in values] == [torch.Size([0]), torch.Size([2])]
    assert [tensor.dtype for tensor in values] == [torch.float32, torch.float32]
    assert [item[1:] for item in broadcasts] == [
        (1, "actor-critic", True),
        (1, "actor-critic", True),
    ]


def test_actor_critic_broadcast_handles_empty_bf16_policy_template(tmp_path):
    init_file = tmp_path / "actor-critic-gloo-init"
    mp.spawn(
        _actor_critic_dtype_broadcast_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    for rank in range(2):
        result = torch.load(tmp_path / f"actor-critic-rank-{rank}.pt", weights_only=True)
        assert result["dtypes"] == ["torch.float32", "torch.float32"]
        assert result["shapes"] == [(0,), (2,)]
        assert torch.equal(result["nonempty"], torch.tensor([1.25, -0.5]))
