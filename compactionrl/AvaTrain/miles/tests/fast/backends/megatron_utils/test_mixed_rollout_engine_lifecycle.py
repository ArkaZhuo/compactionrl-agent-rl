"""Regression tests for mixed colocated/distributed rollout lifecycles."""

from argparse import Namespace
from unittest.mock import MagicMock, patch


_UW_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"


def _make_args():
    return Namespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        hf_checkpoint="/fake/path",
        lora_rank=0,
        megatron_to_hf_mode="bridge",
        pause_generation_mode="retract",
        rollout_num_gpus_per_engine=1,
        update_weight_buffer_size=1 << 30,
    )


@patch(f"{_UW_MODULE}.post_process_weights")
@patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
@patch(f"{_UW_MODULE}.connect_rollout_engines_from_distributed", return_value=MagicMock())
@patch(f"{_UW_MODULE}.get_parallel_state")
@patch(f"{_UW_MODULE}.ray")
@patch(f"{_UW_MODULE}.dist")
@patch(f"{_UW_MODULE}.HfWeightIteratorBase")
def test_mixed_update_resumes_all_rollout_engines(
    mock_iterator_base,
    mock_dist,
    mock_ray,
    mock_parallel_state,
    _mock_connect_distributed,
    _mock_gloo_group,
    _mock_post_process,
):
    """A 4+4 split must resume distributed engines as well as colocated ones."""
    from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

    mock_dist.get_world_size.return_value = 4
    mock_dist.get_rank.return_value = 0
    mock_dist.new_group.return_value = MagicMock()
    mock_parallel_state.return_value = Namespace(
        intra_dp_cp=Namespace(rank=0),
        tp=Namespace(rank=0),
        pp=Namespace(rank=0),
    )

    empty_iterator = MagicMock()
    empty_iterator.get_hf_weight_chunks.return_value = iter([])
    mock_iterator_base.create.return_value = empty_iterator

    engines = [MagicMock(name=f"engine_{index}") for index in range(8)]
    for index, engine in enumerate(engines):
        engine.begin_weight_update.remote.return_value = ("begin", index)

    def fake_ray_get(refs):
        if isinstance(refs, list) and refs and refs[0][0] == "begin":
            return [{"success": True, "supported": False} for _ in refs]
        return refs

    mock_ray.get.side_effect = fake_ray_get

    updater = UpdateWeightFromTensor(
        args=_make_args(),
        model=[MagicMock()],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
        is_lora=False,
    )
    updater.connect_rollout_engines(
        engines,
        rollout_engine_lock=MagicMock(),
        engine_gpu_counts=[1] * 8,
        engine_gpu_offsets=list(range(8)),
    )

    assert updater.all_rollout_engines == engines
    assert updater.rollout_engines == engines[:4]
    assert updater.distributed_rollout_engines == engines[4:]

    updater.update_weights()

    for engine in engines:
        engine.continue_generation.remote.assert_called_once_with()

