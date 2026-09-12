import logging
from pathlib import Path

import torch

from miles.utils.routed_experts import is_boxed_ray_ref, resolve_routed_experts
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def load_debug_rollout_data(args, rollout_id: int):
    data = torch.load(
        args.load_debug_rollout_data.format(rollout_id=rollout_id),
        weights_only=False,
    )["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    return data


def _sample_to_debug_dict(args, sample: Sample):
    data = sample.to_dict()
    routed_experts = data.get("rollout_routed_experts")
    if routed_experts is not None and (is_boxed_ray_ref(routed_experts) or isinstance(routed_experts, str)):
        data["rollout_routed_experts"] = resolve_routed_experts(
            routed_experts,
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )
    return data


def save_debug_rollout_data(args, data, rollout_id, evaluation: bool):
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO may improve the format
        if evaluation:
            dump_data = dict(
                samples=[
                    _sample_to_debug_dict(args, sample)
                    for dataset_name, info in data.items()
                    for sample in info["samples"]
                ]
            )
        else:
            dump_data = dict(
                samples=[_sample_to_debug_dict(args, sample) for sample in data],
            )

        torch.save(dict(rollout_id=rollout_id, **dump_data), path)
