from typing import Any

import numpy as np
import pybase64


def is_boxed_ray_ref(value: Any) -> bool:
    if not hasattr(value, "inner"):
        return False
    try:
        import ray
    except ImportError:
        return False
    return isinstance(value.inner, ray.ObjectRef)


def decode_routed_experts(
    routed_experts: str,
    num_tokens: int,
    num_layers: int,
    moe_router_topk: int,
) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode(routed_experts.encode("ascii")),
        dtype=np.int32,
    ).reshape(
        num_tokens,
        num_layers,
        moe_router_topk,
    )


def resolve_routed_experts(
    routed_experts: Any,
    num_tokens: int,
    num_layers: int,
    moe_router_topk: int,
):
    if is_boxed_ray_ref(routed_experts):
        import ray

        routed_experts = ray.get(routed_experts.inner)
    if isinstance(routed_experts, str):
        return decode_routed_experts(routed_experts, num_tokens, num_layers, moe_router_topk)
    return routed_experts
