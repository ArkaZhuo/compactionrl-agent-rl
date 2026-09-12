from argparse import Namespace

import pytest
from tests.fast.rollout.inference_rollout.integration.utils import (
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.rollout.inference_rollout.inference_rollout_train import sampling_request_size
from miles.utils.misc import function_registry

_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "wrong"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "wrong"},
]

_BASE_ARGV = [
    "--over-sampling-batch-size",
    "4",
    "--dynamic-sampling-filter-path",
    "test:filter_by_reward",
]


def _over_sampling_config(rollout_batch_size: int):
    return integration_env_config(["--rollout-batch-size", str(rollout_batch_size)] + _BASE_ARGV, data_rows=_DATA_ROWS)


@pytest.mark.parametrize(
    "rollout_env,expected_rounds",
    [
        pytest.param(_over_sampling_config(1), 1, id="one_round"),
        pytest.param(_over_sampling_config(2), 2, id="two_rounds"),
    ],
    indirect=["rollout_env"],
)
def test_over_sampling_rounds(rollout_env, expected_rounds):
    env = rollout_env

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    assert all(group[0].reward == 1 for group in out.samples)

    requests_count = len(env.mock_server.request_log)
    expected_requests = expected_rounds * env.args.over_sampling_batch_size
    assert requests_count == expected_requests, f"Expected {expected_rounds} round(s) = {expected_requests} requests"


def test_refill_without_dynamic_filter_only_requests_missing_groups():
    args = Namespace(over_sampling_batch_size=16)

    assert sampling_request_size(args, dynamic_filter=None, missing_groups=16) == 16
    assert sampling_request_size(args, dynamic_filter=None, missing_groups=1) == 1


def test_dynamic_filter_preserves_configured_sampling_granularity():
    args = Namespace(over_sampling_batch_size=16)

    assert sampling_request_size(args, dynamic_filter=lambda *_: True, missing_groups=1) == 16
