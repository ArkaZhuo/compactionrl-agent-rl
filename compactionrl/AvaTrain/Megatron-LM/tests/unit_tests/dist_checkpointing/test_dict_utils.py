# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.dist_checkpointing.dict_utils import merge


def test_merge_optimizer_param_state_appends_local_padding_entries():
    common_state = {
        "optimizer": {
            "param_state": [
                {"step": 1},
                {"step": 1},
            ]
        }
    }
    local_state = {
        "optimizer": {
            "param_state": [
                {"padding": False},
                {"padding": False},
                {"padding": True},
            ]
        }
    }

    assert merge(common_state, local_state) == {
        "optimizer": {
            "param_state": [
                {"step": 1, "padding": False},
                {"step": 1, "padding": False},
                {"padding": True},
            ]
        }
    }


def test_merge_still_rejects_unrelated_lists_with_different_lengths():
    with pytest.raises(ValueError, match="different lengths"):
        merge({"model": [dict()]}, {"model": [dict(), dict()]})
