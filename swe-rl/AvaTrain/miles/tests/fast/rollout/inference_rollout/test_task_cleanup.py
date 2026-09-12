import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_train import abort


@pytest.mark.asyncio
async def test_group_exception_cancels_and_cleans_siblings(monkeypatch):
    sibling_started = asyncio.Event()
    sibling_cleaned = asyncio.Event()

    async def fake_generate_and_rm(state, sample, sampling_params, evaluation=False):
        if sample.index == 0:
            await sibling_started.wait()
            raise RuntimeError("generation failed")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cleaned.set()

    monkeypatch.setattr(inference_rollout_common, "generate_and_rm", fake_generate_and_rm)
    state = SimpleNamespace(
        args=SimpleNamespace(
            sglang_enable_deterministic_inference=False,
            rollout_seed=0,
            group_rm=False,
        ),
        aborted=False,
    )
    group = [SimpleNamespace(index=0), SimpleNamespace(index=1)]

    with pytest.raises(RuntimeError, match="generation failed"):
        await inference_rollout_common.generate_and_rm_group(state, group, sampling_params={})

    assert sibling_cleaned.is_set()


@pytest.mark.asyncio
async def test_abort_cancels_non_partial_pending_groups(monkeypatch):
    task_started = asyncio.Event()
    task_cleaned = asyncio.Event()

    async def pending_group():
        task_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            task_cleaned.set()

    async def no_workers(args):
        return []

    monkeypatch.setattr(
        "miles.rollout.inference_rollout.inference_rollout_train.get_worker_urls",
        no_workers,
    )
    task = asyncio.create_task(pending_group())
    await task_started.wait()
    state = SimpleNamespace(args=SimpleNamespace(partial_rollout=False), aborted=False)

    aborted_samples = await abort(state, {task}, rollout_id=0)

    assert aborted_samples == []
    assert task.cancelled()
    assert task_cleaned.is_set()
