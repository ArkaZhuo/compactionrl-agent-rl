import asyncio
import logging
import os
from argparse import Namespace
from collections.abc import Callable

import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils import dumper_utils
from miles.utils.http_utils import get, post
from miles.utils.misc import as_completed_async, load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _nonnegative_env(name: str) -> int:
    raw = os.environ.get(name, "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value


async def _trip_rollout_fuse(state: GenerateState, pendings: set, pbar, message: str) -> None:
    """Cancel task-owned resources before failing a bounded rollout.

    These limits are opt-in environment controls used by CompactionRL.  Local
    task cancellation is sufficient here: generate_and_rm_group propagates it
    into generate.py, whose finally blocks close the proxy and kill the live
    Sandbox.  The fatal exception then lets the launcher tear down the rollout
    workers instead of spending hours replenishing an unusable batch.
    """
    logger.error(message)
    state.aborted = True
    for task in pendings:
        task.cancel()
    if pendings:
        await asyncio.gather(*pendings, return_exceptions=True)
    pbar.close()
    state.reset()
    raise RuntimeError(message)


async def abort(state: GenerateState, pendings: set, rollout_id: int) -> list[list[Sample]]:
    args = state.args

    assert not state.aborted
    state.aborted = True

    urls = await get_worker_urls(args)
    logger.info(f"Abort request for {urls}")
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])

    # A non-partial rollout has no use for surplus groups after the target
    # batch is complete. Waiting for them can stall a training step on a slow
    # Agent or grader even though enough samples were already collected.
    # Cancellation propagates through generate_and_rm_group, whose cleanup
    # path releases live Sandboxes before returning.
    if not args.partial_rollout:
        for task in pendings:
            task.cancel()
        if pendings:
            await asyncio.gather(*pendings, return_exceptions=True)
            logger.info("Cancelled and cleaned up %d surplus rollout groups", len(pendings))
        return []

    # Partial rollouts retain unfinished responses for the next iteration, so
    # they still need to finish after the inference requests are aborted.
    aborted_samples = []
    async for group in as_completed_async(pendings):
        # Collect partial samples into the data buffer.
        for sample in group:
            if sample.response and "start_rollout_id" not in sample.metadata:
                sample.metadata["start_rollout_id"] = rollout_id
        aborted_samples.append(group)

    logger.info(f"Collected {sum(len(x) for x in aborted_samples)} partial samples into the data buffer")

    return aborted_samples


def sampling_request_size(args: Namespace, dynamic_filter: Callable | None, missing_groups: int) -> int:
    """Return the refill size without creating unnecessary surplus work.

    Dynamic filtering deliberately samples in configured chunks. Without a
    filter, requesting more than the exact deficit only creates surplus
    Agent/grader work after an infrastructure failure.
    """
    if dynamic_filter is None:
        return min(args.over_sampling_batch_size, missing_groups)
    return args.over_sampling_batch_size


async def get_worker_urls(args: Namespace):
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        return response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        return [worker["url"] for worker in response["workers"]]


def submit_generate_tasks(state: GenerateState, samples: list[list[Sample]]):
    return [
        asyncio.create_task(
            # submit a group of samples as a single task.
            generate_and_rm_group(
                state,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
            )
        )
        for group in samples
    ]


async def generate_rollout_async(
    state: GenerateState, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    args = state.args
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    # instantiate data filters
    dynamic_filter = load_function(args.dynamic_sampling_filter_path)

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size
    max_attempts = _nonnegative_env("COMPACTION_ROLLOUT_MAX_ATTEMPTS")
    max_discarded = _nonnegative_env("COMPACTION_ROLLOUT_MAX_DISCARDED")
    if max_attempts and max_attempts < target_data_size:
        raise ValueError(
            "COMPACTION_ROLLOUT_MAX_ATTEMPTS must be zero (disabled) or at least "
            f"rollout_batch_size={target_data_size}, got {max_attempts}"
        )

    pendings = set()
    data = []
    all_data = []
    attempted_groups = 0
    discarded_groups = 0
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while len(data) + len(pendings) < target_data_size:
            if max_discarded and discarded_groups >= max_discarded:
                await _trip_rollout_fuse(
                    state,
                    pendings,
                    pbar,
                    "CompactionRL rollout fuse tripped after too many discarded groups: "
                    f"accepted={len(data)}/{target_data_size} attempted={attempted_groups} "
                    f"discarded={discarded_groups} limit={max_discarded}",
                )
            remaining_attempts = max_attempts - attempted_groups if max_attempts else None
            if remaining_attempts is not None and remaining_attempts <= 0:
                await _trip_rollout_fuse(
                    state,
                    pendings,
                    pbar,
                    "CompactionRL rollout fuse exhausted its attempt budget: "
                    f"accepted={len(data)}/{target_data_size} attempted={attempted_groups} "
                    f"discarded={discarded_groups} limit={max_attempts}",
                )
            # get samples from the buffer and submit the generation requests.
            missing_groups = target_data_size - len(data) - len(pendings)
            request_size = sampling_request_size(args, dynamic_filter, missing_groups)
            if remaining_attempts is not None:
                request_size = min(request_size, remaining_attempts)
            samples = data_source(request_size)
            submitted = submit_generate_tasks(state, samples)
            if not submitted:
                await _trip_rollout_fuse(
                    state,
                    pendings,
                    pbar,
                    "rollout data source returned no samples while the batch was incomplete: "
                    f"accepted={len(data)}/{target_data_size}",
                )
            attempted_groups += len(submitted)
            pendings.update(submitted)

        # wait for the generation to finish
        logger.debug(f"[rollout] Waiting on {len(pendings)} pending tasks, data={len(data)}/{target_data_size}")
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        logger.debug(f"[rollout] asyncio.wait returned: {len(done)} done, {len(pendings)} pending")
        for task in done:
            try:
                group: list[Sample] = task.result()
            except Exception as e:
                logger.error(f"[rollout] Task raised exception: {e!r}", exc_info=True)
                discarded_groups += 1
                continue

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                discarded_groups += 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(state, pendings, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()

    if f := load_function(args.rollout_sample_filter_path):
        f(args, data)
    # There can be circumstances where users want to process all samples including filtered ones.
    if f := load_function(args.rollout_all_samples_process_path):
        f(args, all_samples, data_source)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [sample for group in data for sample in group],
        url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        sampling_params=state.sampling_params,
    )

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
