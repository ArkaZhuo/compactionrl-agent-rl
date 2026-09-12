"""CompactionRL token-level and cross-trajectory advantage estimation."""

from __future__ import annotations

import hashlib
from typing import Any

import torch


# CP=2 is the production 8-GPU layout. Higher CP sizes remain blocked until
# their full-response gather/reslice path has been exercised on real hardware.
SUPPORTED_CONTEXT_PARALLEL_SIZES = frozenset({1, 2})
_CP_DTYPE_IDS = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}
_CP_ID_DTYPES = {value: key for key, value in _CP_DTYPE_IDS.items()}


def _cp_layout_header(
    tensors: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    sample_ids: list[tuple[str, int]],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    device: torch.device,
) -> torch.Tensor:
    """Return a fixed-size header that is safe to compare before batch collectives."""
    qkv_format_id = {"thd": 0, "bshd": 1}.get(qkv_format, -1)
    return torch.tensor(
        [
            len(tensors),
            len(total_lengths),
            len(response_lengths),
            len(sample_ids),
            -1 if max_seq_lens is None else len(max_seq_lens),
            max(response_lengths, default=0),
            qkv_format_id,
        ],
        dtype=torch.int64,
        device=device,
    )


def _cp_layout_rows(
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
    sample_ids: list[tuple[str, int]],
    device: torch.device,
) -> torch.Tensor:
    """Return the exact per-sample sequence layout after batch size is validated."""
    if max_seq_lens is None:
        max_seq_lens = [-1] * len(total_lengths)
    identity_rows = []
    for trajectory_id, segment_index in sample_ids:
        digest = hashlib.blake2b(
            f"{trajectory_id}\0{segment_index}".encode("utf-8"),
            digest_size=16,
        ).digest()
        identity_rows.append(
            (
                int.from_bytes(digest[:8], "little") & (2**63 - 1),
                int.from_bytes(digest[8:], "little") & (2**63 - 1),
            )
        )
    return torch.tensor(
        [
            (total_length, response_length, max_seq_len, identity[0], identity[1])
            for total_length, response_length, max_seq_len, identity in zip(
                total_lengths,
                response_lengths,
                max_seq_lens,
                identity_rows,
                strict=True,
            )
        ],
        dtype=torch.int64,
        device=device,
    )


def _build_local_cp_response_batch(
    tensors: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
) -> torch.Tensor:
    """Place this CP rank's zigzag chunks in a padded full-response batch."""
    from miles.backends.training_utils.cp_utils import get_logits_and_tokens_offset_with_cp

    if not tensors:
        raise RuntimeError("CompactionRL cannot gather an empty CP tensor batch")
    if any(tensor.ndim != 1 for tensor in tensors):
        shapes = [tuple(tensor.shape) for tensor in tensors]
        raise RuntimeError(f"CompactionRL CP values must be one-dimensional, got {shapes}")

    batch = len(tensors)
    max_response_length = max(response_lengths)
    local_batch = torch.zeros(
        batch,
        max_response_length,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    for index, (tensor, total_length, response_length) in enumerate(
        zip(tensors, total_lengths, response_lengths, strict=True)
    ):
        max_seq_len = max_seq_lens[index] if max_seq_lens is not None else None
        _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
            total_length,
            response_length,
            qkv_format,
            max_seq_len,
        )
        first_length = logits_offset[0][1] - logits_offset[0][0]
        chunks = (tensor[:first_length], tensor[first_length:])
        prompt_logit_start = total_length - response_length - 1
        expected_local_length = 0
        for chunk, (offset_start, offset_end) in zip(chunks, logits_offset, strict=True):
            expected_chunk_length = offset_end - offset_start
            if chunk.numel() != expected_chunk_length:
                raise RuntimeError(
                    "CompactionRL CP local value layout mismatch: "
                    f"sample={index}, local={tensor.numel()}, chunk={chunk.numel()}, "
                    f"expected_chunk={expected_chunk_length}, total={total_length}, "
                    f"response={response_length}"
                )
            expected_local_length += expected_chunk_length
            if expected_chunk_length == 0:
                continue
            response_start = offset_start - prompt_logit_start
            response_end = offset_end - prompt_logit_start
            if response_start < 0 or response_end > response_length:
                raise RuntimeError(
                    "CompactionRL CP response offset is out of range: "
                    f"sample={index}, offset=({response_start}, {response_end}), "
                    f"response={response_length}"
                )
            local_batch[index, response_start:response_end] = chunk
        if tensor.numel() != expected_local_length:
            raise RuntimeError(
                "CompactionRL CP local value length mismatch: "
                f"sample={index}, actual={tensor.numel()}, expected={expected_local_length}"
            )
    return local_batch


def _synchronize_cuda_collective(tensor: torch.Tensor) -> None:
    """Finish a CP collective before ranks enter an overlapping process group."""
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _resolve_cp_batch_dtype(
    tensors: list[torch.Tensor],
    device: torch.device,
    cp_group: Any,
    cp_size: int,
    expected_batch: int,
) -> torch.dtype:
    """Validate local counts and resolve dtype with one rank-uniform descriptor."""
    import torch.distributed as dist

    local_dtype_id = -1
    if tensors:
        dtype = tensors[0].dtype
        if dtype not in _CP_DTYPE_IDS:
            raise RuntimeError(f"CompactionRL CP values have unsupported dtype={dtype}")
        if any(tensor.dtype != dtype for tensor in tensors):
            raise RuntimeError("CompactionRL CP values use different local dtypes")
        local_dtype_id = _CP_DTYPE_IDS[dtype]

    descriptor = torch.tensor([len(tensors), local_dtype_id], dtype=torch.int64, device=device)
    descriptors = [torch.empty_like(descriptor) for _ in range(cp_size)]
    dist.all_gather(descriptors, descriptor, group=cp_group)
    descriptor_rows = [item.detach().cpu().tolist() for item in descriptors]
    if any(count not in {0, expected_batch} for count, _ in descriptor_rows):
        raise RuntimeError(
            "CompactionRL CP local tensor counts must be zero or match the response batch: "
            f"expected={expected_batch}, descriptors={descriptor_rows}"
        )
    populated_ids = {dtype_id for count, dtype_id in descriptor_rows if count > 0 and dtype_id >= 0}
    if len(populated_ids) != 1:
        raise RuntimeError(
            "CompactionRL CP ranks did not provide one consistent value dtype: "
            f"{descriptor_rows}"
        )
    return _CP_ID_DTYPES[populated_ids.pop()]


def _gather_full_responses_with_cp(
    tensors: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    sample_ids: list[tuple[str, int]],
    reference_tensor: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Gather a variable-length response batch with fixed-shape collectives.

    A per-sample collective is unsafe here: one rank getting a different list
    order turns the next calls into different NCCL tensor sizes and hangs until
    the watchdog aborts the job.  Validate a fixed-size layout descriptor, then
    reduce one padded batch so every rank issues the same collective sequence.
    """
    import torch.distributed as dist

    from miles.backends.training_utils.parallel import get_parallel_state

    parallel_state = get_parallel_state()
    cp_group = parallel_state.cp.group
    cp_size = parallel_state.cp.size
    if cp_size == 1:
        return tensors

    expected_batch = len(response_lengths)
    if tensors:
        device = tensors[0].device
    elif reference_tensor is not None:
        device = reference_tensor.device
    elif torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    dtype = _resolve_cp_batch_dtype(tensors, device, cp_group, cp_size, expected_batch)
    if not tensors:
        # A short response can live entirely on CP local-rank 0. Megatron then
        # returns no non-loss tensors on its peer, but that peer must still issue
        # every CP collective in the same order. The layout validation below
        # rejects the placeholder if this rank was actually expected to own data.
        tensors = [torch.empty(0, dtype=dtype, device=device) for _ in response_lengths]

    header = _cp_layout_header(
        tensors,
        total_lengths,
        response_lengths,
        sample_ids,
        qkv_format,
        max_seq_lens,
        device,
    )
    headers = [torch.empty_like(header) for _ in range(cp_size)]
    dist.all_gather(headers, header, group=cp_group)
    if any(not torch.equal(item, headers[0]) for item in headers[1:]):
        layouts = [item.detach().cpu().tolist() for item in headers]
        raise RuntimeError(f"CompactionRL CP ranks received different layout headers: {layouts}")
    if not (len(tensors) == len(total_lengths) == len(response_lengths) == len(sample_ids)) or (
        max_seq_lens is not None and len(max_seq_lens) != len(tensors)
    ):
        raise RuntimeError(f"CompactionRL CP local sequence lists are misaligned: {header.tolist()}")
    if qkv_format not in {"thd", "bshd"}:
        raise RuntimeError(f"CompactionRL CP received unsupported qkv_format={qkv_format!r}")

    layout = _cp_layout_rows(total_lengths, response_lengths, max_seq_lens, sample_ids, device)
    layouts = [torch.empty_like(layout) for _ in range(cp_size)]
    dist.all_gather(layouts, layout, group=cp_group)
    if any(not torch.equal(item, layouts[0]) for item in layouts[1:]):
        cpu_layouts = [item.detach().cpu().tolist() for item in layouts]
        raise RuntimeError(f"CompactionRL CP ranks received different sequence layouts: {cpu_layouts}")

    local_error: str | None = None
    try:
        local_batch = _build_local_cp_response_batch(
            tensors,
            total_lengths,
            response_lengths,
            qkv_format,
            max_seq_lens,
        )
    except (RuntimeError, ValueError, AssertionError, IndexError) as error:
        local_error = str(error)
        local_batch = torch.zeros(
            len(response_lengths),
            max(response_lengths),
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )

    layout_valid = torch.tensor([local_error is None], dtype=torch.int32, device=tensors[0].device)
    dist.all_reduce(layout_valid, op=dist.ReduceOp.MIN, group=cp_group)
    if not bool(layout_valid.item()):
        raise RuntimeError(local_error or "CompactionRL CP peer has an invalid local value layout")

    dist.all_reduce(local_batch, op=dist.ReduceOp.SUM, group=cp_group)
    # A synchronous NCCL API call only guarantees that the operation has been
    # enqueued.  Some CP ranks can have no local response tokens and therefore
    # no data dependency on local_batch; without an explicit device wait those
    # ranks may enter TP training while their CP peers still consume the result.
    # CP and TP groups overlap, so that rank-dependent ordering can deadlock.
    _synchronize_cuda_collective(local_batch)
    return [local_batch[index, :response_length] for index, response_length in enumerate(response_lengths)]


def _chunked_variable_lambda_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    lambdas: torch.Tensor,
    gamma: float,
    chunk_size: int = 128,
) -> torch.Tensor:
    """GAE over compacted assistant-token sequences with one lambda per sample."""
    batch, max_len = rewards.shape
    device, dtype = rewards.device, rewards.dtype
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(batch, 1, device=device, dtype=dtype)], dim=1
    )
    deltas = rewards + gamma * next_values - values
    reversed_deltas = torch.flip(deltas, dims=[1])
    pad = (chunk_size - reversed_deltas.shape[1] % chunk_size) % chunk_size
    if pad:
        reversed_deltas = torch.nn.functional.pad(reversed_deltas, (0, pad))
    padded_len = reversed_deltas.shape[1]
    chunks = reversed_deltas.view(batch, padded_len // chunk_size, chunk_size)

    positions = torch.arange(chunk_size, device=device)
    diff = positions[None, :] - positions[:, None]
    nonnegative = diff >= 0
    powers = diff.clamp_min(0).to(dtype)
    weights = torch.where(
        nonnegative.unsqueeze(0),
        (gamma * lambdas).view(batch, 1, 1).pow(powers.view(1, chunk_size, chunk_size)),
        torch.zeros(batch, chunk_size, chunk_size, device=device, dtype=dtype),
    )
    local = torch.bmm(chunks, weights)
    pow_vec = (gamma * lambdas).view(batch, 1).pow(
        torch.arange(1, chunk_size + 1, device=device, dtype=dtype).view(1, chunk_size)
    )

    result_reversed = torch.zeros(batch, padded_len, device=device, dtype=dtype)
    state = torch.zeros(batch, device=device, dtype=dtype)
    num_chunks = padded_len // chunk_size
    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size
        end = start + chunk_size
        current = local[:, chunk_index, :] + state.unsqueeze(1) * pow_vec
        result_reversed[:, start:end] = current
        state = current[:, -1]

    if pad:
        result_reversed = result_reversed[:, :-pad]
    advantages = torch.flip(result_reversed, dims=[1])
    valid = torch.arange(max_len, device=device).view(1, max_len) < lengths.view(batch, 1)
    return torch.where(valid, advantages, torch.zeros_like(advantages))


@torch.no_grad()
def compute_compaction_advantages(args: Any, rollout_data: dict[str, Any], kl: list[torch.Tensor]):
    """Compute GAE on full responses and return tensors in the local CP layout."""
    from miles.backends.training_utils.parallel import get_parallel_state

    cp_size = get_parallel_state().cp.size
    if cp_size not in SUPPORTED_CONTEXT_PARALLEL_SIZES:
        raise NotImplementedError(
            f"CompactionRL custom GAE supports context parallel sizes "
            f"{sorted(SUPPORTED_CONTEXT_PARALLEL_SIZES)}, got {cp_size}"
        )

    values = rollout_data.get("values")
    if values is None:
        raise RuntimeError("CompactionRL PPO requires critic values")
    if not values and cp_size == 1:
        return [], []

    loss_masks = rollout_data["loss_masks"]
    rewards = rollout_data["rewards"]
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    max_seq_lens = rollout_data.get("max_seq_lens")
    expected_batch = len(response_lengths)
    metadata = rollout_data.get("metadata")
    if metadata is None or len(metadata) != expected_batch:
        raise RuntimeError("CompactionRL segment metadata is missing or misaligned")

    seen_segments: set[tuple[str, int]] = set()
    sample_ids: list[tuple[str, int]] = []
    optimized_positions_cpu: list[torch.Tensor] = []
    for index, (item, mask) in enumerate(zip(metadata, loss_masks, strict=True)):
        if not isinstance(item, dict) or item.get("compaction") is not True:
            raise RuntimeError(f"CompactionRL metadata at index {index} is not a compacted segment")
        trajectory_id = item.get("trajectory_id")
        segment_index = item.get("segment_index")
        if not isinstance(trajectory_id, str) or not trajectory_id or not isinstance(segment_index, int):
            raise RuntimeError(f"CompactionRL metadata at index {index} has invalid segment identity")
        identity = (trajectory_id, segment_index)
        if identity in seen_segments:
            raise RuntimeError(f"duplicate CompactionRL segment in local batch: {identity}")
        seen_segments.add(identity)
        sample_ids.append(identity)
        # CUDA nonzero synchronizes the host with all previously enqueued CUDA
        # work. Keep this metadata-only operation on CPU and do it before the CP
        # gathers so it cannot starve ProcessGroupNCCL's heartbeat thread.
        positions_cpu = torch.nonzero(mask.detach().to(device="cpu", dtype=torch.bool), as_tuple=False).flatten()
        optimized_positions_cpu.append(positions_cpu)
        optimized_tokens = int(positions_cpu.numel())
        if item.get("optimized_tokens") != optimized_tokens or optimized_tokens <= 0:
            raise RuntimeError(
                f"CompactionRL optimized-token metadata mismatch for {identity}: "
                f"declared={item.get('optimized_tokens')}, actual={optimized_tokens}"
            )
        future_tokens = item.get("future_optimized_tokens")
        if not isinstance(future_tokens, int) or future_tokens < 0:
            raise RuntimeError(f"CompactionRL future-token metadata is invalid for {identity}: {future_tokens}")

    if not (
        len(total_lengths) == len(response_lengths) == len(loss_masks) == expected_batch
        and len(values) in {0, expected_batch}
        and len(kl) in {0, expected_batch}
        and (max_seq_lens is None or len(max_seq_lens) == expected_batch)
    ):
        raise RuntimeError("CompactionRL sequence lengths are missing or misaligned")

    if cp_size > 1:
        gather_kl = float(args.kl_coef) != 0.0
        reference_tensor = loss_masks[0] if loss_masks else None
        full_values = _gather_full_responses_with_cp(
            values,
            total_lengths,
            response_lengths,
            args.qkv_format,
            max_seq_lens,
            sample_ids,
            reference_tensor,
        )
        full_kl = (
            _gather_full_responses_with_cp(
                kl,
                total_lengths,
                response_lengths,
                args.qkv_format,
                max_seq_lens,
                sample_ids,
                reference_tensor,
            )
            if gather_kl
            else [None] * len(full_values)
        )
    else:
        full_values = values
        full_kl = kl

    gamma = float(args.gamma)
    alpha = float(getattr(args, "compaction_gae_alpha", 1.5))
    if alpha <= 0:
        raise ValueError("compaction_gae_alpha must be positive")

    train_values: list[torch.Tensor] = []
    train_rewards: list[torch.Tensor] = []
    lengths: list[int] = []
    for value, mask, reward, kl_value, response_length, positions_cpu in zip(
        full_values,
        loss_masks,
        rewards,
        full_kl,
        response_lengths,
        optimized_positions_cpu,
        strict=True,
    ):
        mask = mask.to(device=value.device).bool()
        if (
            value.numel() != response_length
            or mask.numel() != response_length
            or (kl_value is not None and kl_value.numel() != response_length)
        ):
            raise RuntimeError("value, KL, and loss-mask lengths differ in CompactionRL GAE")
        positions = positions_cpu.to(device=value.device)
        if positions.numel() == 0:
            raise RuntimeError("CompactionRL segment has no optimized tokens")
        selected_values = value.reshape(-1)[positions]
        if kl_value is None:
            selected_rewards = torch.zeros(positions.numel(), device=value.device, dtype=torch.float32)
        else:
            selected_rewards = -float(args.kl_coef) * kl_value.reshape(-1)[positions].to(torch.float32)
            selected_rewards = selected_rewards.clone()
        selected_rewards[-1] += float(reward)
        train_values.append(selected_values)
        train_rewards.append(selected_rewards.to(dtype=selected_values.dtype))
        lengths.append(int(positions.numel()))

    batch = len(lengths)
    max_len = max(lengths)
    device = train_values[0].device
    dtype = train_values[0].dtype
    padded_values = torch.zeros(batch, max_len, device=device, dtype=dtype)
    padded_rewards = torch.zeros(batch, max_len, device=device, dtype=dtype)
    for index, (sample_values, sample_rewards) in enumerate(zip(train_values, train_rewards, strict=True)):
        length = lengths[index]
        padded_values[index, :length] = sample_values
        padded_rewards[index, :length] = sample_rewards

    length_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
    lambda_tensor = 1.0 - 1.0 / (alpha * length_tensor.to(dtype).clamp_min(1.0))
    local_advantages = _chunked_variable_lambda_gae(
        padded_rewards,
        padded_values,
        length_tensor,
        lambda_tensor,
        gamma,
    )

    future = torch.tensor(
        [int(item.get("future_optimized_tokens", 0)) for item in metadata],
        device=device,
        dtype=dtype,
    )
    cross_factors = (gamma * lambda_tensor).pow(future)
    corrected = local_advantages * cross_factors.view(batch, 1)

    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for index, (value, mask, total_length, response_length, positions_cpu) in enumerate(
        zip(
            full_values,
            loss_masks,
            total_lengths,
            response_lengths,
            optimized_positions_cpu,
            strict=True,
        )
    ):
        mask = mask.to(device=value.device).bool().reshape(-1)
        full_advantage = torch.zeros_like(value, dtype=corrected.dtype).reshape(-1)
        full_return = value.detach().clone().to(dtype=corrected.dtype).reshape(-1)
        positions = positions_cpu.to(device=value.device)
        length = lengths[index]
        full_advantage[positions] = corrected[index, :length]
        full_return[positions] = full_advantage[positions] + value.reshape(-1)[positions]
        full_advantage = full_advantage.reshape_as(value)
        full_return = full_return.reshape_as(value)
        if cp_size > 1:
            from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp

            max_seq_len = max_seq_lens[index] if max_seq_lens is not None else None
            full_advantage = slice_log_prob_with_cp(
                full_advantage, total_length, response_length, args.qkv_format, max_seq_len
            )
            full_return = slice_log_prob_with_cp(
                full_return, total_length, response_length, args.qkv_format, max_seq_len
            )
        if full_advantage.shape != values[index].shape or full_return.shape != values[index].shape:
            raise RuntimeError("CompactionRL CP output shape does not match the local critic layout")
        advantages.append(full_advantage)
        returns.append(full_return)
    return advantages, returns
