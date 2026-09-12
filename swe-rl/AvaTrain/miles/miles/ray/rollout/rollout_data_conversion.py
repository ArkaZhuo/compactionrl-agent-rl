import itertools
import logging

from miles.rollout.filter_hub.snr_filter import group_reward_variance, snr_aware_filter, variance_metrics

logger = logging.getLogger(__name__)


def postprocess_rollout_data(args, data, train_parallel_config):
    metadata = {}

    variances = [group_reward_variance(args, group) for group in data]
    metrics = variance_metrics(variances)
    if args.snr_filter_keep_ratio is not None:
        data, snr_metrics = snr_aware_filter(args, data, variances)
        metrics = metrics | snr_metrics
    metadata["metrics"] = metrics

    # flatten the data if it is a list of lists
    while isinstance(data[0], list):
        data = list(itertools.chain.from_iterable(data))

    if not args.disable_rollout_trim_samples:
        global_batch_size = args.global_batch_size
        if args.use_dynamic_global_batch_size:
            logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
            dynamic_global_batch_size = _compute_dynamic_global_batch_size(
                args, train_parallel_config=train_parallel_config, num_samples=len(data)
            )
            metadata["dynamic_global_batch_size"] = dynamic_global_batch_size
            global_batch_size = dynamic_global_batch_size

        if len(data) % global_batch_size != 0:
            trim_len = (len(data) // global_batch_size) * global_batch_size
            if trim_len == 0:
                raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
            origin_data_length = len(data)
            data = data[:trim_len]
            logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
        logger.info(f"Final collected {len(data)} samples from rollout to train")

    return data, metadata


def _compute_dynamic_global_batch_size(args, train_parallel_config, num_samples: int) -> int:
    """Calculate dynamic global_batch_size that splits num_samples across the
    configured number of training steps.

    Strategy: global_batch_size = (num_samples // num_steps_per_rollout) rounded down
    to a multiple of dp_size, so num_samples // global_batch_size == num_steps_per_rollout
    (a single step when num_steps_per_rollout is unset).
    """
    dp_size = train_parallel_config["dp_size"]
    original_gbs = args.global_batch_size
    num_steps = args.num_steps_per_rollout or 1

    # Round down to a multiple of dp_size so each step splits evenly across dp ranks.
    dynamic_gbs = (num_samples // num_steps // dp_size) * dp_size

    if dynamic_gbs == 0:
        # Too few samples, use at least dp_size
        dynamic_gbs = dp_size
        logger.warning(
            f"num_samples={num_samples} too small for num_steps={num_steps} x dp_size={dp_size}, "
            f"using dp_size as global_batch_size"
        )

    # Calculate how many samples will be discarded
    wasted = num_samples - dynamic_gbs * num_steps

    if dynamic_gbs != original_gbs or wasted > 0:
        logger.info(
            f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
            f"(num_samples={num_samples}, dp_size={dp_size}, "
            f"num_steps={num_steps}, wasted={wasted})"
        )

    return dynamic_gbs
