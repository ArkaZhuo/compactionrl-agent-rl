#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Resume the 2026-09-05 four-GPU GRPO baseline from its latest complete
# checkpoint. Keep the original run's training-critical settings explicit so
# later launcher-default changes cannot silently alter the continuation.
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/checkpoints/swe_dev_secondary_qwen35_4b_4gpu_20260905_124822}"
export CKPT_STEP="${CKPT_STEP:-119}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export SKIP_UPDATE_GRAD_NORM_THRESHOLD="${SKIP_UPDATE_GRAD_NORM_THRESHOLD:-3.0}"
export TRAIN_DATASET="${TRAIN_DATASET:-swe-dev-secondary}"
export SANDBOX_CONCURRENCY_SECONDARY="${SANDBOX_CONCURRENCY_SECONDARY:-64}"
export RUNTIME_MODE="${RUNTIME_MODE:-image}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

exec bash "${SCRIPT_DIR}/run.sh" train-4gpu
