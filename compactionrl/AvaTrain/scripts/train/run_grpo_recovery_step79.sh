#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

LOAD_CHECKPOINT="/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/checkpoints/swe_dev_dual_qwen35_4b_4gpu_20260803_190350"
SAVE_DIR="/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/checkpoints/swe_dev_dual_qwen35_4b_4gpu_recovery_from79_kl001_gn3"

exec env \
  RUNTIME_MODE=image \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  TRAIN_DATASET=swe-dev-dual \
  LOAD_CHECKPOINT="${LOAD_CHECKPOINT}" \
  CKPT_STEP=79 \
  SAVE_DIR="${SAVE_DIR}" \
  NUM_ROLLOUT=500 \
  KL_LOSS_COEF=0.001 \
  SKIP_UPDATE_GRAD_NORM_THRESHOLD=3.0 \
  bash "${AVA_ROOT}/scripts/train/run.sh" train-4gpu
