#!/usr/bin/env bash
# Short command for the verified CompactionRL PPO continuation.
# This entry point is intentionally pinned to the latest complete secondary
# checkpoint pair from the corrected run.
# Use run_compactionrl_ppo.sh directly for experiments with other settings.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STORAGE_ROOT="${COMPACTION_STORAGE_ROOT:-/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/compactionrl}"
CHECKPOINT_ROOT="${STORAGE_ROOT}/checkpoints"

export RUNTIME_MODE="${RUNTIME_MODE:-image}"
export TRAIN_DATASET="${TRAIN_DATASET:-swe-dev-secondary}"
# Miles executes range(start_rollout_id, num_rollout). Loading iteration 109
# sets start_rollout_id=110, so 201 is required to run through rollout 200.
export NUM_ROLLOUT="${NUM_ROLLOUT:-201}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export COMPACTION_STORAGE_ROOT="${STORAGE_ROOT}"
export SANDBOX_CONCURRENCY_SECONDARY="${SANDBOX_CONCURRENCY_SECONDARY:-64}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-64}"
export SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-cache_aware}"
export SGLANG_ROUTER_REQUEST_TIMEOUT_SECS="${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS:-240}"
export SGLANG_ROUTER_MAX_ATTEMPTS="${SGLANG_ROUTER_MAX_ATTEMPTS:-1}"
export COMPACTION_PROXY_HTTP_TIMEOUT_SEC="${COMPACTION_PROXY_HTTP_TIMEOUT_SEC:-300}"
export COMPACTION_PROXY_HTTP_MAX_ATTEMPTS="${COMPACTION_PROXY_HTTP_MAX_ATTEMPTS:-1}"
export QWEN_CODE_API_TIMEOUT_MS="${QWEN_CODE_API_TIMEOUT_MS:-360000}"
export QWEN_CODE_MAX_RETRIES="${QWEN_CODE_MAX_RETRIES:-0}"
# generate.py disables Qwen Code's housekeeping side queries. Keep strict mode
# so any remaining real SubAgent/auxiliary request rejects the unordered
# trajectory instead of silently entering PPO.
export COMPACTION_AUXILIARY_MODE="${COMPACTION_AUXILIARY_MODE:-reject}"

# The stable dual-run checkpoints were produced with these values. Keep resume
# behavior reproducible even if the general launcher defaults change later.
export ACTOR_LR=5e-7
export CRITIC_LR=1e-6
export CRITIC_PPO_EPOCHS=1
export CKPT_STEP="${RESUME_CKPT_STEP:-109}"

# Step 109 is the latest complete paired state before the sustained reward and
# optimized-token decline observed after step 120 in this continuation.
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${CHECKPOINT_ROOT}/compactionrl_swe_dev_secondary_qwen35_4b_ppo_actor_resume_from_94_20260902_154148}"
export CRITIC_LOAD_CHECKPOINT="${CRITIC_LOAD_CHECKPOINT:-${CHECKPOINT_ROOT}/compactionrl_swe_dev_secondary_qwen35_4b_ppo_critic_resume_from_94_20260902_154148}"

[[ -d "${LOAD_CHECKPOINT}" ]] || {
  echo "Missing actor checkpoint: ${LOAD_CHECKPOINT}" >&2
  exit 2
}
[[ -d "${CRITIC_LOAD_CHECKPOINT}" ]] || {
  echo "Missing critic checkpoint: ${CRITIC_LOAD_CHECKPOINT}" >&2
  exit 2
}

exec bash "${SCRIPT_DIR}/run_compactionrl_ppo.sh" resume-8gpu
