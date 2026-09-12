#!/usr/bin/env bash
# Agentic SWE PPO for Qwen3.5-4B on four or eight GPUs.
#
# PPO needs a trainable critic. The top-level launcher assigns half of the
# visible GPUs to each model and uses the configured TP/DP layout within each
# half, while colocated rollout uses all visible GPUs as independent TP=1
# SGLang engines. This file is deliberately separate from the GRPO runner.

set -euo pipefail
export PYTHONUNBUFFERED=1
ulimit -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
AVA_ROOT="$(cd -- "${MILES_DIR}/.." &>/dev/null && pwd)"
RUNTIME_MODE="${RUNTIME_MODE:-venv}"
if [[ "${RUNTIME_MODE}" == image ]]; then
   PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
else
   PYTHON_BIN="${PYTHON_BIN:-${AVA_ROOT}/.venv/bin/python}"
fi
[[ -x "${PYTHON_BIN}" ]] || {
   echo "Missing selected Python: ${PYTHON_BIN} (RUNTIME_MODE=${RUNTIME_MODE})" >&2
   exit 1
}

source "${MILES_DIR}/scripts/models/qwen3.5-4B.sh"

TASK_FILE="$(realpath "${1:?usage: $0 TASK_FILE.jsonl}")"
if [[ "${RUNTIME_MODE}" == image ]]; then
   ROLLOUT_PYTHONPATH="${MILES_DIR}/../Megatron-LM:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
else
   ROLLOUT_PYTHONPATH="${MILES_DIR}/../Megatron-LM:${MILES_DIR}/../sglang/python:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

NUM_GPUS="${NUM_GPUS:-4}"
ACTOR_NUM_GPUS="${ACTOR_NUM_GPUS:-2}"
CRITIC_NUM_GPUS="${CRITIC_NUM_GPUS:-2}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-2}"
TRAIN_SEQUENCE_PARALLEL="${TRAIN_SEQUENCE_PARALLEL:-1}"
[[ $((ACTOR_NUM_GPUS + CRITIC_NUM_GPUS)) -eq "${NUM_GPUS}" ]] || {
   echo "PPO requires ACTOR_NUM_GPUS + CRITIC_NUM_GPUS == NUM_GPUS" >&2
   exit 2
}
[[ $((ACTOR_NUM_GPUS % TRAIN_TP_SIZE)) -eq 0 ]] || {
   echo "ACTOR_NUM_GPUS must be divisible by TRAIN_TP_SIZE" >&2
   exit 2
}
[[ $((CRITIC_NUM_GPUS % TRAIN_TP_SIZE)) -eq 0 ]] || {
   echo "CRITIC_NUM_GPUS must be divisible by TRAIN_TP_SIZE" >&2
   exit 2
}
[[ "${TRAIN_SEQUENCE_PARALLEL}" =~ ^[01]$ ]] || {
   echo "TRAIN_SEQUENCE_PARALLEL must be 0 or 1" >&2
   exit 2
}

NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-32}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.70}"
SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-cache_aware}"
ROUTER_CACHE_THRESHOLD="${ROUTER_CACHE_THRESHOLD:-0.30}"
ROUTER_BALANCE_ABS_THRESHOLD="${ROUTER_BALANCE_ABS_THRESHOLD:-8}"
ROUTER_BALANCE_REL_THRESHOLD="${ROUTER_BALANCE_REL_THRESHOLD:-1.25}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-16}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-536870912}"
OPTIMIZER_CPU_OFFLOAD_FRACTION="${OPTIMIZER_CPU_OFFLOAD_FRACTION:-0.4}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
# Both colocated SGLang and Megatron use TorchMemorySaver, which does not yet
# support expandable_segments.  Ignore an inherited allocator override.
unset PYTORCH_CUDA_ALLOC_CONF
RAY_GCS_PORT="${RAY_GCS_PORT:-6385}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8275}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$((NUM_GPUS * 2))}"
DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH-}"
RESET_ROLLOUT_DATASET_STATE="${RESET_ROLLOUT_DATASET_STATE:-0}"
CKPT_FULLY_PARALLEL_SAVE="${CKPT_FULLY_PARALLEL_SAVE:-0}"
CKPT_ASSUME_CONSTANT_STRUCTURE="${CKPT_ASSUME_CONSTANT_STRUCTURE:-1}"
MCORE_DIST_CKPT_THREAD_COUNT="${MCORE_DIST_CKPT_THREAD_COUNT:-1}"
MCORE_DIST_CKPT_WRITE_ATTEMPTS="${MCORE_DIST_CKPT_WRITE_ATTEMPTS:-3}"
export MCORE_DIST_CKPT_THREAD_COUNT MCORE_DIST_CKPT_WRITE_ATTEMPTS

[[ "${RAY_NUM_CPUS}" =~ ^[1-9][0-9]*$ ]] || {
   echo "RAY_NUM_CPUS must be a positive integer" >&2
   exit 2
}
[[ "${CKPT_FULLY_PARALLEL_SAVE}" =~ ^[01]$ ]] || {
   echo "CKPT_FULLY_PARALLEL_SAVE must be 0 or 1" >&2
   exit 2
}
[[ "${CKPT_ASSUME_CONSTANT_STRUCTURE}" =~ ^[01]$ ]] || {
   echo "CKPT_ASSUME_CONSTANT_STRUCTURE must be 0 or 1" >&2
   exit 2
}
[[ "${RESET_ROLLOUT_DATASET_STATE}" =~ ^[01]$ ]] || {
   echo "RESET_ROLLOUT_DATASET_STATE must be 0 or 1" >&2
   exit 2
}

PPO_EPOCHS="${PPO_EPOCHS:-1}"
PPO_EPS_CLIP="${PPO_EPS_CLIP:-0.2}"
PPO_VALUE_CLIP="${PPO_VALUE_CLIP:-0.2}"
PPO_GAMMA="${PPO_GAMMA:-1.0}"
PPO_LAMBDA="${PPO_LAMBDA:-1.0}"
KL_COEF="${KL_COEF:-0.001}"
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-1}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-1e-5}"

[[ "${KL_COEF}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
   echo "KL_COEF must be a non-negative decimal number" >&2
   exit 2
}

SAVE_DIR="${SAVE_DIR:-/root/agentic_swe_ppo_actor_ckpt}"
CRITIC_SAVE_DIR="${CRITIC_SAVE_DIR:-/root/agentic_swe_ppo_critic_ckpt}"
CRITIC_LOAD="${CRITIC_LOAD_CHECKPOINT:-${REF_LOAD:?set REF_LOAD (torch_dist checkpoint)}}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:?set HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --critic-load "${CRITIC_LOAD}"
   --critic-save "${CRITIC_SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
   CKPT_ARGS+=(--load "${LOAD_CHECKPOINT}")
fi
if [[ "${CKPT_FULLY_PARALLEL_SAVE}" == 0 ]]; then
   CKPT_ARGS+=(--no-ckpt-fully-parallel-save)
fi
if [[ "${CKPT_ASSUME_CONSTANT_STRUCTURE}" == 1 ]]; then
   CKPT_ARGS+=(--ckpt-assume-constant-structure)
fi

ROLLOUT_ARGS=(
   --prompt-data "${TASK_FILE}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --custom-generate-function-path generate.generate
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1.0
   --rollout-top-p 0.95
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [[ "${RESET_ROLLOUT_DATASET_STATE}" == 1 ]]; then
   ROLLOUT_ARGS+=(--reset-rollout-dataset-state)
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TRAIN_TP_SIZE}"
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)
if [[ "${TRAIN_SEQUENCE_PARALLEL}" == 1 ]]; then
   PERF_ARGS+=(--sequence-parallel)
fi

PPO_ARGS=(
   --advantage-estimator ppo
   --ppo-epochs "${PPO_EPOCHS}"
   --eps-clip "${PPO_EPS_CLIP}"
   --value-clip "${PPO_VALUE_CLIP}"
   --gamma "${PPO_GAMMA}"
   --lambd "${PPO_LAMBDA}"
   --normalize-advantages
   --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
   --critic-lr "${CRITIC_LR}"
   --kl-coef "${KL_COEF}"
   --entropy-coef 0.0
)

# A resumed checkpoint may have been created with a longer planned run than
# the requested continuation window.  Preserve the checkpoint's scheduler
# state instead of rejecting a harmless NUM_ROLLOUT difference (for example,
# 500 planned rollouts versus a 200-rollout continuation).
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
   PPO_ARGS+=(--use-checkpoint-opt-param-scheduler)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   # Full GPU Adam states leave too little room for a long-trajectory backward
   # above TorchMemorySaver's safety margin. Hybrid CPU offload removes persistent
   # optimizer state from the long-trajectory peak while preserving FP32
   # optimizer math and both PPO optimization epochs.
   --optimizer-cpu-offload
   --optimizer-offload-fraction "${OPTIMIZER_CPU_OFFLOAD_FRACTION}"
   --use-precision-aware-optimizer
   --lr "${ACTOR_LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --router-policy "${SGLANG_ROUTER_POLICY}"
   --router-cache-threshold "${ROUTER_CACHE_THRESHOLD}"
   --router-balance-abs-threshold "${ROUTER_BALANCE_ABS_THRESHOLD}"
   --router-balance-rel-threshold "${ROUTER_BALANCE_REL_THRESHOLD}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

cd "${MILES_DIR}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export PYTHONPATH="${ROLLOUT_PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --num-cpus "${RAY_NUM_CPUS}" \
   --port="${RAY_GCS_PORT}" --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - <<'PY'
import json
import os

keys = [
    "PYTHONPATH", "PATH", "CUDA_DEVICE_MAX_CONNECTIONS",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "MILES_BALANCE_SANDBOX_PROJECTS",
    "MILES_PLACEMENT_GROUP_TIMEOUT_SEC",
    "AGENT_MAX_TURNS", "AGENT_TIMEOUT_SEC", "AGENT_MAX_TOKENS_PER_TURN",
    "QWEN_CODE_API_TIMEOUT_MS", "QWEN_CODE_MAX_RETRIES",
    "MCORE_DIST_CKPT_THREAD_COUNT", "MCORE_DIST_CKPT_WRITE_ATTEMPTS",
    "SWE_GRADER_TIMEOUT_SEC",
    "SANDBOX_CREATE_MAX_ATTEMPTS", "SANDBOX_CREATE_RETRY_BASE_SEC",
    "SANDBOX_CREATE_RETRY_MAX_SEC", "SANDBOX_CONCURRENCY_PRIMARY",
    "SANDBOX_CONCURRENCY_SECONDARY", "SWEBENCH_RUNTIME_DIR",
    "SBX_API_KEY", "SBX_API_URL", "SBX_API_KEY_PRIMARY", "SBX_API_URL_PRIMARY",
    "SBX_API_KEY_SECONDARY", "SBX_API_URL_SECONDARY",
]
print(json.dumps({"env_vars": {key: os.environ.get(key, "") for key in keys}}))
PY
)"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${PYTHON_BIN}" train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}" \
   --critic-num-nodes 1 \
   --critic-num-gpus-per-node "${CRITIC_NUM_GPUS}" \
   --rollout-num-gpus "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PPO_ARGS[@]}" \
   "${DISTRIBUTED_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
