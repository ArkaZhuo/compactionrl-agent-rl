#!/bin/bash
# Agentic SWE RL: Qwen3.5-4B with an unmodified qwen-code CLI.
#
# Prerequisites (see README.md):
#   SBX_API_KEY / SBX_API_URL            sandbox platform credentials

set -e
export PYTHONUNBUFFERED=1
# NCCL aborts can otherwise leave multi-gigabyte core files in the source tree.
ulimit -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
AVA_ROOT="$(cd -- "${MILES_DIR}/.." &>/dev/null && pwd)"
RUNTIME_MODE="${RUNTIME_MODE:-venv}"
if [[ "${RUNTIME_MODE}" == "image" ]]; then
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
if [[ "${RUNTIME_MODE}" == "image" ]]; then
   # Keep the official image's SGLang Python package paired with its compiled
   # sglang-kernel; only overlay local Megatron/Miles Agentic SWE sources.
   ROLLOUT_PYTHONPATH="${MILES_DIR}/../Megatron-LM:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
else
   ROLLOUT_PYTHONPATH="${MILES_DIR}/../Megatron-LM:${MILES_DIR}/../sglang/python:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
NUM_GPUS="${NUM_GPUS:-4}"
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-536870912}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"
SKIP_UPDATE_GRAD_NORM_THRESHOLD="${SKIP_UPDATE_GRAD_NORM_THRESHOLD:-}"
CKPT_STEP="${CKPT_STEP:-}"
# Fully-parallel distributed checkpoint saving performs additional NCCL object
# collectives after Miles has repeatedly destroyed/reloaded its train process
# groups.  A rank skew there can deadlock the whole job before any shard is
# written.  This workflow resumes with the same four-GPU topology, so prefer the
# stable non-reshardable save path.  Set this to 1 only when a future checkpoint
# must be resharded onto a different DP topology.
CKPT_FULLY_PARALLEL_SAVE="${CKPT_FULLY_PARALLEL_SAVE:-0}"
CKPT_ASSUME_CONSTANT_STRUCTURE="${CKPT_ASSUME_CONSTANT_STRUCTURE:-1}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH-miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"

[[ "${CKPT_FULLY_PARALLEL_SAVE}" =~ ^[01]$ ]] || {
   echo "CKPT_FULLY_PARALLEL_SAVE must be 0 or 1" >&2
   exit 1
}
[[ "${CKPT_ASSUME_CONSTANT_STRUCTURE}" =~ ^[01]$ ]] || {
   echo "CKPT_ASSUME_CONSTANT_STRUCTURE must be 0 or 1" >&2
   exit 1
}
[[ "${RAY_NUM_CPUS}" =~ ^[1-9][0-9]*$ ]] || {
   echo "RAY_NUM_CPUS must be a positive integer" >&2
   exit 1
}
[[ "${KL_LOSS_COEF}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || {
   echo "KL_LOSS_COEF must be a non-negative number" >&2
   exit 1
}
if [[ -n "${SKIP_UPDATE_GRAD_NORM_THRESHOLD}" ]]; then
   [[ "${SKIP_UPDATE_GRAD_NORM_THRESHOLD}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || {
      echo "SKIP_UPDATE_GRAD_NORM_THRESHOLD must be a non-negative number" >&2
      exit 1
   }
fi
if [[ -n "${CKPT_STEP}" ]]; then
   [[ "${CKPT_STEP}" =~ ^[0-9]+$ ]] || {
      echo "CKPT_STEP must be a non-negative integer" >&2
      exit 1
   }
   [[ -n "${LOAD_CHECKPOINT:-}" ]] || {
      echo "CKPT_STEP requires LOAD_CHECKPOINT" >&2
      exit 1
   }
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:?set HF_CHECKPOINT}"
   --ref-load "${REF_LOAD:?set REF_LOAD (torch_dist checkpoint)}"
   --save "${SAVE_DIR:-/root/agentic_swe_ckpt}"
   --save-interval "${SAVE_INTERVAL}"
)
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
   CKPT_ARGS+=(--load "${LOAD_CHECKPOINT}")
fi
if [[ -n "${CKPT_STEP}" ]]; then
   CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
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
   # The proxy uses this token budget for each model request.
   --rollout-max-response-len 16384
   --rollout-temperature 1.0
   --rollout-top-p 0.95
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [[ "${DEBUG_ROLLOUT_ONLY:-0}" == "1" ]]; then
   debug_rollout_data="${DEBUG_ROLLOUT_DATA:-}"
   if [[ -z "${debug_rollout_data}" ]]; then
      debug_rollout_data='/tmp/agentic_swe_rollout_{rollout_id}.pt'
   fi
   ROLLOUT_ARGS+=(
      --debug-rollout-only
      --save-debug-rollout-data "${debug_rollout_data}"
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TRAIN_TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
   # Avoid materializing the full response-by-248320 FP32 logits clone at
   # once. Log-prob softmax is token-independent, so chunking preserves the
   # result while bounding its temporary memory.
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef "${KL_LOSS_COEF}"
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)
if [[ -n "${SKIP_UPDATE_GRAD_NORM_THRESHOLD}" ]]; then
   GRPO_ARGS+=(--skip-update-grad-norm-threshold "${SKIP_UPDATE_GRAD_NORM_THRESHOLD}")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
   # Both parsers must match the model: the tool parser types arguments from the
   # schema, and mistyped arguments break the token round trip (see README).
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

# ray job submit uploads the working directory and runs train.py from it.
cd "${MILES_DIR}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --num-cpus "${RAY_NUM_CPUS}" \
   --port="${RAY_GCS_PORT}" --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ROLLOUT_PYTHONPATH}\",
    \"PATH\": \"${PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
    \"MILES_BALANCE_SANDBOX_PROJECTS\": \"${MILES_BALANCE_SANDBOX_PROJECTS:-0}\",
    \"AGENT_MAX_TURNS\": \"${AGENT_MAX_TURNS:-80}\",
    \"AGENT_TIMEOUT_SEC\": \"${AGENT_TIMEOUT_SEC:-5400}\",
    \"SWE_GRADER_TIMEOUT_SEC\": \"${SWE_GRADER_TIMEOUT_SEC:-600}\",
    \"SANDBOX_CREATE_MAX_ATTEMPTS\": \"${SANDBOX_CREATE_MAX_ATTEMPTS:-120}\",
    \"SANDBOX_CREATE_RETRY_BASE_SEC\": \"${SANDBOX_CREATE_RETRY_BASE_SEC:-2}\",
    \"SANDBOX_CREATE_RETRY_MAX_SEC\": \"${SANDBOX_CREATE_RETRY_MAX_SEC:-30}\",
    \"SANDBOX_CONCURRENCY_PRIMARY\": \"${SANDBOX_CONCURRENCY_PRIMARY:-0}\",
    \"SANDBOX_CONCURRENCY_SECONDARY\": \"${SANDBOX_CONCURRENCY_SECONDARY:-0}\",
    \"SWEBENCH_RUNTIME_DIR\": \"${SWEBENCH_RUNTIME_DIR:-}\",
    \"SBX_API_KEY\": \"${SBX_API_KEY:-}\",
    \"SBX_API_URL\": \"${SBX_API_URL:-}\",
    \"SBX_API_KEY_PRIMARY\": \"${SBX_API_KEY_PRIMARY:-}\",
    \"SBX_API_URL_PRIMARY\": \"${SBX_API_URL_PRIMARY:-}\",
    \"SBX_API_KEY_SECONDARY\": \"${SBX_API_KEY_SECONDARY:-}\",
    \"SBX_API_URL_SECONDARY\": \"${SBX_API_URL_SECONDARY:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${PYTHON_BIN}" train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${DISTRIBUTED_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
