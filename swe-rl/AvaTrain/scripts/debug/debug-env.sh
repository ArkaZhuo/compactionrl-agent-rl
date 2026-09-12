#!/bin/bash
set -euo pipefail
export UV_NO_SYNC=1

WANDB_API_KEY=local-c6d3e5712d547834724d8d98094f340b3c2a869c
WANDB_BASE_URL=https://wandb2.sii.edu.cn

PROJECT_ROOT=${PROJECT_ROOT:-"$(uv run python -c 'from pathlib import Path; import sys; print(Path(sys.prefix).resolve().parent)')"}
AVALANCHE_ROOT=${AVALANCHE_ROOT:-"/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/avalanche"}
RL_REPO_DIR=${RL_REPO_DIR:-"${PROJECT_ROOT}/miles"}

cd "${PROJECT_ROOT}"
export UV_PROJECT="${PROJECT_ROOT}"

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-8}

MODEL_DIR=${MODEL_DIR:-"${AVALANCHE_ROOT}/models/Qwen3.5-35B-A3B"}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-"${AVALANCHE_ROOT}/models/Qwen3.5-35B-A3B_torch_dist_mtp"}
WORK_ROOT_BASE=${WORK_ROOT_BASE:-"${PROJECT_ROOT}/runs/qwen3.5-35b-a3b-rl"}

MATH_ROLLOUT_BATCH_SIZE=${MATH_ROLLOUT_BATCH_SIZE:-16}
MATH_SAMPLES_PER_PROMPT=${MATH_SAMPLES_PER_PROMPT:-16}
MATH_GLOBAL_BATCH_SIZE=${MATH_GLOBAL_BATCH_SIZE:-256}
MATH_STEPS_PER_ROLLOUT=${MATH_STEPS_PER_ROLLOUT:-1}
MATH_MAX_CONTEXT_LEN=${MATH_MAX_CONTEXT_LEN:-16384}
MATH_MAX_RESPONSE_LEN=${MATH_MAX_RESPONSE_LEN:-16384}
MATH_LR=${MATH_LR:-1e-6}
MATH_ADAM_BETA2=${MATH_ADAM_BETA2:-0.98}
MATH_KL_LOSS_COEF=${MATH_KL_LOSS_COEF:-0.0}
MATH_USE_R3=${MATH_USE_R3:-1}
MATH_RESUME_TRAINING=${MATH_RESUME_TRAINING:-auto}
MATH_LOAD_DIR=${MATH_LOAD_DIR:-}
WANDB_PROJECT=${MATH_WANDB_PROJECT:-debug-env}
WANDB_GROUP=${WANDB_GROUP:-debug-env}
WANDB_RUN_ID=${WANDB_RUN_ID:-debug-env}
FORCE_REBUILD_TORCH_DIST=${FORCE_REBUILD_TORCH_DIST:-0}
MEMORY_SNAPSHOT_PATH=${MEMORY_SNAPSHOT_PATH:-qwen35_snapshot.pickle}
MEMORY_SNAPSHOT_NUM_STEPS=${MEMORY_SNAPSHOT_NUM_STEPS:-3}

RUN_DIR_NAME="${WANDB_GROUP//\//_}"
RUN_DIR_NAME="${RUN_DIR_NAME// /_}"
WORK_ROOT=${WORK_ROOT:-${WORK_ROOT_BASE}/${RUN_DIR_NAME}}
DEBUG_ROLLOUT_DATA_DIR=${DEBUG_ROLLOUT_DATA_DIR:-}
if [[ -z "${DEBUG_ROLLOUT_DATA_DIR}" ]]; then
  DEBUG_ROLLOUT_DATA_DIR="${WORK_ROOT}/debug_rollout/rollout_{rollout_id}.pt"
fi
MEMORY_SNAPSHOT_DIR=${MEMORY_SNAPSHOT_DIR:-${WORK_ROOT}/mem_snapshots/train_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
DATA_CACHE_DIR="${WORK_ROOT}/data_cache"
LOG_DIR="${WORK_ROOT}/logs"
SAVE_DIR="${WORK_ROOT}/checkpoints"
NORMALIZED_TRAIN=${NORMALIZED_TRAIN:-"${AVALANCHE_ROOT}/jx_workspace/AvaTrain/exp/dataset/math_for_test.jsonl"}

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}" "$(dirname "${DEBUG_ROLLOUT_DATA_DIR}")"

DEFAULT_MATH_LOAD_DIR=""
if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  DEFAULT_MATH_LOAD_DIR="${SAVE_DIR}"
fi

if [[ "${MATH_RESUME_TRAINING}" == "auto" ]]; then
  if [[ -n "${DEFAULT_MATH_LOAD_DIR}" ]]; then
    MATH_RESUME_TRAINING=1
  else
    MATH_RESUME_TRAINING=0
  fi
fi

if [[ -z "${MATH_LOAD_DIR}" ]] && [[ -n "${DEFAULT_MATH_LOAD_DIR}" ]]; then
  MATH_LOAD_DIR="${DEFAULT_MATH_LOAD_DIR}"
fi

RUN_LOG_PATH="${LOG_DIR}/debug_rollout_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
exec > >(tee "${RUN_LOG_PATH}") 2>&1

cleanup_local_processes() {
  pkill -9 sglang 2>/dev/null || true
  uv run ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
}

get_local_ip() {
  hostname -I 2>/dev/null | awk '{print $1}' || hostname
}

wait_for_full_ray_cluster() {
  local expected_gpus=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
  local attempt
  local status_output

  for attempt in $(seq 1 300); do
    status_output="$(uv run ray status --address="${MASTER_ADDR}:${MASTER_PORT}" 2>&1 || true)"
    if echo "${status_output}" | grep -Eq "/${expected_gpus}(\\.0)? GPU"; then
      echo "Ray cluster is ready with ${expected_gpus} GPUs."
      return 0
    fi
    echo "Waiting for Ray workers to join (${attempt}/300)..."
    sleep 10
  done

  echo "Ray workers did not join the cluster in time." >&2
  uv run ray status --address="${MASTER_ADDR}:${MASTER_PORT}" || true
  return 1
}

start_ray_worker_with_retry() {
  local attempt
  local rc
  local node_name="${WORKER_ID:-${HOSTNAME:-worker-${NODE_RANK}}}"

  for attempt in $(seq 1 150); do
    set +e
    uv run ray start \
      --address="${MASTER_ADDR}:${MASTER_PORT}" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --node-ip-address "${NODE_IP}" \
      --node-name "${node_name}" \
      --dashboard-port="${DASHBOARD_PORT}" \
      --disable-usage-stats
    rc=$?
    set -e

    if [ "${rc}" -eq 0 ]; then
      echo "Ray worker joined on attempt ${attempt}."
      return 0
    fi

    echo "Ray worker join failed on attempt ${attempt}, retrying..."
    uv run ray stop --force 2>/dev/null || true
    sleep 5
  done

  echo "Ray worker failed to join cluster after retries." >&2
  return 1
}

ensure_torch_dist_checkpoint() {
  if [[ "${FORCE_REBUILD_TORCH_DIST}" == "1" ]]; then
    echo "FORCE_REBUILD_TORCH_DIST=1, rebuilding ${TORCH_DIST_DIR}"
    rm -rf "${TORCH_DIST_DIR}"
  fi

  if [ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "Found torch_dist checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi
  if [ -f "${TORCH_DIST_DIR}/common.pt" ] && [ -f "${TORCH_DIST_DIR}/metadata.json" ]; then
    echo "Found torch_dist iteration checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi

  cd "${PROJECT_ROOT}"
  source "${RL_REPO_DIR}/scripts/models/qwen3.5-35B-A3B.sh"
  uv run python -m torch.distributed.run \
    --nproc-per-node "${NUM_GPUS_PER_NODE}" \
    "${RL_REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --save "${TORCH_DIST_DIR}"
}

submit_ray_job() {
  cd "${PROJECT_ROOT}"
  source "${RL_REPO_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

  # NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
  # if [ "${NVLINK_COUNT}" -gt 0 ]; then
  #   HAS_NVLINK=1
  # else
  #   HAS_NVLINK=0
  # fi
  HAS_NVLINK=0

  ROLLOUT_ARGS=(
    --prompt-data "${NORMALIZED_TRAIN}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --rollout-shuffle
    --num-epoch 1
    --rollout-batch-size "${MATH_ROLLOUT_BATCH_SIZE}"
    # --over-sampling-batch-size "${MATH_OVER_SAMPLING_BATCH_SIZE}"
    --n-samples-per-prompt "${MATH_SAMPLES_PER_PROMPT}"
    --rollout-max-context-len "${MATH_MAX_CONTEXT_LEN}"
    --rollout-max-response-len "${MATH_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --global-batch-size "${MATH_GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout "${MATH_STEPS_PER_ROLLOUT}"
    --balance-data
  )

  # if [[ -n "${MATH_DYNAMIC_FILTER_PATH}" ]]; then
  #   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${MATH_DYNAMIC_FILTER_PATH}")
  # fi

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.8
    --sglang-schedule-conservativeness 1.2
    --sglang-max-running-requests 192
    --sglang-schedule-policy lpm
    --sglang-allow-auto-truncate
    --sglang-preferred-sampling-params '{"ignore_eos": true}'
    --sglang-ep-size 1
    --sglang-reasoning-parser qwen3
    --sglang-tool-call-parser qwen3_coder
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-watchdog-timeout 1200
    --sglang-speculative-algorithm NEXTN
    --sglang-speculative-num-steps 1
    --sglang-speculative-eagle-topk 1
    --sglang-speculative-num-draft-tokens 2
    --sglang-enforce-disable-flashinfer-allreduce-fusion
    # --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
    # --sglang-enable-dp-lm-head
  )

  CUSTOM_ARGS=(
    --custom-rm-path scripts.debug.reward_deepmath_mathverify.reward_func
  )

  if [[ "${MATH_USE_R3}" == "1" ]]; then
    CUSTOM_ARGS+=(--use-rollout-routing-replay)
  fi

  WANDB_ARGS=()
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS+=(
      --use-wandb
      --wandb-host "${WANDB_BASE_URL:-https://wandb.ai}"
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP}"
      --wandb-run-id "${WANDB_RUN_ID}"
      --wandb-key "${WANDB_API_KEY}"
      --disable-wandb-random-suffix
    )
  fi

  RUNTIME_ENV_JSON="{\"env_vars\":{\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\",\"UV_NO_SYNC\":\"1\",\"UV_PROJECT\":\"${PROJECT_ROOT}\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_BASE_URL\":\"${WANDB_BASE_URL:-}\"}}"

  uv run ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- uv run python "${RL_REPO_DIR}/train_async.py" \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --debug-rollout-only \
    --save-debug-rollout-data "${DEBUG_ROLLOUT_DATA_DIR}" \
    "${ROLLOUT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
}

NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_ADDR="${MASTER_ADDR:-$(get_local_ip)}"
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
NODE_IP="$(get_local_ip)"

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

cleanup_local_processes

if [[ "${NODE_RANK}" -eq 0 ]]; then
  ensure_torch_dist_checkpoint
  uv run ray start --head \
    --port="${MASTER_PORT}" \
    --node-ip-address "${MASTER_ADDR}" \
    --node-name "${WORKER_ID:-${HOSTNAME:-head-0}}" \
    --num-gpus "${NUM_GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}"
  wait_for_full_ray_cluster
  submit_ray_job
  uv run ray stop --force || true
else
  sleep 5
  start_ray_worker_with_retry
  while uv run ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; do
    sleep 60
  done
fi
