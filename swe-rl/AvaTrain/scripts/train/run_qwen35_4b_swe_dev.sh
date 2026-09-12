#!/usr/bin/env bash

# Qwen3.5-4B Agentic SWE runner over the unmodified Miles/SGLang/Megatron
# checkouts. Only the model shape, four-GPU dense parallelism, local assets,
# and SWE-Dev parser are adapted here.

set -euo pipefail

ACTION="${1:-smoke-1gpu}"
case "${ACTION}" in
  check|smoke-1gpu|train-4gpu|full-4gpu) ;;
  *) echo "usage: $0 {check|smoke-1gpu|train-4gpu|full-4gpu}" >&2; exit 2 ;;
esac

SWE_GRADER_TIMEOUT_SEC="${SWE_GRADER_TIMEOUT_SEC:-600}"
[[ "${SWE_GRADER_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] || \
  fail "SWE_GRADER_TIMEOUT_SEC must be a positive integer"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MILES_DIR="${AVA_ROOT}/miles"
AGENT_DIR="${MILES_DIR}/examples/agentic_swe"
OVERLAY_DIR="${AVA_ROOT}/scripts/agentic_swe_4b"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"

HF_CHECKPOINT="${HF_CHECKPOINT:-${SHARED_ROOT}/model/Qwen/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${SHARED_ROOT}/model/Qwen/Qwen3.5-4B_torch_dist}"
SWE_DEV_DATA="${SWE_DEV_DATA:-${SHARED_ROOT}/swe-rl/data/swe_dev_ready_avatrain_qwen_code_0.21.0.jsonl}"
SWEBENCH_RUNTIME_DIR="${SWEBENCH_RUNTIME_DIR:-${SHARED_ROOT}/.deps/swebench-runtime-py312}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-${SHARED_ROOT}/swe-rl/protocol/avaeval-qwen-code-0.21.0-node-22.23.1-wstunnel-10.6.2}"
INSPIRE_SANDBOX_PYTHONPATH="${INSPIRE_SANDBOX_PYTHONPATH:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox}"
CREDENTIAL_FILE="${SANDBOX_CREDENTIAL_FILE:-${SHARED_ROOT}/.credentials/avatrain_sandbox.sh}"
LOG_ROOT="${LOG_ROOT:-${SHARED_ROOT}/logs/swe-rl/swe-dev-qwen35-4b}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${ACTION}_${RUN_TS}.log}"

if [[ -f "${CREDENTIAL_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CREDENTIAL_FILE}"
fi

select_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    [[ -x "${PYTHON_BIN}" ]] || return 1
    return 0
  fi
  local candidate
  local candidates=(
    "$(command -v python 2>/dev/null || true)"
    "$(command -v python3 2>/dev/null || true)"
    /usr/local/bin/python /usr/local/bin/python3
    /opt/conda/bin/python /opt/venv/bin/python /venv/bin/python
    /root/.venv/bin/python /root/miles/.venv/bin/python
    /usr/bin/python3
  )
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if env -u VIRTUAL_ENV "${candidate}" -c 'import torch, ray, transformers' >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      return 0
    fi
  done
  return 1
}

select_python || {
  echo "ERROR: no image Python can import torch, ray, and transformers." >&2
  echo "Set PYTHON_BIN to the image environment's Python executable." >&2
  exit 1
}
export PYTHON_BIN
unset VIRTUAL_ENV
export PATH="$(dirname -- "${PYTHON_BIN}"):${PROTOCOL_BUNDLE}/linux/bin:${PATH}"
export PYTHONPATH="${OVERLAY_DIR}:${AGENT_DIR}:${AVA_ROOT}/Megatron-LM:${MILES_DIR}:${INSPIRE_SANDBOX_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export SWEBENCH_RUNTIME_DIR

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

fail() { echo "ERROR: $*" >&2; echo "Log: ${LOG_FILE}" >&2; exit 1; }
step() { echo; echo "[$(date -u -Is)] === $* ==="; }

echo "Action          : ${ACTION}"
echo "AvaTrain        : ${AVA_ROOT}"
echo "Python          : ${PYTHON_BIN}"
echo "HF checkpoint   : ${HF_CHECKPOINT}"
echo "torch_dist      : ${REF_LOAD}"
echo "SWE-Dev data    : ${SWE_DEV_DATA}"
echo "SWE-bench       : ${SWEBENCH_RUNTIME_DIR}"
echo "Protocol bundle : ${PROTOCOL_BUNDLE}"
echo "Log             : ${LOG_FILE}"

step "Static inputs"
[[ -s "${HF_CHECKPOINT}/model.safetensors.index.json" ]] || fail "HF checkpoint is incomplete"
[[ -s "${REF_LOAD}/release/.metadata" ]] || fail "torch_dist metadata is missing"
[[ -s "${REF_LOAD}/latest_checkpointed_iteration.txt" ]] || fail "torch_dist tracker is missing"
[[ -s "${SWE_DEV_DATA}" ]] || fail "SWE-Dev JSONL is missing"
[[ -x "${PROTOCOL_BUNDLE}/linux/bin/wstunnel" ]] || fail "pinned wstunnel is missing"
[[ -f "${SWEBENCH_RUNTIME_DIR}/swebench/__init__.py" ]] || fail "SWE-bench runtime is missing"
: "${SBX_API_KEY:?SBX_API_KEY is missing; export it or provide ${CREDENTIAL_FILE}}"
: "${SBX_API_URL:?SBX_API_URL is missing; export it or provide ${CREDENTIAL_FILE}}"
DATA_ROWS="$(wc -l < "${SWE_DEV_DATA}")"
echo "SWE-Dev rows=${DATA_ROWS}"

step "Runtime imports"
"${PYTHON_BIN}" - "${SWEBENCH_RUNTIME_DIR}" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

import numpy
import ray
import torch
import transformers
import sglang
import sgl_kernel
import transformer_engine
import torch_memory_saver
import miles
import megatron
import inspire_sandbox

swebench_runtime = sys.argv[1]
if swebench_runtime not in sys.path:
    sys.path.append(swebench_runtime)
import swebench
import generate

print("python             =", sys.version.split()[0], sys.executable)
print("torch              =", torch.__version__, "cuda=", torch.version.cuda)
print("numpy              =", numpy.__version__)
print("transformers       =", transformers.__version__)
print("ray                =", ray.__version__)
print("sglang             =", importlib.metadata.version("sglang"))
print("sglang-kernel      =", importlib.metadata.version("sglang-kernel"))
print("transformer-engine =", importlib.metadata.version("transformer-engine"))
print("swebench           =", swebench.__version__)
print("visible_gpus       =", torch.cuda.device_count())
assert sys.version_info[:2] == (3, 12)
assert numpy.__version__.startswith("1."), "Miles/Megatron requires NumPy 1.x"
assert torch.cuda.is_available()
preload = Path(torch_memory_saver.__file__).resolve().parent.parent / "torch_memory_saver_hook_mode_preload.abi3.so"
assert preload.is_file(), f"missing torch_memory_saver preload: {preload}"
for index in range(torch.cuda.device_count()):
    capability = torch.cuda.get_device_capability(index)
    print(f"GPU {index}: {torch.cuda.get_device_name(index)} capability={capability}")
    assert capability == (9, 0), f"H100/SM90 required, got {capability}"
print("READY: imports and SWE-Dev overlay passed")
PY

if [[ "${ACTION}" == "check" ]]; then
  echo "READY: environment and static inputs passed."
  echo "Log: ${LOG_FILE}"
  exit 0
fi

case "${ACTION}" in
  smoke-1gpu)
    NUM_GPUS=1
    NUM_ROLLOUT=1
    ROLLOUT_BATCH_SIZE=1
    N_SAMPLES_PER_PROMPT=1
    GLOBAL_BATCH_SIZE=1
    SGLANG_SERVER_CONCURRENCY=1
    AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-8}"
    AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-900}"
    DEBUG_ROLLOUT_ONLY=1
    DYNAMIC_SAMPLING_FILTER_PATH=""
    DEBUG_ROLLOUT_DATA="${DEBUG_ROLLOUT_DATA:-${SHARED_ROOT}/swe-rl/smoke/swe_dev_rollout_{rollout_id}.pt}"
    ;;
  train-4gpu)
    NUM_GPUS=4
    NUM_ROLLOUT=1
    ROLLOUT_BATCH_SIZE=1
    N_SAMPLES_PER_PROMPT=4
    GLOBAL_BATCH_SIZE=4
    SGLANG_SERVER_CONCURRENCY=4
    AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-12}"
    AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-1200}"
    DEBUG_ROLLOUT_ONLY=0
    DYNAMIC_SAMPLING_FILTER_PATH=""
    ;;
  full-4gpu)
    [[ "${DATA_ROWS}" -ge 1000 ]] || fail "full SWE-Dev requires 1000 ready rows/templates; found ${DATA_ROWS}"
    NUM_GPUS=4
    NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
    SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-32}"
    AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-80}"
    AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-5400}"
    DEBUG_ROLLOUT_ONLY=0
    DYNAMIC_SAMPLING_FILTER_PATH="miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    ;;
esac

VISIBLE_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
[[ "${VISIBLE_GPUS}" -ge "${NUM_GPUS}" ]] || fail "${ACTION} requires ${NUM_GPUS} visible GPUs; found ${VISIBLE_GPUS}"
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  fail "a Ray runtime is already active; stop only your own stale Ray runtime before retrying"
fi

export NUM_GPUS NUM_ROLLOUT ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE
export SGLANG_SERVER_CONCURRENCY AGENT_MAX_TURNS AGENT_TIMEOUT_SEC SWE_GRADER_TIMEOUT_SEC DEBUG_ROLLOUT_ONLY
export DYNAMIC_SAMPLING_FILTER_PATH DEBUG_ROLLOUT_DATA
export HF_CHECKPOINT REF_LOAD SBX_API_KEY SBX_API_URL
export SAVE_DIR="${SAVE_DIR:-${SHARED_ROOT}/swe-rl/checkpoints/swe_dev_qwen35_4b_${ACTION}_${RUN_TS}}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"

source "${MILES_DIR}/scripts/models/qwen3.5-4B.sh"
DISTRIBUTED_ARGS=()

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)
ROLLOUT_ARGS=(
  --prompt-data "${SWE_DEV_DATA}"
  --input-key prompt --label-key label --metadata-key metadata
  --custom-generate-function-path generate.generate
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len 16384
  --rollout-temperature 1.0 --rollout-top-p 0.95
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
  ROLLOUT_ARGS+=(--debug-rollout-only --save-debug-rollout-data "${DEBUG_ROLLOUT_DATA}")
fi
PERF_ARGS=(
  --tensor-model-parallel-size 1 --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
)
GRPO_ARGS=(
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00
  --kl-loss-type low_var_kl --entropy-coef 0.00
  --eps-clip 0.2 --eps-clip-high 0.28 --use-tis
)
OPTIMIZER_ARGS=(
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.98
)
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.85
  --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
)
MISC_ARGS=(
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash
)

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export RUNTIME_PYTHONPATH="${PYTHONPATH}"
RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - <<'PY'
import json
import os
keys = [
    "PYTHONPATH", "PATH", "CUDA_DEVICE_MAX_CONNECTIONS",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "AGENT_MAX_TURNS",
    "AGENT_TIMEOUT_SEC", "SWE_GRADER_TIMEOUT_SEC", "SWEBENCH_RUNTIME_DIR", "SBX_API_KEY", "SBX_API_URL",
]
print(json.dumps({"env_vars": {key: os.environ.get(key, "") for key in keys}}))
PY
)"

RAY_GCS_PORT="${RAY_GCS_PORT:-6385}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8275}"
ray_started=0
cleanup() {
  if [[ "${ray_started}" == "1" && "${KEEP_RAY:-0}" != "1" ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

step "Launch ${ACTION}"
cd "${MILES_DIR}"
unset RAY_ADDRESS
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" \
  --port="${RAY_GCS_PORT}" --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"
ray_started=1

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" --colocate \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" "${GRPO_ARGS[@]}" "${DISTRIBUTED_ARGS[@]}" \
  "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"

step "Result"
if [[ "${ACTION}" == "smoke-1gpu" ]]; then
  output="${DEBUG_ROLLOUT_DATA/\{rollout_id\}/0}"
  [[ -s "${output}" ]] || fail "rollout output was not created: ${output}"
  ls -lh "${output}"
  echo "READY: one real Qwen3.5-4B SWE-Dev rollout completed."
else
  [[ -d "${SAVE_DIR}" ]] || fail "checkpoint directory was not created: ${SAVE_DIR}"
  echo "READY: ${ACTION} completed and checkpoint output exists."
fi
echo "Log: ${LOG_FILE}"
