#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-check}"
case "${ACTION}" in
  check|rollout-1gpu|train-4gpu|train-8gpu) ;;
  *) echo "usage: $0 {check|rollout-1gpu|train-4gpu|train-8gpu}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="${AVA_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CREDENTIAL_FILE="${SANDBOX_CREDENTIAL_FILE:-${SCRIPT_DIR}/.env}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
MODEL_ROOT="${MODEL_ROOT:-${SHARED_ROOT}/model/Qwen}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/Qwen3.5-4B_torch_dist}"
SDK_ROOT="${INSPIRE_SANDBOX_PYTHONPATH:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox}"
VENV_DIR="${VENV_DIR:-${AVA_ROOT}/.venv}"
RUNTIME_MODE="${RUNTIME_MODE:-venv}"
case "${RUNTIME_MODE}" in
  image)
    PYTHON_BIN="${PYTHON_BIN:-}"
    if [[ -z "${PYTHON_BIN}" ]]; then
      candidates=(
        "$(command -v python 2>/dev/null || true)"
        "$(command -v python3 2>/dev/null || true)"
        /usr/local/bin/python
        /usr/local/bin/python3
        /opt/conda/bin/python
        /opt/venv/bin/python
        /venv/bin/python
        /root/.venv/bin/python
        /root/miles/.venv/bin/python
        /usr/bin/python3
      )
      for candidate in "${candidates[@]}"; do
        [[ -n "${candidate}" && -x "${candidate}" ]] || continue
        if env -u VIRTUAL_ENV PYTHONNOUSERSITE=1 "${candidate}" -c 'import torch, ray' >/dev/null 2>&1; then
          PYTHON_BIN="${candidate}"
          break
        fi
      done
    fi
    PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
    ;;
  venv) PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}" ;;
  *) echo "RUNTIME_MODE must be 'image' or 'venv'" >&2; exit 2 ;;
esac
SWEBENCH_RUNTIME_DIR="${SWEBENCH_RUNTIME_DIR:-${SHARED_ROOT}/.deps/swebench-runtime-py312}"
SGLANG_KERNEL_DIR="${SGLANG_KERNEL_DIR:-${SHARED_ROOT}/.deps/sglang-kernel-0.4.2.post2-cu129}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-${SHARED_ROOT}/swe-rl/protocol/avaeval-qwen-code-0.21.0-node-22.23.1-wstunnel-10.6.2}"
VERIFIED_DATA="${VERIFIED_DATA:-${SHARED_ROOT}/swe-rl/data/swe_verified_500_avatrain_qwen_code_0.21.0.jsonl}"
SWE_DEV_DATA="${SWE_DEV_DATA:-${SHARED_ROOT}/swe-rl/data/swe_dev_1000_avatrain_qwen_code_0.21.0.jsonl}"
SWE_DEV_DUAL_DATA="${SWE_DEV_DUAL_DATA:-${SHARED_ROOT}/swe-rl/data/swe_dev_1000_dual_project_avatrain_qwen_code_0.21.0.jsonl}"
SWE_DEV_SECONDARY_DATA="${SWE_DEV_SECONDARY_DATA:-${SHARED_ROOT}/swe-rl/data/swe_dev_1000_secondary_project_avatrain_qwen_code_0.21.0.jsonl}"
SMOKE_DATA="${SMOKE_DATA:-${SHARED_ROOT}/swe-rl/data/smoke_verified_1_avatrain_qwen_code_0.21.0.jsonl}"
# The formal GRPO experiment is secondary-only. Every row carries its own
# inspire_template and is routed through SBX_API_KEY_SECONDARY.
TRAIN_DATASET="${TRAIN_DATASET:-swe-dev-secondary}"
case "${TRAIN_DATASET}" in
  swe-dev)
    TRAIN_DATA="${TRAIN_DATA:-${SWE_DEV_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-1000}"
    DATASET_TAG="swe_dev"
    ;;
  swe-dev-dual)
    TRAIN_DATA="${TRAIN_DATA:-${SWE_DEV_DUAL_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-1000}"
    DATASET_TAG="swe_dev_dual"
    ;;
  swe-dev-secondary)
    TRAIN_DATA="${TRAIN_DATA:-${SWE_DEV_SECONDARY_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-1000}"
    DATASET_TAG="swe_dev_secondary"
    ;;
  verified)
    TRAIN_DATA="${TRAIN_DATA:-${VERIFIED_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-500}"
    DATASET_TAG="verified"
    ;;
  *) echo "TRAIN_DATASET must be 'swe-dev', 'swe-dev-dual', 'swe-dev-secondary', or 'verified'" >&2; exit 2 ;;
esac
LOG_ROOT="${LOG_ROOT:-${SHARED_ROOT}/logs/swe-rl/${DATASET_TAG}-gpu}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${ACTION}_${RUN_TS}.log}"

# Keep credentials outside the launcher's logged output. An explicitly exported
# environment still works when a different credential file is selected/missing.
if [[ -f "${CREDENTIAL_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CREDENTIAL_FILE}"
fi

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

fail() { echo "ERROR: $*" >&2; echo "Log: ${LOG_FILE}" >&2; exit 1; }
step() { echo; echo "[$(date -u -Is)] === $* ==="; }

export HF_CHECKPOINT REF_LOAD PROTOCOL_BUNDLE SWEBENCH_RUNTIME_DIR SGLANG_KERNEL_DIR PYTHON_BIN
export RUNTIME_MODE
[[ -x "${PYTHON_BIN}" ]] || fail "Selected Python is missing: ${PYTHON_BIN} (RUNTIME_MODE=${RUNTIME_MODE})"
if [[ "${RUNTIME_MODE}" == image ]]; then
  unset VIRTUAL_ENV
  # Use the official Miles image's internally matched Transformers, SGLang,
  # sglang-kernel and CUDA extensions. Overlay only the local training/plugin
  # sources and Sandbox SDK that are required by this Agentic SWE recipe.
  PATH="${PATH#${VENV_DIR}/bin:}"
  export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
  export PYTHONPATH="${AVA_ROOT}/Megatron-LM:${AVA_ROOT}/miles:${AVA_ROOT}/.vendor/mbridge:${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  export PYTHONPATH="${AVA_ROOT}/.vendor/python/transformers-5.9.0:${AVA_ROOT}/Megatron-LM:${AVA_ROOT}/miles:${AVA_ROOT}/.vendor/mbridge:${AVA_ROOT}/sglang/python:${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PATH="${PROTOCOL_BUNDLE}/linux/bin:${PATH}"

echo "Action            : ${ACTION}"
echo "Runtime mode      : ${RUNTIME_MODE}"
echo "Python            : ${PYTHON_BIN}"
echo "AvaTrain          : ${AVA_ROOT}"
echo "HF checkpoint     : ${HF_CHECKPOINT}"
echo "torch_dist        : ${REF_LOAD}"
echo "Resume checkpoint : ${LOAD_CHECKPOINT:-<fresh run>}"
echo "SWE-bench runtime : ${SWEBENCH_RUNTIME_DIR}"
if [[ "${RUNTIME_MODE}" == image ]]; then
  echo "SGLang kernel     : managed by the Miles image"
else
  echo "SGLang kernel     : ${SGLANG_KERNEL_DIR}"
fi
echo "Training dataset  : ${TRAIN_DATASET}"
echo "Training data     : ${TRAIN_DATA}"
echo "Log               : ${LOG_FILE}"

step "Static assets"
[[ -f "${HF_CHECKPOINT}/model.safetensors.index.json" ]] || fail "HF checkpoint is incomplete"
CKPT_TRACKER="${REF_LOAD}/latest_checkpointed_iteration.txt"
CKPT_RELEASE="${REF_LOAD}/release"
[[ -f "${CKPT_TRACKER}" ]] || fail "torch_dist checkpoint tracker is missing: ${CKPT_TRACKER}"
[[ "$(<"${CKPT_TRACKER}")" == release ]] || fail "torch_dist checkpoint tracker must point to release"
[[ -s "${CKPT_RELEASE}/.metadata" ]] || fail "torch_dist metadata is missing: ${CKPT_RELEASE}/.metadata"
CKPT_SHARD="$(find "${CKPT_RELEASE}" -maxdepth 1 -type f -name '*.distcp' -print -quit)"
[[ -n "${CKPT_SHARD}" ]] || fail "torch_dist checkpoint has no weight shard under ${CKPT_RELEASE}"
resume_iteration=""
resume_latest_iteration=""
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
  LOAD_CHECKPOINT="${LOAD_CHECKPOINT%/}"
  resume_tracker="${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt"
  [[ -f "${resume_tracker}" ]] || fail "resume checkpoint tracker is missing: ${resume_tracker}"
  resume_latest_iteration="$(<"${resume_tracker}")"
  [[ "${resume_latest_iteration}" =~ ^[0-9]+$ ]] || \
    fail "resume checkpoint tracker must contain a numeric iteration: ${resume_tracker}"
  if [[ -n "${CKPT_STEP:-}" ]]; then
    [[ "${CKPT_STEP}" =~ ^[0-9]+$ ]] || fail "CKPT_STEP must be a non-negative integer"
    resume_iteration="${CKPT_STEP}"
    (( resume_iteration <= resume_latest_iteration )) || \
      fail "CKPT_STEP=${resume_iteration} exceeds latest checkpoint ${resume_latest_iteration}"
    export CKPT_STEP
  else
    resume_iteration="${resume_latest_iteration}"
  fi
  resume_iter_dir="$(printf '%s/iter_%07d' "${LOAD_CHECKPOINT}" "${resume_iteration}")"
  [[ -s "${resume_iter_dir}/.metadata" ]] || \
    fail "resume checkpoint metadata is missing: ${resume_iter_dir}/.metadata"
  resume_dataset_state="${LOAD_CHECKPOINT}/rollout/global_dataset_state_dict_${resume_iteration}.pt"
  [[ -s "${resume_dataset_state}" ]] || \
    fail "resume dataset state is missing: ${resume_dataset_state}"
  export LOAD_CHECKPOINT
  echo "Resume checkpoint: OK; selected=${resume_iteration}, latest=${resume_latest_iteration}, model and dataset state are complete"
fi
[[ -f "${SWEBENCH_RUNTIME_DIR}/swebench/__init__.py" ]] || fail "SWE-bench Python 3.12 runtime is missing"
if [[ "${RUNTIME_MODE}" == venv ]]; then
  [[ -f "${SGLANG_KERNEL_DIR}/sgl_kernel/version.py" ]] || fail "SGLang cu129 kernel runtime is missing"
fi
[[ -f "${TRAIN_DATA}" ]] || fail "Training data is missing: ${TRAIN_DATA}"
[[ "$(wc -l < "${TRAIN_DATA}")" -eq "${TRAIN_DATA_ROWS}" ]] || \
  fail "${TRAIN_DATASET} data must contain ${TRAIN_DATA_ROWS} rows"
if [[ "${TRAIN_DATASET}" == swe-dev-secondary ]]; then
  "${PYTHON_BIN}" - "${TRAIN_DATA}" "${TRAIN_DATA_ROWS}" <<'PY'
import json
import sys

path = sys.argv[1]
expected_rows = int(sys.argv[2])
templates = set()
with open(path, encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        project = metadata.get("sandbox_project")
        template = metadata.get("inspire_template")
        if project != "secondary":
            raise SystemExit(
                f"secondary dataset row {line_number} has sandbox_project={project!r}"
            )
        if not isinstance(template, str) or not template.strip():
            raise SystemExit(f"secondary dataset row {line_number} has no inspire_template")
        if template in templates:
            raise SystemExit(
                f"secondary dataset row {line_number} repeats inspire_template={template!r}"
            )
        templates.add(template)

if len(templates) != expected_rows:
    raise SystemExit(
        f"secondary dataset expected {expected_rows} unique templates, found {len(templates)}"
    )
print(f"Secondary dataset routing: OK; rows={expected_rows}, unique_templates={len(templates)}")
PY
fi
[[ -f "${SMOKE_DATA}" ]] || fail "single-instance smoke data is missing"
[[ -x "${PROTOCOL_BUNDLE}/linux/bin/wstunnel" ]] || fail "pinned host wstunnel is missing"
case "${TRAIN_DATASET}" in
  swe-dev-secondary)
    : "${SBX_API_KEY_SECONDARY:?SBX_API_KEY_SECONDARY is required for swe-dev-secondary; check ${CREDENTIAL_FILE}}"
    SBX_API_URL_SECONDARY="${SBX_API_URL_SECONDARY:-${SBX_API_URL:-}}"
    : "${SBX_API_URL_SECONDARY:?SBX_API_URL_SECONDARY is required for swe-dev-secondary; check ${CREDENTIAL_FILE}}"
    MILES_BALANCE_SANDBOX_PROJECTS=0
    SANDBOX_CONCURRENCY_PRIMARY=0
    SANDBOX_CONCURRENCY_SECONDARY="${SANDBOX_CONCURRENCY_SECONDARY:-64}"
    [[ "${SANDBOX_CONCURRENCY_SECONDARY}" =~ ^[1-9][0-9]*$ ]] || \
      fail "SANDBOX_CONCURRENCY_SECONDARY must be a positive integer"
    export SBX_API_URL_SECONDARY MILES_BALANCE_SANDBOX_PROJECTS
    export SANDBOX_CONCURRENCY_PRIMARY SANDBOX_CONCURRENCY_SECONDARY
    ;;
  swe-dev-dual)
    : "${SBX_API_KEY:?SBX_API_KEY is required for swe-dev-dual; check ${CREDENTIAL_FILE}}"
    : "${SBX_API_URL:?SBX_API_URL is required for swe-dev-dual; check ${CREDENTIAL_FILE}}"
    : "${SBX_API_KEY_SECONDARY:?SBX_API_KEY_SECONDARY is required for swe-dev-dual; check ${CREDENTIAL_FILE}}"
    SBX_API_URL_SECONDARY="${SBX_API_URL_SECONDARY:-${SBX_API_URL}}"
    # Preserve per-epoch randomization while interleaving the equal primary and
    # secondary partitions, so every even prompt batch routes half of its
    # trajectories to each project instead of randomly overloading one pool.
    MILES_BALANCE_SANDBOX_PROJECTS="${MILES_BALANCE_SANDBOX_PROJECTS:-1}"
    # Gate the complete Agent+grader episode, not merely the create request.
    # Defaults reflect the measured project capacities while remaining
    # overrideable for controlled 32/32 comparisons.
    SANDBOX_CONCURRENCY_PRIMARY="${SANDBOX_CONCURRENCY_PRIMARY:-16}"
    SANDBOX_CONCURRENCY_SECONDARY="${SANDBOX_CONCURRENCY_SECONDARY:-32}"
    [[ "${SANDBOX_CONCURRENCY_PRIMARY}" =~ ^[1-9][0-9]*$ ]] || \
      fail "SANDBOX_CONCURRENCY_PRIMARY must be a positive integer"
    [[ "${SANDBOX_CONCURRENCY_SECONDARY}" =~ ^[1-9][0-9]*$ ]] || \
      fail "SANDBOX_CONCURRENCY_SECONDARY must be a positive integer"
    export SBX_API_URL_SECONDARY MILES_BALANCE_SANDBOX_PROJECTS
    export SANDBOX_CONCURRENCY_PRIMARY SANDBOX_CONCURRENCY_SECONDARY
    ;;
  *)
    : "${SBX_API_KEY:?SBX_API_KEY is required for ${TRAIN_DATASET}; check ${CREDENTIAL_FILE}}"
    : "${SBX_API_URL:?SBX_API_URL is required for ${TRAIN_DATASET}; check ${CREDENTIAL_FILE}}"
    ;;
esac
echo "Static assets: OK; sandbox credentials are set (values hidden)"

step "GPU runtime imports"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
command -v ray >/dev/null || fail "ray CLI is unavailable"
command -v wstunnel >/dev/null || fail "host wstunnel is unavailable"
nvidia-smi -L
"${PYTHON_BIN}" - "${HF_CHECKPOINT}" "${SWEBENCH_RUNTIME_DIR}" <<'PY'
import sys
import importlib.util
from importlib.metadata import version
from pathlib import Path
import numpy as np
import scipy

swebench_runtime = sys.argv[2]
if swebench_runtime not in sys.path:
    # Append, rather than prepend: the target directory also contains dependency
    # wheels which must not override AvaTrain's pinned GPU environment.
    sys.path.append(swebench_runtime)

import torch
import ray
import transformers
from transformers import AutoConfig
import inspire_sandbox
import flashinfer
import megatron
import mbridge
import miles
import sglang
import transformer_engine
import torch_memory_saver
import miles_plugins.models.qwen3_5
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

print("python_exe      =", sys.executable)
print("torch_file      =", torch.__file__)
print("sgl_kernel_spec =", importlib.util.find_spec("sgl_kernel"))
import sgl_kernel
import swebench
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER_PY

print("python          =", sys.version.split()[0])
print("numpy           =", np.__version__)
print("scipy           =", scipy.__version__)
print("torch           =", torch.__version__, "cuda=", torch.version.cuda)
print("ray             =", ray.__version__)
print("transformers    =", transformers.__version__)
print("visible_gpus    =", torch.cuda.device_count())
print("inspire_sandbox =", inspire_sandbox.__file__)
print("swebench       =", swebench.__version__)
print("sglang_kernel  =", version("sglang-kernel"))
tms_preload = Path(torch_memory_saver.__file__).resolve().parent.parent / "torch_memory_saver_hook_mode_preload.abi3.so"
print("tms_preload     =", tms_preload)
config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print("model_type      =", config.model_type)
assert sys.version_info[:2] == (3, 12)
assert np.__version__.startswith("1."), "Miles/Megatron requires numpy<2"
assert torch.cuda.is_available()
assert swebench.__version__ == "4.1.0"
assert MAP_REPO_TO_PARSER_PY
assert config.model_type == "qwen3_5"
assert tms_preload.is_file(), f"Miles torch_memory_saver preload library is missing: {tms_preload}"
PY

if [[ "${ACTION}" == check ]]; then
  echo "READY: GPU runtime, checkpoints, data, protocol, and credentials passed preflight."
  echo "Log: ${LOG_FILE}"
  exit 0
fi

if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  fail "A Ray runtime is already active on this node; use a fresh GPU job or stop only your own Ray job first"
fi

ray_started=0
cleanup() {
  if [[ "${ray_started}" == 1 && "${KEEP_RAY:-0}" != 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

export RAY_GCS_PORT="${RAY_GCS_PORT:-6385}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8275}"

case "${ACTION}" in
  rollout-1gpu)
    [[ "$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')" -ge 1 ]] || fail "one visible GPU is required"
    export NUM_GPUS=1
    export TRAIN_TP_SIZE=1
    export NUM_ROLLOUT=1
    export ROLLOUT_BATCH_SIZE=1
    export N_SAMPLES_PER_PROMPT=1
    export GLOBAL_BATCH_SIZE=1
    export DEBUG_ROLLOUT_ONLY=1
    export DYNAMIC_SAMPLING_FILTER_PATH=""
    export SGLANG_SERVER_CONCURRENCY=1
    export AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-8}"
    export AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-900}"
    if [[ -z "${DEBUG_ROLLOUT_DATA:-}" ]]; then
      # Keep the Miles placeholder literal; embedding it directly in a nested
      # ${var:-default} expression lets Bash consume the first closing brace.
      DEBUG_ROLLOUT_DATA="${SHARED_ROOT}/swe-rl/smoke/${DATASET_TAG}_rollout_"'{rollout_id}.pt'
      export DEBUG_ROLLOUT_DATA
    fi
    if [[ "${TRAIN_DATASET}" == swe-dev-secondary ]]; then
      # Keep the one-trajectory smoke test on the same project and per-row
      # template routing as the formal secondary-only run.
      task_file="${TRAIN_DATA}"
    else
      task_file="${SMOKE_DATA}"
    fi
    ;;
  train-4gpu|train-8gpu)
    if [[ "${ACTION}" == train-8gpu ]]; then
      required_gpus=8
    else
      required_gpus=4
    fi
    visible_gpus="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
    (( visible_gpus >= required_gpus )) || \
      fail "${required_gpus} visible GPUs are required, found ${visible_gpus}"
    export NUM_GPUS="${required_gpus}"
    export RAY_NUM_CPUS="${RAY_NUM_CPUS:-$((NUM_GPUS * 2))}"
    # TP=1 leaves every rank holding the complete 248320-way vocabulary
    # logits. Long SWE prompts then exhaust an 80 GiB H100 during the
    # gradient-bearing actor forward. TP=2 shards that peak; DP is 2 on four
    # GPUs and 4 on eight GPUs with the same global batch/hyperparameters.
    export TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-2}"
    # SWE trajectories can approach the response limit.  Keep the temporary
    # FP32 vocabulary-loss buffer small enough that an unusually long sample
    # does not exhaust the tail of an 80 GiB H100.  Chunking is token-wise and
    # therefore does not change the loss values.
    export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-64}"
    [[ "${LOG_PROBS_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
      fail "LOG_PROBS_CHUNK_SIZE must be a positive integer"
    # The upstream 1 GiB TorchMemorySaver margin rejected a 122 MiB allocation
    # with 1.01 GiB physically free (only ~6 MiB remained above the margin).
    # A 512 MiB guard still leaves substantial CUDA headroom while allowing the
    # bounded loss chunks to run.
    export TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-536870912}"
    [[ "${TRAIN_MEMORY_MARGIN_BYTES}" =~ ^[0-9]+$ ]] || \
      fail "TRAIN_MEMORY_MARGIN_BYTES must be a non-negative integer"
    # The previous 9216-token cap reached 77.06 GiB on an unusually long SWE
    # trajectory and OOMed while preparing an FP32 vocabulary-loss chunk.
    # Use a conservative four-H100 production default for long 200-step runs.
    export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
    [[ "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || \
      fail "MAX_TOKENS_PER_GPU must be a positive integer"
    # These four values are part of the optimizer scheduler state.  Keep the
    # production defaults identical to the original run; smoke-sized defaults
    # make checkpoint resume fail (for example, scheduler total 8 vs 32000).
    export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
    export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
    # A four-GPU checkpoint is about 66 GiB. Saving every 50 steps produces
    # checkpoints at 50/100/150/200 instead of filling the shared filesystem.
    export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
    export DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH-}"
    export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
    export AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-12}"
    export AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-1200}"
    export SWE_GRADER_TIMEOUT_SEC="${SWE_GRADER_TIMEOUT_SEC:-600}"
    export KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"
    export SKIP_UPDATE_GRAD_NORM_THRESHOLD="${SKIP_UPDATE_GRAD_NORM_THRESHOLD:-}"
    [[ "${SWE_GRADER_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] || \
      fail "SWE_GRADER_TIMEOUT_SEC must be a positive integer"
    if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
      if [[ "${resume_iteration}" == "${resume_latest_iteration}" ]]; then
        SAVE_DIR="${SAVE_DIR:-${LOAD_CHECKPOINT}}"
        [[ "${SAVE_DIR%/}" == "${LOAD_CHECKPOINT}" ]] || \
          fail "SAVE_DIR must equal LOAD_CHECKPOINT for an in-place continuation"
      else
        SAVE_DIR="${SAVE_DIR:-${SHARED_ROOT}/swe-rl/checkpoints/${DATASET_TAG}_qwen35_4b_recovery_from${resume_iteration}_${RUN_TS}}"
        [[ "${SAVE_DIR%/}" != "${LOAD_CHECKPOINT}" ]] || \
          fail "rollback recovery must use a new SAVE_DIR so newer checkpoints are not overwritten"
        [[ ! -e "${SAVE_DIR}" ]] || \
          fail "rollback recovery SAVE_DIR already exists: ${SAVE_DIR}"
      fi
      export LOAD_CHECKPOINT SAVE_DIR
    else
      export SAVE_DIR="${SAVE_DIR:-${SHARED_ROOT}/swe-rl/checkpoints/${DATASET_TAG}_qwen35_4b_${NUM_GPUS}gpu_${RUN_TS}}"
    fi
    task_file="${TRAIN_DATA}"
    ;;
esac

step "Launch ${ACTION}"
for positive_name in NUM_ROLLOUT ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE; do
  positive_value="${!positive_name}"
  [[ "${positive_value}" =~ ^[1-9][0-9]*$ ]] || fail "${positive_name} must be a positive integer"
done
rollout_sample_count=$((NUM_ROLLOUT * ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
(( rollout_sample_count % GLOBAL_BATCH_SIZE == 0 )) || \
  fail "NUM_ROLLOUT * ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT must be divisible by GLOBAL_BATCH_SIZE"
train_iteration_count=$((rollout_sample_count / GLOBAL_BATCH_SIZE))
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<inherited>}"
echo "Ray ports=${RAY_GCS_PORT}/${RAY_DASHBOARD_PORT}"
echo "ray_num_cpus=${RAY_NUM_CPUS:-8} placement_group_timeout_sec=${MILES_PLACEMENT_GROUP_TIMEOUT_SEC:-300}"
echo "rollouts=${NUM_ROLLOUT} prompts=${ROLLOUT_BATCH_SIZE} samples_per_prompt=${N_SAMPLES_PER_PROMPT} global_batch=${GLOBAL_BATCH_SIZE}"
echo "train_iterations=${train_iteration_count} scheduler_total_samples=${rollout_sample_count}"
echo "training_tp=${TRAIN_TP_SIZE}"
echo "log_probs_chunk_size=${LOG_PROBS_CHUNK_SIZE:-1024}"
echo "train_memory_margin_bytes=${TRAIN_MEMORY_MARGIN_BYTES:-<default>}"
echo "max_tokens_per_gpu=${MAX_TOKENS_PER_GPU:-<default>}"
echo "checkpoint_fully_parallel=${CKPT_FULLY_PARALLEL_SAVE:-0} checkpoint_constant_structure=${CKPT_ASSUME_CONSTANT_STRUCTURE:-1}"
echo "load_checkpoint=${LOAD_CHECKPOINT:-<fresh run>}"
echo "resume_iteration=${resume_iteration:-<fresh run>} latest_source_iteration=${resume_latest_iteration:-<fresh run>}"
echo "save_checkpoint=${SAVE_DIR:-<unset>}"
echo "kl_loss_coef=${KL_LOSS_COEF:-0.00} skip_update_grad_norm_threshold=${SKIP_UPDATE_GRAD_NORM_THRESHOLD:-disabled}"
echo "agent_max_turns=${AGENT_MAX_TURNS} agent_timeout_sec=${AGENT_TIMEOUT_SEC}"
echo "swe_grader_timeout_sec=${SWE_GRADER_TIMEOUT_SEC:-<default>}"
echo "balance_sandbox_projects=${MILES_BALANCE_SANDBOX_PROJECTS:-0}"
echo "sandbox_concurrency primary=${SANDBOX_CONCURRENCY_PRIMARY:-unlimited} secondary=${SANDBOX_CONCURRENCY_SECONDARY:-unlimited}"
ray_started=1
bash "${AVA_ROOT}/miles/examples/agentic_swe/run_qwen35_4b_4gpu.sh" "${task_file}"

step "Result"
if [[ "${ACTION}" == rollout-1gpu ]]; then
  output="${DEBUG_ROLLOUT_DATA/\{rollout_id\}/0}"
  [[ -s "${output}" ]] || fail "rollout output was not created: ${output}"
  ls -lh "${output}"
  echo "READY: one real Agentic SWE rollout completed."
else
  [[ -d "${SAVE_DIR}" ]] || fail "checkpoint directory was not created: ${SAVE_DIR}"
  find "${SAVE_DIR}" -maxdepth 2 -type f -printf '%p\n' | sed -n '1,20p'
  echo "READY: ${NUM_GPUS}GPU Agentic SWE GRPO run completed (num_rollout=${NUM_ROLLOUT}) and checkpoint output exists."
fi
echo "Log: ${LOG_FILE}"
