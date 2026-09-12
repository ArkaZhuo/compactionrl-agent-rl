#!/usr/bin/env bash
# Independent four/eight-GPU PPO launcher. This script does not source or
# modify the GRPO launcher, and it refuses to start while another Ray runtime
# is active.

set -euo pipefail

ACTION="${1:-check}"
case "${ACTION}" in
  check|smoke-4gpu|train-4gpu|train-secondary-8gpu|resume-secondary-8gpu|resume-4gpu|resume-8gpu) ;;
  *) echo "usage: $0 {check|smoke-4gpu|train-4gpu|train-secondary-8gpu|resume-secondary-8gpu|resume-4gpu|resume-8gpu}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="${AVA_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CREDENTIAL_FILE="${SANDBOX_CREDENTIAL_FILE:-${SCRIPT_DIR}/.env}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
MODEL_ROOT="${MODEL_ROOT:-${SHARED_ROOT}/model/Qwen}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/Qwen3.5-4B_torch_dist}"
SDK_ROOT="${INSPIRE_SANDBOX_PYTHONPATH:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-${SHARED_ROOT}/swe-rl/protocol/avaeval-qwen-code-0.21.0-node-22.23.1-wstunnel-10.6.2}"
SWEBENCH_RUNTIME_DIR="${SWEBENCH_RUNTIME_DIR:-${SHARED_ROOT}/.deps/swebench-runtime-py312}"
BACKEND_RUNNER="${AVA_ROOT}/miles/examples/agentic_swe/run_qwen35_4b_4gpu_ppo.sh"
RUNTIME_MODE="${RUNTIME_MODE:-image}"

case "${RUNTIME_MODE}" in
  image)
    PYTHON_BIN="${PYTHON_BIN:-}"
    if [[ -z "${PYTHON_BIN}" ]]; then
      candidates=(
        "$(command -v python 2>/dev/null || true)"
        "$(command -v python3 2>/dev/null || true)"
        /usr/local/bin/python /usr/local/bin/python3 /opt/conda/bin/python
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
  venv)
    VENV_DIR="${VENV_DIR:-${AVA_ROOT}/.venv}"
    PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
    ;;
  *) echo "RUNTIME_MODE must be 'image' or 'venv'" >&2; exit 2 ;;
esac

SWE_DEV_DUAL_DATA="${SWE_DEV_DUAL_DATA:-${SHARED_ROOT}/swe-rl/data/swe_dev_1000_dual_project_avatrain_qwen_code_0.21.0.jsonl}"
SWE_DEV_SECONDARY_DATA="${SWE_DEV_SECONDARY_DATA:-${SHARED_ROOT}/swe-rl/data/swe_dev_500_secondary_project_avatrain_qwen_code_0.21.0.jsonl}"
SWE_DEV_DATA="${SWE_DEV_DATA:-${SHARED_ROOT}/swe-rl/data/swe_dev_1000_avatrain_qwen_code_0.21.0.jsonl}"
VERIFIED_DATA="${VERIFIED_DATA:-${SHARED_ROOT}/swe-rl/data/swe_verified_500_avatrain_qwen_code_0.21.0.jsonl}"
if [[ "${ACTION}" == train-secondary-8gpu || "${ACTION}" == resume-secondary-8gpu ]]; then
  [[ -z "${TRAIN_DATASET:-}" || "${TRAIN_DATASET}" == swe-dev-secondary ]] || {
    echo "${ACTION} is pinned to TRAIN_DATASET=swe-dev-secondary" >&2
    exit 2
  }
  TRAIN_DATASET=swe-dev-secondary
else
  TRAIN_DATASET="${TRAIN_DATASET:-swe-dev-dual}"
fi
case "${TRAIN_DATASET}" in
  swe-dev-dual)
    TRAIN_DATA="${TRAIN_DATA:-${SWE_DEV_DUAL_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-1000}"
    DATASET_TAG=swe_dev_dual
    ;;
  swe-dev-secondary)
    TRAIN_DATA="${TRAIN_DATA:-${SWE_DEV_SECONDARY_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-500}"
    DATASET_TAG=swe_dev_secondary
    ;;
  swe-dev)
    TRAIN_DATA="${TRAIN_DATA:-${SWE_DEV_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-1000}"
    DATASET_TAG=swe_dev
    ;;
  verified)
    TRAIN_DATA="${TRAIN_DATA:-${VERIFIED_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-500}"
    DATASET_TAG=verified
    ;;
  *) echo "TRAIN_DATASET must be swe-dev-dual, swe-dev-secondary, swe-dev, or verified" >&2; exit 2 ;;
esac

# Short, explicit recovery entrypoint for the current official SWE-Dev PPO run.
# Pin both sides and clear stale save overrides so an exported value from an old
# terminal command cannot attach a mismatched checkpoint.
if [[ "${ACTION}" == resume-4gpu || "${ACTION}" == resume-8gpu ]]; then
  [[ "${TRAIN_DATASET}" == swe-dev-dual ]] || {
    echo "${ACTION} is pinned to TRAIN_DATASET=swe-dev-dual" >&2
    exit 2
  }
  export LOAD_CHECKPOINT="${SHARED_ROOT}/swe-rl/checkpoints/swe_dev_dual_qwen35_4b_ppo_actor_20260805_133102"
  export CRITIC_LOAD_CHECKPOINT="${SHARED_ROOT}/swe-rl/checkpoints/swe_dev_dual_qwen35_4b_ppo_critic_20260805_133102"
  # Optimizer offload changes the checkpoint's nested sub-optimizer layout.
  # Step 19 was saved with 0.1, so strict optimizer-state resume must match it.
  export OPTIMIZER_CPU_OFFLOAD_FRACTION=0.1
  unset SAVE_DIR CRITIC_SAVE_DIR
fi

# Continue the trained PPO pair at iteration 20 while switching the rollout
# manifest from dual-project data to secondary-only data. The model and
# optimizer state resume normally, but the incompatible dual dataset cursor is
# intentionally not restored. Saves go to a new lineage below.
if [[ "${ACTION}" == resume-secondary-8gpu ]]; then
  export LOAD_CHECKPOINT="${SHARED_ROOT}/swe-rl/checkpoints/swe_dev_dual_qwen35_4b_ppo_actor_20260805_133102"
  export CRITIC_LOAD_CHECKPOINT="${SHARED_ROOT}/swe-rl/checkpoints/swe_dev_dual_qwen35_4b_ppo_critic_20260805_133102"
  export OPTIMIZER_CPU_OFFLOAD_FRACTION=0.1
  export RESET_ROLLOUT_DATASET_STATE=1
  unset SAVE_DIR CRITIC_SAVE_DIR
else
  export RESET_ROLLOUT_DATASET_STATE=0
fi

# The dual run's dataset state addresses a different 1000-row manifest.  A
# secondary-only run must start with a fresh dataset state and checkpoint pair.
if [[ "${ACTION}" == train-secondary-8gpu ]]; then
  [[ -z "${LOAD_CHECKPOINT:-}" && -z "${CRITIC_LOAD_CHECKPOINT:-}" && \
     -z "${SAVE_DIR:-}" && -z "${CRITIC_SAVE_DIR:-}" ]] || {
    echo "train-secondary-8gpu requires fresh, automatically isolated actor/critic checkpoints" >&2
    exit 2
  }
fi

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${SHARED_ROOT}/logs/swe-rl/${DATASET_TAG}-ppo-gpu}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${ACTION}_${RUN_TS}.log}"

if [[ -f "${CREDENTIAL_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CREDENTIAL_FILE}"
fi

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

fail() { echo "ERROR: $*" >&2; echo "Log: ${LOG_FILE}" >&2; exit 1; }
step() { echo; echo "[$(date -u -Is)] === $* ==="; }

[[ -x "${PYTHON_BIN}" ]] || fail "selected Python is missing: ${PYTHON_BIN}"
if [[ "${RUNTIME_MODE}" == image ]]; then
  unset VIRTUAL_ENV
  export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
  export PYTHONPATH="${AVA_ROOT}/Megatron-LM:${AVA_ROOT}/miles:${AVA_ROOT}/.vendor/mbridge:${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export VIRTUAL_ENV="$(dirname -- "$(dirname -- "${PYTHON_BIN}")")"
  export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
  export PYTHONPATH="${AVA_ROOT}/.vendor/python/transformers-5.9.0:${AVA_ROOT}/Megatron-LM:${AVA_ROOT}/miles:${AVA_ROOT}/.vendor/mbridge:${AVA_ROOT}/sglang/python:${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PATH="${PROTOCOL_BUNDLE}/linux/bin:${PATH}"

export RUNTIME_MODE PYTHON_BIN HF_CHECKPOINT REF_LOAD SWEBENCH_RUNTIME_DIR
if [[ "${ACTION}" == resume-8gpu || "${ACTION}" == train-secondary-8gpu || \
      "${ACTION}" == resume-secondary-8gpu ]]; then
  # Keep checkpoint TP=2 and expand each trainable model from DP=1 to DP=2.
  # The optimizer checkpoint is dp_reshardable (DP changes are supported),
  # while changing TP would not preserve strict optimizer-state compatibility.
  # Rollout remains eight independent TP=1 engines in the backend runner.
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export NUM_GPUS=8 ACTOR_NUM_GPUS=4 CRITIC_NUM_GPUS=4 TRAIN_TP_SIZE=2
  default_ray_num_cpus=16
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export NUM_GPUS=4 ACTOR_NUM_GPUS=2 CRITIC_NUM_GPUS=2 TRAIN_TP_SIZE=2
  default_ray_num_cpus=8
fi
export NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
export TRAIN_SEQUENCE_PARALLEL="${TRAIN_SEQUENCE_PARALLEL:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
# PPO gives the actor and critic half of the selected GPUs each. Qwen3.5's
# large vocabulary makes long SWE trajectories create multi-GiB logits gradients.
# Smaller packing/log-prob limits reduce avoidable intermediates; the allocator
# setting below is also required because a single trajectory cannot be split by
# max-tokens-per-gpu.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-16}"
export TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-536870912}"
export OPTIMIZER_CPU_OFFLOAD_FRACTION="${OPTIMIZER_CPU_OFFLOAD_FRACTION:-0.4}"
# TorchMemorySaver currently rejects PyTorch's expandable-segments allocator.
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-${default_ray_num_cpus}}"
export MILES_PLACEMENT_GROUP_TIMEOUT_SEC="${MILES_PLACEMENT_GROUP_TIMEOUT_SEC:-300}"
export CKPT_FULLY_PARALLEL_SAVE="${CKPT_FULLY_PARALLEL_SAVE:-0}"
export CKPT_ASSUME_CONSTANT_STRUCTURE="${CKPT_ASSUME_CONSTANT_STRUCTURE:-1}"
# The checkpoint target is a shared filesystem. One shard writer per rank is
# slower but avoids the concurrent large-file writes that previously produced
# PyTorch inline-container short writes.
export MCORE_DIST_CKPT_THREAD_COUNT="${MCORE_DIST_CKPT_THREAD_COUNT:-1}"
export MCORE_DIST_CKPT_WRITE_ATTEMPTS="${MCORE_DIST_CKPT_WRITE_ATTEMPTS:-3}"
export AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-12}"
export AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-1200}"
export AGENT_MAX_TOKENS_PER_TURN="${AGENT_MAX_TOKENS_PER_TURN:-8192}"
# qwen-code otherwise retries a 120-second request three times. The proxy
# cannot reuse the abandoned generation, so one slow turn can waste ~8 minutes.
export QWEN_CODE_API_TIMEOUT_MS="${QWEN_CODE_API_TIMEOUT_MS:-360000}"
export QWEN_CODE_MAX_RETRIES="${QWEN_CODE_MAX_RETRIES:-0}"
export SGLANG_ROUTER_REQUEST_TIMEOUT_SECS="${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS:-240}"
export SGLANG_ROUTER_MAX_ATTEMPTS="${SGLANG_ROUTER_MAX_ATTEMPTS:-1}"
export AGENTIC_PROXY_HTTP_TIMEOUT_SEC="${AGENTIC_PROXY_HTTP_TIMEOUT_SEC:-300}"
export AGENTIC_PROXY_HTTP_MAX_ATTEMPTS="${AGENTIC_PROXY_HTTP_MAX_ATTEMPTS:-1}"
export TUNNEL_READY_TIMEOUT_SEC="${TUNNEL_READY_TIMEOUT_SEC:-60}"
export TUNNEL_READY_PROBE_TIMEOUT_SEC="${TUNNEL_READY_PROBE_TIMEOUT_SEC:-5}"
export SWE_GRADER_TIMEOUT_SEC="${SWE_GRADER_TIMEOUT_SEC:-600}"
export DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH-}"
export ROUTER_CACHE_THRESHOLD="${ROUTER_CACHE_THRESHOLD:-0.30}"
export ROUTER_BALANCE_ABS_THRESHOLD="${ROUTER_BALANCE_ABS_THRESHOLD:-8}"
export ROUTER_BALANCE_REL_THRESHOLD="${ROUTER_BALANCE_REL_THRESHOLD:-1.25}"
if [[ "${TRAIN_DATASET}" == swe-dev-secondary ]]; then
  # Secondary-only data must not enable dual-project interleaving.  Dataset
  # preflight below rejects any accidental primary row before Ray is started.
  export MILES_BALANCE_SANDBOX_PROJECTS=0
  export SANDBOX_CONCURRENCY_PRIMARY=0
  export SANDBOX_CONCURRENCY_SECONDARY="${SANDBOX_CONCURRENCY_SECONDARY:-64}"
  export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.70}"
  export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-12}"
  export SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-round_robin}"
else
  export MILES_BALANCE_SANDBOX_PROJECTS="${MILES_BALANCE_SANDBOX_PROJECTS:-1}"
  # A dual batch contains up to 32 rows from each project. Let all 64 episodes
  # make progress so tool-heavy sandboxes keep feeding the eight rollout GPUs.
  export SANDBOX_CONCURRENCY_PRIMARY="${SANDBOX_CONCURRENCY_PRIMARY:-32}"
  export SANDBOX_CONCURRENCY_SECONDARY="${SANDBOX_CONCURRENCY_SECONDARY:-32}"
  # SGLang shares every GPU with a temporarily resident Megatron actor or
  # critic.  Values >=0.85 have repeatedly OOMed when TorchMemorySaver resumes
  # the static KV pool after a weight update; 0.70 leaves room for that overlap.
  export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.70}"
  export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-32}"
  export SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-cache_aware}"
fi
export SANDBOX_CREATE_MAX_ATTEMPTS="${SANDBOX_CREATE_MAX_ATTEMPTS:-120}"
export SANDBOX_CREATE_RETRY_BASE_SEC="${SANDBOX_CREATE_RETRY_BASE_SEC:-2}"
export SANDBOX_CREATE_RETRY_MAX_SEC="${SANDBOX_CREATE_RETRY_MAX_SEC:-30}"

# Use each rollout batch for one optimization pass. This matches the current
# CompactionRL-style experiment setting and avoids a second backward pass over
# the same long SWE trajectories.
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
export PPO_EPS_CLIP="${PPO_EPS_CLIP:-0.2}"
export PPO_VALUE_CLIP="${PPO_VALUE_CLIP:-0.2}"
export PPO_GAMMA="${PPO_GAMMA:-1.0}"
export PPO_LAMBDA="${PPO_LAMBDA:-1.0}"
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-1}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export CRITIC_LR="${CRITIC_LR:-1e-5}"

if [[ "${ACTION}" == smoke-4gpu ]]; then
  # Two rollouts are intentional: rollout 0 exercises critic-only warmup;
  # rollout 1 exercises critic->actor value synchronization, GAE, both PPO
  # losses, actor weight publication, and paired checkpoint saving.
  [[ -z "${LOAD_CHECKPOINT:-}" && -z "${CRITIC_LOAD_CHECKPOINT:-}" && \
     -z "${SAVE_DIR:-}" && -z "${CRITIC_SAVE_DIR:-}" ]] || \
    fail "smoke-4gpu requires fresh, automatically isolated actor/critic checkpoints"
  export NUM_ROLLOUT=2
  export ROLLOUT_BATCH_SIZE=1
  export N_SAMPLES_PER_PROMPT=1
  export GLOBAL_BATCH_SIZE=1
  export SAVE_INTERVAL=1
  export NUM_CRITIC_ONLY_STEPS=1
  export SANDBOX_CONCURRENCY_PRIMARY=1
  export SANDBOX_CONCURRENCY_SECONDARY=1
fi

for integer_var in NUM_ROLLOUT ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE \
  ROLLOUT_MAX_RESPONSE_LEN \
  SAVE_INTERVAL MAX_TOKENS_PER_GPU LOG_PROBS_CHUNK_SIZE PPO_EPOCHS RAY_NUM_CPUS \
  MILES_PLACEMENT_GROUP_TIMEOUT_SEC SGLANG_SERVER_CONCURRENCY \
  SGLANG_ROUTER_REQUEST_TIMEOUT_SECS SGLANG_ROUTER_MAX_ATTEMPTS \
  AGENTIC_PROXY_HTTP_TIMEOUT_SEC AGENTIC_PROXY_HTTP_MAX_ATTEMPTS \
  MCORE_DIST_CKPT_THREAD_COUNT MCORE_DIST_CKPT_WRITE_ATTEMPTS \
  ROUTER_BALANCE_ABS_THRESHOLD \
  SANDBOX_CONCURRENCY_SECONDARY SANDBOX_CREATE_MAX_ATTEMPTS AGENT_MAX_TURNS \
  AGENT_TIMEOUT_SEC AGENT_MAX_TOKENS_PER_TURN QWEN_CODE_API_TIMEOUT_MS \
  TUNNEL_READY_TIMEOUT_SEC TUNNEL_READY_PROBE_TIMEOUT_SEC \
  SWE_GRADER_TIMEOUT_SEC; do
  [[ "${!integer_var}" =~ ^[1-9][0-9]*$ ]] || fail "${integer_var} must be a positive integer"
done
[[ "${SANDBOX_CONCURRENCY_PRIMARY}" =~ ^[0-9]+$ ]] || \
  fail "SANDBOX_CONCURRENCY_PRIMARY must be a non-negative integer"
[[ "${QWEN_CODE_MAX_RETRIES}" =~ ^[0-9]+$ ]] || \
  fail "QWEN_CODE_MAX_RETRIES must be a non-negative integer"
(( SGLANG_ROUTER_REQUEST_TIMEOUT_SECS < AGENTIC_PROXY_HTTP_TIMEOUT_SEC )) || \
  fail "router timeout must be smaller than proxy timeout"
(( AGENTIC_PROXY_HTTP_TIMEOUT_SEC * 1000 < QWEN_CODE_API_TIMEOUT_MS )) || \
  fail "proxy timeout must be smaller than qwen-code timeout"
(( TUNNEL_READY_PROBE_TIMEOUT_SEC < TUNNEL_READY_TIMEOUT_SEC )) || \
  fail "TUNNEL_READY_PROBE_TIMEOUT_SEC must be smaller than TUNNEL_READY_TIMEOUT_SEC"
[[ "${SGLANG_MEM_FRACTION}" =~ ^0\.[0-9]*[1-9][0-9]*$ ]] || \
  fail "SGLANG_MEM_FRACTION must be greater than 0 and less than 1"
if (( NUM_GPUS == 8 )) && ! awk -v value="${SGLANG_MEM_FRACTION}" \
  'BEGIN { exit !(value <= 0.70) }'; then
  fail "8-GPU colocated PPO requires SGLANG_MEM_FRACTION <= 0.70; higher values OOM while resuming the KV pool"
fi
[[ "${SGLANG_ROUTER_POLICY}" == round_robin || "${SGLANG_ROUTER_POLICY}" == cache_aware ]] || \
  fail "SGLANG_ROUTER_POLICY must be round_robin or cache_aware"
[[ "${ROUTER_CACHE_THRESHOLD:-0.30}" =~ ^0\.[0-9]+$ ]] || \
  fail "ROUTER_CACHE_THRESHOLD must be greater than 0 and less than 1"
[[ "${ROUTER_BALANCE_REL_THRESHOLD:-1.25}" =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]] || \
  fail "ROUTER_BALANCE_REL_THRESHOLD must be at least 1"
[[ "${TRAIN_MEMORY_MARGIN_BYTES}" =~ ^[0-9]+$ ]] || \
  fail "TRAIN_MEMORY_MARGIN_BYTES must be a non-negative integer"
[[ "${OPTIMIZER_CPU_OFFLOAD_FRACTION}" =~ ^(0\.[0-9]+|1(\.0+)?)$ ]] || \
  fail "OPTIMIZER_CPU_OFFLOAD_FRACTION must be in (0, 1]"
[[ "${NUM_CRITIC_ONLY_STEPS}" =~ ^[0-9]+$ ]] || \
  fail "NUM_CRITIC_ONLY_STEPS must be a non-negative integer"
[[ "${CKPT_FULLY_PARALLEL_SAVE}" =~ ^[01]$ ]] || \
  fail "CKPT_FULLY_PARALLEL_SAVE must be 0 or 1"
[[ "${CKPT_ASSUME_CONSTANT_STRUCTURE}" =~ ^[01]$ ]] || \
  fail "CKPT_ASSUME_CONSTANT_STRUCTURE must be 0 or 1"
[[ "${RESET_ROLLOUT_DATASET_STATE}" =~ ^[01]$ ]] || \
  fail "RESET_ROLLOUT_DATASET_STATE must be 0 or 1"
(( NUM_CRITIC_ONLY_STEPS < NUM_ROLLOUT )) || \
  fail "NUM_CRITIC_ONLY_STEPS must be smaller than NUM_ROLLOUT or the actor will never train"
total_trajectories=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
(( total_trajectories % GLOBAL_BATCH_SIZE == 0 )) || \
  fail "ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT must be divisible by GLOBAL_BATCH_SIZE"
if [[ "${TRAIN_DATASET}" == swe-dev-secondary ]]; then
  (( SANDBOX_CONCURRENCY_SECONDARY <= 64 )) || \
    fail "secondary-only sandbox concurrency must not exceed 64"
else
  (( SANDBOX_CONCURRENCY_PRIMARY + SANDBOX_CONCURRENCY_SECONDARY <= total_trajectories )) || \
    fail "dual sandbox concurrency must not exceed trajectories per rollout batch"
fi
echo "Action             : ${ACTION}"
echo "Algorithm          : PPO (GAE + trainable critic)"
echo "Runtime mode       : ${RUNTIME_MODE}"
echo "Python             : ${PYTHON_BIN}"
echo "Training data      : ${TRAIN_DATA}"
echo "Visible GPU IDs    : ${CUDA_VISIBLE_DEVICES}"
echo "Actor/Critic GPUs  : ${ACTOR_NUM_GPUS}/${CRITIC_NUM_GPUS} (TP=${TRAIN_TP_SIZE} each)"
echo "Sequence parallel  : ${TRAIN_SEQUENCE_PARALLEL}"
echo "PPO batch          : ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} sample(s), ${PPO_EPOCHS} epochs"
echo "Memory guard       : tokens=${MAX_TOKENS_PER_GPU}, chunk=${LOG_PROBS_CHUNK_SIZE}, margin=${TRAIN_MEMORY_MARGIN_BYTES} bytes"
echo "Agent token limits : trajectory=${ROLLOUT_MAX_RESPONSE_LEN}, per_turn=${AGENT_MAX_TOKENS_PER_TURN}"
echo "Agent API policy   : timeout_ms=${QWEN_CODE_API_TIMEOUT_MS}, retries=${QWEN_CODE_MAX_RETRIES}"
echo "Request deadlines  : router=${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS}s, proxy=${AGENTIC_PROXY_HTTP_TIMEOUT_SEC}s, qwen=${QWEN_CODE_API_TIMEOUT_MS}ms"
echo "Request attempts   : router=${SGLANG_ROUTER_MAX_ATTEMPTS}, proxy=${AGENTIC_PROXY_HTTP_MAX_ATTEMPTS}, qwen_retries=${QWEN_CODE_MAX_RETRIES}"
echo "Tunnel readiness   : total_timeout=${TUNNEL_READY_TIMEOUT_SEC}s, probe_timeout=${TUNNEL_READY_PROBE_TIMEOUT_SEC}s"
echo "Rollout serving    : mem_fraction=${SGLANG_MEM_FRACTION}, server_concurrency=${SGLANG_SERVER_CONCURRENCY}, router=${SGLANG_ROUTER_POLICY}"
if [[ "${SGLANG_ROUTER_POLICY}" == cache_aware ]]; then
  echo "Router thresholds  : cache=${ROUTER_CACHE_THRESHOLD}, abs=${ROUTER_BALANCE_ABS_THRESHOLD}, rel=${ROUTER_BALANCE_REL_THRESHOLD}"
fi
echo "Optimizer offload  : ${OPTIMIZER_CPU_OFFLOAD_FRACTION} of Adam state/master weights to CPU"
echo "Epoch memory reset : enabled between PPO optimization epochs"
echo "Ray resources      : CPUs=${RAY_NUM_CPUS}, placement timeout=${MILES_PLACEMENT_GROUP_TIMEOUT_SEC}s"
echo "Checkpoint mode    : fully_parallel=${CKPT_FULLY_PARALLEL_SAVE}, constant_structure=${CKPT_ASSUME_CONSTANT_STRUCTURE}"
echo "Checkpoint writers : per_rank=${MCORE_DIST_CKPT_THREAD_COUNT}, attempts=${MCORE_DIST_CKPT_WRITE_ATTEMPTS}"
echo "Checkpoint cadence : every ${SAVE_INTERVAL} steps"
echo "Extra checkpoints  : completed_updates=${MILES_EXTRA_CHECKPOINT_STEPS:-none}"
echo "Dataset state      : $([[ "${RESET_ROLLOUT_DATASET_STATE}" == 1 ]] && echo reset-for-secondary || echo normal)"
echo "GRPO isolation     : separate launcher, logs, actor checkpoints, and critic checkpoints"
echo "Log                : ${LOG_FILE}"

step "Preflight"
[[ -f "${HF_CHECKPOINT}/model.safetensors.index.json" ]] || fail "HF checkpoint is incomplete"
[[ -s "${REF_LOAD}/release/.metadata" ]] || fail "torch_dist metadata is missing"
[[ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]] || fail "torch_dist tracker is missing"
[[ -f "${TRAIN_DATA}" ]] || fail "training data is missing"
[[ "$(wc -l < "${TRAIN_DATA}")" -eq "${TRAIN_DATA_ROWS}" ]] || fail "unexpected training row count"
[[ -x "${PROTOCOL_BUNDLE}/linux/bin/wstunnel" ]] || fail "pinned wstunnel is missing"
[[ -f "${SWEBENCH_RUNTIME_DIR}/swebench/__init__.py" ]] || fail "SWE-bench runtime is missing"
[[ -f "${BACKEND_RUNNER}" ]] || fail "PPO backend runner is missing"
if [[ "${TRAIN_DATASET}" == swe-dev-secondary ]]; then
  : "${SBX_API_KEY_SECONDARY:?SBX_API_KEY_SECONDARY is missing; check ${CREDENTIAL_FILE}}"
  export SBX_API_URL_SECONDARY="${SBX_API_URL_SECONDARY:-${SBX_API_URL:-https://qz-sbx-api.sii.edu.cn}}"
else
  : "${SBX_API_KEY:?SBX_API_KEY is missing; check ${CREDENTIAL_FILE}}"
  : "${SBX_API_URL:?SBX_API_URL is missing; check ${CREDENTIAL_FILE}}"
fi
if [[ "${TRAIN_DATASET}" == swe-dev-dual ]]; then
  : "${SBX_API_KEY_SECONDARY:?SBX_API_KEY_SECONDARY is required for swe-dev-dual}"
  export SBX_API_URL_SECONDARY="${SBX_API_URL_SECONDARY:-${SBX_API_URL}}"
fi

# Even a read-only Torch CUDA import creates a context. Refuse both check and
# train while GRPO owns the node so PPO preflight cannot consume its last VRAM.
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  fail "another Ray runtime is active; PPO check/train will not disturb it"
fi

TRAIN_DATA="${TRAIN_DATA}" TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS}" TRAIN_DATASET="${TRAIN_DATASET}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os

path = os.environ["TRAIN_DATA"]
expected_rows = int(os.environ["TRAIN_DATA_ROWS"])
labels = set()
project_counts = {}
required_metadata = {
    "instance_id", "repo", "base_commit", "inspire_template", "test_patch",
    "FAIL_TO_PASS", "PASS_TO_PASS", "install_config", "sandbox_project",
}
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
assert len(rows) == expected_rows, (len(rows), expected_rows)
for index, row in enumerate(rows, 1):
    assert set(row) >= {"prompt", "label", "metadata"}, index
    assert isinstance(row["prompt"], list) and row["prompt"], index
    assert isinstance(row["label"], str) and row["label"], index
    assert row["label"] not in labels, f"duplicate label at row {index}: {row['label']}"
    labels.add(row["label"])
    metadata = row["metadata"]
    assert isinstance(metadata, dict) and required_metadata <= set(metadata), index
    assert metadata["instance_id"] == row["label"], index
    project = metadata["sandbox_project"]
    assert project in {"primary", "secondary"}, (index, project)
    project_counts[project] = project_counts.get(project, 0) + 1
if os.environ["TRAIN_DATASET"] == "swe-dev-dual":
    assert project_counts.get("primary") == project_counts.get("secondary") == expected_rows // 2, project_counts
elif os.environ["TRAIN_DATASET"] == "swe-dev-secondary":
    assert project_counts == {"secondary": expected_rows}, project_counts
print(f"dataset_schema=OK unique_labels={len(labels)} projects={project_counts}")
PY

"${PYTHON_BIN}" - <<'PY'
import os

import torch
import ray
import miles
import megatron
import inspire_sandbox

assert torch.cuda.is_available()
required_gpus = int(os.environ["NUM_GPUS"])
assert torch.cuda.device_count() >= required_gpus, (torch.cuda.device_count(), required_gpus)
for index in range(required_gpus):
    with torch.cuda.device(index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_fraction = free_bytes / total_bytes
    assert free_fraction >= 0.90, (
        f"GPU {index} is not clean: free={free_bytes / 2**30:.1f} GiB "
        f"total={total_bytes / 2**30:.1f} GiB ({free_fraction:.1%} free)"
    )
    print(
        f"gpu={index} free_gib={free_bytes / 2**30:.1f} "
        f"total_gib={total_bytes / 2**30:.1f} free_fraction={free_fraction:.1%}"
    )
print("visible_gpus=", torch.cuda.device_count())
print("READY: PPO actor/critic runtime imports passed")
PY

if [[ "${ACTION}" == check ]]; then
  echo "READY: independent PPO preflight passed; no training was started."
  echo "Log: ${LOG_FILE}"
  exit 0
fi

validate_checkpoint() {
  local checkpoint="$1"
  local label="$2"
  local tracker="${checkpoint%/}/latest_checkpointed_iteration.txt"
  [[ -f "${tracker}" ]] || fail "${label} tracker is missing: ${tracker}"
  local iteration
  iteration="$(<"${tracker}")"
  [[ "${iteration}" =~ ^[0-9]+$ ]] || fail "${label} tracker is not numeric"
  local iter_dir
  iter_dir="$(printf '%s/iter_%07d' "${checkpoint%/}" "${iteration}")"
  [[ -s "${iter_dir}/.metadata" ]] || fail "${label} metadata is missing: ${iter_dir}/.metadata"
  printf '%s' "${iteration}"
}

if [[ -n "${LOAD_CHECKPOINT:-}" || -n "${CRITIC_LOAD_CHECKPOINT:-}" ]]; then
  [[ -n "${LOAD_CHECKPOINT:-}" && -n "${CRITIC_LOAD_CHECKPOINT:-}" ]] || \
    fail "PPO resume requires both LOAD_CHECKPOINT and CRITIC_LOAD_CHECKPOINT"
  actor_iteration="$(validate_checkpoint "${LOAD_CHECKPOINT}" actor)"
  critic_iteration="$(validate_checkpoint "${CRITIC_LOAD_CHECKPOINT}" critic)"
  [[ "${actor_iteration}" == "${critic_iteration}" ]] || \
    fail "actor iteration ${actor_iteration} != critic iteration ${critic_iteration}"
  (( NUM_ROLLOUT > actor_iteration + 1 )) || \
    fail "NUM_ROLLOUT=${NUM_ROLLOUT} must be greater than next resume step $((actor_iteration + 1))"
  dataset_state="${LOAD_CHECKPOINT%/}/rollout/global_dataset_state_dict_${actor_iteration}.pt"
  [[ -s "${dataset_state}" ]] || fail "PPO source dataset state is missing: ${dataset_state}"
  if [[ "${ACTION}" == resume-secondary-8gpu ]]; then
    export SAVE_DIR="${SHARED_ROOT}/swe-rl/checkpoints/${DATASET_TAG}_qwen35_4b_ppo_actor_resume_from${actor_iteration}_${RUN_TS}"
    export CRITIC_SAVE_DIR="${SHARED_ROOT}/swe-rl/checkpoints/${DATASET_TAG}_qwen35_4b_ppo_critic_resume_from${critic_iteration}_${RUN_TS}"
    [[ ! -e "${SAVE_DIR}" ]] || fail "secondary actor save directory already exists: ${SAVE_DIR}"
    [[ ! -e "${CRITIC_SAVE_DIR}" ]] || fail "secondary critic save directory already exists: ${CRITIC_SAVE_DIR}"
  else
    export SAVE_DIR="${SAVE_DIR:-${LOAD_CHECKPOINT%/}}"
    export CRITIC_SAVE_DIR="${CRITIC_SAVE_DIR:-${CRITIC_LOAD_CHECKPOINT%/}}"
    [[ "${SAVE_DIR%/}" == "${LOAD_CHECKPOINT%/}" ]] || fail "SAVE_DIR must equal LOAD_CHECKPOINT on resume"
    [[ "${CRITIC_SAVE_DIR%/}" == "${CRITIC_LOAD_CHECKPOINT%/}" ]] || \
      fail "CRITIC_SAVE_DIR must equal CRITIC_LOAD_CHECKPOINT on resume"
  fi
else
  checkpoint_kind=ppo
  [[ "${ACTION}" != smoke-4gpu ]] || checkpoint_kind=ppo_smoke
  export SAVE_DIR="${SAVE_DIR:-${SHARED_ROOT}/swe-rl/checkpoints/${DATASET_TAG}_qwen35_4b_${checkpoint_kind}_actor_${RUN_TS}}"
  export CRITIC_SAVE_DIR="${CRITIC_SAVE_DIR:-${SHARED_ROOT}/swe-rl/checkpoints/${DATASET_TAG}_qwen35_4b_${checkpoint_kind}_critic_${RUN_TS}}"
  export CRITIC_LOAD_CHECKPOINT="${REF_LOAD}"
fi

export RAY_GCS_PORT="${RAY_GCS_PORT:-6385}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8275}"
ray_started=1
cleanup() {
  if [[ "${ray_started}" == 1 && "${KEEP_RAY:-0}" != 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

step "Launch PPO (${ACTION})"
echo "rollouts=${NUM_ROLLOUT} prompts=${ROLLOUT_BATCH_SIZE} samples_per_prompt=${N_SAMPLES_PER_PROMPT} global_batch=${GLOBAL_BATCH_SIZE}"
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
  echo "resume_checkpoint_iteration=${actor_iteration} next_step=$((actor_iteration + 1)) dataset_state_reset=${RESET_ROLLOUT_DATASET_STATE}"
  echo "actor_load=${LOAD_CHECKPOINT}"
  echo "critic_load=${CRITIC_LOAD_CHECKPOINT}"
fi
echo "actor_save=${SAVE_DIR}"
echo "critic_save=${CRITIC_SAVE_DIR}"
echo "sandbox_concurrency primary=${SANDBOX_CONCURRENCY_PRIMARY} secondary=${SANDBOX_CONCURRENCY_SECONDARY}"
bash "${BACKEND_RUNNER}" "${TRAIN_DATA}"

step "Result"
[[ -d "${SAVE_DIR}" ]] || fail "actor checkpoint directory was not created"
[[ -d "${CRITIC_SAVE_DIR}" ]] || fail "critic checkpoint directory was not created"
final_actor_iteration="$(validate_checkpoint "${SAVE_DIR}" actor)"
final_critic_iteration="$(validate_checkpoint "${CRITIC_SAVE_DIR}" critic)"
[[ "${final_actor_iteration}" == "${final_critic_iteration}" ]] || \
  fail "completed run has unpaired checkpoints: actor=${final_actor_iteration}, critic=${final_critic_iteration}"
[[ -s "${SAVE_DIR%/}/rollout/global_dataset_state_dict_${final_actor_iteration}.pt" ]] || \
  fail "completed run is missing dataset state for iteration ${final_actor_iteration}"
if [[ "${ACTION}" == smoke-4gpu ]]; then
  [[ "${final_actor_iteration}" == 1 ]] || \
    fail "two-step smoke expected checkpoint iteration 1, got ${final_actor_iteration}"
fi
echo "paired_checkpoint_iteration=${final_actor_iteration}"
echo "READY: PPO ${ACTION} completed with separate actor and critic checkpoints."
echo "Log: ${LOG_FILE}"
