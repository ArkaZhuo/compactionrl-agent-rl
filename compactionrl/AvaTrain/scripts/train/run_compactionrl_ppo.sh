#!/usr/bin/env bash
# Isolated CompactionRL PPO launcher. This file is intentionally separate from
# scripts/train/run_ppo.sh and never loads its checkpoints.

set -euo pipefail

ACTION="${1:-check}"
case "${ACTION}" in
  check|smoke-8gpu|train-8gpu|resume-8gpu|resume-warmup-8gpu|verify-resume20-8gpu) ;;
  *) echo "usage: $0 {check|smoke-8gpu|train-8gpu|resume-8gpu|resume-warmup-8gpu|verify-resume20-8gpu}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="${AVA_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CREDENTIAL_FILE="${SANDBOX_CREDENTIAL_FILE:-${SCRIPT_DIR}/.env}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
COMPACTION_STORAGE_ROOT="${COMPACTION_STORAGE_ROOT:-/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/compactionrl}"
COMPACTION_CKPT_ROOT="${COMPACTION_CKPT_ROOT:-${COMPACTION_STORAGE_ROOT}/checkpoints}"
COMPACTION_DATA_ROOT="${COMPACTION_DATA_ROOT:-${COMPACTION_STORAGE_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${SHARED_ROOT}/model/Qwen}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/Qwen3.5-4B_torch_dist}"
SDK_ROOT="${INSPIRE_SANDBOX_PYTHONPATH:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-${SHARED_ROOT}/swe-rl/protocol/avaeval-qwen-code-0.21.0-node-22.23.1-wstunnel-10.6.2}"
SWEBENCH_RUNTIME_DIR="${SWEBENCH_RUNTIME_DIR:-${SHARED_ROOT}/.deps/swebench-runtime-py312}"
BACKEND_RUNNER="${AVA_ROOT}/miles/examples/compaction_swe/run_qwen35_4b_8gpu_ppo.sh"
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

SWE_DEV_DUAL_DATA="${SWE_DEV_DUAL_DATA:-${COMPACTION_DATA_ROOT}/swe_dev_1000_dual_project_avatrain_qwen_code_0.21.0.jsonl}"
# The secondary-only 1000-row manifest has the same label/instance_id range as
# the dual 1000-row manifest; only sandbox_project is changed to secondary.
SWE_DEV_SECONDARY_DATA="${SWE_DEV_SECONDARY_DATA:-${SHARED_ROOT}/swe-rl/data/swe_dev_1000_secondary_project_avatrain_qwen_code_0.21.0.jsonl}"
SWE_DEV_DATA="${SWE_DEV_DATA:-${COMPACTION_DATA_ROOT}/swe_dev_1000_avatrain_qwen_code_0.21.0.jsonl}"
VERIFIED_DATA="${VERIFIED_DATA:-${COMPACTION_DATA_ROOT}/swe_verified_500_avatrain_qwen_code_0.21.0.jsonl}"
TRAIN_DATASET="${TRAIN_DATASET:-swe-dev-dual}"
case "${TRAIN_DATASET}" in
  swe-dev-dual)
    TRAIN_DATA="${TRAIN_DATA:-${SWE_DEV_DUAL_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-1000}"
    DATASET_TAG=swe_dev_dual
    ;;
  swe-dev-secondary)
    TRAIN_DATA="${TRAIN_DATA:-${SWE_DEV_SECONDARY_DATA}}"
    TRAIN_DATA_ROWS="${TRAIN_DATA_ROWS:-1000}"
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

if [[ "${ACTION}" == resume-8gpu || "${ACTION}" == resume-warmup-8gpu || \
      "${ACTION}" == verify-resume20-8gpu ]]; then
  export RESET_ROLLOUT_DATASET_STATE="${RESET_ROLLOUT_DATASET_STATE:-0}"
else
  unset LOAD_CHECKPOINT CRITIC_LOAD_CHECKPOINT
  export RESET_ROLLOUT_DATASET_STATE=0
fi

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${COMPACTION_STORAGE_ROOT}/logs/${DATASET_TAG}-ppo-gpu}"
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
  export PYTHONPATH="${AVA_ROOT}/Megatron-LM:${AVA_ROOT}/miles:${AVA_ROOT}/miles/examples:${AVA_ROOT}/examples:${AVA_ROOT}/examples/agentic_swe:${AVA_ROOT}/examples/compaction_swe:${AVA_ROOT}/.vendor/mbridge:${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export VIRTUAL_ENV="$(dirname -- "$(dirname -- "${PYTHON_BIN}")")"
  export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
  export PYTHONPATH="${AVA_ROOT}/.vendor/python/transformers-5.9.0:${AVA_ROOT}/Megatron-LM:${AVA_ROOT}/miles:${AVA_ROOT}/miles/examples:${AVA_ROOT}/examples:${AVA_ROOT}/examples/agentic_swe:${AVA_ROOT}/examples/compaction_swe:${AVA_ROOT}/.vendor/mbridge:${AVA_ROOT}/sglang/python:${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PATH="${PROTOCOL_BUNDLE}/linux/bin:${PATH}"

export RUNTIME_MODE PYTHON_BIN HF_CHECKPOINT REF_LOAD SWEBENCH_RUNTIME_DIR
[[ -f "${HF_CHECKPOINT}/config.json" ]] || fail "HF config is missing: ${HF_CHECKPOINT}/config.json"
MODEL_NATIVE_SEQUENCE_LIMIT="$(
  PYTHONPATH="${AVA_ROOT}/miles/examples${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m compaction_swe.model_config "${HF_CHECKPOINT}/config.json"
)" || fail "failed to read the model-native sequence limit"
[[ "${MODEL_NATIVE_SEQUENCE_LIMIT}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid model-native sequence limit: ${MODEL_NATIVE_SEQUENCE_LIMIT}"
# Match the runnable SWE-RL PPO: the proxy's final request boundary is the HF
# model's native context, not an independent 32K/64K experiment parameter.
export COMPACTION_MODEL_SEQUENCE_LIMIT="${MODEL_NATIVE_SEQUENCE_LIMIT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS=8 ACTOR_NUM_GPUS=4 CRITIC_NUM_GPUS=4 TRAIN_TP_SIZE=2
# Dynamic micro-batching separates samples but cannot split one 64K segment.
# Context parallelism shards each long segment across two training GPUs.
export TRAIN_CP_SIZE="${TRAIN_CP_SIZE:-2}"
default_ray_num_cpus=16
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
# One PPO update consumes one rollout batch by default: 64 prompts x 1 sample.
# Compaction may return more segment samples, and --use-dynamic-global-batch-size
# will derive a larger per-update batch from the collected segment count.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
# This framework-level value seeds sampling parameters only. CompactionRL
# enforces the same per-call bound in its proxy and has no global trajectory cap.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
# Qwen Code housekeeping side queries are disabled by generate.py. Keep the
# production run strict so a remaining real SubAgent/auxiliary request cannot
# be mistaken for an ordered single-agent trajectory.
export COMPACTION_AUXILIARY_MODE="${COMPACTION_AUXILIARY_MODE:-reject}"
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
export AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-250}"
export AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-1200}"
export AGENT_MAX_TOKENS_PER_TURN="${AGENT_MAX_TOKENS_PER_TURN:-2048}"
export QWEN_CODE_API_TIMEOUT_MS="${QWEN_CODE_API_TIMEOUT_MS:-360000}"
export QWEN_CODE_MAX_RETRIES="${QWEN_CODE_MAX_RETRIES:-0}"
# Keep the inner request layers bounded and let them fail before qwen-code's
# outer 360s deadline.  max-attempts is deliberately one at both layers: a
# generation that may already have reached a worker is not safe to replay.
export SGLANG_ROUTER_REQUEST_TIMEOUT_SECS="${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS:-240}"
export SGLANG_ROUTER_MAX_ATTEMPTS="${SGLANG_ROUTER_MAX_ATTEMPTS:-1}"
export COMPACTION_PROXY_HTTP_TIMEOUT_SEC="${COMPACTION_PROXY_HTTP_TIMEOUT_SEC:-300}"
export COMPACTION_PROXY_HTTP_MAX_ATTEMPTS="${COMPACTION_PROXY_HTTP_MAX_ATTEMPTS:-1}"
export TUNNEL_READY_TIMEOUT_SEC="${TUNNEL_READY_TIMEOUT_SEC:-60}"
export TUNNEL_READY_PROBE_TIMEOUT_SEC="${TUNNEL_READY_PROBE_TIMEOUT_SEC:-5}"
export TUNNEL_SUPERVISOR_FAILURE_THRESHOLD="${TUNNEL_SUPERVISOR_FAILURE_THRESHOLD:-3}"
export TUNNEL_SUPERVISOR_FAILURE_WINDOW_SEC="${TUNNEL_SUPERVISOR_FAILURE_WINDOW_SEC:-30}"
export TUNNEL_SUPERVISOR_STARTUP_GRACE_SEC="${TUNNEL_SUPERVISOR_STARTUP_GRACE_SEC:-60}"
export TUNNEL_SUPERVISOR_POLL_SEC="${TUNNEL_SUPERVISOR_POLL_SEC:-0.25}"
export SWE_GRADER_TIMEOUT_SEC="${SWE_GRADER_TIMEOUT_SEC:-600}"
# Opt-in guard for this CompactionRL launcher only. A normal completed episode
# with reward=0 is accepted; these counters cover exceptions and explicitly
# filtered groups, preventing an integrity-gate regression from replenishing a
# 64-episode batch forever.
export COMPACTION_ROLLOUT_MAX_ATTEMPTS="${COMPACTION_ROLLOUT_MAX_ATTEMPTS:-256}"
export COMPACTION_ROLLOUT_MAX_DISCARDED="${COMPACTION_ROLLOUT_MAX_DISCARDED:-128}"
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
  export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-64}"
  # Multi-turn qwen-code requests share long prefixes.  Preserve their cache
  # affinity and account active load instead of rotating every request across
  # workers that merely pass the lightweight health probe.
  export SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-cache_aware}"
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
# The paper pretrains the critic for 50 steps. The previous 20-step shortcut
# entered actor training with an unstable critic and must not be the default
# for a fresh run after changing the trajectory/credit-assignment contract.
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-50}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export CRITIC_LR="${CRITIC_LR:-1e-6}"
# Keep the critic on the same single-pass update contract as the actor unless
# an experiment explicitly opts into additional value passes.  A second pass
# over these long trajectories has produced immediate value clipping and
# unstable critic gradients in the CompactionRL runs.
export CRITIC_PPO_EPOCHS="${CRITIC_PPO_EPOCHS:-1}"
# CompactionRL does not publish a KL coefficient. This small nonzero value is
# an explicit Qwen3.5-4B stability setting, not a claimed paper hyperparameter.
export KL_COEF="${KL_COEF:-0.001}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
export COMPACTION_CONTEXT_BUDGET="${COMPACTION_CONTEXT_BUDGET:-65536}"
export COMPACTION_TRIGGER_TOKENS="${COMPACTION_TRIGGER_TOKENS:-10240}"
export COMPACTION_MAX_TOKENS_PER_TURN="${COMPACTION_MAX_TOKENS_PER_TURN:-${AGENT_MAX_TOKENS_PER_TURN}}"
export COMPACTION_SUMMARY_MAX_TOKENS="${COMPACTION_SUMMARY_MAX_TOKENS:-2048}"
export COMPACTION_MAX_COUNT="${COMPACTION_MAX_COUNT:-3}"
export COMPACTION_RECENT_STEPS="${COMPACTION_RECENT_STEPS:-2}"
export COMPACTION_GAE_ALPHA="${COMPACTION_GAE_ALPHA:-1.5}"

if [[ "${ACTION}" == smoke-8gpu ]]; then
  # Two prompts exercise critic->actor value synchronization, GAE, both PPO
  # losses, actor weight publication, and paired checkpoint saving. Compaction
  # segment counts are aligned by dropping complete trajectories only.
  [[ -z "${LOAD_CHECKPOINT:-}" && -z "${CRITIC_LOAD_CHECKPOINT:-}" && \
     -z "${SAVE_DIR:-}" && -z "${CRITIC_SAVE_DIR:-}" ]] || \
    fail "smoke-8gpu requires fresh, automatically isolated actor/critic checkpoints"
  export NUM_ROLLOUT=2
  export ROLLOUT_BATCH_SIZE=2
  export N_SAMPLES_PER_PROMPT=1
  export GLOBAL_BATCH_SIZE=2
  export SAVE_INTERVAL=1
  export NUM_CRITIC_ONLY_STEPS=1
  export SANDBOX_CONCURRENCY_PRIMARY=1
  export SANDBOX_CONCURRENCY_SECONDARY=1
  # Keep the paper's 64K/10K window semantics in smoke as well. A two-prompt
  # smoke validates the training path but is not guaranteed to reach compaction.
  export COMPACTION_CONTEXT_BUDGET=65536
  export COMPACTION_TRIGGER_TOKENS=10240
  export COMPACTION_MAX_TOKENS_PER_TURN=512
  export AGENT_MAX_TOKENS_PER_TURN=512
  export COMPACTION_SUMMARY_MAX_TOKENS=512
elif [[ "${ACTION}" == verify-resume20-8gpu ]]; then
  # Compatibility-only run over rollout 20..49 using the historical critic-19
  # weights and dataset cursor. The old checkpoint predates the current token
  # ledger, so its output is useful for exercising the runtime but must not be
  # promoted to the fresh experiment's checkpoint lineage.
  export NUM_ROLLOUT=50
  export NUM_CRITIC_ONLY_STEPS=20
  export SAVE_INTERVAL=5
  export ACTOR_LR=5e-7
  export CRITIC_LR=1e-6
  export CRITIC_PPO_EPOCHS=1
  export KL_COEF=0.001
fi

if [[ "${ACTION}" == resume-8gpu ]]; then
  [[ "${ACTOR_LR}" == 5e-7 ]] || \
    fail "resume-8gpu requires ACTOR_LR=5e-7; use resume_compactionrl_ppo.sh or set it explicitly"
  [[ "${CRITIC_LR}" == 1e-6 ]] || \
    fail "resume-8gpu requires CRITIC_LR=1e-6"
  [[ "${CRITIC_PPO_EPOCHS}" == 1 ]] || \
    fail "resume-8gpu requires CRITIC_PPO_EPOCHS=1; refusing an incompatible critic resume"
fi

for integer_var in NUM_ROLLOUT ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE \
  TRAIN_CP_SIZE \
  ROLLOUT_MAX_RESPONSE_LEN \
  SAVE_INTERVAL MAX_TOKENS_PER_GPU LOG_PROBS_CHUNK_SIZE PPO_EPOCHS RAY_NUM_CPUS \
  MILES_PLACEMENT_GROUP_TIMEOUT_SEC SGLANG_SERVER_CONCURRENCY \
  SGLANG_ROUTER_REQUEST_TIMEOUT_SECS SGLANG_ROUTER_MAX_ATTEMPTS \
  COMPACTION_PROXY_HTTP_TIMEOUT_SEC COMPACTION_PROXY_HTTP_MAX_ATTEMPTS \
  MCORE_DIST_CKPT_THREAD_COUNT MCORE_DIST_CKPT_WRITE_ATTEMPTS \
  ROUTER_BALANCE_ABS_THRESHOLD \
  SANDBOX_CONCURRENCY_SECONDARY SANDBOX_CREATE_MAX_ATTEMPTS AGENT_MAX_TURNS \
  AGENT_TIMEOUT_SEC AGENT_MAX_TOKENS_PER_TURN QWEN_CODE_API_TIMEOUT_MS \
  TUNNEL_READY_TIMEOUT_SEC TUNNEL_READY_PROBE_TIMEOUT_SEC \
  SWE_GRADER_TIMEOUT_SEC; do
  [[ "${!integer_var}" =~ ^[1-9][0-9]*$ ]] || fail "${integer_var} must be a positive integer"
done
for compaction_var in COMPACTION_CONTEXT_BUDGET COMPACTION_MODEL_SEQUENCE_LIMIT \
  COMPACTION_TRIGGER_TOKENS COMPACTION_MAX_TOKENS_PER_TURN COMPACTION_SUMMARY_MAX_TOKENS \
  COMPACTION_MAX_COUNT COMPACTION_RECENT_STEPS; do
  [[ "${!compaction_var}" =~ ^[0-9]+$ ]] || fail "${compaction_var} must be a non-negative integer"
done
[[ "${COMPACTION_AUXILIARY_MODE}" == reject || "${COMPACTION_AUXILIARY_MODE}" == serve_untracked ]] || \
  fail "COMPACTION_AUXILIARY_MODE must be reject or serve_untracked"
(( COMPACTION_MODEL_SEQUENCE_LIMIT >= COMPACTION_CONTEXT_BUDGET )) || \
  fail "COMPACTION_MODEL_SEQUENCE_LIMIT must be >= COMPACTION_CONTEXT_BUDGET"
(( COMPACTION_TRIGGER_TOKENS < COMPACTION_CONTEXT_BUDGET )) || \
  fail "COMPACTION_TRIGGER_TOKENS must be smaller than COMPACTION_CONTEXT_BUDGET"
(( COMPACTION_TRIGGER_TOKENS >= COMPACTION_MAX_TOKENS_PER_TURN && \
   COMPACTION_TRIGGER_TOKENS >= COMPACTION_SUMMARY_MAX_TOKENS )) || \
  fail "COMPACTION_TRIGGER_TOKENS must reserve both one execution turn and one summary"
(( AGENT_MAX_TOKENS_PER_TURN == COMPACTION_MAX_TOKENS_PER_TURN )) || \
  fail "AGENT_MAX_TOKENS_PER_TURN and COMPACTION_MAX_TOKENS_PER_TURN must match"
(( COMPACTION_MAX_COUNT <= 3 )) || \
  fail "COMPACTION_MAX_COUNT must be <= 3"
[[ "${SANDBOX_CONCURRENCY_PRIMARY}" =~ ^[0-9]+$ ]] || \
  fail "SANDBOX_CONCURRENCY_PRIMARY must be a non-negative integer"
[[ "${QWEN_CODE_MAX_RETRIES}" =~ ^[0-9]+$ ]] || \
  fail "QWEN_CODE_MAX_RETRIES must be a non-negative integer"
[[ "${COMPACTION_ROLLOUT_MAX_ATTEMPTS}" =~ ^[0-9]+$ ]] || \
  fail "COMPACTION_ROLLOUT_MAX_ATTEMPTS must be a non-negative integer"
[[ "${COMPACTION_ROLLOUT_MAX_DISCARDED}" =~ ^[0-9]+$ ]] || \
  fail "COMPACTION_ROLLOUT_MAX_DISCARDED must be a non-negative integer"
(( COMPACTION_ROLLOUT_MAX_ATTEMPTS == 0 || \
   COMPACTION_ROLLOUT_MAX_ATTEMPTS >= ROLLOUT_BATCH_SIZE )) || \
  fail "COMPACTION_ROLLOUT_MAX_ATTEMPTS must be 0 or at least ROLLOUT_BATCH_SIZE"
(( SGLANG_ROUTER_REQUEST_TIMEOUT_SECS < COMPACTION_PROXY_HTTP_TIMEOUT_SEC )) || \
  fail "request timeout order must be router < proxy"
(( COMPACTION_PROXY_HTTP_TIMEOUT_SEC * 1000 < QWEN_CODE_API_TIMEOUT_MS )) || \
  fail "request timeout order must be proxy < qwen-code"
[[ "${TUNNEL_SUPERVISOR_STARTUP_GRACE_SEC}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  fail "TUNNEL_SUPERVISOR_STARTUP_GRACE_SEC must be a non-negative number"
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
[[ "${KL_COEF}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  fail "KL_COEF must be a non-negative decimal number"
[[ "${ENTROPY_COEF}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  fail "ENTROPY_COEF must be a non-negative decimal number"
[[ "${CKPT_FULLY_PARALLEL_SAVE}" =~ ^[01]$ ]] || \
  fail "CKPT_FULLY_PARALLEL_SAVE must be 0 or 1"
[[ "${CKPT_ASSUME_CONSTANT_STRUCTURE}" =~ ^[01]$ ]] || \
  fail "CKPT_ASSUME_CONSTANT_STRUCTURE must be 0 or 1"
[[ "${RESET_ROLLOUT_DATASET_STATE}" =~ ^[01]$ ]] || \
  fail "RESET_ROLLOUT_DATASET_STATE must be 0 or 1"
(( NUM_CRITIC_ONLY_STEPS < NUM_ROLLOUT )) || \
  fail "NUM_CRITIC_ONLY_STEPS must be smaller than NUM_ROLLOUT or the actor will never train"
total_trajectories=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [[ "${ACTION}" == smoke-8gpu ]]; then
  (( total_trajectories % GLOBAL_BATCH_SIZE == 0 )) || \
    fail "smoke rollout batch must be divisible by global batch size"
fi
if [[ "${TRAIN_DATASET}" == swe-dev-secondary ]]; then
  (( SANDBOX_CONCURRENCY_SECONDARY <= 64 )) || \
    fail "secondary-only sandbox concurrency must not exceed 64"
else
  (( SANDBOX_CONCURRENCY_PRIMARY + SANDBOX_CONCURRENCY_SECONDARY <= total_trajectories )) || \
    fail "dual sandbox concurrency must not exceed trajectories per rollout batch"
fi
echo "Action             : ${ACTION}"
echo "Algorithm          : CompactionRL PPO (summary training + cross-trajectory GAE)"
echo "Runtime mode       : ${RUNTIME_MODE}"
echo "Python             : ${PYTHON_BIN}"
echo "Training data      : ${TRAIN_DATA}"
train_model_parallel_size=$((TRAIN_TP_SIZE * TRAIN_CP_SIZE))
(( ACTOR_NUM_GPUS % train_model_parallel_size == 0 )) || \
  fail "ACTOR_NUM_GPUS must be divisible by TRAIN_TP_SIZE * TRAIN_CP_SIZE"
(( CRITIC_NUM_GPUS % train_model_parallel_size == 0 )) || \
  fail "CRITIC_NUM_GPUS must be divisible by TRAIN_TP_SIZE * TRAIN_CP_SIZE"
actor_dp_size=$((ACTOR_NUM_GPUS / train_model_parallel_size))
critic_dp_size=$((CRITIC_NUM_GPUS / train_model_parallel_size))
echo "Actor/Critic GPUs  : ${ACTOR_NUM_GPUS}/${CRITIC_NUM_GPUS} (TP=${TRAIN_TP_SIZE}, CP=${TRAIN_CP_SIZE}, DP=${actor_dp_size}/${critic_dp_size})"
echo "Training horizon   : rollouts=${NUM_ROLLOUT}"
echo "Auxiliary mode     : ${COMPACTION_AUXILIARY_MODE}"
echo "PPO batch          : ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} sample(s), ${PPO_EPOCHS} epochs"
echo "Memory guard       : tokens=${MAX_TOKENS_PER_GPU}, chunk=${LOG_PROBS_CHUNK_SIZE}, margin=${TRAIN_MEMORY_MARGIN_BYTES} bytes"
echo "Agent token limits : global_trajectory=disabled, per_turn=${AGENT_MAX_TOKENS_PER_TURN}"
echo "Agent API policy   : timeout_ms=${QWEN_CODE_API_TIMEOUT_MS}, retries=${QWEN_CODE_MAX_RETRIES}"
echo "Qwen background    : auto_memory=off, auto_dream=off, auto_skill=off, next_speaker=off, native_compaction=off"
echo "Request deadlines  : router=${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS}s, proxy=${COMPACTION_PROXY_HTTP_TIMEOUT_SEC}s, qwen=${QWEN_CODE_API_TIMEOUT_MS}ms"
echo "Request attempts   : router=${SGLANG_ROUTER_MAX_ATTEMPTS}, proxy=${COMPACTION_PROXY_HTTP_MAX_ATTEMPTS}, qwen_retries=${QWEN_CODE_MAX_RETRIES}"
echo "Tunnel readiness   : total_timeout=${TUNNEL_READY_TIMEOUT_SEC}s, probe_timeout=${TUNNEL_READY_PROBE_TIMEOUT_SEC}s"
echo "Tunnel supervisor  : failures=${TUNNEL_SUPERVISOR_FAILURE_THRESHOLD}, window=${TUNNEL_SUPERVISOR_FAILURE_WINDOW_SEC}s, poll=${TUNNEL_SUPERVISOR_POLL_SEC}s"
echo "Tunnel startup grace: ${TUNNEL_SUPERVISOR_STARTUP_GRACE_SEC}s"
echo "Rollout fuse       : attempts=${COMPACTION_ROLLOUT_MAX_ATTEMPTS}, discarded=${COMPACTION_ROLLOUT_MAX_DISCARDED}"
echo "Sequence limits    : working_window=${COMPACTION_CONTEXT_BUDGET}, model_native=${COMPACTION_MODEL_SEQUENCE_LIMIT}"
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
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
  echo "Checkpoint load    : step=${CKPT_STEP:-latest}"
fi
echo "Dataset state      : $([[ "${RESET_ROLLOUT_DATASET_STATE}" == 1 ]] && echo reset-for-secondary || echo normal)"
echo "Compaction config  : window=${COMPACTION_CONTEXT_BUDGET}, trigger_reserve=${COMPACTION_TRIGGER_TOKENS}, trigger_at=$((COMPACTION_CONTEXT_BUDGET - COMPACTION_TRIGGER_TOKENS)), max_count=${COMPACTION_MAX_COUNT}, max_windows=$((COMPACTION_MAX_COUNT + 1))"
echo "PPO update passes  : actor=${PPO_EPOCHS}, critic=${CRITIC_PPO_EPOCHS}, critic_only=${NUM_CRITIC_ONLY_STEPS}"
echo "Learning rates     : actor=${ACTOR_LR}, critic=${CRITIC_LR}"
echo "Policy regularizer : kl_coef=${KL_COEF}, entropy_coef=${ENTROPY_COEF} (4B stability setting)"
echo "Token loss         : calculate_per_token_loss=1, custom_advantages=compaction_swe.advantages.compute_compaction_advantages"
echo "Log                : ${LOG_FILE}"

step "Preflight"
[[ -f "${HF_CHECKPOINT}/model.safetensors.index.json" ]] || fail "HF checkpoint is incomplete"
HF_CHECKPOINT="${HF_CHECKPOINT}" COMPACTION_MODEL_SEQUENCE_LIMIT="${COMPACTION_MODEL_SEQUENCE_LIMIT}" \
"${PYTHON_BIN}" - <<'PY'
import os

from compaction_swe.model_config import native_sequence_limit

native_limit = native_sequence_limit(os.path.join(os.environ["HF_CHECKPOINT"], "config.json"))
requested = int(os.environ["COMPACTION_MODEL_SEQUENCE_LIMIT"])
assert native_limit == requested, (native_limit, requested)
print(f"model_context=OK native={native_limit} requested={requested}")
PY
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
SWE_DEV_DUAL_DATA="${SWE_DEV_DUAL_DATA}" "${PYTHON_BIN}" - <<'PY'
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
    # Keep the exact 1000-instance range when switching only the sandbox
    # project from the dual manifest to secondary.
    reference_path = os.environ["SWE_DEV_DUAL_DATA"]
    with open(reference_path, encoding="utf-8") as reference_handle:
        reference_labels = {
            json.loads(line)["label"]
            for line in reference_handle
            if line.strip()
        }
    assert len(reference_labels) == expected_rows, len(reference_labels)
    assert labels == reference_labels, (
        "secondary-only label range differs from dual manifest: "
        f"only_secondary={sorted(labels - reference_labels)[:5]} "
        f"only_dual={sorted(reference_labels - labels)[:5]}"
    )
    print(f"secondary_range=OK labels={len(labels)} reference={reference_path}")
print(f"dataset_schema=OK unique_labels={len(labels)} projects={project_counts}")
PY

TRAIN_CP_SIZE="${TRAIN_CP_SIZE}" "${PYTHON_BIN}" - <<'PY'
import os

from compaction_swe.advantages import SUPPORTED_CONTEXT_PARALLEL_SIZES

cp_size = int(os.environ["TRAIN_CP_SIZE"])
assert cp_size in SUPPORTED_CONTEXT_PARALLEL_SIZES, (
    f"CompactionRL custom GAE does not support CP={cp_size}; "
    f"supported={sorted(SUPPORTED_CONTEXT_PARALLEL_SIZES)}"
)
print(f"compaction_gae_context_parallel=OK cp={cp_size}")
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
  local latest_iteration iteration
  latest_iteration="$(<"${tracker}")"
  [[ "${latest_iteration}" =~ ^[0-9]+$ ]] || fail "${label} tracker is not numeric"
  if [[ -n "${CKPT_STEP:-}" ]]; then
    [[ "${CKPT_STEP}" =~ ^[0-9]+$ ]] || fail "CKPT_STEP must be a non-negative integer"
    (( CKPT_STEP <= latest_iteration )) || \
      fail "CKPT_STEP=${CKPT_STEP} exceeds ${label} latest checkpoint ${latest_iteration}"
    iteration="${CKPT_STEP}"
  else
    iteration="${latest_iteration}"
  fi
  local iter_dir
  iter_dir="$(printf '%s/iter_%07d' "${checkpoint%/}" "${iteration}")"
  [[ -s "${iter_dir}/.metadata" ]] || fail "${label} metadata is missing: ${iter_dir}/.metadata"
  printf '%s' "${iteration}"
}

if [[ "${ACTION}" == resume-warmup-8gpu || "${ACTION}" == verify-resume20-8gpu ]]; then
  # During critic-only warmup the actor is intentionally unchanged and is not
  # checkpointed. Resume with the initial actor, the completed critic state,
  # and the exact rollout dataset cursor saved alongside the warmup run.
  unset LOAD_CHECKPOINT
  warmup_iteration="${WARMUP_RESUME_ITERATION:-19}"
  [[ "${warmup_iteration}" =~ ^[0-9]+$ ]] || fail "WARMUP_RESUME_ITERATION must be non-negative"
  warmup_actor_state_dir="${WARMUP_ACTOR_STATE_DIR:-${COMPACTION_CKPT_ROOT}/compactionrl_swe_dev_dual_qwen35_4b_ppo_actor_20260817_141608}"
  warmup_critic_checkpoint="${WARMUP_CRITIC_LOAD_CHECKPOINT:-${COMPACTION_CKPT_ROOT}/compactionrl_swe_dev_dual_qwen35_4b_ppo_critic_20260817_141608}"
  critic_iteration="$(validate_checkpoint "${warmup_critic_checkpoint}" critic)"
  [[ "${critic_iteration}" == "${warmup_iteration}" ]] || \
    fail "warmup critic iteration ${critic_iteration} != requested ${warmup_iteration}"
  (( warmup_iteration + 1 == NUM_CRITIC_ONLY_STEPS )) || \
    fail "warmup resume requires iteration+1 == NUM_CRITIC_ONLY_STEPS (${warmup_iteration}+1 != ${NUM_CRITIC_ONLY_STEPS})"
  (( NUM_ROLLOUT > warmup_iteration + 1 )) || \
    fail "NUM_ROLLOUT=${NUM_ROLLOUT} must be greater than next resume step $((warmup_iteration + 1))"
  dataset_state="${warmup_actor_state_dir%/}/rollout/global_dataset_state_dict_${warmup_iteration}.pt"
  [[ -s "${dataset_state}" ]] || fail "warmup dataset state is missing: ${dataset_state}"
  export START_ROLLOUT_ID="$((warmup_iteration + 1))"
  export ROLLOUT_DATASET_STATE_LOAD="${warmup_actor_state_dir%/}"
  export CRITIC_LOAD_CHECKPOINT="${warmup_critic_checkpoint%/}"
  export SAVE_DIR="${SAVE_DIR:-${COMPACTION_CKPT_ROOT}/compactionrl_${DATASET_TAG}_qwen35_4b_ppo_actor_warmup19_${RUN_TS}}"
  export CRITIC_SAVE_DIR="${CRITIC_SAVE_DIR:-${COMPACTION_CKPT_ROOT}/compactionrl_${DATASET_TAG}_qwen35_4b_ppo_critic_warmup19_${RUN_TS}}"
elif [[ -n "${LOAD_CHECKPOINT:-}" || -n "${CRITIC_LOAD_CHECKPOINT:-}" ]]; then
  [[ -n "${LOAD_CHECKPOINT:-}" && -n "${CRITIC_LOAD_CHECKPOINT:-}" ]] || \
    fail "PPO resume requires both LOAD_CHECKPOINT and CRITIC_LOAD_CHECKPOINT"
  actor_iteration="$(validate_checkpoint "${LOAD_CHECKPOINT}" actor)"
  critic_iteration="$(validate_checkpoint "${CRITIC_LOAD_CHECKPOINT}" critic)"
  [[ "${actor_iteration}" == "${critic_iteration}" ]] || \
    fail "actor iteration ${actor_iteration} != critic iteration ${critic_iteration}"
  export CKPT_STEP="${actor_iteration}"
  (( NUM_ROLLOUT > actor_iteration + 1 )) || \
    fail "NUM_ROLLOUT=${NUM_ROLLOUT} must be greater than next resume step $((actor_iteration + 1))"
  dataset_state="${LOAD_CHECKPOINT%/}/rollout/global_dataset_state_dict_${actor_iteration}.pt"
  [[ -s "${dataset_state}" ]] || fail "PPO source dataset state is missing: ${dataset_state}"
  # Loading from an old checkpoint and saving into the same directory is unsafe:
  # a different dataset or experiment can overwrite later iterations in that
  # lineage. Branch the run by default; callers may provide explicit fresh
  # SAVE_DIR/CRITIC_SAVE_DIR paths when they need a particular location.
  resume_output_tag="${RESUME_OUTPUT_TAG:-resume_from_${actor_iteration}_${RUN_TS}}"
  export SAVE_DIR="${SAVE_DIR:-${COMPACTION_CKPT_ROOT}/compactionrl_${DATASET_TAG}_qwen35_4b_ppo_actor_${resume_output_tag}}"
  export CRITIC_SAVE_DIR="${CRITIC_SAVE_DIR:-${COMPACTION_CKPT_ROOT}/compactionrl_${DATASET_TAG}_qwen35_4b_ppo_critic_${resume_output_tag}}"
  [[ "${SAVE_DIR%/}" != "${LOAD_CHECKPOINT%/}" ]] || fail "resume SAVE_DIR must be a fresh branch"
  [[ "${CRITIC_SAVE_DIR%/}" != "${CRITIC_LOAD_CHECKPOINT%/}" ]] || \
    fail "resume CRITIC_SAVE_DIR must be a fresh branch"
  [[ ! -e "${SAVE_DIR}" ]] || fail "resume actor output path already exists: ${SAVE_DIR}"
  [[ ! -e "${CRITIC_SAVE_DIR}" ]] || fail "resume critic output path already exists: ${CRITIC_SAVE_DIR}"
  export ROLLOUT_DATASET_STATE_LOAD="${ROLLOUT_DATASET_STATE_LOAD:-${LOAD_CHECKPOINT%/}}"
else
  checkpoint_kind=ppo
  [[ "${ACTION}" != smoke-8gpu ]] || checkpoint_kind=ppo_smoke
  export SAVE_DIR="${SAVE_DIR:-${COMPACTION_CKPT_ROOT}/compactionrl_${DATASET_TAG}_qwen35_4b_${checkpoint_kind}_actor_${RUN_TS}}"
  export CRITIC_SAVE_DIR="${CRITIC_SAVE_DIR:-${COMPACTION_CKPT_ROOT}/compactionrl_${DATASET_TAG}_qwen35_4b_${checkpoint_kind}_critic_${RUN_TS}}"
  [[ ! -e "${SAVE_DIR}" ]] || fail "fresh actor checkpoint path already exists: ${SAVE_DIR}"
  [[ ! -e "${CRITIC_SAVE_DIR}" ]] || fail "fresh critic checkpoint path already exists: ${CRITIC_SAVE_DIR}"
  export CRITIC_LOAD_CHECKPOINT="${REF_LOAD}"
fi

export RAY_GCS_PORT="${RAY_GCS_PORT:-6387}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8277}"
ray_started=1
cleanup() {
  if [[ "${ray_started}" == 1 && "${KEEP_RAY:-0}" != 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

step "Launch PPO (${ACTION})"
echo "rollouts=${NUM_ROLLOUT} prompts=${ROLLOUT_BATCH_SIZE} samples_per_prompt=${N_SAMPLES_PER_PROMPT} global_batch=${GLOBAL_BATCH_SIZE}"
if [[ "${ACTION}" == resume-warmup-8gpu || "${ACTION}" == verify-resume20-8gpu ]]; then
  echo "warmup_resume_checkpoint_iteration=${warmup_iteration} next_step=${START_ROLLOUT_ID}"
  echo "actor_load=initial:${REF_LOAD}"
  echo "critic_load=${CRITIC_LOAD_CHECKPOINT}"
  echo "dataset_state=${dataset_state}"
elif [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
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
if [[ "${ACTION}" == smoke-8gpu ]]; then
  [[ "${final_actor_iteration}" == 1 ]] || \
    fail "two-step smoke expected checkpoint iteration 1, got ${final_actor_iteration}"
fi
echo "paired_checkpoint_iteration=${final_actor_iteration}"
echo "READY: PPO ${ACTION} completed with separate actor and critic checkpoints."
echo "Log: ${LOG_FILE}"
