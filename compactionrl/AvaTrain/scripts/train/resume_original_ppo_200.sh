#!/usr/bin/env bash
# Resume the 2026-09-05 standard secondary PPO run, NOT CompactionRL.
# A fresh checkpoint view keeps run_ppo.sh's load==save contract without
# overwriting the source experiment. Only the selected iteration is symlinked;
# future saves are real directories in the new lineage. The 250-rollout
# wrapper reuses this implementation with PPO_RESUME_TOTAL_ROLLOUTS=250.
set -euo pipefail

ACTION="${1:-run}"
case "${ACTION}" in
  check|run) ;;
  *) echo "usage: bash $0 {check|run}" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AVA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export SHARED_ROOT=/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk
CHECKPOINT_ROOT="${SHARED_ROOT}/swe-rl/checkpoints"
SOURCE_ACTOR="${CHECKPOINT_ROOT}/swe_dev_secondary_qwen35_4b_ppo_actor_20260905_131341"
SOURCE_CRITIC="${CHECKPOINT_ROOT}/swe_dev_secondary_qwen35_4b_ppo_critic_20260905_131341"
fail() { echo "ERROR: $*" >&2; exit 1; }
PPO_RESUME_ITERATION="${PPO_RESUME_ITERATION:-159}"
case "${PPO_RESUME_ITERATION}" in
  99|159) ;;
  *) fail "PPO_RESUME_ITERATION must be 99 or 159; no automatic checkpoint fallback" ;;
esac
printf -v ITER_DIR 'iter_%07d' "${PPO_RESUME_ITERATION}"
NEXT_ROLLOUT=$((PPO_RESUME_ITERATION + 1))
DATASET_STATE="${SOURCE_ACTOR}/rollout/global_dataset_state_dict_${PPO_RESUME_ITERATION}.pt"
PPO_RESUME_TOTAL_ROLLOUTS="${PPO_RESUME_TOTAL_ROLLOUTS:-200}"
case "${PPO_RESUME_TOTAL_ROLLOUTS}" in
  200|250) ;;
  *) fail "PPO_RESUME_TOTAL_ROLLOUTS must be 200 or 250" ;;
esac
for source_dir in "${SOURCE_ACTOR}" "${SOURCE_CRITIC}"; do
  for filename in .metadata common.pt __0_0.distcp __1_0.distcp; do
    [[ -s "${source_dir}/${ITER_DIR}/${filename}" ]] || \
      fail "Missing checkpoint file: ${source_dir}/${ITER_DIR}/${filename}"
  done
done
[[ -s "${DATASET_STATE}" ]] || fail "Missing dataset cursor: ${DATASET_STATE}"

# Pin the historical model, dataset, optimizer and agent settings explicitly;
# do not inherit an unrelated GRPO/CompactionRL experiment's shell overrides.
unset LOAD_CHECKPOINT CRITIC_LOAD_CHECKPOINT SAVE_DIR CRITIC_SAVE_DIR
unset PYTHON_BIN VENV_DIR LOG_FILE LOG_ROOT RUN_TS RAY_ADDRESS
unset MILES_EXTRA_CHECKPOINT_STEPS
export RUNTIME_MODE=image
export MODEL_ROOT="${SHARED_ROOT}/model/Qwen"
export HF_CHECKPOINT="${MODEL_ROOT}/Qwen3.5-4B"
export REF_LOAD="${MODEL_ROOT}/Qwen3.5-4B_torch_dist"
export PROTOCOL_BUNDLE="${SHARED_ROOT}/swe-rl/protocol/avaeval-qwen-code-0.21.0-node-22.23.1-wstunnel-10.6.2"
export SWEBENCH_RUNTIME_DIR="${SHARED_ROOT}/.deps/swebench-runtime-py312"
export SANDBOX_CREDENTIAL_FILE="${SCRIPT_DIR}/.env"
export TRAIN_DATASET=swe-dev-secondary TRAIN_DATA_ROWS=1000
export TRAIN_DATA="${SHARED_ROOT}/swe-rl/data/swe_dev_1000_secondary_project_avatrain_qwen_code_0.21.0.jsonl"
export SWE_DEV_SECONDARY_DATA="${TRAIN_DATA}"
export CUDA_VISIBLE_DEVICES="${PPO_RESUME_GPU_IDS:-0,1,2,3}"
export NUM_ROLLOUT="${PPO_RESUME_TOTAL_ROLLOUTS}" SAVE_INTERVAL=20
export ROLLOUT_BATCH_SIZE=64 GLOBAL_BATCH_SIZE=64 N_SAMPLES_PER_PROMPT=1
export ROLLOUT_MAX_RESPONSE_LEN=16384 TRAIN_SEQUENCE_PARALLEL=1
export SGLANG_ROUTER_POLICY=cache_aware SGLANG_MEM_FRACTION=0.60
export SGLANG_SERVER_CONCURRENCY=12
export MAX_TOKENS_PER_GPU=1024 LOG_PROBS_CHUNK_SIZE=8
export TRAIN_MEMORY_MARGIN_BYTES=2147483648 OPTIMIZER_CPU_OFFLOAD_FRACTION=0.4
export PPO_EPOCHS=1 PPO_EPS_CLIP=0.2 PPO_VALUE_CLIP=0.2
export PPO_GAMMA=1.0 PPO_LAMBDA=1.0 NUM_CRITIC_ONLY_STEPS=1
export ACTOR_LR=1e-6 CRITIC_LR=1e-5
export AGENT_MAX_TURNS=12 AGENT_TIMEOUT_SEC=1200 AGENT_MAX_TOKENS_PER_TURN=8192
export SGLANG_ROUTER_REQUEST_TIMEOUT_SECS=240 SGLANG_ROUTER_MAX_ATTEMPTS=1
export AGENTIC_PROXY_HTTP_TIMEOUT_SEC=300 AGENTIC_PROXY_HTTP_MAX_ATTEMPTS=1
export QWEN_CODE_API_TIMEOUT_MS=360000 QWEN_CODE_MAX_RETRIES=0
export TUNNEL_READY_TIMEOUT_SEC=60 TUNNEL_READY_PROBE_TIMEOUT_SEC=5
export SWE_GRADER_TIMEOUT_SEC=600
export MILES_BALANCE_SANDBOX_PROJECTS=0 SANDBOX_CONCURRENCY_PRIMARY=0
export SANDBOX_CONCURRENCY_SECONDARY=64
export SANDBOX_CREATE_MAX_ATTEMPTS=120 SANDBOX_CREATE_RETRY_BASE_SEC=2
export SANDBOX_CREATE_RETRY_MAX_SEC=30
export COMPACTION_ROLLOUT_MAX_ATTEMPTS=256 COMPACTION_ROLLOUT_MAX_DISCARDED=128
export DYNAMIC_SAMPLING_FILTER_PATH=""
export ROUTER_CACHE_THRESHOLD=0.30 ROUTER_BALANCE_ABS_THRESHOLD=8
export ROUTER_BALANCE_REL_THRESHOLD=1.25
export CKPT_FULLY_PARALLEL_SAVE=0 CKPT_ASSUME_CONSTANT_STRUCTURE=1
export MCORE_DIST_CKPT_THREAD_COUNT=1 MCORE_DIST_CKPT_WRITE_ATTEMPTS=3
export RAY_NUM_CPUS=8 MILES_PLACEMENT_GROUP_TIMEOUT_SEC=300
export RAY_GCS_PORT=6385 RAY_DASHBOARD_PORT=8275 MASTER_ADDR=127.0.0.1

[[ -s "${TRAIN_DATA}" ]] || fail "Missing training dataset"
[[ "$(wc -l < "${TRAIN_DATA}")" -eq 1000 ]] || fail "Expected 1000 dataset rows"
available_bytes="$(df -B1 --output=avail "${CHECKPOINT_ROOT}" | awk 'NR==2 {print $1}')"
[[ "${available_bytes}" =~ ^[0-9]+$ ]] || fail "Cannot determine checkpoint free space"
# The selected iteration is already saved. Save every 20 completed rollouts, plus
# the final rollout even when the requested horizon is not divisible by 20.
checkpoint_pairs=$((NUM_ROLLOUT / SAVE_INTERVAL - NEXT_ROLLOUT / SAVE_INTERVAL))
if (( NUM_ROLLOUT % SAVE_INTERVAL != 0 )); then
  checkpoint_pairs=$((checkpoint_pairs + 1))
fi
required_gib=$((checkpoint_pairs * 134 + 100))
(( available_bytes >= required_gib * 1024 * 1024 * 1024 )) || \
  fail "Need at least ${required_gib} GiB free for ${checkpoint_pairs} new actor/critic pairs plus headroom"
echo "Static checkpoint files: present (full deserialization is checked during GPU startup)"
echo "Resume: actor=${PPO_RESUME_ITERATION} critic=${PPO_RESUME_ITERATION} dataset_cursor=${PPO_RESUME_ITERATION}; run rollouts ${NEXT_ROLLOUT}..$((NUM_ROLLOUT - 1))"
echo "Checkpoint plan: ${checkpoint_pairs} new pairs; required free space ${required_gib} GiB"
echo "Policy: Qwen3.5-4B, standard PPO, secondary-1000, 4 GPUs (2 actor + 2 critic)"
echo "Checkpoint free space: $((available_bytes / 1024 / 1024 / 1024)) GiB (quota not verified)"
if [[ "${ACTION}" == check ]]; then
  echo "Static check passed; no GPU, sandbox, output-directory or training action performed."
  exit 0
fi

command -v nvidia-smi >/dev/null || fail "Run inside the allocated Miles GPU image, not the CPU workspace"
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  fail "Existing Ray runtime detected; refusing to stop or share it"
fi

# Do not copy 134 GiB of model/optimizer state. Independent small metadata
# files plus an iteration-directory link are sufficient for Megatron loading.
# Keep the selected source iteration until this branch has a new complete pair.
RUN_ROOT="$(mktemp -d "${CHECKPOINT_ROOT}/swe_dev_secondary_ppo_resume${PPO_RESUME_ITERATION}_to${NUM_ROLLOUT}.XXXXXXXX")"
mkdir "${RUN_ROOT}/actor" "${RUN_ROOT}/critic" "${RUN_ROOT}/actor/rollout"
ln -s "${SOURCE_ACTOR}/${ITER_DIR}" "${RUN_ROOT}/actor/${ITER_DIR}"
ln -s "${SOURCE_CRITIC}/${ITER_DIR}" "${RUN_ROOT}/critic/${ITER_DIR}"
printf '%s\n' "${PPO_RESUME_ITERATION}" > "${RUN_ROOT}/actor/latest_checkpointed_iteration.txt"
printf '%s\n' "${PPO_RESUME_ITERATION}" > "${RUN_ROOT}/critic/latest_checkpointed_iteration.txt"
cp -- "${DATASET_STATE}" "${RUN_ROOT}/actor/rollout/global_dataset_state_dict_${PPO_RESUME_ITERATION}.pt"
export LOAD_CHECKPOINT="${RUN_ROOT}/actor" CRITIC_LOAD_CHECKPOINT="${RUN_ROOT}/critic"
export SAVE_DIR="${LOAD_CHECKPOINT}" CRITIC_SAVE_DIR="${CRITIC_LOAD_CHECKPOINT}"
export LOG_ROOT="${RUN_ROOT}/logs"
echo "NEW RUN ROOT: ${RUN_ROOT}"
echo "Source checkpoints are retained; new saves are isolated at ${RUN_ROOT}"
# Deliberately NOT resume-4gpu: that legacy action pins a different dual run.
exec bash "${SCRIPT_DIR}/run_ppo.sh" train-4gpu
