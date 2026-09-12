#!/usr/bin/env bash
# Fresh four-GPU, 200-step standard PPO baseline on the same secondary-1000
# SWE data as the current CompactionRL experiment. This changes rollout
# reliability/memory guards only; PPO/GAE and reward remain standard.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"

# This is a fresh, reproducible lineage. Do not inherit resume/save/data
# overrides exported by an earlier CompactionRL shell command.
unset LOAD_CHECKPOINT CRITIC_LOAD_CHECKPOINT SAVE_DIR CRITIC_SAVE_DIR
unset TRAIN_DATA TRAIN_DATASET RUN_TS LOG_FILE LOG_ROOT DYNAMIC_SAMPLING_FILTER_PATH
unset PYTHON_BIN VENV_DIR

export RUNTIME_MODE=image
export TRAIN_DATASET=swe-dev-secondary
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SWE_DEV_SECONDARY_DATA="${SHARED_ROOT}/swe-rl/data/swe_dev_1000_secondary_project_avatrain_qwen_code_0.21.0.jsonl"
export TRAIN_DATA_ROWS=1000
export NUM_ROLLOUT=200
export ROLLOUT_BATCH_SIZE=64
export GLOBAL_BATCH_SIZE=64
export N_SAMPLES_PER_PROMPT=1
export ROLLOUT_MAX_RESPONSE_LEN=16384
# Save both actor and critic after every 20 completed updates. Keeping all 10
# pairs through step 200 needs about 1.3 TiB, so older pairs may need pruning
# during this run unless additional shared-disk space becomes available.
export SAVE_INTERVAL=20
export SGLANG_ROUTER_POLICY=cache_aware
export SGLANG_MEM_FRACTION=0.60
export SGLANG_SERVER_CONCURRENCY=12
export MAX_TOKENS_PER_GPU=1024
export LOG_PROBS_CHUNK_SIZE=8
export TRAIN_MEMORY_MARGIN_BYTES=2147483648
export OPTIMIZER_CPU_OFFLOAD_FRACTION=0.4
# Preserve the established AvaTrain standard-PPO hyperparameters.
export PPO_EPOCHS=1
export PPO_EPS_CLIP=0.2
export PPO_VALUE_CLIP=0.2
export PPO_GAMMA=1.0
export PPO_LAMBDA=1.0
export NUM_CRITIC_ONLY_STEPS=1
export ACTOR_LR=1e-6
export CRITIC_LR=1e-5
# Deterministic request hierarchy: router 240s < proxy 300s < qwen-code 360s.
export SGLANG_ROUTER_REQUEST_TIMEOUT_SECS=240
export SGLANG_ROUTER_MAX_ATTEMPTS=1
export AGENTIC_PROXY_HTTP_TIMEOUT_SEC=300
export AGENTIC_PROXY_HTTP_MAX_ATTEMPTS=1
export QWEN_CODE_API_TIMEOUT_MS=360000
export QWEN_CODE_MAX_RETRIES=0
export AGENT_MAX_TURNS=12
export AGENT_TIMEOUT_SEC=1200
export AGENT_MAX_TOKENS_PER_TURN=8192
export TUNNEL_READY_TIMEOUT_SEC=60
export TUNNEL_READY_PROBE_TIMEOUT_SEC=5
export SANDBOX_CONCURRENCY_SECONDARY=64
# Bound infrastructure resampling without filtering any complete reward=0
# trajectory. The framework uses these legacy names for all rollout modes.
export COMPACTION_ROLLOUT_MAX_ATTEMPTS=256
export COMPACTION_ROLLOUT_MAX_DISCARDED=128

# Require enough room for the first checkpoint plus one additional pair as
# write headroom. This is intentionally not a full-run capacity assertion:
# the 20-step policy may require manual pruning before the shared disk fills.
available_bytes="$(df -B1 --output=avail "${SHARED_ROOT}" | awk 'NR==2 {gsub(/[[:space:]]/, "", $0); print $0}')"
required_bytes=$((300 * 1024 * 1024 * 1024))
[[ "${available_bytes}" =~ ^[0-9]+$ ]] || {
  echo "Unable to determine free checkpoint space under ${SHARED_ROOT}" >&2
  exit 1
}
(( available_bytes >= required_bytes )) || {
  echo "Insufficient checkpoint space: need at least 300 GiB, have $((available_bytes / 1024 / 1024 / 1024)) GiB" >&2
  exit 1
}
echo "Checkpoint disk preflight: $((available_bytes / 1024 / 1024 / 1024)) GiB free; 20-step saving needs ~1.3 TiB for all 10 pairs"
echo "WARNING: prune older actor+critic checkpoint pairs if shared-disk free space approaches 200 GiB"

exec bash "${SCRIPT_DIR}/run_ppo.sh" train-4gpu
