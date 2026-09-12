#!/usr/bin/env bash
# Start or resume SWE-RL PPO only after the local CompactionRL launcher has
# exited and all eight GPUs have remained idle for one hour.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
COMPACTION_ROOT="${COMPACTION_ROOT:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/compactionrl/AvaTrain}"
LOG_ROOT="${WATCHDOG_LOG_ROOT:-/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/ppo_watchdog/logs}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
IDLE_REQUIRED_SEC="${IDLE_REQUIRED_SEC:-3600}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_INITIAL_COMPACTION="${REQUIRE_INITIAL_COMPACTION:-1}"
PPO_ACTION="${PPO_ACTION:-resume-8gpu}"
PPO_RESUME_ACTOR_CHECKPOINT="${PPO_RESUME_ACTOR_CHECKPOINT:-/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/ppo/checkpoints/swe_dev_dual_qwen35_4b_ppo_actor_fresh_kl0p001_20260818_180622}"
PPO_RESUME_CRITIC_CHECKPOINT="${PPO_RESUME_CRITIC_CHECKPOINT:-/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/ppo/checkpoints/swe_dev_dual_qwen35_4b_ppo_critic_fresh_kl0p001_20260818_180622}"
PPO_RESUME_OPTIMIZER_CPU_OFFLOAD_FRACTION="${PPO_RESUME_OPTIMIZER_CPU_OFFLOAD_FRACTION:-0.4}"

mkdir -p "${LOG_ROOT}"
LOCK_FILE="${WATCHDOG_LOCK_FILE:-${LOG_ROOT}/watch_compaction_then_start_ppo.lock}"
command -v flock >/dev/null 2>&1 || {
  echo "flock is not available" >&2
  exit 1
}
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another PPO watchdog already holds ${LOCK_FILE}; exiting." >&2
  exit 1
fi

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${WATCHDOG_LOG_FILE:-${LOG_ROOT}/watch_compaction_then_start_ppo_${RUN_TS}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "[$(date -u -Is)] $*"; }
fail() { log "ERROR: $*" >&2; exit 1; }

[[ "${POLL_INTERVAL_SEC}" =~ ^[1-9][0-9]*$ ]] || fail "POLL_INTERVAL_SEC must be positive"
[[ "${IDLE_REQUIRED_SEC}" =~ ^[1-9][0-9]*$ ]] || fail "IDLE_REQUIRED_SEC must be positive"
[[ "${EXPECTED_GPUS}" =~ ^[1-9][0-9]*$ ]] || fail "EXPECTED_GPUS must be positive"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${REQUIRE_INITIAL_COMPACTION}" =~ ^[01]$ ]] || fail "REQUIRE_INITIAL_COMPACTION must be 0 or 1"
case "${PPO_ACTION}" in
  train-8gpu) ;;
  resume-8gpu)
    [[ "${PPO_RESUME_ACTOR_CHECKPOINT}" == /* ]] || \
      fail "resume-8gpu requires an absolute PPO_RESUME_ACTOR_CHECKPOINT"
    [[ "${PPO_RESUME_CRITIC_CHECKPOINT}" == /* ]] || \
      fail "resume-8gpu requires an absolute PPO_RESUME_CRITIC_CHECKPOINT"
    [[ "${PPO_RESUME_OPTIMIZER_CPU_OFFLOAD_FRACTION}" =~ ^(0\.[0-9]+|1(\.0+)?)$ ]] || \
      fail "resume-8gpu requires PPO_RESUME_OPTIMIZER_CPU_OFFLOAD_FRACTION in (0, 1]"
    ;;
  *) fail "PPO_ACTION must be train-8gpu or resume-8gpu" ;;
esac
command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1 || fail "nvidia-smi is not available: ${NVIDIA_SMI_BIN}"
[[ -f "${AVA_ROOT}/scripts/train/run_ppo.sh" ]] || fail "PPO launcher is missing"

compaction_running() {
  local proc pid cmdline cwd
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [[ "${pid}" != "$$" ]] || continue
    [[ -r "${proc}/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "${proc}/cmdline" 2>/dev/null || true)"
    [[ "${cmdline}" == *"scripts/train/run_compactionrl_ppo.sh"* ]] || continue
    cwd="$(readlink -f "${proc}/cwd" 2>/dev/null || true)"
    if [[ "${cwd}" == "${COMPACTION_ROOT}" || "${cwd}" == "${COMPACTION_ROOT}/"* || \
          "${cmdline}" == *"${COMPACTION_ROOT}/scripts/train/run_compactionrl_ppo.sh"* ]]; then
      printf '%s\n' "${pid}"
      return 0
    fi
  done
  return 1
}

read_gpu_utils() {
  local output line value
  GPU_UTILS=()
  if ! output="$("${NVIDIA_SMI_BIN}" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    return 1
  fi
  while IFS= read -r line; do
    value="${line//[[:space:]%]/}"
    [[ "${value}" =~ ^[0-9]+$ ]] || return 1
    GPU_UTILS+=("${value}")
  done <<< "${output}"
  [[ "${#GPU_UTILS[@]}" -eq "${EXPECTED_GPUS}" ]]
}

all_gpus_idle() {
  local value
  read_gpu_utils || return 1
  for value in "${GPU_UTILS[@]}"; do
    (( value == 0 )) || return 1
  done
}

log "Watchdog started: compaction_root=${COMPACTION_ROOT} expected_gpus=${EXPECTED_GPUS} idle_required=${IDLE_REQUIRED_SEC}s poll=${POLL_INTERVAL_SEC}s"
log "PPO target: action=${PPO_ACTION}, total_rollouts=200, save_interval=10, kl_coef=0.001"
if [[ "${PPO_ACTION}" == resume-8gpu ]]; then
  log "PPO resume pair: actor=${PPO_RESUME_ACTOR_CHECKPOINT} critic=${PPO_RESUME_CRITIC_CHECKPOINT}"
fi
if initial_compaction_pid="$(compaction_running)"; then
  log "Observed the current CompactionRL launcher (pid=${initial_compaction_pid}); waiting for it to exit"
elif [[ "${REQUIRE_INITIAL_COMPACTION}" == 1 ]]; then
  fail "no active CompactionRL launcher was found; start this watchdog on the GPU node while the current run is alive"
else
  log "No active CompactionRL launcher was found; initial-process guard was explicitly disabled"
fi

idle_since=0
while true; do
  now="$(date +%s)"
  if compaction_pid="$(compaction_running)"; then
    if (( idle_since != 0 )); then
      log "CompactionRL is active again (pid=${compaction_pid}); resetting idle timer"
    fi
    idle_since=0
    sleep "${POLL_INTERVAL_SEC}"
    continue
  fi

  if ! all_gpus_idle; then
    if (( idle_since != 0 )); then
      log "GPU activity, GPU-count mismatch, or query failure detected; resetting idle timer"
    fi
    idle_since=0
    sleep "${POLL_INTERVAL_SEC}"
    continue
  fi

  if (( idle_since == 0 )); then
    idle_since="${now}"
    log "CompactionRL is absent and all ${EXPECTED_GPUS} GPUs are at 0%; starting idle timer"
  else
    idle_elapsed=$((now - idle_since))
    log "Idle condition remains true: ${idle_elapsed}/${IDLE_REQUIRED_SEC}s"
    if (( idle_elapsed >= IDLE_REQUIRED_SEC )); then
      break
    fi
  fi
  sleep "${POLL_INTERVAL_SEC}"
done

# Recheck both gates immediately before touching Ray or launching PPO.
if compaction_pid="$(compaction_running)"; then
  fail "CompactionRL restarted during final check (pid=${compaction_pid}); refusing to launch PPO"
fi
all_gpus_idle || fail "GPUs are no longer all idle during final check; refusing to launch PPO"

PPO_RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
PPO_SAVE_ROOT=/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/ppo/checkpoints
mkdir -p "${PPO_SAVE_ROOT}"
PPO_ENV=(
  KL_COEF=0.001
  NUM_ROLLOUT=200
  SAVE_INTERVAL=10
  SGLANG_MEM_FRACTION=0.70
  SGLANG_SERVER_CONCURRENCY=32
  SANDBOX_CONCURRENCY_PRIMARY=32
  SANDBOX_CONCURRENCY_SECONDARY=32
  RAY_NUM_CPUS=16
  MCORE_DIST_CKPT_THREAD_COUNT=1
  MCORE_DIST_CKPT_WRITE_ATTEMPTS=3
  LOG_ROOT=/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/ppo/logs/swe_dev_dual-ppo-gpu
)
if [[ "${PPO_ACTION}" == train-8gpu ]]; then
  PPO_ENV+=(
    SAVE_DIR=${PPO_SAVE_ROOT}/swe_dev_dual_qwen35_4b_ppo_actor_fresh_kl0p001_${PPO_RUN_TS}
    CRITIC_SAVE_DIR=${PPO_SAVE_ROOT}/swe_dev_dual_qwen35_4b_ppo_critic_fresh_kl0p001_${PPO_RUN_TS}
  )
else
  PPO_ENV+=(
    PPO_RESUME_ACTOR_CHECKPOINT=${PPO_RESUME_ACTOR_CHECKPOINT}
    PPO_RESUME_CRITIC_CHECKPOINT=${PPO_RESUME_CRITIC_CHECKPOINT}
    PPO_RESUME_OPTIMIZER_CPU_OFFLOAD_FRACTION=${PPO_RESUME_OPTIMIZER_CPU_OFFLOAD_FRACTION}
  )
fi

log "Both gates passed. Preparing PPO action=${PPO_ACTION}."
if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'DRY-RUN: cd %q && ray stop --force; env' "${AVA_ROOT}"
  printf ' %q' "${PPO_ENV[@]}"
  printf ' bash scripts/train/run_ppo.sh %q\n' "${PPO_ACTION}"
  exit 0
fi

cd "${AVA_ROOT}"
ray stop --force || true
sleep 5
log "Launching PPO now"
env "${PPO_ENV[@]}" bash scripts/train/run_ppo.sh "${PPO_ACTION}"
