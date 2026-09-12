#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
LOG_DIR="${LOG_DIR:-${QB_ROOT}/logs/swe-rl/swe-dev-secondary-inspire}"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/build_secondary_500_${RUN_TS}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/build_secondary_500.pid}"
BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-4}"
DRIVER="${ROOT_DIR}/scripts/agentic_swe/prepare_swe_dev_dual_project.sh"

mkdir -p "${LOG_DIR}"
if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "ERROR: secondary Template build is already running as PID ${old_pid}" >&2
    exit 1
  fi
fi

if [[ -z "${SBX_API_KEY_SECONDARY:-}" ]]; then
  read -rsp "Secondary SBX API key: " SBX_API_KEY_SECONDARY
  echo
  export SBX_API_KEY_SECONDARY
fi
: "${SBX_API_KEY_SECONDARY:?secondary API key must not be empty}"
export SBX_API_URL_SECONDARY="${SBX_API_URL_SECONDARY:-https://qz-sbx-api.sii.edu.cn}"

{
  echo "[$(date -u -Is)] Preparing deterministic secondary 500-instance partition"
  bash "${DRIVER}" partition
  echo "[$(date -u -Is)] Building one secondary Template as a gate"
  bash "${DRIVER}" build-one
  echo "[$(date -u -Is)] Creating a real Sandbox from the gate Template"
  bash "${DRIVER}" verify-one
  echo "[$(date -u -Is)] Gate passed; launching remaining secondary Templates"
} 2>&1 | tee -a "${LOG_FILE}"

BUILD_CONCURRENCY="${BUILD_CONCURRENCY}" \
  nohup bash "${DRIVER}" build-all >>"${LOG_FILE}" 2>&1 &
build_pid=$!
echo "${build_pid}" >"${PID_FILE}"
unset SBX_API_KEY_SECONDARY

echo "STARTED secondary Template build PID=${build_pid} concurrency=${BUILD_CONCURRENCY}"
echo "Log: ${LOG_FILE}"
echo "Status: bash ${DRIVER} status"
echo "Follow: tail -F ${LOG_FILE}"
