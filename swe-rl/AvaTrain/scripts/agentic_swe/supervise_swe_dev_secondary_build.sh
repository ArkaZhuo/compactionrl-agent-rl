#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
DRIVER="${ROOT_DIR}/scripts/agentic_swe/prepare_swe_dev_dual_project.sh"
MANIFEST="${SECONDARY_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_dev_secondary_500_qwen_code_0.21.0.json}"
BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-4}"
MAX_RETRY_ROUNDS="${MAX_RETRY_ROUNDS:-5}"
INITIAL_BUILD_PID="${INITIAL_BUILD_PID:-}"

: "${SBX_API_KEY_SECONDARY:?SBX_API_KEY_SECONDARY is required}"
export SBX_API_URL_SECONDARY="${SBX_API_URL_SECONDARY:-https://qz-sbx-api.sii.edu.cn}"

if [[ -n "${INITIAL_BUILD_PID}" ]]; then
  echo "[$(date -u -Is)] Waiting for initial build PID ${INITIAL_BUILD_PID}"
  while kill -0 "${INITIAL_BUILD_PID}" 2>/dev/null; do
    sleep 30
  done
fi

manifest_counts() {
  python3 - "${MANIFEST}" <<'PY'
import collections
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
counts = collections.Counter(entry.get("status", "unknown") for entry in manifest.get("templates", {}).values())
print(counts.get("ready", 0), counts.get("failed", 0), counts.get("building", 0))
PY
}

for ((round=1; round<=MAX_RETRY_ROUNDS; round++)); do
  read -r ready failed building < <(manifest_counts)
  echo "[$(date -u -Is)] Retry audit round=${round} ready=${ready} failed=${failed} building=${building}"
  if [[ "${ready}" -eq 500 ]]; then
    break
  fi
  BUILD_CONCURRENCY="${BUILD_CONCURRENCY}" bash "${DRIVER}" build-all
  sleep 15
done

read -r ready failed building < <(manifest_counts)
if [[ "${ready}" -ne 500 || "${failed}" -ne 0 || "${building}" -ne 0 ]]; then
  echo "ERROR: secondary build incomplete after ${MAX_RETRY_ROUNDS} retries: ready=${ready} failed=${failed} building=${building}" >&2
  exit 1
fi

bash "${DRIVER}" merge
echo "READY secondary Templates complete and dual-project training data exported"
