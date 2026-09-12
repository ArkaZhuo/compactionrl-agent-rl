#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
CREDENTIAL_FILE="${CREDENTIAL_FILE:-${ROOT_DIR}/scripts/train/.sandbox_credentials.sh}"
MANIFEST="${TEMPLATE_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_dev_1000_qwen_code_0.21.0.json}"
SOURCE="${SWE_DEV_SOURCE_DATA:-${QB_ROOT}/swe-dev/data/swe_dev_rft_public_1000_skip_pytmc_283.jsonl}"
PUBLIC_SOURCE="${SWE_DEV_PUBLIC_DATA:-${QB_ROOT}/swe-dev/data/swe_dev_rft_public_registry_available.jsonl}"
MISSING_SOURCE="${SWE_DEV_MISSING_DATA:-${QB_ROOT}/swe-dev/data/swe_dev_rft_public_registry_missing.jsonl}"
OUTPUT="${OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/swe_dev_1000_avatrain_qwen_code_0.21.0.jsonl}"
SMOKE_OUTPUT="${SMOKE_OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/smoke_swe_dev_1_avatrain_qwen_code_0.21.0.jsonl}"
BASE_PID_FILE="${BASE_PID_FILE:-${QB_ROOT}/logs/swe-rl/swe-dev-inspire-orchestrator/build-base.pid}"
PUBLIC_CONCURRENCY="${PUBLIC_CONCURRENCY:-8}"
REBUILD_CONCURRENCY="${REBUILD_CONCURRENCY:-4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RUNTIME_ATTEMPTS="${RUNTIME_ATTEMPTS:-120}"
RUNTIME_RETRY_DELAY="${RUNTIME_RETRY_DELAY:-30}"
SMOKE_INSTANCE="${SMOKE_INSTANCE:-asottile__tokenize-rt-3}"
SDK_PATH="${INSPIRE_SANDBOX_PYTHONPATH:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox}"
SWEBENCH_RUNTIME="${SWEBENCH_PYTHONPATH:-${QB_ROOT}/.deps/swebench-runtime-py310}"

for path in "${CREDENTIAL_FILE}" "${SOURCE}" "${PUBLIC_SOURCE}" "${MISSING_SOURCE}"; do
  [[ -f "${path}" ]] || { echo "ERROR missing required file: ${path}" >&2; exit 1; }
done

# The credential file is private and is sourced without printing its contents.
source "${CREDENTIAL_FILE}"
: "${SBX_API_KEY:?SBX_API_KEY is not set by the credential file}"
: "${SBX_API_URL:?SBX_API_URL is not set by the credential file}"

cd "${ROOT_DIR}"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

dataset_counts() {
  local dataset="$1"
  python3 - "${dataset}" "${MANIFEST}" <<'PY'
import json, sys
from collections import Counter

def source_metadata(row):
    metadata = row.get("metadata") or {}
    return metadata.get("remote_env_info") or metadata

with open(sys.argv[1], encoding="utf-8") as source:
    ids = {
        source_metadata(json.loads(line))["instance_id"]
        for line in source
        if line.strip()
    }
with open(sys.argv[2], encoding="utf-8") as source:
    templates = (json.load(source).get("templates") or {})
states = Counter((templates.get(instance_id) or {}).get("status", "unseen") for instance_id in ids)
print(len(ids), states["ready"], states["building"], states["failed"], states["unseen"])
PY
}

shared_ready() {
  python3 - "${MANIFEST}" <<'PY'
import json, sys
try:
    manifest = json.load(open(sys.argv[1], encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit(1)
shared = manifest.get("shared_environment") or {}
raise SystemExit(0 if shared.get("status") == "ready" else 1)
PY
}

run_public_batch() {
  SWE_DEV_SOURCE_DATA="${PUBLIC_SOURCE}" BUILD_CONCURRENCY="${PUBLIC_CONCURRENCY}" \
    bash scripts/agentic_swe/prepare_swe_dev_inspire.sh build-all
}

run_rebuild_batch() {
  SWE_DEV_MISSING_DATA="${SOURCE}" BUILD_CONCURRENCY="${REBUILD_CONCURRENCY}" \
    bash scripts/agentic_swe/prepare_swe_dev_rebuilt_inspire.sh build-all
}

retry_command() {
  local label="$1"
  local function_name="$2"
  local attempt
  for attempt in $(seq 1 "${RUNTIME_ATTEMPTS}"); do
    echo "[$(timestamp)] ${label} attempt=${attempt}/${RUNTIME_ATTEMPTS}"
    if "${function_name}"; then
      return 0
    fi
    echo "[$(timestamp)] WARN ${label} failed; retrying in ${RUNTIME_RETRY_DELAY}s"
    sleep "${RUNTIME_RETRY_DELAY}"
  done
  echo "ERROR ${label} failed after ${RUNTIME_ATTEMPTS} attempts" >&2
  return 1
}

verify_base() {
  bash scripts/agentic_swe/prepare_swe_dev_rebuilt_inspire.sh verify-base
}

build_smoke() {
  INSTANCE_ID="${SMOKE_INSTANCE}" SWE_DEV_MISSING_DATA="${MISSING_SOURCE}" \
    bash scripts/agentic_swe/prepare_swe_dev_rebuilt_inspire.sh build-one
}

verify_smoke() {
  INSTANCE_ID="${SMOKE_INSTANCE}" SWE_DEV_MISSING_DATA="${MISSING_SOURCE}" \
    bash scripts/agentic_swe/prepare_swe_dev_rebuilt_inspire.sh verify-one
}

export_smoke() {
  INSTANCE_ID="${SMOKE_INSTANCE}" SMOKE_OUTPUT_DATA="${SMOKE_OUTPUT}" \
    bash scripts/agentic_swe/prepare_swe_dev_inspire.sh export-one
}

grade_smoke() {
  PYTHONPATH="${ROOT_DIR}/miles/examples/agentic_swe:${SWEBENCH_RUNTIME}:${SDK_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 miles/examples/agentic_swe/verify_inspire_grader.py --data "${SMOKE_OUTPUT}"
}

retry_batch() {
  local label="$1"
  local expected="$2"
  local dataset="$3"
  local function_name="$4"
  local attempt total ready building failed unseen
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    read -r total ready building failed unseen <<<"$(dataset_counts "${dataset}")"
    echo "[$(timestamp)] ${label} before attempt=${attempt}: total=${total} ready=${ready} building=${building} failed=${failed} unseen=${unseen}"
    if [[ "${total}" == "${expected}" && "${ready}" == "${expected}" ]]; then
      return 0
    fi
    "${function_name}"
    read -r total ready building failed unseen <<<"$(dataset_counts "${dataset}")"
    echo "[$(timestamp)] ${label} after attempt=${attempt}: total=${total} ready=${ready} building=${building} failed=${failed} unseen=${unseen}"
    if [[ "${total}" == "${expected}" && "${ready}" == "${expected}" ]]; then
      return 0
    fi
    sleep 30
  done
  echo "ERROR ${label} remains incomplete after ${MAX_ATTEMPTS} attempts" >&2
  return 1
}

echo "[$(timestamp)] SWE-Dev Template supervisor started"
echo "public_concurrency=${PUBLIC_CONCURRENCY} rebuild_concurrency=${REBUILD_CONCURRENCY} max_attempts=${MAX_ATTEMPTS}"

if [[ -s "${BASE_PID_FILE}" ]]; then
  base_pid="$(cat "${BASE_PID_FILE}")"
  echo "[$(timestamp)] Waiting for shared base build pid=${base_pid}"
  while kill -0 "${base_pid}" 2>/dev/null; do sleep 20; done
fi

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  if shared_ready; then break; fi
  echo "[$(timestamp)] Building shared Python 3.9 base attempt=${attempt}/${MAX_ATTEMPTS}"
  bash scripts/agentic_swe/prepare_swe_dev_rebuilt_inspire.sh build-base
done
shared_ready || { echo "ERROR shared Python 3.9 base is not ready" >&2; exit 1; }

# Docker Hub access from the remote builder is not reliable. Use the validated
# local setup_repo.sh contexts for every remaining instance; ready entries are
# resumably skipped.
echo "[$(timestamp)] Building rebuilt smoke instance=${SMOKE_INSTANCE}"
retry_command build-smoke build_smoke
retry_batch rebuilt-all 1000 "${SOURCE}" run_rebuild_batch

echo "[$(timestamp)] Verifying shared Python 3.9 base"
retry_command verify-base verify_base

echo "[$(timestamp)] Verifying rebuilt smoke instance=${SMOKE_INSTANCE}"
retry_command verify-smoke verify_smoke
export_smoke

echo "[$(timestamp)] Running real SWE-Dev grader smoke instance=${SMOKE_INSTANCE}"
retry_command grader-smoke grade_smoke

read -r total ready building failed unseen <<<"$(dataset_counts "${SOURCE}")"
echo "[$(timestamp)] final manifest: total=${total} ready=${ready} building=${building} failed=${failed} unseen=${unseen}"
if [[ "${total} ${ready} ${building} ${failed} ${unseen}" != "1000 1000 0 0 0" ]]; then
  echo "ERROR final manifest gate failed" >&2
  exit 1
fi

bash scripts/agentic_swe/prepare_swe_dev_inspire.sh export-all
python3 - "${OUTPUT}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(rows) == 1000, len(rows)
assert len({row["label"] for row in rows}) == 1000
assert all(row["metadata"].get("inspire_template") for row in rows)
print(f"READY final Agentic SWE-Dev data rows={len(rows)} output={sys.argv[1]}")
PY

echo "[$(timestamp)] READY all 1000 SWE-Dev Templates built and final data exported"
