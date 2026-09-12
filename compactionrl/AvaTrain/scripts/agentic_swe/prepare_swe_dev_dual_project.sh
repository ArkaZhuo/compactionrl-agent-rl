#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-status}"
QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
SOURCE="${SWE_DEV_SOURCE_DATA:-${QB_ROOT}/swe-dev/data/swe_dev_rft_public_1000_skip_pytmc_283.jsonl}"
PRIMARY_DATA="${PRIMARY_DATA:-${QB_ROOT}/swe-rl/data/swe_dev_1000_avatrain_qwen_code_0.21.0.jsonl}"
SECONDARY_SOURCE="${SECONDARY_SOURCE:-${QB_ROOT}/swe-dev/data/swe_dev_secondary_500.jsonl}"
SECONDARY_MANIFEST="${SECONDARY_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_dev_secondary_500_qwen_code_0.21.0.json}"
DUAL_DATA="${DUAL_DATA:-${QB_ROOT}/swe-rl/data/swe_dev_1000_dual_project_avatrain_qwen_code_0.21.0.jsonl}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-${QB_ROOT}/swe-rl/protocol/avaeval-qwen-code-0.21.0-node-22.23.1-wstunnel-10.6.2}"
SDK_ROOT="${INSPIRE_SANDBOX_PYTHONPATH:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox}"
SECONDARY_COUNT="${SECONDARY_COUNT:-500}"
BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-4}"
NAME_PREFIX="${SECONDARY_NAME_PREFIX:-sywang-fzk-swedev2-qc0210}"
PY_TOOL="${ROOT_DIR}/miles/examples/agentic_swe/prepare_swe_dev_dual_project.py"
REBUILD_TOOL="${ROOT_DIR}/miles/examples/agentic_swe/prepare_swe_dev_rebuilt_templates.py"
BUILD_ROOT="${SWE_DEV_BUILD_ROOT:-${QB_ROOT}/swe-dev/build}"
ENV_CONTEXT="${SWE_DEV_ENV_CONTEXT:-${BUILD_ROOT}/full_1001_buildkit/contexts/sweb.env.py.x86_64.bf6f45fb1552f3254153e6__latest}"
BASE_TEMPLATE="${SWE_DEV_SECONDARY_BASE_TEMPLATE:-sywang-fzk-swedev2-py39-qc0210-v1}"

partition_tool() {
  python3 "${PY_TOOL}" "$1" \
    --source "${SOURCE}" \
    --secondary-source "${SECONDARY_SOURCE}" \
    --secondary-manifest "${SECONDARY_MANIFEST}" \
    --primary-data "${PRIMARY_DATA}" \
    --output "${DUAL_DATA}" \
    --secondary-count "${SECONDARY_COUNT}"
}

template_tool() {
  : "${SBX_API_KEY_SECONDARY:?set SBX_API_KEY_SECONDARY in the current shell}"
  : "${SBX_API_URL_SECONDARY:=${SBX_API_URL:-https://qz-sbx-api.sii.edu.cn}}"
  SBX_API_KEY="${SBX_API_KEY_SECONDARY}" \
  SBX_API_URL="${SBX_API_URL_SECONDARY}" \
  PYTHONPATH="${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_inspire_templates.py" "$@" \
      --dataset "${SECONDARY_SOURCE}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${SECONDARY_MANIFEST}" \
      --name-prefix "${NAME_PREFIX}" \
      --image-namespace swerebench
}

rebuild_tool() {
  : "${SBX_API_KEY_SECONDARY:?set SBX_API_KEY_SECONDARY in the current shell}"
  : "${SBX_API_URL_SECONDARY:=${SBX_API_URL:-https://qz-sbx-api.sii.edu.cn}}"
  SBX_API_KEY="${SBX_API_KEY_SECONDARY}" \
  SBX_API_URL="${SBX_API_URL_SECONDARY}" \
  PYTHONPATH="${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 "${REBUILD_TOOL}" "$@" \
      --dataset "${SECONDARY_SOURCE}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${SECONDARY_MANIFEST}" \
      --shared-root "${QB_ROOT}" \
      --env-context "${ENV_CONTEXT}" \
      --build-root "${BUILD_ROOT}" \
      --base-template "${BASE_TEMPLATE}" \
      --name-prefix "${NAME_PREFIX}" \
      --image-namespace swerebench
}

manifest_counts() {
  python3 - "${SECONDARY_MANIFEST}" "${SECONDARY_COUNT}" <<'PY'
import collections
import json
import sys

path, selected = sys.argv[1], int(sys.argv[2])
try:
    manifest = json.load(open(path, encoding="utf-8"))
except FileNotFoundError:
    print(0, 0, 0, selected)
    raise SystemExit
counts = collections.Counter(
    entry.get("status", "missing")
    for entry in (manifest.get("templates") or {}).values()
)
known = counts["ready"] + counts["failed"] + counts["building"]
print(counts["ready"], counts["failed"], counts["building"], max(0, selected - known))
PY
}

shared_environment_ready() {
  python3 - "${SECONDARY_MANIFEST}" "${BASE_TEMPLATE}" <<'PY'
import json
import sys

try:
    manifest = json.load(open(sys.argv[1], encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit(1)
shared = manifest.get("shared_environment") or {}
raise SystemExit(0 if shared.get("status") == "ready" and shared.get("alias") == sys.argv[2] else 1)
PY
}

complete_build() {
  local ready failed building missing
  read -r ready failed building missing < <(manifest_counts)
  echo "SECONDARY_AUDIT ready=${ready} failed=${failed} building=${building} missing=${missing}"

  # The first pass uses the exact SWE image whenever the secondary project can
  # pull it.  Resume that pass if it was interrupted before all 500 entries
  # received a terminal state.
  if (( missing > 0 || building > 0 )); then
    template_tool build --concurrency "${BUILD_CONCURRENCY}"
    read -r ready failed building missing < <(manifest_counts)
    echo "SECONDARY_PUBLIC_PASS ready=${ready} failed=${failed} building=${building} missing=${missing}"
  fi

  # Some swerebench images are not readable from every Inspire project.  All
  # 500 local setup contexts have been prevalidated, so rebuild only non-ready
  # entries from the shared Python 3.9 base instead of retrying permanent 401s.
  if (( failed > 0 && missing == 0 && building == 0 )); then
    if ! shared_environment_ready; then
      rebuild_tool build-base
      rebuild_tool verify-base
    fi
    rebuild_tool build --concurrency "${BUILD_CONCURRENCY}"
  fi
}

case "${ACTION}" in
  partition)
    partition_tool partition
    ;;
  build-one)
    partition_tool partition
    first_id="$(python3 -c 'import json,sys; print(json.loads(next(open(sys.argv[1])))["metadata"]["remote_env_info"]["instance_id"])' "${SECONDARY_SOURCE}")"
    template_tool build --instance-id "${first_id}" --concurrency 1 --fail-fast
    ;;
  verify-one)
    first_id="$(python3 -c 'import json,sys; print(json.loads(next(open(sys.argv[1])))["metadata"]["remote_env_info"]["instance_id"])' "${SECONDARY_SOURCE}")"
    template_tool verify --instance-id "${first_id}"
    ;;
  build-all)
    partition_tool partition
    complete_build
    ;;
  merge)
    partition_tool merge
    ;;
  status)
    partition_tool status
    ;;
  *)
    echo "usage: $0 {partition|build-one|verify-one|build-all|merge|status}" >&2
    exit 2
    ;;
esac
