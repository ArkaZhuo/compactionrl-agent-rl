#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-check}"
QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"

DATASET="${SWE_DEV_MISSING_DATA:-${QB_ROOT}/swe-dev/data/swe_dev_rft_public_registry_missing.jsonl}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-$(bash "${ROOT_DIR}/scripts/agentic_swe/prepare_protocol_bundle.sh" path)}"
TEMPLATE_MANIFEST="${TEMPLATE_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_dev_1000_qwen_code_0.21.0.json}"
BUILD_ROOT="${SWE_DEV_BUILD_ROOT:-${QB_ROOT}/swe-dev/build}"
ENV_CONTEXT="${SWE_DEV_ENV_CONTEXT:-${BUILD_ROOT}/full_1001_buildkit/contexts/sweb.env.py.x86_64.bf6f45fb1552f3254153e6__latest}"
BASE_TEMPLATE="${SWE_DEV_BASE_TEMPLATE:-sywang-fzk-swedev-py39-qc0210-v1}"
INSTANCE_ID="${INSTANCE_ID:-asottile__tokenize-rt-3}"
BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-2}"

SDK_PATH="${INSPIRE_SANDBOX_PYTHONPATH:-${QB_ROOT/\/inspire\/qb-ilm/\/inspire\/hdd}/.deps/inspire-sandbox}"
if [[ ! -d "${SDK_PATH}/inspire_sandbox" ]]; then
  SDK_PATH="/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox"
fi

LOG_DIR="${LOG_DIR:-${QB_ROOT}/logs/swe-rl/swe-dev-inspire-rebuilt}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${ACTION}_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "action=${ACTION}"
echo "dataset=${DATASET}"
echo "template_manifest=${TEMPLATE_MANIFEST}"
echo "build_root=${BUILD_ROOT}"
echo "env_context=${ENV_CONTEXT}"
echo "base_template=${BASE_TEMPLATE}"
echo "instance_id=${INSTANCE_ID}"
echo "log=${LOG_FILE}"

[[ -f "${DATASET}" ]] || { echo "Required file does not exist: ${DATASET}" >&2; exit 1; }
[[ -f "${ENV_CONTEXT}/setup_env.sh" ]] || { echo "Missing shared setup_env.sh" >&2; exit 1; }
bash "${ROOT_DIR}/scripts/agentic_swe/prepare_protocol_bundle.sh" check

require_credentials() {
  : "${SBX_API_KEY:?export SBX_API_KEY before using Inspire Sandbox}"
  : "${SBX_API_URL:?export SBX_API_URL before using Inspire Sandbox}"
}

rebuild_tool() {
  PYTHONPATH="${SDK_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_swe_dev_rebuilt_templates.py" "$@" \
      --dataset "${DATASET}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${TEMPLATE_MANIFEST}" \
      --shared-root "${QB_ROOT}" \
      --env-context "${ENV_CONTEXT}" \
      --build-root "${BUILD_ROOT}" \
      --base-template "${BASE_TEMPLATE}"
}

case "${ACTION}" in
  check)
    rebuild_tool validate
    ;;
  build-base)
    require_credentials
    rebuild_tool build-base
    ;;
  verify-base)
    require_credentials
    rebuild_tool verify-base
    ;;
  build-one)
    require_credentials
    rebuild_tool build --instance-id "${INSTANCE_ID}" --concurrency 1 --fail-fast
    ;;
  verify-one)
    require_credentials
    rebuild_tool verify --instance-id "${INSTANCE_ID}"
    ;;
  build-all)
    require_credentials
    rebuild_tool build --concurrency "${BUILD_CONCURRENCY}"
    ;;
  *)
    echo "usage: $0 {check|build-base|verify-base|build-one|verify-one|build-all}" >&2
    exit 2
    ;;
esac

echo "DONE action=${ACTION} log=${LOG_FILE}"
