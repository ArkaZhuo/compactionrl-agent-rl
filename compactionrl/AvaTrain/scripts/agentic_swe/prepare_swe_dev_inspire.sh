#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-check}"

QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
DATASET="${SWE_DEV_SOURCE_DATA:-${QB_ROOT}/swe-dev/data/swe_dev_rft_public_1000_skip_pytmc_283.jsonl}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-$(bash "${ROOT_DIR}/scripts/agentic_swe/prepare_protocol_bundle.sh" path)}"
TEMPLATE_MANIFEST="${TEMPLATE_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_dev_1000_qwen_code_0.21.0.json}"
OUTPUT_DATA="${OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/swe_dev_1000_avatrain_qwen_code_0.21.0.jsonl}"
READY_OUTPUT_DATA="${READY_OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/swe_dev_ready_avatrain_qwen_code_0.21.0.jsonl}"
SMOKE_OUTPUT_DATA="${SMOKE_OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/smoke_swe_dev_1_avatrain_qwen_code_0.21.0.jsonl}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-swerebench}"
INSTANCE_ID="${INSTANCE_ID:-15five__scim2-filter-parser-20}"
BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-4}"
NAME_PREFIX="${NAME_PREFIX:-sywang-fzk-swedev-qc0210}"

SDK_PATH="${INSPIRE_SANDBOX_PYTHONPATH:-${QB_ROOT/\/inspire\/qb-ilm/\/inspire\/hdd}/.deps/inspire-sandbox}"
if [[ ! -d "${SDK_PATH}/inspire_sandbox" ]]; then
  SDK_PATH="/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox"
fi

LOG_DIR="${LOG_DIR:-${QB_ROOT}/logs/swe-rl/swe-dev-inspire}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${ACTION}_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "action=${ACTION}"
echo "dataset=${DATASET}"
echo "protocol_bundle=${PROTOCOL_BUNDLE}"
echo "template_manifest=${TEMPLATE_MANIFEST}"
echo "output_data=${OUTPUT_DATA}"
echo "image_namespace=${IMAGE_NAMESPACE}"
echo "instance_id=${INSTANCE_ID}"
echo "log=${LOG_FILE}"

[[ -f "${DATASET}" ]] || { echo "Required file does not exist: ${DATASET}" >&2; exit 1; }
bash "${ROOT_DIR}/scripts/agentic_swe/prepare_protocol_bundle.sh" check

require_credentials() {
  : "${SBX_API_KEY:?export SBX_API_KEY before using Inspire Sandbox}"
  : "${SBX_API_URL:?export SBX_API_URL before using Inspire Sandbox}"
}

template_tool() {
  PYTHONPATH="${SDK_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_inspire_templates.py" "$@" \
      --dataset "${DATASET}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${TEMPLATE_MANIFEST}" \
      --name-prefix "${NAME_PREFIX}" \
      --image-namespace "${IMAGE_NAMESPACE}"
}

case "${ACTION}" in
  status)
    python3 "${ROOT_DIR}/scripts/agentic_swe/swe_dev_inspire_status.py" \
      --source "${DATASET}" \
      --registry-manifest "${QB_ROOT}/swe-dev/data/swe_dev_registry_audit.json" \
      --template-manifest "${TEMPLATE_MANIFEST}" \
      --prewarm-progress "${QB_ROOT}/swe-dev/build/prewarm_public_1000_skip_pytmc_20260730/progress.json" \
      --build-root "${QB_ROOT}/swe-dev/build" \
      --public-concurrency "${BUILD_CONCURRENCY}"
    ;;
  check)
    template_tool validate
    if [[ -n "${SBX_API_KEY:-}" && -n "${SBX_API_URL:-}" ]]; then
      echo "sandbox_credentials=set"
    else
      echo "sandbox_credentials=unset"
    fi
    ;;
  build-one)
    require_credentials
    template_tool build --instance-id "${INSTANCE_ID}" --concurrency 1 --fail-fast
    ;;
  verify-one)
    require_credentials
    template_tool verify --instance-id "${INSTANCE_ID}"
    ;;
  build-all)
    require_credentials
    template_tool build --concurrency "${BUILD_CONCURRENCY}"
    ;;
  export-one)
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_swe_dev_data.py" \
      --source "${DATASET}" \
      --template-manifest "${TEMPLATE_MANIFEST}" \
      --output "${SMOKE_OUTPUT_DATA}" \
      --image-namespace "${IMAGE_NAMESPACE}" \
      --instance-id "${INSTANCE_ID}"
    ;;
  export-ready)
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_swe_dev_data.py" \
      --source "${DATASET}" \
      --template-manifest "${TEMPLATE_MANIFEST}" \
      --output "${READY_OUTPUT_DATA}" \
      --image-namespace "${IMAGE_NAMESPACE}" \
      --allow-missing-templates
    ;;
  export-all)
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_swe_dev_data.py" \
      --source "${DATASET}" \
      --template-manifest "${TEMPLATE_MANIFEST}" \
      --output "${OUTPUT_DATA}" \
      --image-namespace "${IMAGE_NAMESPACE}"
    ;;
  *)
    echo "usage: $0 {status|check|build-one|verify-one|build-all|export-one|export-ready|export-all}" >&2
    exit 2
    ;;
esac

echo "DONE action=${ACTION} log=${LOG_FILE}"
