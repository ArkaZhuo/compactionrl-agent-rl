#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-check}"

QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
DATASET="${VERIFIED_SOURCE_DATA:-${QB_ROOT}/swe-bench/data/verified_all_500.jsonl}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-$(bash "${ROOT_DIR}/scripts/agentic_swe/prepare_protocol_bundle.sh" path)}"
TEMPLATE_MANIFEST="${TEMPLATE_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_verified_qwen_code_0.21.0.json}"
OUTPUT_DATA="${OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/swe_verified_500_avatrain_qwen_code_0.21.0.jsonl}"
INSTANCE_ID="${INSTANCE_ID:-astropy__astropy-12907}"
BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-4}"

SDK_PATH="${INSPIRE_SANDBOX_PYTHONPATH:-${QB_ROOT/\/inspire\/qb-ilm/\/inspire\/hdd}/.deps/inspire-sandbox}"
if [[ ! -d "${SDK_PATH}/inspire_sandbox" ]]; then
  SDK_PATH="/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox"
fi

LOG_DIR="${LOG_DIR:-${QB_ROOT}/logs/swe-rl/inspire-templates}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${ACTION}_$(date +%Y%m%d_%H%M%S).log}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "action=${ACTION}"
echo "dataset=${DATASET}"
echo "protocol_bundle=${PROTOCOL_BUNDLE}"
echo "template_manifest=${TEMPLATE_MANIFEST}"
echo "output_data=${OUTPUT_DATA}"
echo "instance_id=${INSTANCE_ID}"
echo "log=${LOG_FILE}"

require_file() {
  [[ -f "$1" ]] || {
    echo "Required file does not exist: $1" >&2
    exit 1
  }
}

require_sandbox_credentials() {
  : "${SBX_API_KEY:?export SBX_API_KEY before using Inspire Sandbox}"
  : "${SBX_API_URL:?export SBX_API_URL before using Inspire Sandbox}"
}

template_tool() {
  PYTHONPATH="${SDK_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_inspire_templates.py" "$@"
}

require_file "${DATASET}"
bash "${ROOT_DIR}/scripts/agentic_swe/prepare_protocol_bundle.sh" check

case "${ACTION}" in
  check)
    template_tool validate \
      --dataset "${DATASET}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${TEMPLATE_MANIFEST}"
    if [[ -n "${SBX_API_KEY:-}" && -n "${SBX_API_URL:-}" ]]; then
      echo "sandbox_credentials=set"
    else
      echo "sandbox_credentials=unset"
    fi
    ;;
  build-one)
    require_sandbox_credentials
    template_tool build \
      --dataset "${DATASET}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${TEMPLATE_MANIFEST}" \
      --instance-id "${INSTANCE_ID}" \
      --concurrency 1 \
      --fail-fast
    ;;
  verify-one)
    require_sandbox_credentials
    template_tool verify \
      --dataset "${DATASET}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${TEMPLATE_MANIFEST}" \
      --instance-id "${INSTANCE_ID}"
    ;;
  build-all)
    require_sandbox_credentials
    template_tool build \
      --dataset "${DATASET}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${TEMPLATE_MANIFEST}" \
      --concurrency "${BUILD_CONCURRENCY}"
    ;;
  export-data)
    require_file "${TEMPLATE_MANIFEST}"
    if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
      converter_python="${ROOT_DIR}/.venv/bin/python"
      converter_pythonpath=""
    elif [[ -n "${SWEBENCH_PYTHONPATH:-}" ]]; then
      converter_python="python3"
      converter_pythonpath="${SWEBENCH_PYTHONPATH}"
    else
      echo "A Python 3.12 AvaTrain environment or SWEBENCH_PYTHONPATH is required for export-data." >&2
      exit 1
    fi
    PYTHONPATH="${converter_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${converter_python}" "${ROOT_DIR}/miles/examples/agentic_swe/prepare_verified_data.py" \
        --source "${DATASET}" \
        --template-manifest "${TEMPLATE_MANIFEST}" \
        --output "${OUTPUT_DATA}"
    ;;
  *)
    echo "usage: $0 {check|build-one|verify-one|build-all|export-data}" >&2
    exit 2
    ;;
esac

echo "DONE action=${ACTION} log=${LOG_FILE}"

