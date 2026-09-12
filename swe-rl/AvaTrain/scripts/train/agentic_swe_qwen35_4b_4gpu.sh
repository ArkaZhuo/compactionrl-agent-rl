#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_FILE="$(realpath "${1:?usage: $0 TASK_FILE.jsonl}")"
MODEL_ROOT="${MODEL_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/model/Qwen}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/Qwen3.5-4B_torch_dist}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/protocol/avaeval-qwen-code-0.21.0-node-22.23.1-wstunnel-10.6.2}"
export HF_CHECKPOINT REF_LOAD

: "${HF_CHECKPOINT:?set HF_CHECKPOINT}"
: "${REF_LOAD:?set REF_LOAD}"
: "${SBX_API_KEY:?set SBX_API_KEY}"
: "${SBX_API_URL:?set SBX_API_URL}"

if [[ ! -x "${PROTOCOL_BUNDLE}/linux/bin/wstunnel" ]]; then
    echo "Pinned host wstunnel is missing: ${PROTOCOL_BUNDLE}/linux/bin/wstunnel" >&2
    exit 1
fi
export PATH="${PROTOCOL_BUNDLE}/linux/bin:${PATH}"

command -v wstunnel >/dev/null || {
    echo "wstunnel is required; install it or set it on PATH" >&2
    exit 1
}

cd "${ROOT_DIR}"
exec uv run --locked --extra swe bash miles/examples/agentic_swe/run_qwen35_4b_4gpu.sh "${TASK_FILE}"
