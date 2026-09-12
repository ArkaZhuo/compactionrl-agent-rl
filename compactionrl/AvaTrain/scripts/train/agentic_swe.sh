#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_FILE="$(realpath "${1:?usage: $0 TASK_FILE.jsonl}")"

: "${HF_CHECKPOINT:?set HF_CHECKPOINT}"
: "${REF_LOAD:?set REF_LOAD}"
: "${SBX_API_KEY:?set SBX_API_KEY}"
: "${SBX_API_URL:?set SBX_API_URL}"

command -v wstunnel >/dev/null || {
    echo "wstunnel is required; install it or set it on PATH" >&2
    exit 1
}

cd "${ROOT_DIR}"
exec uv run --locked --extra swe bash miles/examples/agentic_swe/run_qwen35_35b_a3b.sh "${TASK_FILE}"
