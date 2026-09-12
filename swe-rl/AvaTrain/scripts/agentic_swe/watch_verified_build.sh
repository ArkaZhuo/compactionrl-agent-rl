#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
PID_FILE="${BUILD_PID_FILE:-${QB_ROOT}/swe-rl/templates/build_all.pid}"
MANIFEST="${TEMPLATE_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_verified_qwen_code_0.21.0.json}"
OUTPUT_DATA="${OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/swe_verified_500_avatrain_qwen_code_0.21.0.jsonl}"
SWEBENCH_PYTHONPATH="${SWEBENCH_PYTHONPATH:-/tmp/avatrain_swebench_py310}"
INTERVAL="${WATCH_INTERVAL:-30}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "Missing build PID file: ${PID_FILE}" >&2
  exit 1
fi
build_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
if [[ ! "${build_pid}" =~ ^[0-9]+$ ]]; then
  echo "Invalid build PID: ${build_pid}" >&2
  exit 1
fi

status_counts() {
  python3 - "${MANIFEST}" <<'PY'
import json
import sys
from collections import Counter

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
counts = Counter(entry.get("status", "missing") for entry in manifest.get("templates", {}).values())
print(
    f"ready={counts['ready']} building={counts['building']} "
    f"failed={counts['failed']} total={len(manifest.get('templates', {}))}"
)
PY
}

echo "Watching Verified template build pid=${build_pid} manifest=${MANIFEST}"
while kill -0 "${build_pid}" 2>/dev/null; do
  printf '%s ' "$(date -Is)"
  status_counts
  sleep "${INTERVAL}"
done

counts="$(status_counts)"
echo "$(date -Is) build process exited; ${counts}"
if [[ "${counts}" != "ready=500 building=0 failed=0 total=500" ]]; then
  echo "Template build did not finish cleanly; refusing to export incomplete data." >&2
  exit 1
fi
if [[ ! -d "${SWEBENCH_PYTHONPATH}/swebench" ]]; then
  echo "Missing SWE-bench Python runtime: ${SWEBENCH_PYTHONPATH}" >&2
  exit 1
fi

SWEBENCH_PYTHONPATH="${SWEBENCH_PYTHONPATH}" \
  bash "${ROOT_DIR}/scripts/agentic_swe/prepare_verified_inspire.sh" export-data

python3 - "${OUTPUT_DATA}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
assert len(rows) == 500, len(rows)
assert len({row["label"] for row in rows}) == 500
assert all(row["metadata"]["inspire_template"] for row in rows)
assert all(row["metadata"]["install_config"]["test_cmd"] for row in rows)
print(f"READY full Agentic SWE data rows=500 output={path}")
PY
