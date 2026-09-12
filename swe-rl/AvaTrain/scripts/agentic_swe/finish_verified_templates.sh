#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
MANIFEST="${TEMPLATE_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_verified_qwen_code_0.21.0.json}"
OUTPUT_DATA="${OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/swe_verified_500_avatrain_qwen_code_0.21.0.jsonl}"
SWEBENCH_PYTHONPATH="${SWEBENCH_PYTHONPATH:-/tmp/avatrain_swebench_py310}"
current_pid="${1:?usage: $0 CURRENT_BUILD_PID}"

: "${SBX_API_KEY:?SBX_API_KEY must be inherited in memory}"
: "${SBX_API_URL:?SBX_API_URL must be inherited in memory}"

counts() {
  python3 - "${MANIFEST}" <<'PY'
import json, sys
from collections import Counter
d=json.load(open(sys.argv[1], encoding="utf-8"))
c=Counter(v.get("status", "missing") for v in d.get("templates", {}).values())
print(c["ready"], c["building"], c["failed"], len(d.get("templates", {})))
PY
}

echo "Waiting for active retry pid=${current_pid}"
while kill -0 "${current_pid}" 2>/dev/null; do sleep 20; done
read -r ready building failed total <<<"$(counts)"
echo "Active retry exited: ready=${ready} building=${building} failed=${failed} total=${total}"

# The existing watcher handles export when the active retry reaches 500/500.
if [[ "${ready} ${building} ${failed} ${total}" == "500 0 0 500" ]]; then
  echo "READY active retry completed all templates; primary watcher will export data"
  exit 0
fi

for attempt in 1 2 3; do
  log="${QB_ROOT}/logs/swe-rl/inspire-templates/automatic-retry-${attempt}_$(date +%Y%m%d_%H%M%S).log"
  echo "Starting automatic retry ${attempt}/3 log=${log}"
  BUILD_CONCURRENCY=8 LOG_FILE="${log}" \
    bash "${ROOT_DIR}/scripts/agentic_swe/prepare_verified_inspire.sh" build-all
  read -r ready building failed total <<<"$(counts)"
  echo "Retry ${attempt}/3 result: ready=${ready} building=${building} failed=${failed} total=${total}"
  if [[ "${ready} ${building} ${failed} ${total}" == "500 0 0 500" ]]; then
    break
  fi
  sleep 30
done

if [[ "${ready} ${building} ${failed} ${total}" != "500 0 0 500" ]]; then
  echo "ERROR templates remain incomplete after automatic retries" >&2
  exit 1
fi

SWEBENCH_PYTHONPATH="${SWEBENCH_PYTHONPATH}" \
  bash "${ROOT_DIR}/scripts/agentic_swe/prepare_verified_inspire.sh" export-data
python3 - "${OUTPUT_DATA}" <<'PY'
import json, sys
rows=[json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(rows) == 500
assert len({row["label"] for row in rows}) == 500
print(f"READY full Agentic SWE data rows={len(rows)} output={sys.argv[1]}")
PY
