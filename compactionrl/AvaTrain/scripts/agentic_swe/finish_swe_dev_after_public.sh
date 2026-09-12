#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PUBLIC_BUILD_PID="${PUBLIC_BUILD_PID:-1624021}"
MANIFEST="${SWE_DEV_TEMPLATE_MANIFEST:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/templates/swe_dev_1000_qwen_code_0.21.0.json}"
FINAL_DATA="${SWE_DEV_FINAL_DATA:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/data/swe_dev_1000_avatrain_qwen_code_0.21.0.jsonl}"

echo "Waiting for public template build PID ${PUBLIC_BUILD_PID}"
while kill -0 "${PUBLIC_BUILD_PID}" 2>/dev/null; do
  sleep 30
done

cd "${AVA_ROOT}"
source scripts/train/.sandbox_credentials.sh

echo "Public build exited; rebuilding Project-MONAI__MONAI-640"
INSTANCE_ID=Project-MONAI__MONAI-640 \
  bash scripts/agentic_swe/prepare_swe_dev_rebuilt_inspire.sh build-one

python3 - "${MANIFEST}" <<'PY'
import collections
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
counts = collections.Counter(entry.get("status", "unknown") for entry in manifest["templates"].values())
print(f"Template status before export: {dict(counts)}")
if counts != {"ready": 1000}:
    raise SystemExit("Refusing export until all 1000 templates are ready")
PY

bash scripts/agentic_swe/prepare_swe_dev_inspire.sh export-all
test -s "${FINAL_DATA}"
echo "READY final_swe_dev_data=${FINAL_DATA}"
