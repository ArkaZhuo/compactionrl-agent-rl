#!/usr/bin/env bash
set -euo pipefail

# Build a complete 1000-instance template set in the secondary Inspire
# project. The old secondary-500 manifest is used only as a resumable seed.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
QB_ROOT="${QB_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
SOURCE="${SWE_DEV_SOURCE_DATA:-${QB_ROOT}/swe-dev/data/swe_dev_rft_public_1000_skip_pytmc_283.jsonl}"
DUAL_DATA="${DUAL_DATA:-${QB_ROOT}/swe-rl/data/swe_dev_1000_dual_project_avatrain_qwen_code_0.21.0.jsonl}"
OUTPUT_DATA="${OUTPUT_DATA:-${QB_ROOT}/swe-rl/data/swe_dev_1000_secondary_project_avatrain_qwen_code_0.21.0.jsonl}"
SEED_MANIFEST="${SEED_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_dev_secondary_500_qwen_code_0.21.0.json}"
MANIFEST="${TEMPLATE_MANIFEST:-${QB_ROOT}/swe-rl/templates/swe_dev_secondary_1000_qwen_code_0.21.0.json}"
PROTOCOL_BUNDLE="${PROTOCOL_BUNDLE:-${QB_ROOT}/swe-rl/protocol/avaeval-qwen-code-0.21.0-node-22.23.1-wstunnel-10.6.2}"
BUILD_ROOT="${SWE_DEV_BUILD_ROOT:-${QB_ROOT}/swe-dev/build}"
ENV_CONTEXT="${SWE_DEV_ENV_CONTEXT:-${BUILD_ROOT}/full_1001_buildkit/contexts/sweb.env.py.x86_64.bf6f45fb1552f3254153e6__latest}"
BASE_TEMPLATE="${SWE_DEV_SECONDARY_BASE_TEMPLATE:-sywang-fzk-swedev2-py39-qc0210-v1}"
NAME_PREFIX="${SECONDARY_NAME_PREFIX:-sywang-fzk-swedev2-qc0210}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-swerebench}"
PUBLIC_CONCURRENCY="${PUBLIC_CONCURRENCY:-4}"
REBUILD_CONCURRENCY="${REBUILD_CONCURRENCY:-4}"
PUBLIC_ROUNDS="${PUBLIC_ROUNDS:-3}"
REBUILD_ROUNDS="${REBUILD_ROUNDS:-3}"
SDK_PATH="${INSPIRE_SANDBOX_PYTHONPATH:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox}"
LOG_DIR="${LOG_DIR:-${QB_ROOT}/logs/swe-rl/swe-dev-secondary-inspire}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/build_secondary_1000_$(date -u +%Y%m%d_%H%M%S).log}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

: "${SBX_API_KEY_SECONDARY:?set SBX_API_KEY_SECONDARY to the secondary project API key}"
export SBX_API_URL_SECONDARY="${SBX_API_URL_SECONDARY:-https://qz-sbx-api.sii.edu.cn}"

for path in "${SOURCE}" "${DUAL_DATA}" "${SEED_MANIFEST}" "${PROTOCOL_BUNDLE}/manifest.json"; do
  [[ -f "${path}" ]] || { echo "ERROR missing required file: ${path}" >&2; exit 1; }
done
[[ -d "${ENV_CONTEXT}" ]] || { echo "ERROR missing build context root: ${ENV_CONTEXT}" >&2; exit 1; }
[[ -d "${SDK_PATH}/inspire_sandbox" ]] || { echo "ERROR missing Inspire SDK: ${SDK_PATH}" >&2; exit 1; }

if [[ ! -e "${MANIFEST}" ]]; then
  cp -- "${SEED_MANIFEST}" "${MANIFEST}"
  chmod 600 "${MANIFEST}"
  echo "SEEDED manifest=${MANIFEST} from=${SEED_MANIFEST}"
fi

export PYTHONPATH="${SDK_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

source_count() {
  python3 - "${SOURCE}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
ids = [(row.get("metadata") or {}).get("remote_env_info", {}).get("instance_id") for row in rows]
if len(rows) != 1000 or len(set(ids)) != 1000 or any(not item for item in ids):
    raise SystemExit(f"source must contain 1000 unique remote_env_info.instance_id rows; got rows={len(rows)} ids={len(set(ids))}")
print(len(rows))
PY
}

status_counts() {
  python3 - "${SOURCE}" "${MANIFEST}" <<'PY'
import collections, json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
ids = [(row.get("metadata") or {}).get("remote_env_info", {}).get("instance_id") for row in rows]
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
templates = manifest.get("templates") or {}
states = collections.Counter((templates.get(item) or {}).get("status", "missing") for item in ids)
print("1000 %d %d %d %d" % (states["ready"], states["building"], states["failed"], states["missing"]))
PY
}

print_status() {
  local total ready building failed missing
  read -r total ready building failed missing <<<"$(status_counts)"
  echo "STATUS total=${total} ready=${ready} building=${building} failed=${failed} missing=${missing}"
}

public_build() {
  SBX_API_KEY="${SBX_API_KEY_SECONDARY}" SBX_API_URL="${SBX_API_URL_SECONDARY}" \
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_inspire_templates.py" build \
      --dataset "${SOURCE}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${MANIFEST}" \
      --name-prefix "${NAME_PREFIX}" \
      --image-namespace "${IMAGE_NAMESPACE}" \
      --concurrency "${PUBLIC_CONCURRENCY}"
}

rebuild_build() {
  SBX_API_KEY="${SBX_API_KEY_SECONDARY}" SBX_API_URL="${SBX_API_URL_SECONDARY}" \
    python3 "${ROOT_DIR}/miles/examples/agentic_swe/prepare_swe_dev_rebuilt_templates.py" build \
      --dataset "${SOURCE}" \
      --protocol-bundle "${PROTOCOL_BUNDLE}" \
      --manifest "${MANIFEST}" \
      --shared-root "${QB_ROOT}" \
      --env-context "${ENV_CONTEXT}" \
      --build-root "${BUILD_ROOT}" \
      --base-template "${BASE_TEMPLATE}" \
      --name-prefix "${NAME_PREFIX}" \
      --image-namespace "${IMAGE_NAMESPACE}" \
      --concurrency "${REBUILD_CONCURRENCY}"
}

export_secondary_data() {
  python3 - "${DUAL_DATA}" "${MANIFEST}" "${OUTPUT_DATA}" <<'PY'
import json, os, sys
source, manifest_path, output = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
templates = manifest.get("templates") or {}
rows = [json.loads(line) for line in open(source, encoding="utf-8") if line.strip()]
if len(rows) != 1000 or len({row.get("label") for row in rows}) != 1000:
    raise SystemExit("dual data must contain 1000 unique rows")
converted = []
for row in rows:
    label = row.get("label")
    entry = templates.get(label) or {}
    if entry.get("status") != "ready" or not entry.get("alias"):
        raise SystemExit(f"template is not ready for {label}: {entry}")
    metadata = dict(row.get("metadata") or {})
    metadata["inspire_template"] = entry["alias"]
    metadata["sandbox_project"] = "secondary"
    converted.append({**row, "metadata": metadata})
tmp = output + ".tmp"
os.makedirs(os.path.dirname(output), exist_ok=True)
with open(tmp, "w", encoding="utf-8") as handle:
    for row in converted:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
os.replace(tmp, output)
print(f"READY secondary_data rows={len(converted)} output={output}")
PY
}

source_count >/dev/null
echo "START secondary-template-build source=1000 manifest=${MANIFEST} log=${LOG_FILE}"
echo "public_concurrency=${PUBLIC_CONCURRENCY} rebuild_concurrency=${REBUILD_CONCURRENCY} base_template=${BASE_TEMPLATE}"
print_status

for round in $(seq 1 "${PUBLIC_ROUNDS}"); do
  read -r _ ready _ failed missing <<<"$(status_counts)"
  [[ "${ready}" == "1000" ]] && break
  echo "PUBLIC_PASS round=${round}/${PUBLIC_ROUNDS} ready=${ready} failed=${failed} missing=${missing}"
  public_build || echo "WARN public build pass returned non-zero; continuing to inspect manifest"
  print_status
done

read -r _ ready _ failed missing <<<"$(status_counts)"
if [[ "${ready}" != "1000" ]]; then
  echo "REBUILD_REQUIRED ready=${ready} failed=${failed} missing=${missing}"
  for round in $(seq 1 "${REBUILD_ROUNDS}"); do
    read -r _ ready _ failed missing <<<"$(status_counts)"
    [[ "${ready}" == "1000" ]] && break
    echo "REBUILD_PASS round=${round}/${REBUILD_ROUNDS} ready=${ready} failed=${failed} missing=${missing}"
    rebuild_build || echo "WARN rebuild pass returned non-zero; continuing to inspect manifest"
    print_status
  done
fi

read -r _ ready building failed missing <<<"$(status_counts)"
if [[ "${ready} ${building} ${failed} ${missing}" != "1000 0 0 0" ]]; then
  echo "ERROR secondary template gate failed: ready=${ready} building=${building} failed=${failed} missing=${missing}" >&2
  exit 1
fi
export_secondary_data
echo "READY secondary templates=1000 data=${OUTPUT_DATA} manifest=${MANIFEST}"
