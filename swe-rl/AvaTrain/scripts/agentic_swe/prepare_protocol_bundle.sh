#!/usr/bin/env bash
set -euo pipefail

# Reproducible host/sandbox bundle for the Agentic SWE qwen-code route.
# Versions are pinned to releases available when the original example landed.

QWEN_CODE_VERSION="0.21.0"
NODE_VERSION="22.23.1"
WSTUNNEL_VERSION="10.6.2"

QWEN_CODE_SHA512="878b7c72b1f5593292e08deea25390193cef1aeee25bd0eea887a06aafaff324574f24e4355ee4ef80004473d6621e4324bd5963f6420ec3ae3ec36168b967a2"
NODE_SHA256="9749e988f437343b7fa832c69ded82a312e41a03116d766797ac14f6f9eee578"
WSTUNNEL_SHA256="db6064cca0515b67f8652e201cff8e27553b8cbb7216b2e19241311e34868e6e"

DEFAULT_OUTPUT_ROOT="/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/protocol"
BUNDLE_NAME="avaeval-qwen-code-${QWEN_CODE_VERSION}-node-${NODE_VERSION}-wstunnel-${WSTUNNEL_VERSION}"
OUTPUT_ROOT="${PROTOCOL_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${BUNDLE_NAME}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7892}"

QWEN_URL="https://registry.npmjs.org/@qwen-code/qwen-code/-/qwen-code-${QWEN_CODE_VERSION}.tgz"
NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
WSTUNNEL_URL="https://github.com/erebe/wstunnel/releases/download/v${WSTUNNEL_VERSION}/wstunnel_${WSTUNNEL_VERSION}_linux_amd64.tar.gz"

usage() {
  echo "usage: $0 [prepare|check|path]" >&2
}

verify_bundle() {
  [[ -x "${OUTPUT_DIR}/frameworks/qwen_code/bin/qwen" ]]
  [[ -x "${OUTPUT_DIR}/frameworks/qwen_code/node/bin/node" ]]
  [[ -f "${OUTPUT_DIR}/frameworks/qwen_code/lib/qwen-code/cli-entry.js" ]]
  [[ -x "${OUTPUT_DIR}/linux/bin/wstunnel" ]]
  [[ -f "${OUTPUT_DIR}/manifest.json" ]]

  local qwen_version node_version wstunnel_version
  qwen_version="$(${OUTPUT_DIR}/frameworks/qwen_code/bin/qwen --version | head -n 1)"
  node_version="$(${OUTPUT_DIR}/frameworks/qwen_code/node/bin/node --version)"
  wstunnel_version="$(${OUTPUT_DIR}/linux/bin/wstunnel --version | head -n 1)"
  [[ "${qwen_version}" == *"${QWEN_CODE_VERSION}"* ]]
  [[ "${node_version}" == "v${NODE_VERSION}" ]]
  [[ "${wstunnel_version}" == *"${WSTUNNEL_VERSION}"* ]]
  echo "qwen-code=${qwen_version}"
  echo "node=${node_version}"
  echo "wstunnel=${wstunnel_version}"
  echo "READY protocol_bundle=${OUTPUT_DIR}"
}

action="${1:-prepare}"
case "${action}" in
  path)
    echo "${OUTPUT_DIR}"
    exit 0
    ;;
  check)
    verify_bundle
    exit 0
    ;;
  prepare) ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ -d "${OUTPUT_DIR}" ]]; then
  verify_bundle
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
tmp_dir="$(mktemp -d "${OUTPUT_ROOT}/.${BUNDLE_NAME}.building.XXXXXX")"
cleanup() {
  if [[ -d "${tmp_dir:-}" ]]; then
    rm -rf -- "${tmp_dir}"
  fi
}
trap cleanup EXIT

download() {
  local url="$1"
  local output="$2"
  HTTPS_PROXY="${PROXY_URL}" HTTP_PROXY="${PROXY_URL}" \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 30 \
      --output "${output}" "${url}"
}

echo "Downloading pinned Agentic SWE protocol components through ${PROXY_URL}"
download "${QWEN_URL}" "${tmp_dir}/qwen-code.tgz"
download "${NODE_URL}" "${tmp_dir}/node.tar.xz"
download "${WSTUNNEL_URL}" "${tmp_dir}/wstunnel.tar.gz"

echo "${QWEN_CODE_SHA512}  ${tmp_dir}/qwen-code.tgz" | sha512sum --check --status
echo "${NODE_SHA256}  ${tmp_dir}/node.tar.xz" | sha256sum --check --status
echo "${WSTUNNEL_SHA256}  ${tmp_dir}/wstunnel.tar.gz" | sha256sum --check --status

bundle_dir="${tmp_dir}/bundle"
mkdir -p \
  "${bundle_dir}/frameworks/qwen_code/bin" \
  "${bundle_dir}/frameworks/qwen_code/lib" \
  "${bundle_dir}/frameworks/qwen_code/node/bin" \
  "${bundle_dir}/linux/bin"

mkdir -p "${tmp_dir}/qwen-unpack" "${tmp_dir}/node-unpack" "${tmp_dir}/wstunnel-unpack"
tar -xzf "${tmp_dir}/qwen-code.tgz" -C "${tmp_dir}/qwen-unpack"
tar -xJf "${tmp_dir}/node.tar.xz" -C "${tmp_dir}/node-unpack"
tar -xzf "${tmp_dir}/wstunnel.tar.gz" -C "${tmp_dir}/wstunnel-unpack"

mv "${tmp_dir}/qwen-unpack/package" "${bundle_dir}/frameworks/qwen_code/lib/qwen-code"
cp "${tmp_dir}/node-unpack/node-v${NODE_VERSION}-linux-x64/bin/node" \
  "${bundle_dir}/frameworks/qwen_code/node/bin/node"
cp "${tmp_dir}/wstunnel-unpack/wstunnel" "${bundle_dir}/linux/bin/wstunnel"

chmod 0755 \
  "${bundle_dir}/frameworks/qwen_code/node/bin/node" \
  "${bundle_dir}/linux/bin/wstunnel"

# The wrapper is deliberately relative-path based so the exact protocol root can
# be copied into every sandbox image without changing the qwen-code package.
printf '%s\n' \
  '#!/bin/sh' \
  'set -eu' \
  'qwen_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)' \
  'exec "$qwen_root/node/bin/node" "$qwen_root/lib/qwen-code/cli-entry.js" "$@"' \
  > "${bundle_dir}/frameworks/qwen_code/bin/qwen"
chmod 0755 "${bundle_dir}/frameworks/qwen_code/bin/qwen"

python3 - "${bundle_dir}/manifest.json" <<PY
import json
import sys

manifest = {
    "schema_version": 1,
    "qwen_code": {
        "version": "${QWEN_CODE_VERSION}",
        "source": "${QWEN_URL}",
        "sha512": "${QWEN_CODE_SHA512}",
    },
    "node": {
        "version": "${NODE_VERSION}",
        "source": "${NODE_URL}",
        "sha256": "${NODE_SHA256}",
    },
    "wstunnel": {
        "version": "${WSTUNNEL_VERSION}",
        "source": "${WSTUNNEL_URL}",
        "sha256": "${WSTUNNEL_SHA256}",
    },
    "sandbox_paths": {
        "qwen": "/__avaeval_agentic_protocol_v1__/frameworks/qwen_code/bin/qwen",
        "wstunnel": "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel",
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(manifest, output, indent=2, sort_keys=True)
    output.write("\n")
PY

mv "${bundle_dir}" "${OUTPUT_DIR}"
verify_bundle
