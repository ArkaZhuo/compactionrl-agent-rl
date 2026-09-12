#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="${AVA_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${SHARED_ROOT}/model/Qwen/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${SHARED_ROOT}/model/Qwen/Qwen3.5-4B_torch_dist}"
SDK_ROOT="${INSPIRE_SANDBOX_PYTHONPATH:-/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox}"
SWEBENCH_RUNTIME_DIR="${SWEBENCH_RUNTIME_DIR:-${SHARED_ROOT}/.deps/swebench-runtime-py312}"
LOG_ROOT="${LOG_ROOT:-${SHARED_ROOT}/logs/swe-rl/miles-image-check}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/miles_image_runtime_${RUN_TS}.log}"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date -u -Is)] Miles image runtime audit"
echo "AvaTrain       : ${AVA_ROOT}"
echo "HF checkpoint  : ${HF_CHECKPOINT}"
echo "torch_dist     : ${REF_LOAD}"
echo "Log            : ${LOG_FILE}"
echo "VIRTUAL_ENV    : ${VIRTUAL_ENV:-<unset>} (ignored by this audit)"
echo "SBX_API_KEY    : $([[ -n "${SBX_API_KEY:-}" ]] && echo set || echo unset)"
echo "SBX_API_URL    : $([[ -n "${SBX_API_URL:-}" ]] && echo set || echo unset)"

echo
echo "=== Image and GPU ==="
for item in CUDA_VERSION SGLANG_IMAGE_TAG SGLANG_BUILD_COMMIT; do
  printf '%-20s = %s\n' "${item}" "${!item:-<unset>}"
done
[[ -r /etc/os-release ]] && sed -n '1,8p' /etc/os-release
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "FAIL nvidia-smi is missing"
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 4
else
  echo "WARN nvcc is missing"
fi

# Do not use the partially constructed AvaTrain .venv. Pick an image-level
# interpreter that can import the image's own PyTorch installation.
SYSTEM_PYTHON="${SYSTEM_PYTHON:-}"
if [[ -z "${SYSTEM_PYTHON}" ]]; then
  candidates=("$(command -v python3 2>/dev/null || true)" /usr/bin/python3 /usr/local/bin/python3 "$(command -v python 2>/dev/null || true)")
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    [[ "${candidate}" == "${AVA_ROOT}/.venv/"* ]] && continue
    if env -u VIRTUAL_ENV PYTHONPATH= PYTHONNOUSERSITE=1 "${candidate}" -c 'import torch' >/dev/null 2>&1; then
      SYSTEM_PYTHON="${candidate}"
      break
    fi
  done
fi

echo
echo "=== System Python baseline ==="
if [[ -z "${SYSTEM_PYTHON}" || ! -x "${SYSTEM_PYTHON}" ]]; then
  echo "RESULT=FAIL"
  echo "FAIL No image-level Python interpreter can import torch"
  echo "Log: ${LOG_FILE}"
  exit 1
fi
echo "SYSTEM_PYTHON=${SYSTEM_PYTHON}"

env -u VIRTUAL_ENV \
  PYTHONPATH= \
  PYTHONNOUSERSITE=1 \
  "${SYSTEM_PYTHON}" - "${HF_CHECKPOINT}" <<'PY'
import importlib
import importlib.metadata
import os
import sys
import traceback
from pathlib import Path

hf_checkpoint = sys.argv[1]
failures = []
warnings = []

print("python_executable =", sys.executable)
print("python_version    =", sys.version.split()[0])

def check_import(module_name, dist_name=None):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version is None and dist_name:
            try:
                version = importlib.metadata.version(dist_name)
            except Exception:
                version = "unknown"
        print(f"OK   {module_name:24s} version={version or 'unknown'} file={getattr(module, '__file__', None)}")
        return module
    except Exception as exc:
        print(f"FAIL {module_name:24s} {type(exc).__name__}: {exc}")
        failures.append(module_name)
        return None

torch = check_import("torch", "torch")
numpy = check_import("numpy", "numpy")
scipy = check_import("scipy", "scipy")
ray = check_import("ray", "ray")
transformers = check_import("transformers", "transformers")
check_import("sglang", "sglang")
check_import("sgl_kernel", "sglang-kernel")
check_import("transformer_engine", "transformer-engine")
check_import("megatron", "megatron-core")
check_import("miles", "miles")
torch_memory_saver = check_import("torch_memory_saver", "torch-memory-saver")

if torch_memory_saver is not None:
    tms_preload = (
        Path(torch_memory_saver.__file__).resolve().parent.parent
        / "torch_memory_saver_hook_mode_preload.abi3.so"
    )
    print("torch_memory_saver preload =", tms_preload)
    if not tms_preload.is_file():
        print("FAIL torch_memory_saver preload library is missing")
        failures.append("torch_memory_saver preload")

if torch is not None:
    print("torch_cuda       =", torch.version.cuda)
    print("cuda_available   =", torch.cuda.is_available())
    print("visible_gpus     =", torch.cuda.device_count())
    if not torch.cuda.is_available():
        failures.append("torch.cuda")

if numpy is not None and not numpy.__version__.startswith("1."):
    print("FAIL Miles/Megatron requires numpy<2")
    failures.append("numpy<2")

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    print("OK   Qwen3.5 FLA kernels imported")
except Exception as exc:
    print(f"FAIL Qwen3.5 FLA imports: {type(exc).__name__}: {exc}")
    failures.append("qwen3.5-fla")

if transformers is not None and os.path.isfile(os.path.join(hf_checkpoint, "config.json")):
    try:
        config = transformers.AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
        print("model_type       =", config.model_type)
        if config.model_type != "qwen3_5":
            failures.append("qwen3_5-config")
    except Exception as exc:
        print(f"FAIL Qwen3.5 AutoConfig: {type(exc).__name__}: {exc}")
        failures.append("qwen3_5-config")
else:
    print("FAIL HF checkpoint config is unavailable")
    failures.append("hf-checkpoint")

print("BASELINE_FAILURES=", ",".join(dict.fromkeys(failures)) if failures else "none")
sys.exit(1 if failures else 0)
PY
baseline_rc=$?

echo
echo "=== Local AvaTrain overlay ==="
OVERLAY_PYTHONPATH="${AVA_ROOT}/.vendor/python/transformers-5.9.0:${AVA_ROOT}/Megatron-LM:${AVA_ROOT}/miles:${AVA_ROOT}/.vendor/mbridge:${AVA_ROOT}/sglang/python:${AVA_ROOT}/miles/examples/agentic_swe:${SDK_ROOT}:${SWEBENCH_RUNTIME_DIR}"
env -u VIRTUAL_ENV \
  PYTHONPATH="${OVERLAY_PYTHONPATH}" \
  PYTHONNOUSERSITE=1 \
  "${SYSTEM_PYTHON}" - "${HF_CHECKPOINT}" <<'PY'
import sys

failures = []
for statement, label in [
    ("import inspire_sandbox", "inspire_sandbox"),
    ("import swebench", "swebench"),
    ("import miles_plugins.models.qwen3_5", "local qwen3_5 plugin"),
    ("from fla.modules import FusedRMSNormGated, ShortConvolution", "FLA modules"),
    ("from fla.ops.gated_delta_rule import chunk_gated_delta_rule", "FLA gated delta rule"),
    ("import generate", "Agentic SWE generate"),
]:
    try:
        exec(statement, {})
        print("OK  ", label)
    except Exception as exc:
        print("FAIL", label, f"{type(exc).__name__}: {exc}")
        failures.append(label)

print("OVERLAY_FAILURES=", ",".join(failures) if failures else "none")
sys.exit(1 if failures else 0)
PY
overlay_rc=$?

echo
echo "=== Shared assets ==="
asset_failures=0
for asset in \
  "${HF_CHECKPOINT}/model.safetensors.index.json" \
  "${REF_LOAD}/latest_checkpointed_iteration.txt" \
  "${REF_LOAD}/release/.metadata" \
  "${SHARED_ROOT}/swe-rl/data/smoke_verified_1_avatrain_qwen_code_0.21.0.jsonl" \
  "${SHARED_ROOT}/swe-rl/data/swe_verified_500_avatrain_qwen_code_0.21.0.jsonl"; do
  if [[ -s "${asset}" ]]; then
    echo "OK   ${asset}"
  else
    echo "FAIL ${asset}"
    asset_failures=$((asset_failures + 1))
  fi
done

echo
echo "=== Result ==="
if [[ "${baseline_rc}" -eq 0 && "${overlay_rc}" -eq 0 && "${asset_failures}" -eq 0 ]]; then
  echo "RESULT=READY"
  rc=0
else
  echo "RESULT=NEEDS_ADAPTATION baseline_rc=${baseline_rc} overlay_rc=${overlay_rc} asset_failures=${asset_failures}"
  rc=1
fi
echo "Log: ${LOG_FILE}"
exit "${rc}"
