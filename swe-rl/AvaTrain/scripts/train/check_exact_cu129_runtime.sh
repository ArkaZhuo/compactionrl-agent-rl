#!/usr/bin/env bash

# Read-only audit for the exact AvaTrain/Miles H100 (CUDA 12.9) image.
# This script never installs, removes, or modifies packages.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVA_ROOT="${AVA_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"
ALLOW_NO_GPU="${ALLOW_NO_GPU:-0}"

if [[ -z "${PYTHON_BIN}" ]]; then
  candidates=(
    "$(command -v python 2>/dev/null || true)"
    "$(command -v python3 2>/dev/null || true)"
    /usr/local/bin/python
    /usr/local/bin/python3
    /opt/conda/bin/python
    /opt/venv/bin/python
    /venv/bin/python
    /root/.venv/bin/python
    /root/miles/.venv/bin/python
    /usr/bin/python3
  )
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if env -u VIRTUAL_ENV PYTHONNOUSERSITE=1 "${candidate}" -c 'import torch, ray' >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

EXPECTED_MILES_COMMIT="12ddddd9a4ac0e488319d48dfc5ef3a594b3c2e6"
EXPECTED_SGLANG_COMMIT="a72831a0a2d68a79bc3f8257262b5cba6bc0ce54"
EXPECTED_MEGATRON_COMMIT="b6c451dbae34f28e85e3239a4a454f11023c158c"

failures=0

fail() {
  echo "FAIL $*"
  failures=$((failures + 1))
}

echo "Exact AvaTrain cu129 runtime audit"
echo "Python       : ${PYTHON_BIN}"
echo "AvaTrain     : ${AVA_ROOT}"
echo "ALLOW_NO_GPU : ${ALLOW_NO_GPU}"
echo

echo "=== GPU and driver ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi -L; then
    fail "nvidia-smi cannot access a GPU"
  fi
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null || true
elif [[ "${ALLOW_NO_GPU}" == "1" ]]; then
  echo "SKIP nvidia-smi is unavailable (ALLOW_NO_GPU=1)"
else
  fail "nvidia-smi is unavailable"
fi

echo
echo "=== Python, CUDA, packages and compiled extensions ==="
if [[ ! -x "${PYTHON_BIN}" ]]; then
  fail "Python is missing or not executable: ${PYTHON_BIN}"
  python_rc=1
else
  env -u VIRTUAL_ENV \
    PYTHONNOUSERSITE=1 \
    EXPECTED_ALLOW_NO_GPU="${ALLOW_NO_GPU}" \
    "${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

expected_versions = {
    "torch": "2.11.0+cu129",
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "transformers": "5.6.0",
    "sglang": "0.5.13.dev31+ga72831a",
    "sglang-kernel": "0.4.2.post2+cu129",
    "transformer-engine": "2.10.0",
    "ray": "2.56.1",
    "miles": "0.1.0",
    "megatron-core": "0.16.0rc0",
}
allow_no_gpu = os.environ.get("EXPECTED_ALLOW_NO_GPU") == "1"
failures = []


def fail(message):
    print(f"FAIL {message}")
    failures.append(message)


print("python_executable =", sys.executable)
print("python_version    =", sys.version.split()[0])
if sys.version.split()[0] != "3.12.3":
    fail(f"python expected 3.12.3, found {sys.version.split()[0]}")

for dist_name, expected in expected_versions.items():
    try:
        actual = importlib.metadata.version(dist_name)
    except Exception as exc:
        fail(f"{dist_name} metadata unavailable: {type(exc).__name__}: {exc}")
        continue
    status = "OK  " if actual == expected else "FAIL"
    print(f"{status} {dist_name:20s} expected={expected:28s} actual={actual}")
    if actual != expected:
        failures.append(f"{dist_name} version")

modules = [
    "torch",
    "numpy",
    "scipy",
    "transformers",
    "ray",
    "sglang",
    "sgl_kernel",
    "transformer_engine",
    "transformer_engine.pytorch",
    "megatron.core",
    "miles",
    "torch_memory_saver",
]
loaded = {}
for module_name in modules:
    if (
        module_name == "sgl_kernel"
        and allow_no_gpu
        and loaded.get("torch") is not None
        and not loaded["torch"].cuda.is_available()
    ):
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            fail("sgl_kernel module is unavailable")
        else:
            print(f"SKIP live import {module_name} without a GPU; module={spec.origin}")
        continue
    try:
        loaded[module_name] = importlib.import_module(module_name)
        print(f"OK   import {module_name}")
    except Exception as exc:
        fail(f"import {module_name}: {type(exc).__name__}: {exc}")

torch = loaded.get("torch")
if torch is not None:
    print("torch.version.cuda =", torch.version.cuda)
    if torch.version.cuda != "12.9":
        fail(f"torch CUDA expected 12.9, found {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print("cuda_available    =", cuda_available)
    print("visible_gpus      =", torch.cuda.device_count())
    if not cuda_available:
        if allow_no_gpu:
            print("SKIP live GPU/SM90 validation (ALLOW_NO_GPU=1)")
        else:
            fail("torch.cuda is unavailable")
    else:
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            capability = torch.cuda.get_device_capability(index)
            print(f"GPU {index}: {name}; compute_capability={capability[0]}.{capability[1]}")
            if capability != (9, 0):
                fail(f"GPU {index} expected H100/SM90 capability 9.0, found {capability}")

sgl_kernel = loaded.get("sgl_kernel")
if sgl_kernel is not None:
    kernel_root = Path(sgl_kernel.__file__).resolve().parent
else:
    kernel_spec = importlib.util.find_spec("sgl_kernel")
    kernel_root = Path(kernel_spec.origin).resolve().parent if kernel_spec and kernel_spec.origin else None
if kernel_root is not None:
    sm90_common_ops = list((kernel_root / "sm90").glob("common_ops*.so"))
    print("sgl_kernel_root   =", kernel_root)
    print("sm90_common_ops   =", sm90_common_ops)
    if not sm90_common_ops:
        fail("SGLang SM90 common_ops library is missing")

torch_memory_saver = loaded.get("torch_memory_saver")
if torch_memory_saver is not None:
    preload = (
        Path(torch_memory_saver.__file__).resolve().parent.parent
        / "torch_memory_saver_hook_mode_preload.abi3.so"
    )
    print("tms_preload       =", preload)
    if not preload.is_file():
        fail("torch_memory_saver preload library is missing")

for statement, label in [
    ("from fla.modules import FusedRMSNormGated, ShortConvolution", "FLA modules"),
    ("from fla.ops.gated_delta_rule import chunk_gated_delta_rule", "FLA gated delta rule"),
    ("from sglang.srt.entrypoints.engine import Engine", "SGLang Engine"),
]:
    if label == "SGLang Engine" and allow_no_gpu and (torch is None or not torch.cuda.is_available()):
        print("SKIP SGLang Engine live import without a GPU (ALLOW_NO_GPU=1)")
        continue
    try:
        exec(statement, {})
        print(f"OK   {label}")
    except Exception as exc:
        fail(f"{label}: {type(exc).__name__}: {exc}")

print("PYTHON_FAILURES  =", len(failures))
raise SystemExit(1 if failures else 0)
PY
  python_rc=$?
fi

echo
echo "=== Pinned source commits ==="
check_commit() {
  local label="$1"
  local expected="$2"
  shift 2
  local found=0
  local path actual

  for path in "$@"; do
    # Worktrees and submodules commonly store .git as a pointer file rather
    # than a directory, so ask Git directly whether the checkout is valid.
    git -C "${path}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || continue
    found=1
    if ! actual="$(git -C "${path}" rev-parse HEAD 2>/dev/null)"; then
      fail "${label}: cannot read Git commit at ${path}"
      continue
    fi
    if [[ "${actual}" == "${expected}" ]]; then
      echo "OK   ${label}: ${actual} (${path})"
    else
      fail "${label}: expected ${expected}, found ${actual} (${path})"
    fi
  done

  if [[ "${found}" -eq 0 ]]; then
    fail "${label}: source checkout not found"
  fi
}

check_commit \
  "Miles" "${EXPECTED_MILES_COMMIT}" \
  /root/miles "${AVA_ROOT}/miles"
check_commit \
  "SGLang" "${EXPECTED_SGLANG_COMMIT}" \
  /sgl-workspace/sglang "${AVA_ROOT}/sglang"
check_commit \
  "Megatron-LM" "${EXPECTED_MEGATRON_COMMIT}" \
  /root/Megatron-LM "${AVA_ROOT}/Megatron-LM"

if [[ "${python_rc}" -ne 0 ]]; then
  failures=$((failures + 1))
fi

echo
echo "=== Result ==="
if [[ "${failures}" -eq 0 ]]; then
  echo "RESULT=READY"
  echo "READY: exact AvaTrain H100/cu129 runtime matched."
  exit 0
fi

echo "RESULT=NOT_READY failures=${failures}"
echo "NOT_READY: this environment does not exactly match the pinned H100/cu129 runtime."
exit 1
