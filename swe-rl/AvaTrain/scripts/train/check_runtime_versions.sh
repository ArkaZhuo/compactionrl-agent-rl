#!/usr/bin/env bash

# Concise, read-only GPU image inventory. Every result is one key=value line.

set -uo pipefail

value_or_none() {
  local value="${1:-}"
  if [[ -n "${value}" ]]; then
    printf '%s\n' "${value}"
  else
    printf 'none\n'
  fi
}

os_name=""
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  os_name="${PRETTY_NAME:-}"
fi
echo "OS=$(value_or_none "${os_name}")"

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -)"
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
  gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)"
  compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | paste -sd, -)"
else
  gpu_name=""
  driver=""
  gpu_memory=""
  compute_cap=""
fi
echo "GPU=$(value_or_none "${gpu_name}")"
echo "GPU_MEMORY_MIB=$(value_or_none "${gpu_memory}")"
echo "GPU_COMPUTE_CAP=$(value_or_none "${compute_cap}")"
echo "NVIDIA_DRIVER=$(value_or_none "${driver}")"
echo "CUDA_ENV=$(value_or_none "${CUDA_VERSION:-}")"

nvcc_version=""
if command -v nvcc >/dev/null 2>&1; then
  nvcc_version="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -n 1)"
fi
echo "NVCC=$(value_or_none "${nvcc_version}")"

cudart="$(ldconfig -p 2>/dev/null | awk '/libcudart\.so/{print $NF; exit}')"
echo "CUDART=$(value_or_none "${cudart}")"
echo "SGLANG_IMAGE_TAG=$(value_or_none "${SGLANG_IMAGE_TAG:-}")"
echo "SGLANG_BUILD_COMMIT=$(value_or_none "${SGLANG_BUILD_COMMIT:-}")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  candidates=(
    "$(command -v python 2>/dev/null || true)"
    "$(command -v python3 2>/dev/null || true)"
    /usr/local/bin/python /usr/local/bin/python3
    /opt/conda/bin/python /opt/venv/bin/python /venv/bin/python
    /root/.venv/bin/python /root/miles/.venv/bin/python
    /usr/bin/python3
  )
  # Prefer an environment containing Torch; otherwise report the first Python.
  fallback=""
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    [[ -n "${fallback}" ]] || fallback="${candidate}"
    if env -u VIRTUAL_ENV "${candidate}" -c 'import torch' >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
  PYTHON_BIN="${PYTHON_BIN:-${fallback}}"
fi

if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
  python_version="$("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || true)"
  echo "PYTHON=$(value_or_none "${python_version}")"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null
import importlib.metadata
import importlib.util
from pathlib import Path


def dist_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "none"


for key, dist in (
    ("PIP", "pip"),
    ("NUMPY", "numpy"),
    ("SCIPY", "scipy"),
    ("TORCH", "torch"),
    ("TORCHVISION", "torchvision"),
    ("RAY", "ray"),
    ("TRANSFORMERS", "transformers"),
    ("SGLANG", "sglang"),
    ("SGLANG_KERNEL", "sglang-kernel"),
    ("TRANSFORMER_ENGINE", "transformer-engine"),
    ("FLASH_ATTN", "flash-attn"),
    ("FLASH_LINEAR_ATTENTION", "flash-linear-attention"),
    ("MAMBA_SSM", "mamba-ssm"),
    ("CAUSAL_CONV1D", "causal-conv1d"),
    ("TORCH_MEMORY_SAVER", "torch-memory-saver"),
    ("MILES", "miles"),
    ("MEGATRON_CORE", "megatron-core"),
):
    print(f"{key}={dist_version(dist)}")

try:
    import torch
    print(f"TORCH_CUDA={torch.version.cuda or 'none'}")
    print(f"CUDA_AVAILABLE={str(torch.cuda.is_available()).lower()}")
    print(f"TORCH_VISIBLE_GPUS={torch.cuda.device_count()}")
except Exception:
    print("TORCH_CUDA=none")
    print("CUDA_AVAILABLE=none")
    print("TORCH_VISIBLE_GPUS=none")

kernel_spec = importlib.util.find_spec("sgl_kernel")
if kernel_spec and kernel_spec.origin:
    kernel_root = Path(kernel_spec.origin).resolve().parent
    sm90 = next((kernel_root / "sm90").glob("common_ops*.so"), None)
    print(f"SGLANG_SM90_SO={sm90 if sm90 else 'none'}")
else:
    print("SGLANG_SM90_SO=none")

tms_spec = importlib.util.find_spec("torch_memory_saver")
if tms_spec and tms_spec.origin:
    preload = Path(tms_spec.origin).resolve().parent.parent / "torch_memory_saver_hook_mode_preload.abi3.so"
    print(f"TMS_PRELOAD_SO={preload if preload.is_file() else 'none'}")
else:
    print("TMS_PRELOAD_SO=none")
PY
else
  echo "PYTHON=none"
  echo "PYTHON_BIN=none"
  for key in PIP NUMPY SCIPY TORCH TORCHVISION RAY TRANSFORMERS SGLANG SGLANG_KERNEL TRANSFORMER_ENGINE FLASH_ATTN FLASH_LINEAR_ATTENTION MAMBA_SSM CAUSAL_CONV1D TORCH_MEMORY_SAVER MILES MEGATRON_CORE TORCH_CUDA CUDA_AVAILABLE TORCH_VISIBLE_GPUS SGLANG_SM90_SO TMS_PRELOAD_SO; do
    echo "${key}=none"
  done
fi

for command_name in uv conda micromamba ray; do
  command_path="$(command -v "${command_name}" 2>/dev/null || true)"
  echo "${command_name^^}_CLI=$(value_or_none "${command_path}")"
done
