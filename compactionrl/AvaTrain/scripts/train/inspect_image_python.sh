#!/usr/bin/env bash

# Read-only inventory for locating the Python runtime inside a GPU image.

set -uo pipefail

echo "=== Container identity ==="
printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"
for name in CUDA_VERSION SGLANG_IMAGE_TAG SGLANG_BUILD_COMMIT PATH PYTHONPATH VIRTUAL_ENV CONDA_PREFIX UV_PROJECT_ENVIRONMENT; do
  printf '%-24s %s\n' "${name}=" "${!name:-<unset>}"
done
[[ -r /etc/os-release ]] && sed -n '1,12p' /etc/os-release

echo
echo "=== GPU and CUDA ==="
command -v nvidia-smi || true
nvidia-smi -L 2>&1 || true
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1 || true
command -v nvcc || true
nvcc --version 2>&1 | tail -n 4 || true

echo
echo "=== Environment managers ==="
for command_name in conda micromamba mamba uv pip pip3 ray; do
  printf '%-12s ' "${command_name}"
  command -v "${command_name}" 2>/dev/null || echo '<missing>'
done
if command -v uv >/dev/null 2>&1; then
  uv --version 2>&1 || true
  uv python list 2>&1 | sed -n '1,80p' || true
fi

echo
echo "=== Python executables ==="
declare -a candidates=()
while IFS= read -r path; do
  [[ -n "${path}" ]] && candidates+=("${path}")
done < <(
  {
    type -aP python 2>/dev/null || true
    type -aP python3 2>/dev/null || true
    find /usr/local /opt /root /venv /workspace /sgl-workspace \
      -maxdepth 6 -type f \( -name python -o -name python3 \) -perm -111 \
      2>/dev/null || true
  } | awk '!seen[$0]++'
)

if [[ "${#candidates[@]}" -eq 0 ]]; then
  echo "No Python executables found in standard image locations."
fi

for python_bin in "${candidates[@]}"; do
  echo
  echo "--- ${python_bin} ---"
  "${python_bin}" - <<'PY' 2>&1 || true
import importlib.util
import sys

print("executable =", sys.executable)
print("version    =", sys.version.replace("\n", " "))
print("prefix     =", sys.prefix)
print("base_prefix=", sys.base_prefix)
print("sys.path:")
for entry in sys.path:
    print(" ", entry)
for name in ("torch", "ray", "transformers", "sglang", "sgl_kernel", "miles", "megatron"):
    spec = importlib.util.find_spec(name)
    print(f"spec {name:14s} = {getattr(spec, 'origin', None) if spec else None}")
try:
    import torch
    print("TORCH_READY =", torch.__version__, "cuda=", torch.version.cuda, "available=", torch.cuda.is_available())
except Exception as exc:
    print("TORCH_ERROR =", type(exc).__name__, str(exc))
PY
done

echo
echo "=== Installed package files on disk ==="
for root in /usr/local/lib /usr/lib /opt /root /venv /workspace /sgl-workspace; do
  [[ -e "${root}" ]] || continue
  find "${root}" -maxdepth 8 -type f \
    \( -path '*/torch/__init__.py' \
       -o -path '*/ray/__init__.py' \
       -o -path '*/sgl_kernel/__init__.py' \
       -o -path '*/transformers/__init__.py' \) \
    -print 2>/dev/null || true
done

echo
echo "=== Exact target-image fingerprints ==="
echo "Expected CUDA_VERSION       : 12.9.1"
echo "Expected SGLANG_IMAGE_TAG   : lmsysorg/sglang:v0.5.12-cu129"
echo "Expected Python             : 3.12.3"
echo "Expected Torch              : 2.11.0+cu129"
echo "Expected SGLang             : 0.5.13.dev31+ga72831a"
echo "Expected sglang-kernel      : 0.4.2.post2+cu129"
