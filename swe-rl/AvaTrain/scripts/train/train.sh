

runtime_env_add() {
    local kv; local -a a=()
    for kv in "$@"; do
        [[ -n $kv ]] || continue
        a+=(--arg "${kv%%=*}" "${kv#*=}")
    done
    (( ${#a[@]} )) || return 0
    RUNTIME_ENV_JSON=$(jq -c "${a[@]}" '.env_vars += $ARGS.named' <<<"$RUNTIME_ENV_JSON")
  }

get_local_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || hostname
}

cleanup_local_processes() {
    pkill -9 sglang 2>/dev/null || true
    uv run --no-sync ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
}


# =======================================================================================


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT_PATH="${SCRIPT_DIR}/../../miles/train_async.py"
RUNTIME_ENV_JSON=${RUNTIME_ENV_JSON:-'{"env_vars":{}}'}


runtime_env_add \
    "SGLANG_CACHE_ROOT=/root/.cache/sglang_cache" \
    "TVM_FFI_CACHE_DIR=/root/.cache/tvm_ffi_cache" \
    "FLASHINFER_CACHE_DIR=/root/.cache/flashinfer_cache"

SGLANG_CACHE_ROOT="/root/.cache/sglang_cache"
TVM_FFI_CACHE_DIR="/root/.cache/tvm_ffi_cache"
FLASHINFER_CACHE_DIR="/root/.cache/flashinfer_cache"

cleanup_local_processes

source "${SCRIPT_DIR}/setup_ray.sh"
source "${SCRIPT_DIR}/build_cli.sh"
source "${SCRIPT_DIR}/setup_logging.sh"


submit_ray_job() {
  source "${MODEL_ARGS_SCRIPT}"
  build_cli_args
  runtime_env_add "${CUSTOM_ENV[@]:-}"
  

  ray_cmd=(
    uv run --no-sync ray job submit
      --address="http://127.0.0.1:${DASHBOARD_PORT}"
      --runtime-env-json="${RUNTIME_ENV_JSON}"
      --
        uv run --no-sync "${TRAIN_SCRIPT_PATH}"
        "${MODEL_ARGS[@]}"
        "${CLI_ARGS[@]}"
  )

  if [[ -n "${BACKUP_SCRIPT_DIR:-}" ]]; then
    { echo '#!/usr/bin/env bash'; printf '%q ' "${ray_cmd[@]}"; echo; } \
      > "${BACKUP_SCRIPT_DIR}/ray_cmd.sh"
    chmod +x "${BACKUP_SCRIPT_DIR}/ray_cmd.sh"
    echo "Ray command saved to ${BACKUP_SCRIPT_DIR}/ray_cmd.sh"
  fi

  "${ray_cmd[@]}"
}


if [[ "${RANK}" -eq 0 ]]; then
  setup_experiment_dir
  snapshot_script
  redirect_run_logs
  run_head_node
  if submit_ray_job; then
    uv run --no-sync ray stop --force || true
  else
    echo "[WARN] submit_ray_job failed (exit=$?), keeping ray alive for debugging" >&2
    sleep 1h
  fi
else
  redirect_run_logs
  run_worker_node
fi
