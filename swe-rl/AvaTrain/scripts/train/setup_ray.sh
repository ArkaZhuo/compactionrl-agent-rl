setup_ray_cluster_identity() {
    RANK=${RANK:-0}
    MASTER_ADDR="${MASTER_ADDR:-$(get_local_ip)}"
    MASTER_PORT=${MASTER_PORT:-6379}
    DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
    NO_PROXY="127.0.0.1,localhost,${MASTER_ADDR}"
    runtime_env_add \
    "MASTER_ADDR=${MASTER_ADDR}" \
    "MASTER_PORT=${MASTER_PORT}" \
    "NO_PROXY=${NO_PROXY}"
}


wait_for_full_ray_cluster() {
    local expected_gpus=$(( PET_NNODES * PET_NPROC_PER_NODE ))
    local url="http://127.0.0.1:${DASHBOARD_PORT}/api/cluster_status"
    local attempt body gpus nodes
    for ((attempt = 1; attempt <= 100; attempt++)); do
      if body=$(curl -fs --max-time 5 "$url"); then
        gpus=$(jq -r '(.data.clusterStatus.loadMetricsReport.usage.GPU[1] // 0)|floor' <<<"$body")
        nodes=$(jq -r '.data.clusterStatus.autoscalerReport.activeNodes | length' <<<"$body")
        if (( nodes >= PET_NNODES && gpus >= expected_gpus )); then
          echo "Ray cluster ready: ${nodes}/${PET_NNODES} nodes, ${gpus}/${expected_gpus} GPUs."
          return 0
        fi
      fi
      echo "Waiting for Ray workers (${attempt}/100)..."
      sleep 10
    done
    echo "Ray workers did not join in time." >&2
    uv run --no-sync ray status --address="${MASTER_ADDR}:${MASTER_PORT}" || true
    return 1
}

start_ray_worker_with_retry() {
    local node_name="${WORKER_ID:-${HOSTNAME:-worker-${RANK}}}"
    local attempt

    for ((attempt = 1; attempt <= 100; attempt++)); do
        if uv run --no-sync ray start \
            --address="${MASTER_ADDR}:${MASTER_PORT}" \
            --num-gpus "${PET_NPROC_PER_NODE}" \
            --node-ip-address "$(get_local_ip)" \
            --node-name "${node_name}" \
            --dashboard-port="${DASHBOARD_PORT}" \
            --disable-usage-stats; then
            echo "Ray worker joined on attempt ${attempt}."
            return 0
        fi

        echo "Ray worker join failed on attempt ${attempt}, retrying..."
        uv run --no-sync ray stop --force 2>/dev/null || true
        sleep 5
    done

    echo "Ray worker failed to join cluster after retries." >&2
    return 1
}


run_worker_node() {
    sleep 5
    start_ray_worker_with_retry
    while uv run --no-sync ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; do
        sleep 60
    done
}

run_head_node() {
    nvlink_count=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
    HAS_NVLINK=$(( nvlink_count > 0 ? 1 : 0 ))
    runtime_env_add \
    "NCCL_NVLS_ENABLE=${HAS_NVLINK}"
    
    uv run --no-sync ray start --head \
        --port="${MASTER_PORT}" \
        --node-ip-address "${MASTER_ADDR}" \
        --node-name "${WORKER_ID:-${HOSTNAME:-head-0}}" \
        --num-gpus "${PET_NPROC_PER_NODE}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${DASHBOARD_PORT}"
    wait_for_full_ray_cluster
}


setup_ray_cluster_identity