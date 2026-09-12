setup_experiment_dir() {
    mkdir -p "${EXP_DIR}"
    RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="${EXP_DIR}/${RUN_ID}"
    LOG_DIR="${RUN_DIR}/logs"
    ROLLOUT_DIR="${RUN_DIR}/rollouts"
    SAVE="${RUN_DIR}/checkpoints" # follows the argument name of miles
    SAVE_HF="${RUN_DIR}/checkpoints_hf" # follows the argument name of miles
    BACKUP_SCRIPT_DIR="${RUN_DIR}/scripts"
    mkdir -p "${LOG_DIR}" "${ROLLOUT_DIR}" "${SAVE}" "${BACKUP_SCRIPT_DIR}"
    if [[ "${ENABLE_DEBUG_DUMP:-0}" == "1" ]]; then
        DUMP_DETAILS="${RUN_DIR}/debug"
        mkdir -p "${DUMP_DETAILS}"
        runtime_env_add "DUMP_DETAILS=${DUMP_DETAILS}"
    fi

    runtime_env_add \
        "RUN_DIR=${RUN_DIR}" \
        "LOG_DIR=${LOG_DIR}" \
        "ROLLOUT_DIR=${ROLLOUT_DIR}" \
        "SAVE=${SAVE}" \
        "SAVE_HF=${SAVE_HF}" \
        "WANDB_DIR=${LOG_DIR}" \
        "BACKUP_SCRIPT_DIR=${BACKUP_SCRIPT_DIR}"
}

redirect_run_logs() {
    if [[ "${RANK}" -eq 0 ]]; then
        exec > >(tee "${LOG_DIR}/run.log") 2>&1
    else
        exec >/dev/null 2>&1
    fi
}

snapshot_script() {
    mkdir -p "${BACKUP_SCRIPT_DIR}"
    local -a files=("${BASH_SOURCE[@]}")
    if [[ -n "${SOURCED_FILES:-}" ]]; then
        files+=("${SOURCED_FILES[@]}")       # plus sibling libs that registered themselves
    else
        files+=("${SCRIPT_DIR}"/*.sh)           # fallback: also grab the whole lib dir
    fi

    local f
    for f in "${files[@]}"; do
        [[ -f "$f" ]] && cp -f "$f" "${BACKUP_SCRIPT_DIR}/"
    done

    {
        echo "# Env snapshot for ${RUN_ID} @ $(date '+%F %T')"
        ( set -o posix; set )
    } > "${BACKUP_SCRIPT_DIR}/env_snapshot.sh"
    chmod 600 "${BACKUP_SCRIPT_DIR}/env_snapshot.sh"   # contains *_API_KEY etc.

    echo "Snapshot saved under ${BACKUP_SCRIPT_DIR}"
}