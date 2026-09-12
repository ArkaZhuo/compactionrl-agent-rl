#!/usr/bin/env bash
# Restore the original 2026-09-05 secondary PPO actor, critic and cursor at 99.
# Train rollouts 100..249; save new pairs at 119,139,159,179,199,219,239,249.
# Never auto-select 159 or the source directory's latest tracker.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PPO_RESUME_ITERATION=99
export PPO_RESUME_TOTAL_ROLLOUTS=250
exec bash "${SCRIPT_DIR}/resume_original_ppo_200.sh" "$@"
