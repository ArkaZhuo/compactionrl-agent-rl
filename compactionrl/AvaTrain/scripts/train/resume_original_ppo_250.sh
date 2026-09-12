#!/usr/bin/env bash
# Standard AvaTrain PPO, secondary-1000: restore paired iteration 159 and
# continue through rollout 249 (250 cumulative rollouts, 90 more rollouts).
# Save actor+critic at 179, 199, 219, 239, and the final iteration 249.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PPO_RESUME_TOTAL_ROLLOUTS=250
exec bash "${SCRIPT_DIR}/resume_original_ppo_200.sh" "$@"
