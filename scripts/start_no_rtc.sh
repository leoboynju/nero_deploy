#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: bash scripts/start_no_rtc.sh [client arguments]\n'
    printf 'Start drivers and synchronous ordinary inference, without RTC fields.\n'
    printf 'Execute the first REPLAN_STEPS actions (default 8) of each 16-step prediction.\n'
    printf 'Grippers: opening requires two consecutive samples; closing executes on its first sample.\n'
    printf 'Recovers both arms before the drivers start.\n'
    printf 'Environment: RECOVER_BEFORE_START (default 1), KEEP_DRIVERS (default 0), REPLAN_STEPS, SPEED_PERCENT, CAMERA_WARMUP_SECONDS, CONFIG, PROMPT, MAX_STEPS, PYTHON.\n'
    printf 'Example: REPLAN_STEPS=5 MAX_STEPS=500 bash scripts/start_no_rtc.sh\n'
    exit 0
fi

export RECOVER_BEFORE_START="${RECOVER_BEFORE_START:-1}"
exec bash "$ROOT/scripts/start.sh" --replan-steps "${REPLAN_STEPS:-8}" "$@" --no-grasp-control --inference-mode sync
