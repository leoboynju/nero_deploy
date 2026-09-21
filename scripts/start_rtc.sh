#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: bash scripts/start_rtc.sh [client arguments]\n'
    printf 'Start drivers and asynchronous RTC inference with prefix guidance.\n'
    printf 'Recovers both arms after conflict checks, with ROS drivers stopped.\n'
    printf 'Environment: RECOVER_BEFORE_START (default 1), KEEP_DRIVERS (default 0), CONFIG, PROMPT, MAX_STEPS, PYTHON.\n'
    printf 'Example: MAX_STEPS=500 bash scripts/start_rtc.sh\n'
    exit 0
fi

export RECOVER_BEFORE_START="${RECOVER_BEFORE_START:-1}"
exec bash "$ROOT/scripts/start.sh" "$@" --inference-mode rtc
