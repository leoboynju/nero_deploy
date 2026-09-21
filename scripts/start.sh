#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"
mkdir -p "$LOG_DIR" "$ROOT/outputs/locks"
exec 8>"$ROOT/outputs/locks/stack.lock"
flock -n 8 || { printf 'A Nero deployment stack is already running.\n' >&2; exit 1; }

bash "$ROOT/scripts/bind_can.sh"
printf 'Startup: CAN ready at %ss\n' "$SECONDS"
bash "$ROOT/scripts/start_drivers.sh"
printf 'Startup: recovery/drivers ready at %ss; starting policy client\n' "$SECONDS"
client_pid=""
cleanup() {
    local status=$?
    trap '' INT TERM
    if [[ -n "$client_pid" ]] && kill -0 "$client_pid" 2>/dev/null; then
        kill -TERM "$client_pid" 2>/dev/null || true
        # Bound teardown even if the policy is blocked in a network receive.
        for ((i=0; i<100; i++)); do
            kill -0 "$client_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$client_pid" 2>/dev/null; then
            printf 'Policy client did not exit in 10s; terminating it before driver cleanup.\n' >&2
            kill -KILL "$client_pid" 2>/dev/null || true
        fi
        wait "$client_pid" 2>/dev/null || true
    fi
    if [[ "$status" == "0" && "${KEEP_DRIVERS:-0}" == "1" ]]; then
        printf 'Task complete; drivers retained. Next task may use RECOVER_BEFORE_START=0.\n'
    else
        bash "$ROOT/scripts/stop_all.sh" || return $?
    fi
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
bash "$ROOT/scripts/start_policy_client.sh" "$@" &
client_pid=$!
wait "$client_pid"
client_pid=""
