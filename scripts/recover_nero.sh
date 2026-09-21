#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LEFT_CAN="${LEFT_CAN:-can3}"
RIGHT_CAN="${RIGHT_CAN:-can2}"
SPEED_PERCENT="${SPEED_PERCENT:-30}"

[[ -x "$PYTHON" ]] || { printf 'Python not found: %s. Run scripts/install.sh first.\n' "$PYTHON" >&2; exit 1; }

left_pid=""
right_pid=""
cleanup() {
    for pid in "$left_pid" "$right_pid"; do
        if [[ -n "$pid" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$PYTHON" -m nero_deploy.diagnostics.recover left --can "$LEFT_CAN" \
    --speed-percent "$SPEED_PERCENT" --yes &
left_pid=$!
"$PYTHON" -m nero_deploy.diagnostics.recover right --can "$RIGHT_CAN" \
    --speed-percent "$SPEED_PERCENT" --yes &
right_pid=$!
pending=("$left_pid" "$right_pid")
while ((${#pending[@]})); do
    status=0
    wait -n -p completed "${pending[@]}" || status=$?
    if [[ -z "${completed:-}" ]]; then
        # wait -n can miss already-completed jobs; plain wait retains their
        # statuses. Only wait for a child still tracked by this recovery.
        completed="${pending[0]}"
        status=0
        wait "$completed" || status=$?
    fi
    [[ "$completed" != "$left_pid" ]] || left_pid=""
    [[ "$completed" != "$right_pid" ]] || right_pid=""
    [[ "$status" == "0" ]] || exit "$status"
    pending=()
    [[ -z "$left_pid" ]] || pending+=("$left_pid")
    [[ -z "$right_pid" ]] || pending+=("$right_pid")
done
printf 'dual-arm recovery complete\n'
