#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"
mkdir -p "$LOG_DIR" "$ROOT/outputs/locks"
exec 9>"$ROOT/outputs/locks/drivers.lock"
flock -w 35 9 || { printf 'Another driver lifecycle operation is running.\n' >&2; exit 1; }
python3 "$ROOT/scripts/driver_processes.py" stop --log-dir "$LOG_DIR"
printf 'Nero project driver sessions and orphan nodes stopped.\n'
