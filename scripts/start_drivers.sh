#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROBOT_WS_SETUP="${ROBOT_WS_SETUP:-$ROOT/thirdparty/ros_ws/install/setup.bash}"
PYAGXARM_ROOT="${PYAGXARM_ROOT:-$ROOT/thirdparty/pyAgxArm}"
LEFT_CAN="${LEFT_CAN:-can3}"
RIGHT_CAN="${RIGHT_CAN:-can2}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"
mkdir -p "$LOG_DIR" "$ROOT/outputs/locks"
exec 9>"$ROOT/outputs/locks/drivers.lock"
flock -w 35 9 || { printf 'Another driver lifecycle operation is running.\n' >&2; exit 1; }
[[ -r "$ROBOT_WS_SETUP" ]] || { printf 'ROS driver workspace missing. Run scripts/install.sh first.\n' >&2; exit 1; }

set +u
source "$ROS_SETUP"
source "$ROBOT_WS_SETUP"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export PYTHONPATH="$PYAGXARM_ROOT:$ROOT/src:$ROOT/vendor/openpi-client/src:${PYTHONPATH:-}"

probe_status=0
/usr/bin/python3 "$ROOT/scripts/probe_drivers.py" --mode ready-or-empty --timeout 5 || probe_status=$?
case "$probe_status" in
    0) if [[ "${RECOVER_BEFORE_START:-0}" != "1" ]]; then
           printf 'Existing dual-arm drivers have unique control endpoints and live feedback.\n'
           exit 0
       fi ;;
    3) printf 'An existing policy/controller is publishing commands; driver startup cancelled.\n' >&2; exit 1 ;;
    1|2) ;;
    *) printf 'Driver probe failed (status %s); refusing hardware changes.\n' "$probe_status" >&2; exit 1 ;;
esac

# Silence is not proof that no driver exists. Clean tracked sessions and this
# project's orphan nodes, then wait for their DDS endpoints to disappear.
python3 "$ROOT/scripts/driver_processes.py" stop --log-dir "$LOG_DIR"
/usr/bin/python3 "$ROOT/scripts/probe_drivers.py" --mode empty --timeout 25 || {
    printf 'Control endpoints remain after project cleanup; refusing duplicate drivers. Stop the owning launch/controller (possibly another workspace or host); stop_all.sh only stops this deployment.\n' >&2
    python3 "$ROOT/scripts/driver_processes.py" diagnose >&2 || true
    exit 1
}

python3 "$ROOT/scripts/driver_processes.py" check-environment
if [[ "${RECOVER_BEFORE_START:-0}" == "1" ]]; then
    printf 'Startup: recovering both arms with ROS drivers stopped\n'
    bash "$ROOT/scripts/recover_nero.sh"
    printf 'Startup: recovery complete at %ss in driver startup\n' "$SECONDS"
fi

ready=0
cleanup_failed_start() {
    if [[ "$ready" != "1" ]]; then
        python3 "$ROOT/scripts/driver_processes.py" stop --log-dir "$LOG_DIR"
    fi
}
trap cleanup_failed_start EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 "$ROOT/scripts/driver_processes.py" launch --arm left --can-port "$LEFT_CAN" --log-dir "$LOG_DIR"
python3 "$ROOT/scripts/driver_processes.py" launch --arm right --can-port "$RIGHT_CAN" --log-dir "$LOG_DIR"
/usr/bin/python3 "$ROOT/scripts/probe_drivers.py" --mode ready --timeout 20
ready=1
printf 'Drivers ready with one subscriber per control topic; logs=%s\n' "$LOG_DIR"
