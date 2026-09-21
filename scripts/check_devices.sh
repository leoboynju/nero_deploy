#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s\n' '=== RealSense ==='
bash "$ROOT/scripts/bind_cameras.sh"
printf '%s\n' '=== CAN ==='
for interface in "${LEFT_CAN:-can3}" "${RIGHT_CAN:-can2}"; do
    ip -details link show "$interface"
done
printf '%s\n' '=== ROS topics ==='
ROBOT_WS_SETUP="${ROBOT_WS_SETUP:-$ROOT/thirdparty/ros_ws/install/setup.bash}"
[[ -r "$ROBOT_WS_SETUP" ]] || { printf 'ROS driver workspace missing. Run scripts/install.sh first.\n' >&2; exit 1; }
set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "$ROBOT_WS_SETUP"
set -u
ros2 topic list
printf '%s\n' '=== ROS feedback ==='
for arm in left right; do
    if timeout 3 ros2 topic echo --once "/${arm}_arm/feedback/joint_states" >/dev/null 2>&1; then
        printf '%s arm feedback: LIVE\n' "$arm"
    else
        printf '%s arm feedback: STALE OR MISSING\n' "$arm"
    fi
done
