#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROBOT_WS_SETUP="${ROBOT_WS_SETUP:-$ROOT/thirdparty/ros_ws/install/setup.bash}"
CONFIG="${CONFIG:-$ROOT/config/nero.yaml}"
[[ -r "$ROBOT_WS_SETUP" ]] || { printf 'ROS driver workspace missing. Run scripts/install.sh first.\n' >&2; exit 1; }

set +u
source "$ROS_SETUP"
source "$ROBOT_WS_SETUP"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export PYTHONPATH="$ROOT/src:$ROOT/vendor/openpi-client/src:$ROOT/thirdparty/pyAgxArm${PYTHONPATH:+:$PYTHONPATH}"
POLICY_HOST="$("$PYTHON" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["policy"]["host"])' "$CONFIG")"
export NO_PROXY="$POLICY_HOST${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$POLICY_HOST${no_proxy:+,$no_proxy}"

[[ -x "$PYTHON" ]] || { printf 'Python not found: %s. Run scripts/install.sh first.\n' "$PYTHON" >&2; exit 1; }
args=(--config "$CONFIG" --execute)
if [[ -n "${CAMERA_WARMUP_SECONDS:-}" ]]; then
    args+=(--camera-warmup-seconds "$CAMERA_WARMUP_SECONDS")
fi
if [[ -n "${PROMPT:-}" ]]; then
    args+=(--prompt "$PROMPT")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
    args+=(--max-steps "$MAX_STEPS")
fi
exec "$PYTHON" -m nero_deploy.cli "${args[@]}" "$@"
