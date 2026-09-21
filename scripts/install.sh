#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
    if command -v python3.10 >/dev/null 2>&1; then
        PYTHON="$(command -v python3.10)"
    else
        PYTHON="python3"
    fi
fi
VENV="${VENV:-$ROOT/.venv}"

command -v "$PYTHON" >/dev/null || { printf 'Python not found: %s\n' "$PYTHON" >&2; exit 1; }
if [[ ! -x "$VENV/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$PYTHON" --system-site-packages "$VENV"
    else
        "$PYTHON" -m venv --system-site-packages "$VENV"
    fi
fi

pip_install() {
    env -u http_proxy -u https_proxy -u all_proxy \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        "$VENV/bin/python" -m pip install --no-build-isolation "$@"
}

pip_install -e "$ROOT/thirdparty/pyAgxArm"
pip_install -e "$ROOT[dev]"

set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
set -u
colcon --log-base "$ROOT/thirdparty/ros_ws/log" build \
    --base-paths "$ROOT/thirdparty/ros_ws/src" \
    --build-base "$ROOT/thirdparty/ros_ws/build" \
    --install-base "$ROOT/thirdparty/ros_ws/install" \
    --symlink-install

printf 'Environment ready: %s\n' "$VENV"
printf 'ROS driver workspace: %s\n' "$ROOT/thirdparty/ros_ws/install/setup.bash"
printf 'Run: PYTHON=%s bash scripts/start_policy_client.sh\n' "$VENV/bin/python"
