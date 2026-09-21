#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/config/nero.yaml}"
[[ -x "$PYTHON" ]] || { printf 'Python not found: %s. Run scripts/install.sh first.\n' "$PYTHON" >&2; exit 1; }

"$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import pyrealsense2 as rs
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
expected = {name: values["serial"] for name, values in config["cameras"].items() if isinstance(values, dict)}
devices = {}
for device in rs.context().query_devices():
    serial = device.get_info(rs.camera_info.serial_number)
    name = device.get_info(rs.camera_info.name)
    devices[serial] = name
print("Detected RealSense devices:")
for serial, name in devices.items():
    print(f"  {serial}: {name}")
missing = [f"{role}={serial}" for role, serial in expected.items() if serial not in devices]
if missing:
    raise SystemExit("Missing configured cameras: " + ", ".join(missing))
print("Camera binding:")
for role, serial in expected.items():
    print(f"  {role}: {serial}")
PY
