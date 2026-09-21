#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/config/nero.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/camera_views}"
[[ -x "$PYTHON" ]] || { printf 'Python not found: %s. Run scripts/install.sh first.\n' "$PYTHON" >&2; exit 1; }

exec "$PYTHON" "$ROOT/src/tests/capture_camera_views.py" \
    --config "$CONFIG" --output-dir "$OUTPUT_DIR" "$@"
