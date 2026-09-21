#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/config/nero.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/transport}"
SAMPLES="${SAMPLES:-3}"
[[ -x "$PYTHON" ]] || { printf 'Python not found: %s. Run scripts/install.sh first.\n' "$PYTHON" >&2; exit 1; }

export PYTHONPATH="$ROOT/src:$ROOT/vendor/openpi-client/src:$ROOT/thirdparty/pyAgxArm${PYTHONPATH:+:$PYTHONPATH}"
POLICY_HOST="$("$PYTHON" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["policy"]["host"])' "$CONFIG")"
export NO_PROXY="$POLICY_HOST${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$POLICY_HOST${no_proxy:+,$no_proxy}"
exec "$PYTHON" - "$CONFIG" "$OUTPUT_DIR" "$SAMPLES" <<'PY'
import sys
from pathlib import Path
import shutil
import yaml
from nero_deploy.cameras import RealSenseCamera
from nero_deploy.policy_client import PolicyClient
from openpi_client import msgpack_numpy
import numpy as np
from PIL import Image

config = yaml.safe_load(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2])
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True, exist_ok=True)
cams = config["cameras"]
cameras = {name: RealSenseCamera(item["serial"], cams["width"], cams["height"], cams["fps"], auto_exposure=bool(cams.get("auto_exposure", True)), exposure_us=cams.get("exposure_us")) for name, item in cams.items() if isinstance(item, dict)}
try:
    time = __import__("time")
    time.sleep(float(cams.get("warmup_seconds", 5)))
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if all(camera.snapshot()[0] is not None for camera in cameras.values()): break
        time.sleep(0.05)
    else:
        raise RuntimeError("timed out waiting for three camera frames")
    client = PolicyClient(config["policy"]["host"], int(config["policy"]["port"]), config["policy"].get("image_size", 224))
    records = []
    for index in range(int(sys.argv[3])):
        frames = {name: camera.snapshot()[0] for name, camera in cameras.items()}
        if any(frame is None for frame in frames.values()): raise RuntimeError("camera frame unavailable")
        state = np.zeros(16, dtype=np.float32)
        sent_frames = client.prepare_images(frames)
        result = client.infer(sent_frames, state, config["policy"]["prompt"])
        request = {"images": sent_frames, "state": state, "prompt": config["policy"]["prompt"]}
        request_bytes = len(msgpack_numpy.packb(request))
        for name, frame in frames.items():
            np.save(output / f"sample_{index:03d}_{name}_raw.npy", frame)
            Image.fromarray(frame).save(output / f"sample_{index:03d}_{name}_raw.png")
            np.save(output / f"sample_{index:03d}_{name}_sent.npy", sent_frames[name])
            Image.fromarray(sent_frames[name]).save(output / f"sample_{index:03d}_{name}_sent.png")
        records.append({"index": index, "request_bytes": request_bytes, "camera_settings": {name: camera.settings() for name, camera in cameras.items()}, "raw_images": {name: {"shape": list(frame.shape), "dtype": str(frame.dtype), "bytes": int(frame.nbytes), "mean": float(frame.mean())} for name, frame in frames.items()}, "sent_images": {name: {"shape": list(frame.shape), "dtype": str(frame.dtype), "bytes": int(frame.nbytes), "mean": float(frame.mean())} for name, frame in sent_frames.items()}, "actions_shape": list(np.asarray(result["actions"]).shape)})
    (output / "manifest.yaml").write_text(yaml.safe_dump({"camera_order": list(cameras), "records": records}, sort_keys=False))
    print(f"transport test passed: {output}")
finally:
    for camera in cameras.values(): camera.close()
PY
