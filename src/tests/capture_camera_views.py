#!/usr/bin/env python3
"""Capture labeled RealSense views without ROS, policy inference, or arm control."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nero_deploy.cameras import RealSenseCamera
from nero_deploy.config import load_config

ROLES = ("left_wrist", "right_wrist", "third_person")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/nero.yaml")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/camera_views",
        help="Parent directory for a new timestamped capture directory",
    )
    parser.add_argument("--timeout", type=float, default=20, help="Frame wait timeout after warmup (seconds)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cc = cfg["cameras"]
    serials = [str(cc[role]["serial"]) for role in ROLES]
    warmup = float(cc.get("warmup_seconds", 5))
    max_age = float(cfg.get("control", {}).get("max_source_age", 0.25))
    if len(set(serials)) != len(ROLES):
        raise ValueError("Each camera role must have a different serial number")
    if not math.isfinite(warmup) or warmup < 0:
        raise ValueError("warmup_seconds must be finite and nonnegative")
    if not all(math.isfinite(value) and value > 0 for value in (args.timeout, max_age)):
        raise ValueError("timeout and max_source_age must be finite and positive")
    if cc.get("exposure_us") is not None and cc.get("auto_exposure", True):
        raise ValueError("exposure_us requires auto_exposure=false")

    cameras = {}
    try:
        for role, serial in zip(ROLES, serials):
            print(f"Opening {role}: {serial}", flush=True)
            cameras[role] = RealSenseCamera(
                serial, int(cc["width"]), int(cc["height"]), int(cc["fps"]),
                auto_exposure=bool(cc.get("auto_exposure", True)),
                exposure_us=cc.get("exposure_us"),
            )
        print(f"Warming up for {warmup:g}s...", flush=True)
        time.sleep(warmup)
        deadline = time.monotonic() + args.timeout
        while True:
            snapshots = {role: camera.snapshot() for role, camera in cameras.items()}
            if all(frame is not None and age <= max_age for frame, age in snapshots.values()):
                break
            if time.monotonic() >= deadline:
                status = ", ".join(
                    f"{role}: {'no frame' if frame is None else f'age={age:.3f}s'}"
                    for role, (frame, age) in snapshots.items()
                )
                raise TimeoutError(f"Timed out waiting for fresh frames ({status})")
            time.sleep(0.05)
    finally:
        for role, camera in cameras.items():
            try:
                camera.close()
            except Exception as exc:
                print(f"Warning: could not close {role}: {exc}", file=sys.stderr)

    output = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    # The camera wrapper returns RGB; no BGR conversion or image flipping is needed.
    images = [Image.fromarray(snapshots[role][0]) for role in ROLES]
    overview = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images) + 48), "white")
    draw = ImageDraw.Draw(overview)
    x = 0
    for role, serial, image in zip(ROLES, serials, images):
        age = snapshots[role][1]
        path = output / f"{role}_{serial}.png"
        image.save(path)
        overview.paste(image, (x, 48))
        draw.text((x + 8, 6), f"{role} | SN: {serial}", fill="black")
        draw.text((x + 8, 24), f"{image.width}x{image.height} | frame age: {age:.3f}s", fill="black")
        x += image.width
        print(f"Saved {path}", flush=True)
    overview.save(output / "overview.png")
    print(f"Overview: {output / 'overview.png'}", flush=True)
    print("Left to right: left_wrist, right_wrist, third_person. Verify physical placement visually.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Capture cancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        print("Check camera serials, USB connections, permissions, and other camera users.", file=sys.stderr)
        sys.exit(1)
