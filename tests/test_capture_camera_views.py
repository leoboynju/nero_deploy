import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def capture(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "src/tests/capture_camera_views.py"
    spec = importlib.util.spec_from_file_location("capture_camera_views", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(path), "--output-dir", str(tmp_path)])
    monkeypatch.setattr(module, "load_config", lambda _: {
        "cameras": {
            "width": 64, "height": 48, "fps": 30, "warmup_seconds": 0,
            **{role: {"serial": str(index)} for index, role in enumerate(module.ROLES)},
        },
    })
    return module


def test_capture_saves_rgb_views_and_closes_cameras(capture, monkeypatch, tmp_path):
    closed = []

    class Camera:
        def __init__(self, serial, *args, **kwargs):
            self.serial = serial

        def snapshot(self):
            return np.full((48, 64, 3), [255, 0, 0], dtype=np.uint8), 0.01

        def close(self):
            closed.append(self.serial)

    monkeypatch.setattr(capture, "RealSenseCamera", Camera)
    capture.main()
    assert closed == ["0", "1", "2"]
    output, = tmp_path.iterdir()
    assert len(list(output.glob("*.png"))) == 4
    with Image.open(output / "left_wrist_0.png") as image:
        assert image.size == (64, 48)
        assert image.getpixel((0, 0)) == (255, 0, 0)
    with Image.open(output / "overview.png") as image:
        assert image.size == (192, 96)


def test_partial_startup_closes_open_camera(capture, monkeypatch, tmp_path):
    closed = []

    class Camera:
        def __init__(self, serial, *args, **kwargs):
            if serial == "1":
                raise RuntimeError("device busy")
            self.serial = serial

        def close(self):
            closed.append(self.serial)

    monkeypatch.setattr(capture, "RealSenseCamera", Camera)
    with pytest.raises(RuntimeError, match="device busy"):
        capture.main()
    assert closed == ["0"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("frame", [None, np.zeros((48, 64, 3), dtype=np.uint8)])
def test_missing_or_stale_frames_timeout(capture, monkeypatch, tmp_path, frame):
    closed = []

    class Camera:
        def __init__(self, serial, *args, **kwargs):
            self.serial = serial

        def snapshot(self):
            return frame, 10.0

        def close(self):
            closed.append(self.serial)

    ticks = iter([0, 21])
    monkeypatch.setattr(capture.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(capture, "RealSenseCamera", Camera)
    with pytest.raises(TimeoutError, match="fresh frames"):
        capture.main()
    assert closed == ["0", "1", "2"]
    assert not list(tmp_path.iterdir())
