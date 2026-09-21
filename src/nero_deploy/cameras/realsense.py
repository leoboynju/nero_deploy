from __future__ import annotations

import threading
import time

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        *,
        auto_exposure: bool = True,
        exposure_us: float | None = None,
    ) -> None:
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.serial = serial
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        self._color_sensor = next(
            sensor
            for sensor in profile.get_device().query_sensors()
            if sensor.supports(rs.camera_info.name)
            and "RGB" in sensor.get_info(rs.camera_info.name).upper()
        )
        if self._color_sensor.supports(rs.option.enable_auto_exposure):
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
        if exposure_us is not None:
            if auto_exposure:
                raise ValueError("exposure_us requires auto_exposure=false")
            if not self._color_sensor.supports(rs.option.exposure):
                raise RuntimeError(f"camera {serial} does not support manual exposure")
            self._color_sensor.set_option(rs.option.exposure, float(exposure_us))
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._captured_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def _capture(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())
            with self._lock:
                self._frame = bgr[..., ::-1].copy()
                self._captured_at = time.monotonic()

    def snapshot(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._frame is None:
                return None, float("inf")
            return np.ascontiguousarray(self._frame), time.monotonic() - self._captured_at

    def settings(self) -> dict[str, float | bool | None]:
        result: dict[str, float | bool | None] = {}
        if self._color_sensor.supports(rs.option.enable_auto_exposure):
            result["auto_exposure"] = bool(self._color_sensor.get_option(rs.option.enable_auto_exposure))
        if self._color_sensor.supports(rs.option.exposure):
            exposure_us = float(self._color_sensor.get_option(rs.option.exposure))
            result["exposure_us"] = exposure_us
            result["exposure_seconds"] = exposure_us / 1_000_000.0
        if self._color_sensor.supports(rs.option.gain):
            result["gain"] = float(self._color_sensor.get_option(rs.option.gain))
        return result

    def close(self) -> None:
        self._stop.set()
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass
        self._thread.join(timeout=2)
