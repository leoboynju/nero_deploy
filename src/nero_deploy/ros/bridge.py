from __future__ import annotations

import threading
import time

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


class DualArmBridge(Node):
    def __init__(self) -> None:
        super().__init__("nero_deploy_policy_client")
        self._lock = threading.Lock()
        self._states: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._state_times = {"left": 0.0, "right": 0.0}
        self._gripper_forces = {"left": float("nan"), "right": float("nan")}
        self._joint_publishers = {}
        self._gripper_publishers = {}
        for arm in ("left", "right"):
            self.create_subscription(
                JointState,
                f"/{arm}_arm/feedback/joint_states",
                lambda message, name=arm: self._on_state(name, message),
                10,
            )
            self._joint_publishers[arm] = self.create_publisher(
                JointState, f"/{arm}_arm/control/move_j", 10
            )
            self._gripper_publishers[arm] = self.create_publisher(
                JointState, f"/{arm}_arm/control/joint_states", 10
            )

    def _on_state(self, arm: str, message: JointState) -> None:
        if len(message.position) < 8:
            return
        with self._lock:
            self._states[arm] = np.asarray(message.position[:8], dtype=np.float32)
            self._state_times[arm] = time.monotonic()
            self._gripper_forces[arm] = float(message.effort[7]) if len(message.effort) >= 8 else float("nan")

    def gripper_forces(self) -> np.ndarray:
        """Reported efforts from JointState; unavailable measurements remain NaN."""
        with self._lock:
            return np.asarray([self._gripper_forces[arm] for arm in ("left", "right")])

    def state(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._states["left"] is None or self._states["right"] is None:
                return None, float("inf")
            state = np.concatenate([self._states["left"], self._states["right"]])
            age = max(time.monotonic() - self._state_times[arm] for arm in ("left", "right"))
            return state, age

    def readiness(self) -> dict[str, bool]:
        with self._lock:
            return {arm: self._states[arm] is not None for arm in ("left", "right")}

    def publish(self, action: np.ndarray, gripper_open: dict[str, bool]) -> None:
        stamp = self.get_clock().now().to_msg()
        for offset, arm in ((0, "left"), (8, "right")):
            joints = JointState()
            joints.header.stamp = stamp
            joints.name = JOINT_NAMES
            joints.position = [float(value) for value in action[offset : offset + 7]]
            self._joint_publishers[arm].publish(joints)

            gripper = JointState()
            gripper.header.stamp = stamp
            gripper.name = ["gripper"]
            gripper.position = [0.1 if gripper_open[arm] else 0.0]
            self._gripper_publishers[arm].publish(gripper)

    def validate_exclusive_control(self) -> None:
        errors = []
        for arm in ("left", "right"):
            for topic in (f"/{arm}_arm/control/move_j", f"/{arm}_arm/control/joint_states"):
                publishers = self.count_publishers(topic)
                subscribers = self.count_subscribers(topic)
                if publishers != 1 or subscribers != 1:
                    errors.append(f"{topic}: publishers={publishers}, subscribers={subscribers}")
        if errors:
            raise RuntimeError("exclusive control check failed: " + "; ".join(errors))
