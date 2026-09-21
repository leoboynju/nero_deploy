from __future__ import annotations

import numpy as np


JOINT_INDICES = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)


class DualArmJointEMA:
    """EMA filter for the 14 arm joints; gripper channels are untouched."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._previous: np.ndarray | None = None

    def initialize(self, action: np.ndarray) -> None:
        self._previous = np.asarray(action, dtype=np.float32).copy()

    def commit(self, command: np.ndarray) -> None:
        """Anchor the next update to the command actually sent after downstream clamps."""
        self.initialize(command)

    def apply(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if self._previous is None:
            self.initialize(action)
            return action
        action[JOINT_INDICES] = (
            self.alpha * action[JOINT_INDICES]
            + (1.0 - self.alpha) * self._previous[JOINT_INDICES]
        )
        self._previous = action.copy()
        return action


class GripperDebouncer:
    """Confirm consecutive OPEN requests; a CLOSE request takes effect immediately."""

    def __init__(self, confirm_steps: int) -> None:
        if confirm_steps < 1:
            raise ValueError("gripper_confirm_steps must be positive")
        self.confirm_steps = confirm_steps
        self._open = np.ones(2, dtype=bool)
        self._counts = np.zeros(2, dtype=np.int64)

    def initialize(self, state: np.ndarray) -> None:
        # ROS feedback and published gripper targets are widths in metres (0..0.1).
        self.reset(np.asarray(state)[[7, 15]] >= 0.05)

    def reset(self, opened) -> None:
        self._open = np.asarray(opened, dtype=bool).copy()
        self._counts[:] = 0

    def update(self, action: np.ndarray, *, current_open=None) -> dict[str, bool]:
        if current_open is not None:
            # RTC can defer actual actuation until the arm reaches its grasp pose.
            self._open = np.asarray(current_open, dtype=bool).copy()
        requested = np.asarray(action)[[7, 15]] >= 0.5
        self._counts = np.where(requested & ~self._open, self._counts + 1, 0)
        self._open = requested & (self._open | (self._counts >= self.confirm_steps))
        self._counts[self._open] = 0
        return {"left": bool(self._open[0]), "right": bool(self._open[1])}
