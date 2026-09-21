from __future__ import annotations

import threading
import time

import numpy as np


class LatestActionChunk:
    """Replace queued actions atomically whenever a newer inference arrives."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._actions: np.ndarray | None = None
        self._cursor = 0
        self._sequence = 0

    def replace(self, actions: np.ndarray) -> int:
        actions = np.asarray(actions, dtype=np.float32).copy()
        with self._condition:
            self._actions = actions
            self._cursor = 0
            self._sequence += 1
            self._condition.notify_all()
            return self._sequence

    def next(self, timeout: float | None = None) -> tuple[np.ndarray, int] | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._actions is None or self._cursor >= len(self._actions):
                if timeout is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
            action = self._actions[self._cursor].copy()
            sequence = self._sequence
            self._cursor += 1
            return action, sequence

    def remaining(self) -> np.ndarray | None:
        with self._condition:
            if self._actions is None or self._cursor >= len(self._actions):
                return None
            return self._actions[self._cursor :].copy()
