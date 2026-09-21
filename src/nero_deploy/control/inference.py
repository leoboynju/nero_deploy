"""Explicit synchronous, RTC, and unguided-asynchronous execution modes."""

from __future__ import annotations

import logging
import time

import numpy as np

from openpi_client.async_rtc_action_chunk_broker import AsyncRTCActionChunkBroker
from openpi_client.base_policy import BasePolicy


logger = logging.getLogger(__name__)


class SynchronousActionChunkBroker(BasePolicy):
    """Predict synchronously, then execute the first N actions before replanning."""

    def __init__(self, policy, replan_steps: int = 8) -> None:
        if not 1 <= replan_steps <= 16:
            raise ValueError("sync replan_steps must be between 1 and 16")
        self._policy = policy
        self._replan_steps = replan_steps
        self._chunk = None
        self._last_result = None
        self._index = 0
        self._chunk_id = 0
        self._closed = False

    def _log_execution(self) -> None:
        if self._index:
            logger.info("Sync executed #%d: indices=0..%d (zero-based), count=%d",
                        self._chunk_id, self._index - 1, self._index)

    def infer(self, observation: dict) -> dict:
        if self._closed:
            raise RuntimeError("Synchronous broker is closed")
        if self._chunk is None or self._index >= self._replan_steps:
            self._log_execution()
            # No prefix guidance is sent, even if an upstream caller supplied RTC fields.
            request = {key: value for key, value in observation.items() if not key.startswith("rtc_")}
            started = time.monotonic()
            result = self._policy.infer(request)
            actions = np.asarray(result["actions"])
            if actions.shape != (16, 16) or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite Nero action chunk with shape (16, 16), got {actions.shape}")
            self._chunk = np.array(actions, copy=True)
            self._last_result = dict(result)
            self._index = 0
            self._chunk_id += 1
            logger.info("Sync inference #%d: infer_ms=%.1f, execute_indices=0..%d",
                        self._chunk_id, (time.monotonic() - started) * 1000, self._replan_steps - 1)

        result = dict(self._last_result)
        result["actions"] = self._chunk[self._index].copy()
        # Keep the shared action-trace schema. These fields are local diagnostics,
        # not RTC conditioning fields passed to the server.
        result.update({
            "inference_mode": "sync",
            "rtc_chunk_id": self._chunk_id,
            "rtc_action_index": self._index,
            "rtc_chunk_changed": self._index == 0,
            "rtc_switch_delta": None,
            "rtc_pending": False,
            "rtc_queue_length": self._replan_steps - self._index - 1,
            "rtc_inference_delay": 0,
        })
        self._index += 1
        return result

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if reset is not None:
            reset()
        self._chunk = None
        self._last_result = None
        self._index = 0
        # Keep IDs monotonic across event-driven replans for unambiguous traces.

    def close(self) -> None:
        if not self._closed:
            self._log_execution()
        self._closed = True


def resolve_inference_settings(policy_cfg: dict, mode: str | None, replan_steps: int | None) -> tuple[str, int]:
    rtc_cfg = policy_cfg.get("rtc", {})
    selected = mode or policy_cfg.get("inference_mode") or ("rtc" if rtc_cfg.get("enabled", True) else "async")
    if selected not in ("rtc", "sync", "async"):
        raise ValueError(f"Unknown inference mode: {selected}")
    steps = replan_steps if replan_steps is not None else int(policy_cfg.get("sync_replan_steps", 8))
    if not 1 <= steps <= 16:
        raise ValueError("sync replan_steps must be between 1 and 16")
    if selected != "sync" and replan_steps is not None:
        raise ValueError("--replan-steps only applies to --inference-mode sync")
    return selected, steps


def create_inference_broker(policy, policy_cfg: dict, control_hz: float, mode: str, replan_steps: int):
    if mode == "sync":
        return SynchronousActionChunkBroker(policy, replan_steps=replan_steps)
    if mode not in ("rtc", "async"):
        raise ValueError(f"Unknown inference mode: {mode}")
    rtc_cfg = policy_cfg.get("rtc", {})
    return AsyncRTCActionChunkBroker(
        policy,
        action_horizon=16,
        execution_horizon=int(rtc_cfg.get("execution_horizon", 8)),
        control_hz=control_hz,
        trigger_horizon=int(rtc_cfg.get("trigger_horizon", rtc_cfg.get("execution_horizon", 8))),
        enable_rtc=mode == "rtc",
    )
