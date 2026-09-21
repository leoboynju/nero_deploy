"""Coordinate discrete gripper events with joint tracking and bounded holds."""

from dataclasses import dataclass
import logging

import numpy as np

from .filter import GripperDebouncer


logger = logging.getLogger(__name__)
ARMS = ("left", "right")


@dataclass(frozen=True)
class GraspConfig:
    enabled: bool = True
    confirm_seconds: float = 0.08  # Legacy config accepted; confirmation now uses open=2, close=1 samples.
    min_hold_seconds: float = 0.30
    reach_tolerance_rad: float = 0.025
    reach_stable_seconds: float = 0.06
    reach_timeout_seconds: float = 1.5
    close_timeout_seconds: float = 1.0
    settle_seconds: float = 0.12
    closed_width_m: float = 0.005
    contact_force_n: float = 0.5

    def __post_init__(self):
        values = [value for name, value in vars(self).items() if name not in ("enabled", "confirm_seconds")]
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Grasp timing, tolerance and feedback thresholds must be finite and positive")
        if self.close_timeout_seconds < self.min_hold_seconds:
            raise ValueError("close_timeout_seconds must be >= min_hold_seconds")


def resolve_grasp_config(mode, options, override=None):
    """RTC execution experiments must never change the no-RTC baseline."""
    if mode != "rtc":
        if override is True:
            raise ValueError("--grasp-control is only available in rtc mode; no-RTC uses the original gripper path")
        return GraspConfig(enabled=False)
    settings = dict(options)
    if override is not None:
        settings["enabled"] = override
    return GraspConfig(**settings)


@dataclass
class GraspDecision:
    action: np.ndarray
    gripper_open: dict[str, bool]
    held: bool
    replan: bool
    phase: str
    event: str = ""


class GraspCoordinator:
    def __init__(self, config: GraspConfig):
        self.config = config
        self.phase = "tracking"
        self._open = np.ones(2, dtype=bool)
        self._candidate = self._open.copy()
        self._gripper_filter = GripperDebouncer(confirm_steps=2)
        self._candidate_joints = np.zeros((2, 7), dtype=np.float32)
        self._cooldown_until = -np.inf
        self._target = None
        self._closing = np.zeros(2, dtype=bool)
        self._since = 0.0
        self._reach_since = None
        self._cached_result = None

    @property
    def active(self):
        return self.phase != "tracking"

    def initialize(self, state, now):
        self._open = np.asarray(state)[[7, 15]] >= 0.05
        self._candidate = self._open.copy()
        self._gripper_filter.reset(self._open)

    def infer(self, broker, observation):
        """Do not advance the policy cursor while executing a latched grasp pose."""
        if not self.active:
            self._cached_result = broker.infer(observation)
        result = dict(self._cached_result)
        if self.active:
            result["rtc_chunk_changed"] = False
            result.pop("rtc_switch_delta", None)
        return result

    def after_publish(self, broker, decision):
        if decision.replan:
            broker.reset()
            self._cached_result = None

    def _decision(self, action, held=False, replan=False, event=""):
        if event:
            logger.info("Grasp event=%s phase=%s arms=%s", event, self.phase,
                        [arm for arm, selected in zip(ARMS, self._closing) if selected])
        return GraspDecision(np.array(action, dtype=np.float32, copy=True),
                             dict(zip(ARMS, map(bool, self._open))), held, replan, self.phase, event)

    def _finish(self, now, event):
        self.phase = "tracking"
        self._cooldown_until = now + self.config.min_hold_seconds if event == "reach_timeout" else now
        self._candidate = self._open.copy()
        self._gripper_filter.reset(self._open)
        # Keep publishing the held pose on this final tick. The caller discards
        # in-flight/queued predictions and replans from fresh feedback next tick.
        return self._decision(self._target, held=True, replan=True, event=event)

    def step(self, action, state, now, forces=None):
        state = np.asarray(state)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Grasp coordinator requires finite 16D feedback")
        cfg = self.config
        if self.phase == "approach":
            errors = [np.max(np.abs(self._target[offset:offset + 7] - state[offset:offset + 7]))
                      for offset, selected in zip((0, 8), self._closing) if selected]
            if max(errors) <= cfg.reach_tolerance_rad:
                if self._reach_since is None:
                    self._reach_since = now
                if now - self._reach_since + 1e-9 >= cfg.reach_stable_seconds:
                    self._open[self._closing] = False
                    self.phase = "closing"
                    self._since = now
                    self._initial_width = state[[7, 15]].copy()
                    self._width_ref = self._initial_width.copy()
                    self._width_since = np.full(2, now)
                    return self._decision(self._target, held=True, event="close_sent")
            else:
                self._reach_since = None
            if now - self._since >= cfg.reach_timeout_seconds:
                return self._finish(now, "reach_timeout")
            return self._decision(self._target, held=True)

        if self.phase in ("closing", "opening"):
            width = state[[7, 15]]
            moved = np.abs(width - self._width_ref) > 0.001
            self._width_ref[moved] = width[moved]
            self._width_since[moved] = now
            force = np.full(2, np.nan) if forces is None else np.asarray(forces)
            opening_phase = self.phase == "opening"
            progressed = (width - self._initial_width if opening_phase else self._initial_width - width) >= 0.002
            stable = now - self._width_since >= cfg.settle_seconds
            contact = np.isfinite(force) & (force >= cfg.contact_force_n)
            complete = ((width >= 0.095) | (progressed & stable) if opening_phase
                        else (width <= cfg.closed_width_m)
                        | (progressed & stable & (contact | ~np.isfinite(force))))
            if now - self._since >= cfg.min_hold_seconds and complete[self._closing].all():
                return self._finish(now, "open_settled" if opening_phase else "close_settled")
            if now - self._since >= cfg.close_timeout_seconds:
                # A timeout is not evidence of a successful grasp. Keep the last
                # gripper command, resume visual replanning, and allow a confirmed release.
                return self._finish(now, "open_timeout" if opening_phase else "close_timeout")
            return self._decision(self._target, held=True)

        if action is None or np.shape(action) != (16,) or not np.isfinite(action).all():
            raise ValueError("Tracking requires a finite 16D policy action")
        requested = np.asarray(action)[[7, 15]] >= 0.5
        changed = requested != self._candidate
        for arm, offset in enumerate((0, 8)):
            if changed[arm] and not requested[arm]:
                # Close is accepted on its first sample; preserve that grasp pose.
                self._candidate_joints[arm] = np.asarray(action)[offset:offset + 7]
        self._candidate = requested.copy()
        filtered = self._gripper_filter.update(action, current_open=self._open)
        confirmed = np.asarray([filtered[arm] for arm in ARMS])
        ready = confirmed != self._open
        opening = ready & confirmed
        self._open[opening] = True
        closing = ready & ~confirmed
        if closing.any() and now >= self._cooldown_until:
            # Latch the actual close-event pose, not subsequent lift/retract targets.
            # Hold both arms to preserve bimanual timing while the gripper responds.
            self._target = np.array(action, dtype=np.float32, copy=True)
            for arm, offset in enumerate((0, 8)):
                if closing[arm]:
                    self._target[offset:offset + 7] = self._candidate_joints[arm]
            self._closing = closing
            self.phase = "approach"
            self._since = now
            self._reach_since = None
            return self._decision(self._target, held=True, event="close_pending")
        if opening.any():
            self._target = np.array(action, dtype=np.float32, copy=True)
            self._closing = opening  # Active gripper channels for this motion phase.
            self.phase = "opening"
            self._since = now
            self._initial_width = state[[7, 15]].copy()
            self._width_ref = self._initial_width.copy()
            self._width_since = np.full(2, now)
            return self._decision(self._target, held=True, event="open_sent")
        return self._decision(action, event="open_sent" if opening.any() else "")
