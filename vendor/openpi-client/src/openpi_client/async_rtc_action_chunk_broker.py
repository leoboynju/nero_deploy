import copy
from collections import deque
import logging
import math
import queue
import threading
import time
from typing import Dict, Optional

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


class AsyncRTCActionChunkBroker(_base_policy.BasePolicy):
    """Run inference while executing one action per call.

    Call ``infer(observation)`` immediately before executing its returned action,
    and supply a fresh observation on the next control tick. Chunk indices advance
    by returned actions, never by elapsed wall time (which includes blocking waits).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int = 16,
        execution_horizon: int = 8,
        control_hz: float = 30.0,
        trigger_horizon: Optional[int] = None,
        enable_rtc: bool = True,
    ):
        if action_horizon != 16 or execution_horizon != 8:
            raise ValueError("AsyncRTCActionChunkBroker currently requires action_horizon=16 and execution_horizon=8")
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self._policy = policy
        self._action_horizon = action_horizon
        self._execution_horizon = execution_horizon
        self._control_hz = control_hz
        self._trigger_horizon = execution_horizon if trigger_horizon is None else trigger_horizon
        self._enable_rtc = enable_rtc
        if not 1 <= self._trigger_horizon <= action_horizon:
            raise ValueError("trigger_horizon must be between 1 and action_horizon")

        self._actions: Optional[np.ndarray] = None
        self._last_result: Optional[Dict] = None
        self._pending = False
        self._request_id = 0
        self._active_request_id: Optional[int] = None
        self._step = 0
        self._request_step = 0
        self._chunk_id = 0
        self._chunk_start_step = 0
        self._chunk_first_index = 0
        self._chunk_returned = 0
        self._estimated_delay_steps = 0
        self._delay_history = deque(maxlen=10)
        self._request_queue = queue.Queue(maxsize=1)
        # A stale result after reset must never block publication of the next result.
        self._result_queue = queue.Queue()
        self._policy_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._inference_worker, name="openpi-rtc-inference", daemon=True)
        self._worker.start()

    def _inference_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                request_id, observation, started_at = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            worker_started_at = time.monotonic()
            try:
                with self._policy_lock:
                    if request_id != self._active_request_id or self._stop_event.is_set():
                        continue
                    result = self._policy.infer(observation)
                item = (request_id, started_at, worker_started_at, time.monotonic(), result, None)
            except BaseException as error:  # Propagate inference failures to the control thread.
                item = (request_id, started_at, worker_started_at, time.monotonic(), None, error)
            self._result_queue.put(item)

    def _submit(self, obs: Dict, previous_actions: Optional[np.ndarray]) -> None:
        request = copy.deepcopy(obs)
        if self._enable_rtc and previous_actions is not None and len(previous_actions):
            request["rtc_prev_actions"] = np.array(previous_actions, copy=True)
            request["rtc_inference_delay"] = self._estimated_delay_steps
            # This wire field is the prefix attention horizon, i.e. the actual
            # overlap length, which may differ from the default execution horizon.
            request["rtc_execution_horizon"] = len(previous_actions)
            if self._estimated_delay_steps >= len(previous_actions):
                logger.warning(
                    "RTC delay estimate %d reaches the %d-step overlap; the control loop may need to wait",
                    self._estimated_delay_steps,
                    len(previous_actions),
                )

        self._request_id += 1
        request_id = self._request_id
        started_at = time.monotonic()
        self._active_request_id = request_id
        self._request_step = self._step
        self._pending = True
        self._request_queue.put_nowait((request_id, request, started_at))

    def _consume_result(self, *, wait: bool) -> bool:
        while True:
            try:
                item = self._result_queue.get(timeout=None if wait and self._pending else 0)
            except queue.Empty:
                return False
            if item[0] == self._active_request_id:
                break

        consumed_at = time.monotonic()
        request_id, started_at, worker_started_at, completed_at, result, error = item
        self._pending = False
        self._active_request_id = None
        if error is not None:
            raise RuntimeError("RTC inference worker failed") from error

        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or actions.shape[0] != self._action_horizon or actions.shape[1] < 1:
            raise ValueError(f"Expected action chunk with shape ({self._action_horizon}, A), got {actions.shape}")

        worker_queue_ms = (worker_started_at - started_at) * 1000
        client_round_trip_ms = (completed_at - worker_started_at) * 1000
        result_poll_ms = (consumed_at - completed_at) * 1000
        total_adoption_ms = (consumed_at - started_at) * 1000
        server_infer_ms = float(result.get("server_timing", {}).get("infer_ms", 0.0))
        transport_ms = max(0.0, client_round_trip_ms - server_infer_ms)

        discard_steps = self._step - self._request_step
        # Predict latency conservatively for guidance only. This estimate must not
        # determine the action cursor: startup/queue-starvation waits execute no actions.
        delay_steps = max(discard_steps, math.ceil((completed_at - started_at) * self._control_hz))
        self._delay_history.append(delay_steps)
        self._estimated_delay_steps = max(self._delay_history)
        if discard_steps >= len(actions):
            logger.warning("Discarding RTC result #%d: all %d actions have expired", request_id, len(actions))
            return False
        switch_delta = None
        overlap_remaining = 0 if self._actions is None else len(self._actions)
        if overlap_remaining:
            # Compare OLD and NEW targets for the SAME control tick, not the old
            # last command against a future target. These are policy-space values.
            switch_delta = np.array(actions[discard_steps] - self._actions[0], copy=True)
        if self._chunk_returned:
            self._log_chunk_execution()
        self._actions = np.array(actions[discard_steps:], copy=True)
        self._chunk_id = request_id
        self._chunk_start_step = self._request_step
        self._chunk_first_index = discard_steps
        self._chunk_returned = 0
        self._last_result = dict(result)
        self._last_result["rtc_switch_delta"] = switch_delta
        self._last_result["rtc_timing"] = {
            "worker_queue_ms": worker_queue_ms,
            "client_round_trip_ms": client_round_trip_ms,
            "server_infer_ms": server_infer_ms,
            "transport_ms": transport_ms,
            "result_poll_ms": result_poll_ms,
            "total_adoption_ms": total_adoption_ms,
            "inference_delay_steps": self._estimated_delay_steps,
            "discard_steps": discard_steps,
        }
        logger.info(
            "RTC timing #%d: worker_queue_ms=%.1f, client_round_trip_ms=%.1f, server_infer_ms=%.1f, "
            "transport_ms=%.1f, result_poll_ms=%.1f, total_adoption_ms=%.1f, inference_delay_steps=%d, "
            "discard_steps=%d",
            request_id,
            worker_queue_ms,
            client_round_trip_ms,
            server_infer_ms,
            transport_ms,
            result_poll_ms,
            total_adoption_ms,
            self._estimated_delay_steps,
            discard_steps,
        )
        logger.info(
            "RTC adopt #%d: observation_step=%d, control_step=%d, start_index=%d, "
            "available_steps=%d, old_overlap_remaining=%d",
            request_id,
            self._request_step,
            self._step,
            discard_steps,
            len(self._actions),
            overlap_remaining,
        )
        return True

    def _log_chunk_execution(self) -> None:
        logger.info(
            "RTC executed #%d: indices=%d..%d (zero-based), count=%d",
            self._chunk_id,
            self._chunk_first_index,
            self._chunk_first_index + self._chunk_returned - 1,
            self._chunk_returned,
        )

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._stop_event.is_set():
            raise RuntimeError("RTC broker is closed")
        self._consume_result(wait=False)

        while self._actions is None or len(self._actions) == 0:
            if not self._pending:
                self._submit(obs, previous_actions=None)
            else:
                logger.warning("RTC action queue exhausted; waiting without advancing the action cursor")
            self._consume_result(wait=True)

        assert self._actions is not None
        assert self._last_result is not None
        # obs describes the state BEFORE executing this tick's action. Submit
        # before popping so the first prefix action has that same timestamp.
        if len(self._actions) <= self._trigger_horizon and not self._pending:
            self._submit(obs, previous_actions=self._actions)

        action = np.array(self._actions[0], copy=True)
        self._actions = self._actions[1:]
        self._step += 1

        result = dict(self._last_result)
        result["actions"] = action
        result["rtc_chunk_id"] = self._chunk_id
        result["rtc_action_index"] = self._step - 1 - self._chunk_start_step
        result["rtc_chunk_changed"] = self._chunk_returned == 0
        if self._chunk_returned:
            result.pop("rtc_switch_delta", None)
        self._chunk_returned += 1
        result["rtc_pending"] = self._pending
        result["rtc_queue_length"] = len(self._actions)
        result["rtc_inference_delay"] = self._estimated_delay_steps
        return result

    @override
    def reset(self) -> None:
        self._actions = None
        self._last_result = None
        self._pending = False
        self._active_request_id = None
        self._step = 0
        self._request_step = 0
        self._chunk_id = 0
        self._chunk_start_step = 0
        self._chunk_first_index = 0
        self._chunk_returned = 0
        self._estimated_delay_steps = 0
        self._delay_history.clear()
        for pending_queue in (self._request_queue, self._result_queue):
            while True:
                try:
                    pending_queue.get_nowait()
                except queue.Empty:
                    break
        # Serialize reset with any in-flight websocket call. Request IDs are not
        # reused, so a late result from the old episode will be ignored.
        with self._policy_lock:
            reset = getattr(self._policy, "reset", None)
            if reset is not None:
                reset()

    def close(self) -> None:
        if not self._stop_event.is_set() and self._chunk_returned:
            self._log_chunk_execution()
        self._stop_event.set()
        self._worker.join(timeout=1.0)
