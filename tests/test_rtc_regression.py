from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from openpi_client import async_rtc_action_chunk_broker as rtc


@pytest.fixture
def broker_factory(monkeypatch):
    """Control worker completions and wall time without sleeps or network calls."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(rtc, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)

    def create(**kwargs):
        broker = rtc.AsyncRTCActionChunkBroker(SimpleNamespace(reset=lambda: None), **kwargs)
        return broker, clock

    return create


def _chunk(offset=0):
    return np.repeat((np.arange(16, dtype=np.float32) + offset)[:, None], 16, axis=1)


def _complete(broker, clock, *, seconds=0.2, offset=0, actions=None):
    request_id, obs, started = broker._request_queue.get_nowait()
    clock.now = started + seconds
    result = {"actions": _chunk(offset) if actions is None else actions}
    broker._result_queue.put((request_id, started, started, clock.now, result, None))
    return obs


def _seed(broker):
    broker._actions = _chunk()
    broker._last_result = {"actions": _chunk()}


@pytest.mark.parametrize("seconds", [0.2, 0.6, 5.0])
def test_startup_wait_never_skips_unexecuted_actions(broker_factory, seconds):
    broker, clock = broker_factory()
    broker._submit({"step": 0}, previous_actions=None)
    _complete(broker, clock, seconds=seconds)
    result = broker.infer({"step": 0})
    np.testing.assert_array_equal(result["actions"], _chunk()[0])
    assert result["rtc_timing"]["discard_steps"] == 0
    assert broker._request_queue.empty()


def test_prefix_starts_at_the_observation_tick_including_current_action(broker_factory):
    broker, _ = broker_factory()
    _seed(broker)
    for step in range(8):
        np.testing.assert_array_equal(broker.infer({"step": step})["actions"], _chunk()[step])
    assert broker._request_queue.empty()
    result = broker.infer({"step": 8})
    _, request, _ = broker._request_queue.get_nowait()
    assert request["step"] == 8
    np.testing.assert_array_equal(result["actions"], _chunk()[8])
    np.testing.assert_array_equal(request["rtc_prev_actions"], _chunk()[8:])
    assert request["rtc_execution_horizon"] == 8


def test_adoption_discards_executed_steps_instead_of_wall_clock_estimate(broker_factory):
    broker, clock = broker_factory()
    _seed(broker)
    broker._submit({"step": 0}, previous_actions=_chunk()[:8])
    for step in range(2):
        broker.infer({"step": step})
    _complete(broker, clock, seconds=0.2, offset=100)
    result = broker.infer({"step": 2})
    assert result["rtc_timing"]["discard_steps"] == 2
    np.testing.assert_array_equal(result["actions"], _chunk(100)[2])


def test_starvation_keeps_unexecuted_suffix_and_never_submits_empty_prefix(broker_factory):
    broker, clock = broker_factory()
    _seed(broker)
    broker._actions = _chunk()[:8]
    for step in range(8):
        broker.infer({"step": step})
    assert len(broker._actions) == 0
    _complete(broker, clock, seconds=0.6, offset=100)
    result = broker.infer({"step": 8})
    assert result["rtc_timing"]["discard_steps"] == 8
    np.testing.assert_array_equal(result["actions"], _chunk(100)[8])
    _, request, _ = broker._request_queue.get_nowait()
    assert request["rtc_prev_actions"].shape == (8, 16)
    np.testing.assert_array_equal(request["rtc_prev_actions"][0], result["actions"])


def test_fully_expired_result_is_rejected_instead_of_replaying_last_action(broker_factory):
    broker, clock = broker_factory()
    broker._submit({"step": 0}, previous_actions=_chunk()[:8])
    broker._step = 16
    _complete(broker, clock)
    assert not broker._consume_result(wait=False)
    assert broker._actions is None
    assert not broker._pending


def test_actual_overlap_length_is_sent_for_early_trigger(broker_factory):
    broker, _ = broker_factory(trigger_horizon=12)
    _seed(broker)
    for step in range(5):
        broker.infer({"step": step})
    _, request, _ = broker._request_queue.get_nowait()
    assert request["step"] == 4
    assert request["rtc_execution_horizon"] == 12
    np.testing.assert_array_equal(request["rtc_prev_actions"], _chunk()[4:])


@pytest.mark.parametrize("delays", [(5,), (4, 5, 4, 5), (1, 8, 5, 4)])
def test_multi_chunk_execution_matches_global_action_timestamps(broker_factory, delays):
    """A timestamp-coded oracle detects repeated, skipped or shifted actions."""
    broker, clock = broker_factory()
    _seed(broker)
    inflight = None
    request_number = 0
    executed = {}
    adoptions = []
    for step in range(80):
        clock.now = step / 30
        if inflight is not None and step == inflight[0]:
            _, request_id, request, started, delay = inflight
            broker._result_queue.put(
                (request_id, started, started, clock.now, {"actions": _chunk(request["step"])}, None)
            )
            adoptions.append((request_id, delay))
            inflight = None
        result = broker.infer({"step": step})
        np.testing.assert_array_equal(result["actions"], np.full(16, step))
        executed.setdefault(result["rtc_chunk_id"], []).append(result["rtc_action_index"])
        if result["rtc_chunk_changed"] and step:
            assert result["rtc_action_index"] == adoptions[-1][1]
            if adoptions[-1][1] == 8:
                assert result["rtc_switch_delta"] is None  # The old queue is exhausted.
            else:
                np.testing.assert_array_equal(result["rtc_switch_delta"], np.zeros(16))
        if not broker._request_queue.empty():
            request_id, request, started = broker._request_queue.get_nowait()
            delay = delays[request_number % len(delays)]
            request_number += 1
            # The prefix is the unexecuted old tail, aligned at this observation.
            assert request["step"] == step
            np.testing.assert_array_equal(request["rtc_prev_actions"][:, 0], np.arange(step, step + 8))
            inflight = (step + delay, request_id, request, started, delay)
    for indices in executed.values():
        assert indices == list(range(indices[0], indices[-1] + 1))
    for (request_id, delay), (_, next_delay) in zip(adoptions[:-1], adoptions[1:], strict=True):
        # Native RTC: used_k = request_interval + d_(k+1) - d_k.
        assert executed[request_id] == list(range(delay, 8 + next_delay))
    if delays == (5,):
        assert executed[0] == list(range(13))
        assert executed[1] == list(range(5, 13))
        assert executed[2] == list(range(5, 13))


def test_switch_diagnostic_compares_targets_at_same_tick(broker_factory):
    broker, clock = broker_factory()
    _seed(broker)
    broker._submit({"step": 0}, previous_actions=_chunk()[:8])
    for step in range(5):
        broker.infer({"step": step})
    _complete(broker, clock, offset=100)
    result = broker.infer({"step": 5})
    assert result["rtc_action_index"] == 5
    assert result["rtc_chunk_changed"]
    np.testing.assert_array_equal(result["rtc_switch_delta"], np.full(16, 100))
    following = broker.infer({"step": 6})
    assert following["rtc_action_index"] == 6
    assert not following["rtc_chunk_changed"]
    assert "rtc_switch_delta" not in following


def test_disabling_guidance_preserves_asynchronous_replanning(broker_factory):
    broker, _ = broker_factory(enable_rtc=False)
    _seed(broker)
    for step in range(9):
        broker.infer({"step": step})
    _, request, _ = broker._request_queue.get_nowait()
    assert request == {"step": 8}
    assert broker._pending


def test_reset_ignores_late_results_and_clears_latency_history(broker_factory):
    broker, clock = broker_factory()
    broker._submit({"step": 0}, previous_actions=None)
    stale = broker._request_queue.get_nowait()
    broker._delay_history.append(6)
    broker._estimated_delay_steps = 6
    broker.reset()
    assert broker._estimated_delay_steps == 0
    assert not broker._delay_history
    broker._submit({"step": 0}, previous_actions=None)
    broker._result_queue.put((stale[0], 0.0, 0.0, 0.0, {"actions": _chunk(100)}, None))
    _complete(broker, clock, offset=200)
    np.testing.assert_array_equal(broker.infer({"step": 0})["actions"], _chunk(200)[0])


@pytest.mark.parametrize("shape", [(0, 16), (8, 16), (16,), (16, 0)])
def test_malformed_chunk_is_rejected(broker_factory, shape):
    broker, clock = broker_factory()
    broker._submit({"step": 0}, previous_actions=None)
    _complete(broker, clock, actions=np.zeros(shape))
    with pytest.raises(ValueError, match="Expected action chunk"):
        broker.infer({"step": 0})


def test_worker_reset_is_serialized_with_inflight_policy_call():
    entered = threading.Event()
    release = threading.Event()
    events = queue.Queue()

    class Policy:
        def infer(self, obs):
            if obs["episode"] == 0:
                entered.set()
                assert release.wait(timeout=5)
            events.put(("infer", obs["episode"]))
            return {"actions": _chunk(obs["episode"] * 100)}

        def reset(self):
            events.put(("reset", None))

    broker = rtc.AsyncRTCActionChunkBroker(Policy())
    try:
        broker._submit({"episode": 0}, previous_actions=None)
        assert entered.wait(timeout=5)
        with ThreadPoolExecutor(max_workers=1) as pool:
            resetting = pool.submit(broker.reset)
            release.set()
            resetting.result(timeout=5)
        result = broker.infer({"episode": 1})
        np.testing.assert_array_equal(result["actions"], _chunk(100)[0])
        assert [events.get_nowait() for _ in range(3)] == [("infer", 0), ("reset", None), ("infer", 1)]
    finally:
        release.set()
        broker.close()


def test_worker_exception_is_propagated_and_close_stops_idle_thread():
    class Policy:
        def infer(self, obs):
            raise ValueError("bad RTC request")

    broker = rtc.AsyncRTCActionChunkBroker(Policy())
    try:
        with pytest.raises(RuntimeError, match="RTC inference worker failed") as error:
            broker.infer({})
        assert isinstance(error.value.__cause__, ValueError)
    finally:
        broker.close()
    assert not broker._worker.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        broker.infer({})
