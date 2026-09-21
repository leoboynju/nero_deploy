from types import SimpleNamespace

import numpy as np
import pytest

from nero_deploy.control import inference


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.resets = 0

    def infer(self, observation):
        self.calls.append(observation)
        chunk = np.repeat((100 * len(self.calls) + np.arange(16))[:, None], 16, axis=1)
        return {"actions": chunk.astype(np.float32), "policy_timing": {"infer_ms": 150}}

    def reset(self):
        self.resets += 1


@pytest.mark.parametrize("steps", [5, 8, 16])
def test_sync_executes_leading_actions_then_replans_from_fresh_observation(steps):
    policy = FakePolicy()
    broker = inference.SynchronousActionChunkBroker(policy, replan_steps=steps)
    for tick in range(2 * steps):
        result = broker.infer({"step": tick})
        chunk = tick // steps + 1
        index = tick % steps
        np.testing.assert_array_equal(result["actions"], np.full(16, chunk * 100 + index))
        assert result["rtc_chunk_id"] == chunk
        assert result["rtc_action_index"] == index
        assert result["rtc_chunk_changed"] == (index == 0)
        assert result["rtc_queue_length"] == steps - index - 1
        assert result["rtc_inference_delay"] == 0
        assert not result["rtc_pending"]
        assert result["policy_timing"]["infer_ms"] == 150
    assert policy.calls == [{"step": 0}, {"step": steps}]
    broker.close()


def test_sync_strips_rtc_fields_and_does_not_skip_actions_during_blocking_wait(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(inference, "time", SimpleNamespace(monotonic=lambda: clock.now))
    policy = FakePolicy()
    original_infer = policy.infer

    def delayed(observation):
        clock.now += 0.6
        return original_infer(observation)

    policy.infer = delayed
    broker = inference.SynchronousActionChunkBroker(policy)
    result = broker.infer({"step": 0, "rtc_prev_actions": np.zeros((8, 16)),
                           "rtc_inference_delay": 5, "rtc_execution_horizon": 8})
    assert policy.calls == [{"step": 0}]
    np.testing.assert_array_equal(result["actions"], np.full(16, 100))
    assert result["rtc_action_index"] == 0


def test_sync_reset_and_close():
    policy = FakePolicy()
    broker = inference.SynchronousActionChunkBroker(policy)
    result = broker.infer({"step": 0})
    result["actions"][:] = -100  # Caller edits cannot corrupt the queued chunk.
    np.testing.assert_array_equal(broker.infer({"step": 1})["actions"], np.full(16, 101))
    broker.reset()
    assert policy.resets == 1
    assert broker.infer({"step": 2})["rtc_action_index"] == 0
    broker.close()
    broker.close()
    with pytest.raises(RuntimeError, match="closed"):
        broker.infer({})


@pytest.mark.parametrize("shape", [(8, 16), (16, 7), (16,), (0, 16)])
def test_sync_rejects_malformed_predictions(shape):
    policy = SimpleNamespace(infer=lambda obs: {"actions": np.zeros(shape)})
    with pytest.raises(ValueError, match="Expected finite Nero action chunk"):
        inference.SynchronousActionChunkBroker(policy).infer({})


@pytest.mark.parametrize("steps", [0, 17, -1])
def test_invalid_replan_length_is_rejected(steps):
    with pytest.raises(ValueError, match="between 1 and 16"):
        inference.resolve_inference_settings({}, "sync", steps)


def test_explicit_modes_override_yaml_and_keep_legacy_unguided_async_choice(monkeypatch):
    cfg = {"rtc": {"enabled": False, "execution_horizon": 8, "trigger_horizon": 8}}
    assert inference.resolve_inference_settings(cfg, None, None) == ("async", 8)
    assert inference.resolve_inference_settings(cfg, "rtc", None) == ("rtc", 8)
    assert inference.resolve_inference_settings(cfg, "sync", 5) == ("sync", 5)
    monkeypatch.setattr(inference, "AsyncRTCActionChunkBroker", lambda policy, **kwargs: SimpleNamespace(**kwargs))
    assert inference.create_inference_broker(FakePolicy(), cfg, 30, "rtc", 8).enable_rtc
    assert not inference.create_inference_broker(FakePolicy(), cfg, 30, "async", 8).enable_rtc
    assert isinstance(inference.create_inference_broker(FakePolicy(), cfg, 30, "sync", 8),
                      inference.SynchronousActionChunkBroker)
    with pytest.raises(ValueError, match="only applies"):
        inference.resolve_inference_settings(cfg, "rtc", 5)


def test_sync_output_is_compatible_with_action_tracing(tmp_path):
    from nero_deploy.diagnostics.action_trace import ActionTrace

    trace = ActionTrace(tmp_path, {"inference_mode": "sync", "sync_replan_steps": 8})
    result = inference.SynchronousActionChunkBroker(FakePolicy()).infer({})
    trace.record(timestamp=0.0, state=np.zeros(16), policy_action=result["actions"],
                 ema_action=np.zeros(16), command=np.zeros(16), result=result, state_age=0.01)
    with np.load(trace.save(), allow_pickle=False) as data:
        assert data["action_index"].tolist() == [0]
