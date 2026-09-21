import numpy as np
import time

from openpi_client.async_rtc_action_chunk_broker import AsyncRTCActionChunkBroker


class FakePolicy:
    def __init__(self) -> None:
        self.calls = []

    def infer(self, observation):
        self.calls.append(observation)
        value = float(len(self.calls))
        return {"actions": np.full((16, 16), value, dtype=np.float32)}


def test_broker_returns_single_action_and_replans_with_rtc_fields() -> None:
    policy = FakePolicy()
    broker = AsyncRTCActionChunkBroker(policy, control_hz=1000.0)
    try:
        observation = {"images": {}, "state": np.zeros(16), "prompt": "test"}
        first = broker.infer(observation)
        assert first["actions"].shape == (16,)
        for _ in range(8):
            broker.infer(observation)
        deadline = time.monotonic() + 1.0
        while len(policy.calls) < 2 and time.monotonic() < deadline:
            broker.infer(observation)
            time.sleep(0.01)
        assert len(policy.calls) >= 2
        assert "rtc_prev_actions" in policy.calls[-1]
        assert policy.calls[-1]["rtc_prev_actions"].shape[1] == 16
    finally:
        broker.close()


def test_broker_can_disable_server_side_rtc() -> None:
    policy = FakePolicy()
    broker = AsyncRTCActionChunkBroker(policy, enable_rtc=False)
    try:
        observation = {"images": {}, "state": np.zeros(16), "prompt": "test"}
        for _ in range(9):
            broker.infer(observation)
        assert all("rtc_prev_actions" not in call for call in policy.calls)
    finally:
        broker.close()
