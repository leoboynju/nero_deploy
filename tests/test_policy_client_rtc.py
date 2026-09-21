from types import SimpleNamespace

import numpy as np
import pytest

from nero_deploy import policy_client


def _observation():
    return {
        "images": {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in ("left_wrist", "right_wrist", "third_person")},
        "state": np.zeros(16),
        "prompt": "test",
        "rtc_prev_actions": np.zeros((8, 16)),
        "rtc_inference_delay": 3,
        "rtc_execution_horizon": 8,
    }


def test_rtc_errors_are_not_silently_retried_as_ordinary_inference(monkeypatch):
    calls = []
    connections = []

    def infer(observation):
        calls.append(observation)
        raise RuntimeError("RTC gradient failure")

    def connect(*args):
        connections.append(args)
        return SimpleNamespace(infer=infer)

    monkeypatch.setattr(policy_client.websocket_client_policy, "WebsocketClientPolicy", connect)
    client = policy_client.PolicyClient("test", 8000, image_size=None)
    with pytest.raises(RuntimeError, match="RTC gradient failure"):
        client.infer(_observation())
    assert len(connections) == 1
    assert len(calls) == 1
    assert "rtc_prev_actions" in calls[0]


def test_rtc_fields_and_output_clamping_are_preserved(monkeypatch):
    calls = []

    def infer(observation):
        calls.append(observation)
        return {"actions": np.ones((16, 16)), "policy_timing": {"infer_ms": 100},
                "rtc_guided_action_indices": [*range(7), *range(8, 15)]}

    monkeypatch.setattr(
        policy_client.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *args: SimpleNamespace(infer=infer),
    )
    monkeypatch.setattr(policy_client, "clamp_action_chunk", lambda actions, margin: (actions * 0.5, 14, 0.5))
    result = policy_client.PolicyClient("test", 8000, image_size=None).infer(_observation())
    assert calls[0]["rtc_prev_actions"].shape == (8, 16)
    assert calls[0]["rtc_prev_actions"].dtype == np.float32
    assert calls[0]["rtc_inference_delay"] == 3
    assert result["safety_clipped_values"] == 14
    assert result["policy_timing"]["infer_ms"] == 100
    np.testing.assert_array_equal(result["actions"], np.full((16, 16), 0.5))


def test_old_server_without_joint_mask_confirmation_is_rejected(monkeypatch):
    monkeypatch.setattr(policy_client.websocket_client_policy, "WebsocketClientPolicy",
                        lambda *args: SimpleNamespace(infer=lambda obs: {"actions": np.zeros((16, 16))}))
    with pytest.raises(RuntimeError, match="Restart the updated policy server"):
        policy_client.PolicyClient("test", 8000, image_size=None).infer(_observation())
