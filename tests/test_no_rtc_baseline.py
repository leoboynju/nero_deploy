import importlib.util
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nero_deploy.control.grasp import resolve_grasp_config


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_shared_rtc_grasp_settings_cannot_enable_no_rtc_coordination(mode):
    # Even irrelevant invalid RTC knobs must not affect the working baseline.
    assert not resolve_grasp_config(mode, {"enabled": True, "confirm_seconds": 0}).enabled
    with pytest.raises(ValueError, match="only available in rtc"):
        resolve_grasp_config(mode, {"enabled": True}, True)
    assert resolve_grasp_config("rtc", {"enabled": True}).enabled
    assert not resolve_grasp_config("rtc", {"enabled": True}, False).enabled


@pytest.mark.parametrize("gripper_values,expected", [([0.0, 1.0, 0.0, 1.0, 1.0], [False, False, False, False, True]),
                                                   ([0.49, 0.5, 0.51], [False, False, True])])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_actual_no_rtc_cli_confirms_only_opening_and_preserves_joint_feedback(monkeypatch, gripper_values, expected, mode):
    steps = len(gripper_values)
    published = []
    feedback = np.zeros(16, dtype=np.float32)
    feedback[[7, 15]] = 0.1
    joint_indices = [*range(7), *range(8, 15)]
    calls = {"state": 0, "infer": 0}

    class Bridge:
        def state(self):
            calls["state"] += 1
            return feedback.copy(), 0.0

        def gripper_forces(self):
            return np.full(2, np.nan)

        def validate_exclusive_control(self):
            pass

        def publish(self, action, grippers):
            published.append((action.copy(), dict(grippers)))

        def destroy_node(self):
            pass

    class Camera:
        def __init__(self, *args, **kwargs):
            pass

        def snapshot(self):
            return np.zeros((2, 2, 3), dtype=np.uint8), 0.0

        def close(self):
            pass

    class Policy:
        def __init__(self, *args):
            pass

        def infer(self, observation):
            calls["infer"] += 1
            # Feedback changes during blocking inference. The original baseline
            # seeds EMA from the captured observation, not this newer reading.
            feedback[joint_indices] = 0.2
            actions = np.full((16, 16), 0.25, dtype=np.float32)
            actions[:, [7, 15]] = 1.0
            actions[:steps, 15] = gripper_values
            actions[:steps, 7] = [1, 0, 1, 1, 0][:steps]
            return {"actions": actions}

    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(init=lambda: None, spin=lambda node: None, try_shutdown=lambda: None))
    monkeypatch.setitem(sys.modules, "rclpy.executors", SimpleNamespace(
        ExternalShutdownException=type("ExternalShutdownException", (Exception,), {}),
    ))
    monkeypatch.setitem(sys.modules, "nero_deploy.cameras", SimpleNamespace(RealSenseCamera=Camera))
    monkeypatch.setitem(sys.modules, "nero_deploy.ros", SimpleNamespace(DualArmBridge=Bridge))
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    path = Path(__file__).resolve().parents[1] / "src/nero_deploy/cli.py"
    spec = importlib.util.spec_from_file_location("nero_deploy._baseline_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "threading", SimpleNamespace(
        Event=cli.threading.Event,
        Thread=lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda timeout: None),
    ))
    config = {
        "policy": {"host": "test", "port": 8000, "prompt": "test", "rtc": {"enabled": True}},
        "cameras": {"width": 2, "height": 2, "fps": 30, "warmup_seconds": 0,
                    **{name: {"serial": "test"} for name in ("left_wrist", "right_wrist", "third_person")}},
        "control": {"hz": 1000, "max_steps": steps, "max_source_age": 0.25,
                    "max_joint_step": 0.3, "joint_limit_margin": 0.01,
                    "joint_ema": {"enabled": True, "alpha": 0.35},
                    "gripper_confirm_steps": 0, "action_trace": {"enabled": False},
                    "grasp": {"enabled": True, "confirm_seconds": 0}},
    }
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "PolicyClient", Policy)

    def unexpected_coordinator(*args):
        raise AssertionError("No-RTC must not construct GraspCoordinator")

    monkeypatch.setattr(cli, "GraspCoordinator", unexpected_coordinator)
    monkeypatch.setattr(sys, "argv", ["cli", "--inference-mode", mode, "--execute", "--max-steps", str(steps)])
    cli.main()
    assert calls == {"state": steps + 1, "infer": 1}  # Readiness + one feedback read per tick.
    assert [grippers["right"] for _, grippers in published] == expected
    assert [grippers["left"] for _, grippers in published] == [True, False, False, True, False][:steps]
    np.testing.assert_allclose(published[0][0][joint_indices], 0.0875, atol=1e-7)
