import numpy as np
import pytest

from nero_deploy.control.safety import clamp_action, clamp_action_chunk, clamp_action_step, validate_action_chunk


def test_valid_dual_arm_chunk() -> None:
    validate_action_chunk(np.zeros((16, 16), dtype=np.float32), np.zeros(16), 0.3, 0.01)


def test_rejects_wrong_shape() -> None:
    with pytest.raises(RuntimeError, match="shape"):
        validate_action_chunk(np.zeros((16, 8)), np.zeros(16), 0.3, 0.01)


def test_rejects_joint_jump() -> None:
    actions = np.zeros((16, 16), dtype=np.float32)
    actions[0, 0] = 0.5
    with pytest.raises(RuntimeError, match="jump"):
        validate_action_chunk(actions, np.zeros(16), 0.3, 0.01)


def test_clamps_out_of_range_joints_and_preserves_grippers() -> None:
    actions = np.zeros((16, 16), dtype=np.float32)
    actions[:, 0] = 9.0
    actions[:, 7] = 1.0
    actions[:, 15] = 1.0
    clamped, count, max_clip = clamp_action_chunk(actions, 0.01)
    assert count == 16
    assert max_clip > 0.0
    assert clamped[:, 0].max() == pytest.approx(2.70526 - 0.01)
    np.testing.assert_allclose(clamped[:, [7, 15]], 1.0)
    validate_action_chunk(clamped, np.zeros(16, dtype=np.float32), 10.0, 0.01)


def test_clamps_single_broker_action() -> None:
    action = np.zeros(16, dtype=np.float32)
    action[0] = 9.0
    action[7] = 1.0
    clamped, count, _ = clamp_action(action, 0.01)
    assert count == 1
    assert clamped[0] == pytest.approx(2.70526 - 0.01)
    assert clamped[7] == 1.0


def test_clamps_single_action_step_to_feedback() -> None:
    action = np.zeros(16, dtype=np.float32)
    action[0] = 1.0
    action[7] = 1.0
    clamped, count, _ = clamp_action_step(action, np.zeros(16, dtype=np.float32), 0.3)
    assert count == 1
    assert clamped[0] == pytest.approx(0.3)
    assert clamped[7] == 1.0
