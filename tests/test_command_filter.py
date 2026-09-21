import numpy as np

from nero_deploy.control.filter import DualArmJointEMA, GripperDebouncer
from nero_deploy.control.safety import clamp_action_step


def test_ema_next_update_uses_limited_command_instead_of_unexecuted_target():
    state = np.zeros(16, dtype=np.float32)
    ema = DualArmJointEMA(0.35)
    ema.initialize(state)
    proposed = ema.apply(np.ones(16))
    command, clipped, _ = clamp_action_step(proposed, state, 0.1)
    assert clipped == 14
    ema.commit(command)
    # Once the goal returns to zero, decay from the sent 0.1, not the unsent 0.35.
    following = ema.apply(np.zeros(16))
    np.testing.assert_allclose(following[:7], 0.065)
    np.testing.assert_allclose(following[8:15], 0.065)


def test_gripper_alternating_predictions_do_not_toggle_commands():
    gripper = GripperDebouncer(2)
    gripper.initialize(np.zeros(16))
    action = np.zeros(16)
    for value in [0.51, 0.49] * 10:
        action[[7, 15]] = value
        assert gripper.update(action) == {"left": False, "right": False}
    action[7] = 1
    assert gripper.update(action) == {"left": False, "right": False}
    assert gripper.update(action) == {"left": True, "right": False}
    action[7] = 0
    assert gripper.update(action) == {"left": False, "right": False}


def test_gripper_initialization_uses_physical_feedback_widths():
    gripper = GripperDebouncer(2)
    state = np.zeros(16)
    state[7] = 0.1
    gripper.initialize(state)
    assert gripper.update(np.zeros(16)) == {"left": False, "right": False}


def test_open_confirmation_is_independent_per_arm_and_cancelled_by_close():
    gripper = GripperDebouncer(2)
    gripper.initialize(np.zeros(16))
    action = np.zeros(16)
    for requested, expected in [((1, 0), (False, False)), ((0, 1), (False, False)),
                                ((1, 1), (False, True)), ((1, 0), (True, False))]:
        action[[7, 15]] = requested
        assert tuple(gripper.update(action).values()) == expected
