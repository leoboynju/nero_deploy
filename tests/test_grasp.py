import numpy as np
import pytest

from nero_deploy.control.grasp import GraspConfig, GraspCoordinator


def setup_close():
    state = np.zeros(16)
    state[[7, 15]] = 0.1
    action = np.zeros(16)
    action[:7] = 0.1
    action[15] = 1
    control = GraspCoordinator(GraspConfig())
    control.initialize(state, 0)
    return control, state, action


def test_close_latches_pose_waits_for_arm_and_cannot_be_reversed_by_next_chunk():
    control, state, action = setup_close()
    pending = control.step(action, state, 0)
    assert pending.held and pending.event == "close_pending"
    np.testing.assert_allclose(pending.action[:7], 0.1)
    action[:7] = 0.8
    np.testing.assert_allclose(control.step(action, state, 0.1).action[:7], 0.1)
    action[:7] = 1.0  # A future lift target must not change the latched grasp pose.
    action[7] = 1.0  # Nor may a new chunk reopen the pending/closing gripper.
    held = control.step(action, state, 0.4)
    np.testing.assert_allclose(held.action[:7], 0.1)
    assert held.gripper_open["left"]
    state[:7] = 0.1
    control.step(None, state, 0.5)
    closed = control.step(None, state, 0.57)
    assert closed.event == "close_sent" and not closed.gripper_open["left"]
    state[7] = 0.04  # Contact with an object; never require zero width.
    assert not control.step(action, state, 0.65, forces=[1, np.nan]).replan
    finished = control.step(None, state, 0.9, forces=[1, np.nan])
    assert finished.replan and finished.held and finished.event == "close_settled"
    assert not finished.gripper_open["left"]
    assert not control.active


def test_pending_grasp_times_out_without_closing_when_arm_cannot_reach():
    control, state, action = setup_close()
    control.step(action, state, 0)
    control.step(action, state, 0.1)
    result = control.step(None, state, 1.7)
    assert result.replan and result.event == "reach_timeout"
    assert result.gripper_open["left"]


def test_no_feedback_motion_is_not_reported_as_success_and_release_still_works():
    control, state, action = setup_close()
    state[:7] = 0.1
    control.step(action, state, 0)
    control.step(action, state, 0.1)
    control.step(None, state, 0.2)
    control.step(None, state, 0.27)
    # Even a nonzero force without width progress cannot falsely confirm closure.
    assert not control.step(None, state, 0.8, forces=[1, 1]).replan
    result = control.step(None, state, 1.3, forces=[1, 1])
    assert result.event == "close_timeout" and result.replan
    assert not result.gripper_open["left"]
    action[7] = 1
    control.step(action, state, 1.4)
    release = control.step(action, state, 1.5)
    assert release.event == "open_sent" and release.held
    assert release.gripper_open["left"]
    assert control.step(None, state, 1.9).event == "open_settled"


def test_single_open_pulses_do_not_release_closed_gripper():
    control, state, action = setup_close()
    state[7] = 0
    control.initialize(state, 0)
    for index in range(20):
        action[7] = 1 - index % 2
        result = control.step(action, state, index / 30)
        assert not result.held and not result.gripper_open["left"]
    action[7] = 1
    assert not control.step(action, state, 1.0).gripper_open["left"]
    released = control.step(action, state, 1.001)
    assert released.gripper_open["left"] and released.event == "open_sent"


def test_held_ticks_do_not_supply_second_open_confirmation():
    control, state, action = setup_close()
    state[7] = 0
    control.initialize(state, 0)
    action[7] = 1
    action[15] = 0
    action[8:15] = 0.2
    assert control.step(action, state, 0).event == "close_pending"
    for now in (0.1, 0.2, 0.3, 0.4):
        assert not control.step(None, state, now).gripper_open["left"]
    assert control.step(None, state, 1.6).replan
    action[15] = 1
    assert not control.step(action, state, 1.7).gripper_open["left"]
    assert control.step(action, state, 1.701).event == "open_sent"


def test_force_during_motion_does_not_count_as_completed_contact():
    control, state, action = setup_close()
    state[:7] = 0.1
    control.step(action, state, 0)
    control.step(action, state, 0.1)
    control.step(None, state, 0.2)
    control.step(None, state, 0.27)
    state[7] = 0.07
    assert not control.step(None, state, 0.65, forces=[1, 1]).replan
    state[7] = 0.06
    assert not control.step(None, state, 0.78, forces=[1, 1]).replan
    assert control.step(None, state, 0.95, forces=[1, 1]).event == "close_settled"


def test_both_arms_must_reach_before_simultaneous_closure():
    control, state, action = setup_close()
    action[15] = 0
    action[8:15] = 0.2
    control.step(action, state, 0)
    control.step(action, state, 0.1)
    state[:7] = 0.1
    assert all(control.step(None, state, 0.5).gripper_open.values())
    state[8:15] = 0.2
    control.step(None, state, 0.6)
    assert not any(control.step(None, state, 0.67).gripper_open.values())


def test_held_ticks_do_not_consume_actions_and_resume_discards_old_queue():
    from nero_deploy.control.inference import SynchronousActionChunkBroker

    control, state, action = setup_close()
    calls = []

    class Policy:
        def infer(self, observation):
            calls.append(observation)
            return {"actions": np.repeat(action[None], 16, axis=0)}

    broker = SynchronousActionChunkBroker(Policy(), 8)
    control.infer(broker, {"step": 0})
    control.step(action, state, 0)
    result = control.infer(broker, {"step": 1})
    control.step(action, state, 0.1)
    assert result["rtc_action_index"] == 0
    for tick in range(2, 6):
        held = control.infer(broker, {"step": tick})
        assert held["rtc_action_index"] == 0
        assert not held["rtc_chunk_changed"]
    assert calls == [{"step": 0}]
    finished = control.step(None, state, 1.7)
    control.after_publish(broker, finished)
    fresh = control.infer(broker, {"step": 6})
    assert fresh["rtc_action_index"] == 0
    assert fresh["rtc_chunk_id"] > result["rtc_chunk_id"]
    assert calls == [{"step": 0}, {"step": 6}]


@pytest.mark.parametrize("kwargs", [{"settle_seconds": 0}, {"reach_tolerance_rad": -1},
                                   {"close_timeout_seconds": 0.1}, {"min_hold_seconds": float("nan")}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GraspConfig(**kwargs)
