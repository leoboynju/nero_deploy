import importlib.util
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def recovery(monkeypatch):
    # Never import the real SDK or construct a CAN bus, even during collection.
    sdk = SimpleNamespace(
        AgxArmFactory=SimpleNamespace(create_arm=Mock()),
        ArmModel=SimpleNamespace(NERO="nero"), NeroFW=SimpleNamespace(DEFAULT="default"),
        create_agx_arm_config=Mock(return_value={}),
    )
    monkeypatch.setitem(sys.modules, "pyAgxArm", sdk)
    path = Path(__file__).resolve().parents[1] / "src/nero_deploy/diagnostics/recover.py"
    spec = importlib.util.spec_from_file_location("_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = SimpleNamespace(now=0.0, tick=lambda: None)

    def sleep(seconds):
        assert seconds >= 0
        clock.now = round(clock.now + seconds, 9)
        clock.tick()

    module.time = SimpleNamespace(
        time=lambda: 1000 + clock.now, monotonic=lambda: clock.now, sleep=sleep,
    )
    feedback = module._Feedback()
    robot = SimpleNamespace(has_comm_error=lambda: False)
    return SimpleNamespace(module=module, clock=clock, feedback=feedback, robot=robot, sdk=sdk)


def frame(can_id, timestamp, data):
    return SimpleNamespace(
        arbitration_id=can_id, timestamp=timestamp, data=bytearray(data), dlc=8,
        is_extended_id=False, is_remote_frame=False, is_error_frame=False,
    )


def emit(run, joints=None, *, ids=None, stamp=None, status=(1, 0, 1, 0, 0, 0, 0)):
    joints = run.module.TARGETS["left"] if joints is None else joints
    angles = [round(math.degrees(q) * 1000) for q in joints]
    payloads = (
        struct.pack(">6BH", *status), struct.pack(">2i", *angles[:2]),
        struct.pack(">2i", *angles[2:4]), struct.pack(">2i", *angles[4:6]),
        struct.pack(">i4x", angles[6]),
    )
    stamp = run.module.time.time() if stamp is None else stamp
    for can_id, data in zip(run.module._FEEDBACK_IDS, payloads):
        if ids is None or can_id in ids:
            run.feedback(frame(can_id, stamp, data))


def wait(run, timeout=1.0):
    return run.module._wait_for_target(
        run.robot, run.feedback, run.module.TARGETS["left"], timeout,
    )


def test_cached_finished_status_cannot_complete(recovery):
    emit(recovery, [0] * 7)
    with pytest.raises(RuntimeError, match="timed out"):
        wait(recovery)
    assert recovery.clock.now == 1.0


def test_fresh_finished_status_far_from_target_cannot_complete(recovery):
    recovery.clock.tick = lambda: emit(recovery, [0] * 7)
    with pytest.raises(RuntimeError, match="timed out"):
        wait(recovery)


@pytest.mark.parametrize("missing", [0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A9])
@pytest.mark.parametrize("preexisting", [False, True])
def test_missing_or_stale_stream_never_completes(recovery, missing, preexisting):
    if preexisting:
        emit(recovery)
    ids = set(recovery.module._FEEDBACK_IDS) - {missing}
    recovery.clock.tick = lambda: emit(recovery, ids=ids)
    with pytest.raises(RuntimeError, match="timed out"):
        wait(recovery)


@pytest.mark.parametrize("frozen", [0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A9])
def test_one_stream_freezing_after_first_valid_sample_blocks_dwell(recovery, frozen):
    def tick():
        ids = None if recovery.clock.now == 0.05 else set(recovery.module._FEEDBACK_IDS) - {frozen}
        emit(recovery, ids=ids)

    recovery.clock.tick = tick
    with pytest.raises(RuntimeError, match="timed out"):
        wait(recovery)


@pytest.mark.parametrize("kind", ["frozen", "duplicate_timestamp", "precommand", "future", "old", "skew"])
def test_timestamp_and_counter_freshness(recovery, kind):
    def tick():
        now = recovery.clock.now
        if kind == "frozen":
            if now == 0.05:
                emit(recovery)
        elif kind == "duplicate_timestamp":
            emit(recovery, stamp=1000.05)
        elif kind == "precommand":
            emit(recovery, stamp=999.99)
        elif kind == "future":
            emit(recovery, stamp=1000 + now + 1)
        elif kind == "old":
            emit(recovery, stamp=1000 + now - 0.3)
        else:
            emit(recovery)
            emit(recovery, ids={0x2A5}, stamp=1000 + now - 0.15)

    recovery.clock.tick = tick
    with pytest.raises(RuntimeError, match="timed out"):
        wait(recovery)


def test_already_at_target_needs_fresh_advancing_dwell_without_moving_transition(recovery):
    emit(recovery)
    recovery.clock.tick = lambda: emit(recovery)
    joints, errors, dwell, samples = wait(recovery)
    assert joints == pytest.approx(recovery.module.TARGETS["left"], abs=1e-5)
    assert max(errors) < 1e-5
    assert dwell >= 0.5 and samples >= 2
    assert recovery.clock.now >= 0.55


@pytest.mark.parametrize("offset", [0.0099, 0.0101])
def test_target_tolerance_is_applied_to_every_joint(recovery, offset):
    joints = [q + offset for q in recovery.module.TARGETS["left"]]
    recovery.clock.tick = lambda: emit(recovery, joints, status=(1, 0, 1, 0, 0, 255, 0))
    if offset < 0.01:
        _, errors, _, _ = wait(recovery)
        assert all(error <= 0.01 for error in errors)
    else:
        with pytest.raises(RuntimeError, match="timed out"):
            wait(recovery)


@pytest.mark.parametrize("excursion", ["outside_target", "inside_tolerance", "moving"])
def test_transient_crossing_or_instability_restarts_dwell(recovery, excursion):
    def tick():
        joints = list(recovery.module.TARGETS["left"])
        status = (1, 0, 1, 0, 0, 0, 0)
        if 0.25 <= recovery.clock.now < 0.4:
            if excursion == "moving":
                status = (1, 0, 1, 0, 1, 0, 0)
            else:
                joints[0] += 0.02 if excursion == "outside_target" else 0.005
        emit(recovery, joints, status=status)

    recovery.clock.tick = tick
    _, _, dwell, _ = wait(recovery, 1.2)
    assert recovery.clock.now >= 0.9
    assert dwell >= 0.5


@pytest.mark.parametrize("status", [
    (1, 0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0, 0),
    (1, 0, 4, 0, 0, 0, 0), (1, 0, 6, 0, 0, 0, 0),
    (1, 0, 1, 1, 0, 0, 0),
])
def test_moving_between_polls_then_normal_restarts_dwell(recovery, status):
    def tick():
        if recovery.clock.now == 0.3:
            emit(recovery, ids={0x2A1}, status=status, stamp=recovery.module.time.time() - 0.001)
        emit(recovery)

    recovery.clock.tick = tick
    _, _, dwell, _ = wait(recovery, 1.2)
    assert recovery.clock.now >= 0.85
    assert dwell >= 0.5


@pytest.mark.parametrize("joint", range(7))
@pytest.mark.parametrize("offset", [0.005, 0.02])
def test_joint_excursion_between_polls_then_target_restarts_dwell(recovery, joint, offset):
    def tick():
        if recovery.clock.now == 0.3:
            joints = list(recovery.module.TARGETS["left"])
            joints[joint] += offset
            can_id = recovery.module._FEEDBACK_IDS[1 + joint // 2]
            emit(recovery, joints, ids={can_id}, stamp=recovery.module.time.time() - 0.001)
        emit(recovery)

    recovery.clock.tick = tick
    _, _, dwell, _ = wait(recovery, 1.2)
    assert recovery.clock.now >= 0.85
    assert dwell >= 0.5


def test_repeated_hidden_movement_cannot_complete(recovery):
    def tick():
        emit(recovery, ids={0x2A1}, status=(1, 0, 1, 0, 1, 0, 0),
             stamp=recovery.module.time.time() - 0.001)
        emit(recovery)

    recovery.clock.tick = tick
    with pytest.raises(RuntimeError, match="timed out"):
        wait(recovery)


@pytest.mark.parametrize("joint", range(7))
def test_opposite_extrema_within_one_poll_exceed_stable_range(recovery, joint):
    def tick():
        if recovery.clock.now == 0.3:
            for offset in (0.0015, -0.0015):
                joints = list(recovery.module.TARGETS["left"])
                joints[joint] += offset
                emit(recovery, joints, ids={recovery.module._FEEDBACK_IDS[1 + joint // 2]},
                     stamp=recovery.module.time.time() - (0.002 if offset > 0 else 0.001))
        emit(recovery)

    recovery.clock.tick = tick
    wait(recovery, 1.2)
    assert recovery.clock.now >= 0.85


def test_extrema_accumulate_even_without_a_new_complete_sample(recovery):
    def tick():
        now = recovery.clock.now
        if now in (0.2, 0.25):
            joints = list(recovery.module.TARGETS["left"])
            joints[0] += 0.0015 if now == 0.2 else -0.0015
            emit(recovery, joints, ids={0x2A5}, stamp=recovery.module.time.time() - 0.001)
        # Status only advances on alternate polls; the positive excursion must
        # still contribute to the dwell range before the next complete sample.
        ids = None if round(now * 20) % 2 else set(recovery.module._FEEDBACK_IDS) - {0x2A1}
        emit(recovery, ids=ids)

    recovery.clock.tick = tick
    _, _, dwell, _ = wait(recovery, 1.2)
    assert recovery.clock.now >= 0.75
    assert dwell >= 0.5


@pytest.mark.parametrize("kind", ["precommand", "stale", "future", "replayed"])
def test_old_frame_followed_by_fresh_frame_cannot_sustain_dwell(recovery, kind):
    def tick():
        if recovery.clock.now == 0.3:
            stamp = {"precommand": 999.0, "stale": 1000.04,
                     "future": 1000.31, "replayed": 1000.25}[kind]
            emit(recovery, ids={0x2A9}, stamp=stamp)
        emit(recovery)

    recovery.clock.tick = tick
    wait(recovery, 1.2)
    assert recovery.clock.now >= 0.85


@pytest.mark.parametrize("status", [
    (1, 1, 1, 0, 0, 0, 0), (1, 7, 1, 0, 0, 0, 0),
    (1, 0, 1, 0, 0, 0, 1), (1, 0, 1, 0, 0, 0, 0x4000),
])
def test_fault_is_latched_even_if_followed_by_normal_feedback(recovery, status):
    def tick():
        emit(recovery, status=status)
        emit(recovery)

    recovery.clock.tick = tick
    with pytest.raises(RuntimeError, match="arm fault"):
        wait(recovery)


@pytest.mark.parametrize("status", [
    (0, 0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0),
    (1, 0, 4, 0, 0, 0, 0), (1, 0, 6, 0, 0, 0, 0),
    (1, 0, 1, 1, 0, 0, 0), (1, 0, 1, 0, 255, 0, 0),
])
def test_unexpected_control_mode_teaching_or_motion_not_accepted(recovery, status):
    recovery.clock.tick = lambda: emit(recovery, status=status)
    with pytest.raises(RuntimeError, match="timed out"):
        wait(recovery)


@pytest.mark.parametrize("field,value", [
    ("data", b"\0" * 7), ("data", b"\0" * 9), ("dlc", 7),
    ("is_extended_id", True), ("is_remote_frame", True), ("is_error_frame", True),
    ("timestamp", float("nan")), ("timestamp", float("inf")), ("timestamp", 0),
])
def test_malformed_feedback_fails_closed_without_callback_exception(recovery, field, value):
    def tick():
        bad = frame(0x2A5, recovery.module.time.time(), b"\0" * 8)
        setattr(bad, field, value)
        recovery.feedback(bad)
        emit(recovery)

    recovery.clock.tick = tick
    with pytest.raises(RuntimeError, match="malformed recovery feedback"):
        wait(recovery)


def test_snapshots_own_immutable_values_and_ignore_unrelated_frames(recovery):
    original = frame(0x2A5, 1000.1, struct.pack(">2i", 1000, -2000))
    recovery.feedback(original)
    before, *_ = recovery.feedback.snapshot()
    original.data[:] = b"\0" * 8
    original.timestamp = 9000
    recovery.feedback(frame(0x123, float("nan"), b""))
    assert recovery.feedback.snapshot()[0] == before
    recovery.feedback(original)
    after, *_ = recovery.feedback.snapshot()
    assert before[1].values == pytest.approx((math.radians(1), math.radians(-2)))
    assert before[1].timestamp == 1000.1
    assert after[1].count == before[1].count + 1
    assert after[1].values == (0, 0)


def test_interval_evidence_is_consumed_together_and_next_interval_is_independent(recovery):
    recovery.feedback.begin_wait()
    recovery.clock.now = 0.05
    emit(recovery, status=(1, 0, 1, 0, 1, 0, 0))
    recovery.clock.now = 0.06
    emit(recovery)
    frames, error, valid, extrema = recovery.feedback.snapshot()
    assert error is None and not valid
    assert all(bounds is not None for bounds in extrema)
    assert frames[0].values[:5] == (1, 0, 1, 0, 0)
    assert recovery.feedback.snapshot() == (frames, None, True, (None,) * 4)

    recovery.clock.now = 0.07
    joints = list(recovery.module.TARGETS["left"])
    joints[0] += 0.005
    emit(recovery, joints, ids={0x2A5})
    recovery.clock.now = 0.08
    emit(recovery, ids={0x2A5})
    _, _, valid, bounds = recovery.feedback.snapshot()
    assert valid
    assert bounds[0][1][0] - bounds[0][0][0] == pytest.approx(0.005, abs=2e-5)
    assert bounds[1:] == (None,) * 3
    assert frames[1].values[0] == pytest.approx(recovery.module.TARGETS["left"][0], abs=1e-5)


@pytest.mark.parametrize("arm_status", [0, 7])
def test_begin_wait_discards_precommand_interval_evidence(recovery, arm_status):
    # An earlier collection epoch must not supply samples or violations to a
    # new command's dwell. Delayed old frames after begin_wait remain invalid.
    recovery.feedback.begin_wait()
    recovery.clock.now = 0.05
    emit(recovery, [0] * 7, status=(1, arm_status, 1, 0, 1, 0, 0))
    recovery.clock.tick = lambda: emit(recovery)
    _, _, dwell, _ = wait(recovery)
    assert 0.6 <= recovery.clock.now < 0.7
    assert dwell >= 0.5


@pytest.mark.parametrize("kind", ["short", "long", "nan", "inf"])
def test_nonfinite_or_wrong_length_measured_joints_rejected(recovery, kind):
    # Integer CAN payloads cannot encode NaN; defend the verification boundary too.
    def tick():
        emit(recovery)
        with recovery.feedback._lock:
            f = recovery.feedback._frames[0x2A5]
            if kind in ("nan", "inf"):
                recovery.feedback._frames[0x2A5] = f._replace(values=(float(kind), f.values[1]))
            else:
                f = recovery.feedback._frames[0x2A9]
                values = () if kind == "short" else f.values + (0.0,)
                recovery.feedback._frames[0x2A9] = f._replace(values=values)

    recovery.clock.tick = tick
    with pytest.raises(RuntimeError, match="timed out"):
        wait(recovery)


@pytest.fixture
def main_run(recovery, monkeypatch):
    run = recovery
    run.events = []
    run.moved = False
    run.outcome = "success"
    run.arm = "left"

    def register(callback):
        run.events.append("register")
        run.feedback = callback

    def connect():
        assert run.events[-1] == "register"
        run.events.append("connect")
        emit(run, [0] * 7)

    def move_j(target):
        run.events.append(("move_j", tuple(target)))
        run.moved = True

    run.robot = SimpleNamespace(
        OPTIONS=SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="gripper")),
        get_context=lambda: SimpleNamespace(register_parser_packet_fun=register),
        init_effector=lambda kind: SimpleNamespace(
            move_gripper_m=lambda **kwargs: run.events.append(("gripper", kwargs))),
        connect=connect, disconnect=lambda: run.events.append("disconnect"),
        enable=lambda: run.events.append("enable") or True,
        set_speed_percent=lambda value: run.events.append(("speed", value)),
        move_j=move_j, has_comm_error=lambda: run.outcome == "comm_error" and run.moved,
    )
    run.sdk.AgxArmFactory.create_arm.return_value = run.robot

    def tick():
        if run.moved and run.outcome == "success":
            emit(run, run.module.TARGETS[run.arm])
        elif run.moved and run.outcome == "fault":
            emit(run, status=(1, 7, 1, 0, 0, 0, 0))

    run.clock.tick = tick
    monkeypatch.setattr(sys, "argv", ["recover", "left", "--timeout=1", "--yes"])
    return run


@pytest.mark.parametrize("arm", ["left", "right"])
def test_main_preserves_commands_and_verifies_before_single_disconnect(main_run, capsys, monkeypatch, arm):
    run = main_run
    run.arm = arm
    monkeypatch.setattr(sys, "argv", ["recover", arm, "--timeout=1", "--yes"])
    run.module.main()
    assert run.events == [
        "register", "connect", "enable", ("speed", 30),
        ("gripper", {"value": 0.1, "force": 1.0}),
        ("move_j", tuple(run.module.TARGETS[arm])), "disconnect",
    ]
    run.sdk.create_agx_arm_config.assert_called_once_with(
        robot="nero", firmeware_version="default", interface="socketcan",
        channel="can3" if arm == "left" else "can2",
    )
    run.sdk.AgxArmFactory.create_arm.assert_called_once_with({})
    assert run.clock.now >= 1.05
    output = capsys.readouterr()
    assert "recovery complete" in output.out
    assert "measured=" in output.out and "max_error_rad=" in output.out
    assert "verified fresh and settled" in output.out
    assert not output.err


@pytest.mark.parametrize("outcome", ["timeout", "comm_error", "fault"])
def test_main_failure_disconnects_without_extra_motion_commands(main_run, capsys, outcome):
    run = main_run
    run.outcome = outcome
    with pytest.raises(RuntimeError):
        run.module.main()
    assert run.events.count("connect") == run.events.count("disconnect") == 1
    assert run.events[-1] == "disconnect"
    assert [event[0] for event in run.events if isinstance(event, tuple)] == ["speed", "gripper", "move_j"]
    output = capsys.readouterr()
    assert "recovery complete" not in output.out
    assert "disconnect is not confirmation that the arm stopped" in output.err


@pytest.mark.parametrize("method", ["connect", "enable", "set_speed_percent", "move_j"])
def test_main_original_command_exception_disconnects_once(main_run, capsys, method):
    run = main_run
    original = getattr(run.robot, method)

    def fail(*args):
        original(*args)
        raise RuntimeError("fake command failure")

    setattr(run.robot, method, fail)
    with pytest.raises(RuntimeError, match="fake command failure"):
        run.module.main()
    assert run.events.count("connect") == run.events.count("disconnect") == 1
    assert run.events[-1] == "disconnect"
    moves = [event for event in run.events if isinstance(event, tuple) and event[0] == "move_j"]
    assert len(moves) == (method == "move_j")
    output = capsys.readouterr()
    assert "recovery complete" not in output.out
    assert "disconnect is not confirmation that the arm stopped" in output.err


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_timeout_validated_before_config_or_factory(main_run, monkeypatch, capsys, value):
    run = main_run
    monkeypatch.setattr(sys, "argv", ["recover", "left", f"--timeout={value}", "--yes"])
    with pytest.raises(SystemExit) as exc:
        run.module.main()
    assert exc.value.code == 2
    run.sdk.create_agx_arm_config.assert_not_called()
    run.sdk.AgxArmFactory.create_arm.assert_not_called()
    assert run.events == []
    assert "--timeout must be finite and positive" in capsys.readouterr().err
