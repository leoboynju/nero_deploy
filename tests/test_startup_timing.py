import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


class BrokerReached(Exception):
    pass


class ExternalShutdownException(Exception):
    pass


@pytest.fixture
def startup(monkeypatch):
    names = ("left_wrist", "right_wrist", "third_person")
    config = {
        "policy": {"host": "offline.invalid", "port": 8000, "prompt": "test"},
        "cameras": {
            "width": 2, "height": 2, "fps": 30, "warmup_seconds": 3,
            **{name: {"serial": name} for name in names},
        },
        "control": {
            "hz": 30, "max_steps": 1, "joint_limit_margin": 0.01,
            "max_source_age": 0.25, "max_joint_step": 0.1,
            "action_trace": {"enabled": False},
        },
    }
    run = SimpleNamespace(
        now=100.0, connection_seconds=1.0, warmup_seconds=3.0,
        stop_after=None, events=[], waits=[], state_calls=0, config=config,
        ros_active=False, handlers={}, cameras=[], published=[],
    )

    def record(name):
        run.events.append((name, run.now))

    def advance(seconds):
        assert seconds >= 0
        run.now += seconds

    class Stop:
        stopped = False

        def set(self):
            if self.stopped:
                return
            self.stopped = True
            record("stop")

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            record("wait_start")
            run.waits.append(timeout)
            assert timeout >= 0
            if run.stop_after is not None:
                assert 0 < run.stop_after < timeout
                advance(run.stop_after)
                self.set()
            else:
                advance(timeout)
            record("wait_end")
            return self.stopped

    class Thread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            assert daemon

        def start(self):
            record("thread_start")
            run.thread = self

        def join(self, timeout):
            assert timeout == 2
            assert not run.ros_active
            self.target(*self.args)
            record("thread_join")

    class Bridge:
        def __init__(self):
            record("bridge_init")
            run.bridge = self

        def state(self):
            run.state_calls += 1
            # Readiness is deliberately later than camera construction/startup.
            return (None if run.state_calls == 1 else np.zeros(16)), 0.0

        def gripper_forces(self):
            return np.full(2, np.nan)

        def validate_exclusive_control(self):
            assert run.events[-1][0] == "wait_end"
            assert not run.stop.is_set()
            assert run.now == pytest.approx(
                run.ready_at + max(run.connection_seconds, run.warmup_seconds)
            )
            record("exclusive_control")

        def publish(self, *args):
            run.published.append(args)
            record("publish")

        def destroy_node(self):
            record("bridge_destroy")

    class Camera:
        def __init__(self, serial, *args, **kwargs):
            self.name = serial
            run.cameras.append(self)
            record(f"camera_init:{serial}")
            advance(0.25)

        def snapshot(self):
            if self.name == names[-1]:
                run.ready_at = run.now
                record("ready")
            return object(), 0.0

        def close(self):
            record(f"camera_close:{self.name}")

    class Policy:
        def __init__(self, *args):
            record("connection_start")
            advance(run.connection_seconds)
            record("connection_end")

        def infer(self, *args):
            pytest.fail("Inference must not run before the broker sentinel")

    def create_broker(*args):
        assert run.events[-1][0] == "exclusive_control"
        assert not run.stop.is_set()
        assert run.now == pytest.approx(
            run.ready_at + max(run.connection_seconds, run.warmup_seconds)
        )
        record("broker")
        # Stop at the boundary that could start background inference.
        raise BrokerReached

    def init():
        assert not run.ros_active
        run.ros_active = True
        record("ros_init")

    def try_shutdown():
        run.ros_active = False
        record("ros_shutdown")

    def spin(node):
        assert node is run.bridge
        assert not run.ros_active
        raise ExternalShutdownException

    def install_handler(signum, handler):
        assert run.ros_active, "Custom handlers must be installed after rclpy.init"
        run.handlers[signum] = handler

    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(
        init=init, spin=spin, try_shutdown=try_shutdown,
        shutdown=Mock(side_effect=RuntimeError("rclpy.shutdown already called")),
    ))
    monkeypatch.setitem(sys.modules, "rclpy.executors", SimpleNamespace(
        ExternalShutdownException=ExternalShutdownException,
    ))
    monkeypatch.setitem(sys.modules, "nero_deploy.cameras", SimpleNamespace(RealSenseCamera=Camera))
    monkeypatch.setitem(sys.modules, "nero_deploy.ros", SimpleNamespace(DualArmBridge=Bridge))
    monkeypatch.setitem(sys.modules, "nero_deploy.policy_client", SimpleNamespace(PolicyClient=Policy))
    monkeypatch.setitem(sys.modules, "nero_deploy.control.inference", SimpleNamespace(
        create_inference_broker=create_broker,
        resolve_inference_settings=lambda cfg, mode, steps: (mode or "sync", steps or 8),
    ))
    path = Path(__file__).resolve().parents[1] / "src/nero_deploy/cli.py"
    spec = importlib.util.spec_from_file_location("nero_deploy._startup_timing_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    run.stop = Stop()
    # Replace only the isolated CLI's bindings, not process-wide time/threading.
    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=lambda: run.now, sleep=advance))
    monkeypatch.setattr(cli, "threading", SimpleNamespace(Event=lambda: run.stop, Thread=Thread))
    monkeypatch.setattr(cli, "signal", SimpleNamespace(SIGINT=2, SIGTERM=15, signal=install_handler))
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(sys, "argv", ["cli", "--execute"])
    run.cli = cli
    return run


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
@pytest.mark.parametrize("source", ["config", "cli"])
@pytest.mark.parametrize("connection_seconds,warmup_seconds", [(1.0, 3.0), (5.0, 3.0), (1.0, 0.0)])
def test_connection_overlaps_warmup_before_broker(startup, monkeypatch, mode, source,
                                                connection_seconds, warmup_seconds):
    startup.connection_seconds = connection_seconds
    startup.warmup_seconds = warmup_seconds
    argv = ["cli", "--execute", "--inference-mode", mode]
    if source == "cli":
        startup.config["cameras"]["warmup_seconds"] = 30
        argv += ["--camera-warmup-seconds", str(warmup_seconds)]
    else:
        startup.config["cameras"]["warmup_seconds"] = warmup_seconds
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(BrokerReached):
        startup.cli.main()

    times = dict(startup.events)
    assert times["ready"] > times["ros_init"]
    assert startup.state_calls == 2
    assert times["connection_start"] == times["ready"]
    if warmup_seconds:
        assert times["connection_start"] < times["ready"] + warmup_seconds
    assert times["connection_end"] == pytest.approx(times["ready"] + connection_seconds)
    assert times["wait_start"] == times["connection_end"]
    assert startup.waits == pytest.approx([max(0.0, warmup_seconds - connection_seconds)])
    expected = times["ready"] + max(connection_seconds, warmup_seconds)
    assert times["wait_end"] == pytest.approx(expected)
    assert times["exclusive_control"] == pytest.approx(expected)
    assert times["broker"] == pytest.approx(expected)
    assert [name for name, _ in startup.events][-10:] == [
        "wait_end", "exclusive_control", "broker", "stop",
        "camera_close:left_wrist", "camera_close:right_wrist", "camera_close:third_person",
        "ros_shutdown", "thread_join", "bridge_destroy",
    ]


def test_stop_during_warmup_prevents_broker_and_cleans_up(startup, caplog):
    caplog.set_level("INFO")
    startup.stop_after = 0.5

    startup.cli.main()

    times = dict(startup.events)
    assert startup.waits == pytest.approx([2.0])
    assert startup.stop.is_set()
    assert times["stop"] == pytest.approx(times["ready"] + 1.5)
    assert times["stop"] < times["ready"] + startup.warmup_seconds
    assert "exclusive_control" not in times
    assert "broker" not in times
    assert "no policy commands published yet" in caplog.text
    assert "publishing first policy action" not in caplog.text
    assert startup.published == []
    assert [name for name, _ in startup.events][-8:] == [
        "stop", "wait_end",
        "camera_close:left_wrist", "camera_close:right_wrist", "camera_close:third_person",
        "ros_shutdown", "thread_join", "bridge_destroy",
    ]


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
@pytest.mark.parametrize("source", ["cli", "config"])
def test_invalid_warmup_rejected_before_ros_init(startup, monkeypatch, capsys, value, source):
    if source == "cli":
        monkeypatch.setattr(sys, "argv", ["cli", f"--camera-warmup-seconds={value}"])
    else:
        startup.config["cameras"]["warmup_seconds"] = float(value)

    with pytest.raises(SystemExit) as exc:
        startup.cli.main()

    assert exc.value.code == 2
    assert "camera warmup seconds must be finite and non-negative" in capsys.readouterr().err
    assert startup.events == []
    assert startup.waits == []


@pytest.fixture
def lifecycle(startup, monkeypatch):
    startup.warmup_seconds = 0
    startup.config["cameras"]["warmup_seconds"] = 0
    startup.broker = SimpleNamespace(
        infer=Mock(return_value={"actions": np.zeros(16)}), close=Mock(),
    )
    startup.create_broker = Mock(return_value=startup.broker)
    monkeypatch.setattr(startup.cli, "create_inference_broker", startup.create_broker)
    return startup


@pytest.mark.parametrize("execute", [True, False])
def test_broker_closed_once_on_normal_exit(lifecycle, monkeypatch, execute, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(sys, "argv", ["cli", "--execute"] if execute else ["cli"])

    lifecycle.cli.main()

    lifecycle.broker.infer.assert_called_once()
    lifecycle.broker.close.assert_called_once_with()
    assert len(lifecycle.published) == int(execute)
    assert caplog.text.count("publishing first policy action") == int(execute)
    assert [name for name, _ in lifecycle.events][-3:] == [
        "ros_shutdown", "thread_join", "bridge_destroy",
    ]


def test_cleanup_when_ros_already_shutdown(lifecycle):
    def infer(observation):
        lifecycle.cli.rclpy.try_shutdown()
        lifecycle.thread.target(*lifecycle.thread.args)
        return {"actions": np.zeros(16)}

    lifecycle.broker.infer.side_effect = infer

    lifecycle.cli.main()

    lifecycle.cli.rclpy.shutdown.assert_not_called()
    lifecycle.broker.close.assert_called_once_with()
    assert lifecycle.published == []
    assert lifecycle.stop.is_set()
    assert [name for name, _ in lifecycle.events][-3:] == [
        "ros_shutdown", "thread_join", "bridge_destroy",
    ]


@pytest.mark.parametrize("failed_camera", ["left_wrist", "right_wrist", "third_person"])
def test_partial_camera_initialization_cleans_up(startup, monkeypatch, failed_camera):
    camera_type = startup.cli.RealSenseCamera

    def camera(serial, *args, **kwargs):
        if serial == failed_camera:
            raise RuntimeError("camera initialization failed")
        return camera_type(serial, *args, **kwargs)

    monkeypatch.setattr(startup.cli, "RealSenseCamera", camera)

    with pytest.raises(RuntimeError, match="camera initialization failed"):
        startup.cli.main()

    events = [name for name, _ in startup.events]
    for camera in startup.cameras:
        assert events.count(f"camera_close:{camera.name}") == 1
    assert events[-3:] == ["ros_shutdown", "thread_join", "bridge_destroy"]
    assert "connection_start" not in events


@pytest.mark.parametrize("ready", [True, False])
def test_stop_during_readiness_skips_policy_connection(startup, monkeypatch, ready):
    def state(bridge):
        startup.handlers[startup.cli.signal.SIGTERM](15, None)
        return (np.zeros(16) if ready else None), 0.0

    monkeypatch.setattr(startup.cli.DualArmBridge, "state", state)

    startup.cli.main()

    events = [name for name, _ in startup.events]
    assert "connection_start" not in events
    assert "broker" not in events
    assert startup.published == []
    assert events[-3:] == ["ros_shutdown", "thread_join", "bridge_destroy"]


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
@pytest.mark.parametrize("signum", [2, 15])
def test_stop_during_inference_discards_result(lifecycle, monkeypatch, mode, signum):
    monkeypatch.setattr(sys, "argv", ["cli", "--execute", "--inference-mode", mode])

    def infer(observation):
        lifecycle.handlers[signum](signum, None)
        return {"actions": np.zeros(16)}

    lifecycle.broker.infer.side_effect = infer

    lifecycle.cli.main()

    lifecycle.broker.infer.assert_called_once()
    lifecycle.broker.close.assert_called_once_with()
    assert lifecycle.published == []


def test_stop_during_postprocessing_prevents_publish(lifecycle, monkeypatch):
    monkeypatch.setattr(lifecycle.cli, "validate_action", lambda *args: lifecycle.stop.set())

    lifecycle.cli.main()

    lifecycle.broker.infer.assert_called_once()
    lifecycle.broker.close.assert_called_once_with()
    assert lifecycle.published == []


def test_inference_error_preserved_when_broker_close_fails(lifecycle):
    lifecycle.broker.infer.side_effect = RuntimeError("inference failed")
    lifecycle.broker.close.side_effect = RuntimeError("broker close failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        lifecycle.cli.main()

    lifecycle.broker.close.assert_called_once_with()
    assert lifecycle.published == []
    assert [name for name, _ in lifecycle.events][-3:] == [
        "ros_shutdown", "thread_join", "bridge_destroy",
    ]


def test_cleanup_failures_still_release_bridge(lifecycle, monkeypatch, caplog):
    def fail():
        raise RuntimeError("cleanup failed")

    trace = SimpleNamespace(record=Mock(), save=Mock(side_effect=fail))
    lifecycle.config["control"]["action_trace"]["enabled"] = True
    monkeypatch.setattr(lifecycle.cli, "ActionTrace", lambda *args: trace)
    camera_close = Mock(side_effect=fail)
    monkeypatch.setattr(lifecycle.cli.RealSenseCamera, "close", camera_close)
    lifecycle.broker.close.side_effect = fail
    join = Mock(side_effect=fail)
    monkeypatch.setattr(lifecycle.cli.threading.Thread, "join", join)

    lifecycle.cli.main()

    trace.save.assert_called_once_with()
    assert camera_close.call_count == 3
    lifecycle.broker.close.assert_called_once_with()
    join.assert_called_once_with(timeout=2)
    assert [name for name, _ in lifecycle.events][-2:] == ["ros_shutdown", "bridge_destroy"]
    assert "Could not clean up inference broker" in caplog.text
    assert "Could not clean up spin thread" in caplog.text
