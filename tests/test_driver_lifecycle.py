import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


processes = load_script("driver_processes")
probe = load_script("probe_drivers")


def fake_node(root):
    node = root / "thirdparty/ros_ws/install/agx_arm_ctrl/lib/agx_arm_ctrl/agx_arm_ctrl_single"
    node.parent.mkdir(parents=True)
    node.write_text(
        "import pathlib, subprocess, sys, time\n"
        "if '--parent' in sys.argv or '--orphan' in sys.argv:\n"
        " child = subprocess.Popen([sys.executable, __file__, '__ns:=/right_arm'], "
        "start_new_session='--orphan' in sys.argv)\n"
        " pathlib.Path(sys.argv[-1]).write_text(str(child.pid))\n"
        " if '--orphan' in sys.argv: raise SystemExit(0)\n"
        "time.sleep(60)\n"
    )
    return node


def child_pid(path):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if path.exists() and path.read_text():
            return int(path.read_text())
        time.sleep(0.02)
    raise AssertionError("Fake child did not start")


def test_sessions_stop_launcher_and_children_without_inheriting_lock(tmp_path):
    import fcntl

    node = fake_node(tmp_path)
    log_dir = tmp_path / "logs"
    lock_path = tmp_path / "stack.lock"
    lock = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.set_inheritable(lock, True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    output = tmp_path / "child.pid"
    try:
        pid = processes.launch(tmp_path, log_dir, "right", [sys.executable, str(node),
                               "__ns:=/right_arm", "--parent", str(output)])
        child = child_pid(output)
        assert os.getpgid(pid) == pid
        assert os.getpgid(child) == pid
        os.close(lock)
        lock = None
        with lock_path.open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = processes.stop_project(tmp_path, log_dir, timeout=0.5)
        assert pid in stopped and child in stopped
        assert processes.process_info(pid) is None
        assert processes.process_info(child) is None
        assert not (log_dir / "driver_right.pid").exists()
    finally:
        if lock is not None:
            os.close(lock)
        processes.stop_project(tmp_path, log_dir, timeout=0.1)


def test_orphan_is_found_without_launcher_or_pid_file(tmp_path):
    node = fake_node(tmp_path)
    output = tmp_path / "child.pid"
    parent = subprocess.Popen([sys.executable, str(node), "__ns:=/right_arm", "--orphan", str(output)])
    try:
        child = child_pid(output)
        parent.wait(timeout=3)
        assert processes.process_info(child) is not None
        assert child in processes.stop_project(tmp_path, tmp_path / "logs", timeout=0.5)
        assert processes.process_info(child) is None
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=3)
        processes.stop_project(tmp_path, tmp_path / "logs", timeout=0.1)


def test_stale_pid_and_other_installation_are_not_killed(tmp_path):
    root = tmp_path / "ours"
    node = fake_node(tmp_path / "other")
    log_dir = root / "logs"
    log_dir.mkdir(parents=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    other = subprocess.Popen([sys.executable, str(node), "__ns:=/right_arm"])
    try:
        (log_dir / "driver_right.pid").write_text(str(unrelated.pid))
        processes.stop_project(root, log_dir, timeout=0.1)
        assert unrelated.poll() is None
        assert other.poll() is None
    finally:
        for process in (unrelated, other):
            process.terminate()
            process.wait(timeout=3)


def report(subscribers=1, messages=2, publishers=0):
    return {arm: {"messages": messages, "feedback_age": 0.01,
                  "feedback_publishers": ["driver"] if subscribers else [],
                  "controls": {topic: {"subscribers": ["driver"] * subscribers,
                                       "publishers": ["controller"] * publishers}
                               for topic in ("move_j", "joint_states")}}
            for arm in ("left", "right")}


def test_probe_requires_live_feedback_and_unique_endpoints():
    assert probe.classify(report(), "ready") == 0
    assert probe.classify(report(subscribers=2), "ready") == 1
    assert probe.classify(report(messages=0), "ready") == 1
    assert probe.classify(report(publishers=1), "ready") == 3
    assert probe.classify(report(), "empty") == 4
    assert probe.classify(report(messages=0), "empty") == 1
    assert probe.classify(report(subscribers=0, messages=0), "empty") == 0
    assert probe.classify(report(subscribers=0, messages=0), "ready-or-empty") == 2
    assert probe.classify(report(), "ready-or-empty") == 0
    assert probe.classify(report(messages=0), "ready-or-empty") == 1
    assert probe.classify(report(subscribers=0, publishers=1), "ready-or-empty") == 3


@pytest.mark.parametrize("status,expected,elapsed", [(2, 2, 3.5), (0, 0, 1.5), (3, 3, 0.1), (4, 4, 1.5), (1, 1, 5)])
def test_probe_early_completion_keeps_discovery_settling(monkeypatch, status, expected, elapsed):
    clock = [0.0]
    destroyed = []
    node = SimpleNamespace(create_subscription=lambda *args: None,
                           get_publishers_info_by_topic=lambda topic: [],
                           get_subscriptions_info_by_topic=lambda topic: [],
                           destroy_node=lambda: destroyed.append(True))

    def spin_once(node, timeout_sec):
        clock[0] = round(clock[0] + timeout_sec, 2)

    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(
        init=lambda: None, create_node=lambda name: node, spin_once=spin_once, try_shutdown=lambda: None))
    monkeypatch.setitem(sys.modules, "rclpy.qos", SimpleNamespace(
        QoSProfile=lambda **kwargs: None, ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1)))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", SimpleNamespace(JointState=object))
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(probe, "classify", lambda report, mode: status)
    monkeypatch.setattr(sys, "argv", ["probe", "--mode", "ready-or-empty", "--timeout", "5"])
    assert probe.main() == expected
    assert clock[0] == pytest.approx(elapsed)
    assert destroyed == [True]


@pytest.mark.parametrize("first,final,expected", [
    (0, 0, ["probe:ready-or-empty"]),
    (3, 0, ["probe:ready-or-empty"]),
    (7, 0, ["probe:ready-or-empty"]),
    (1, 0, ["probe:ready-or-empty", "manager:stop", "probe:empty", "manager:launch:left", "manager:launch:right", "probe:ready"]),
    (2, 0, ["probe:ready-or-empty", "manager:stop", "probe:empty", "manager:launch:left", "manager:launch:right", "probe:ready"]),
    (1, 1, ["probe:ready-or-empty", "manager:stop", "probe:empty", "manager:launch:left", "manager:launch:right", "probe:ready", "manager:stop"]),
])
@pytest.mark.parametrize("recover", [False, True])
def test_startup_reuse_conflict_cleanup_and_failure_paths(tmp_path, first, final, expected, recover):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "start_drivers.sh").write_text((ROOT / "scripts/start_drivers.sh").read_text())
    helper = (
        "import json, os, pathlib, sys\n"
        "p=pathlib.Path(os.environ['CALL_LOG']); rows=json.loads(p.read_text()) if p.exists() else []\n"
        "probe='probe_drivers' in __file__\n"
        "action=sys.argv[sys.argv.index('--mode')+1] if probe else sys.argv[1]\n"
        "label=('probe:' if probe else 'manager:')+action\n"
        "if action=='launch': label+=':'+sys.argv[sys.argv.index('--arm')+1]\n"
        "rows.append(label); p.write_text(json.dumps(rows))\n"
        "if probe and action in ('ready', 'ready-or-empty'): raise SystemExit(int(os.environ['FIRST' if action=='ready-or-empty' else 'FINAL']))\n"
    )
    (scripts / "probe_drivers.py").write_text(helper)
    (scripts / "driver_processes.py").write_text(helper)
    (scripts / "recover_nero.sh").write_text('python3 "$(dirname "$0")/driver_processes.py" recover\n')
    setup = tmp_path / "setup.bash"
    setup.write_text(":\n")
    call_log = tmp_path / "calls.json"
    env = {**os.environ, "ROS_SETUP": str(setup), "ROBOT_WS_SETUP": str(setup),
           "CALL_LOG": str(call_log), "FIRST": str(first), "FINAL": str(final),
           "LOG_DIR": str(tmp_path / "logs"), "RECOVER_BEFORE_START": "1" if recover else "0"}
    if recover and first in (0, 1, 2):
        if first == 0:
            expected = ["probe:ready-or-empty", "manager:stop", "probe:empty",
                        "manager:launch:left", "manager:launch:right", "probe:ready"]
        expected = expected.copy()
        expected.insert(expected.index("probe:empty") + 1, "manager:recover")
    if "probe:empty" in expected:
        expected = expected.copy()
        expected.insert(expected.index("probe:empty") + 1, "manager:check-environment")
    result = subprocess.run(["bash", str(scripts / "start_drivers.sh")], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == (first in (0, 1, 2) and final == 0), result.stderr
    assert json.loads(call_log.read_text()) == expected
