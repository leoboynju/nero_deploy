import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("driver_process_ownership", ROOT / "scripts/driver_processes.py")
processes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processes)


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    prefix = root / "thirdparty/ros_ws/install/agx_arm_ctrl"
    marker = prefix / "share/ament_index/resource_index/packages/agx_arm_ctrl"
    marker.parent.mkdir(parents=True)
    marker.touch()
    monkeypatch.setenv("ROS_DOMAIN_ID", "73")
    env = {**os.environ, "ROS_DOMAIN_ID": "73", "AMENT_PREFIX_PATH": str(prefix),
           "PYTHONPATH": str(root / "thirdparty/pyAgxArm")}
    return root, env


@pytest.fixture
def spawn(tmp_path):
    # These scripts only sleep. Never invoke ROS or use a real driver executable.
    script = tmp_path / "ros2"
    script.write_text("import time\ntime.sleep(60)\n")
    children = []

    def start(env, arm="left", command=None, new_session=True):
        child = subprocess.Popen(
            command or [sys.executable, str(script), "launch", "agx_arm_ctrl",
                        "start_single_agx_arm.launch.py", f"namespace:={arm}_arm"],
            env=env, start_new_session=new_session,
        )
        children.append(child)
        return child

    yield start
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)


@pytest.mark.parametrize("sdk_override", [False, True])
def test_untracked_launcher_blocks_duplicate_and_is_stopped(project, spawn, sdk_override):
    root, env = project
    if sdk_override:
        env["PYAGXARM_ROOT"] = str(root.parent / "custom_sdk")
        env["PYTHONPATH"] = os.pathsep.join((env["PYAGXARM_ROOT"], str(root / "src")))
    child = spawn(env)
    assert processes.owned_launcher(processes.process_info(child.pid), root)
    with pytest.raises(RuntimeError, match="Refusing duplicate left driver"):
        processes.launch(root, root / "logs", "left", ["must-not-execute"])
    assert processes.stop_project(root, root / "logs", timeout=0.2) == [child.pid]
    child.wait(timeout=3)
    assert processes.process_info(child.pid) is None


@pytest.mark.parametrize("tracking", ["legacy", "stale", "current"])
def test_tracked_launcher_only_signals_verified_session(project, spawn, monkeypatch, tracking):
    root, env = project
    child = spawn(env)
    logs = root / "logs"
    logs.mkdir()
    (logs / "driver_left.pid").write_text(str(child.pid))
    if tracking != "legacy":
        identity = {**processes.process_info(child.pid), "root": str(root)}
        if tracking == "stale":
            identity["start"] -= 1
        (logs / "driver_left.identity.json").write_text(json.dumps(identity))
    killpg = os.killpg
    groups = []

    def signal_group(pgid, sig):
        assert pgid == child.pid
        groups.append(pgid)
        killpg(pgid, sig)

    monkeypatch.setattr(processes.os, "killpg", signal_group)
    assert processes.stop_project(root, logs, timeout=0.2) == [child.pid]
    child.wait(timeout=3)
    assert groups == ([child.pid] if tracking == "current" else [])
    assert not (logs / "driver_left.pid").exists()
    assert not (logs / "driver_left.identity.json").exists()


def test_untracked_launcher_does_not_signal_shared_shell_group(project, spawn, monkeypatch):
    root, env = project
    child = spawn(env, new_session=False)
    unrelated = spawn(env, command=[sys.executable, "-c", "import time; time.sleep(60)"],
                      new_session=False)
    assert os.getpgid(child.pid) == os.getpgid(unrelated.pid) == os.getpgrp()
    monkeypatch.setattr(processes.os, "killpg", lambda *args: pytest.fail("Shared group signalled"))
    assert processes.stop_project(root, root / "logs", timeout=0.2) == [child.pid]
    child.wait(timeout=3)
    assert unrelated.poll() is None


@pytest.mark.parametrize("mismatch", ["domain", "project", "sdk_only", "installation", "overlay", "no_ament"])
def test_external_launcher_is_not_owned_or_stopped(project, spawn, mismatch):
    root, env = project
    if mismatch == "domain":
        env["ROS_DOMAIN_ID"] = "74"
    elif mismatch == "project":
        env["PYTHONPATH"] = str(root.parent / "other/src")
    elif mismatch == "sdk_only":
        env["PYAGXARM_ROOT"] = str(root.parent / "custom_sdk")
        env["PYTHONPATH"] = env["PYAGXARM_ROOT"]
    elif mismatch == "no_ament":
        env.pop("AMENT_PREFIX_PATH")
    else:
        prefix = root.parent / "pika_ros/install/agx_arm_ctrl"
        marker = prefix / "share/ament_index/resource_index/packages/agx_arm_ctrl"
        marker.parent.mkdir(parents=True)
        marker.touch()
        env["AMENT_PREFIX_PATH"] = str(prefix) + (
            os.pathsep + env["AMENT_PREFIX_PATH"] if mismatch == "overlay" else "")
    child = spawn(env)
    assert not processes.owned_launcher(processes.process_info(child.pid), root)
    logs = root / "logs"
    logs.mkdir()
    (logs / "driver_left.pid").write_text(str(child.pid))
    assert processes.stop_project(root, logs, timeout=0.1) == []
    assert child.poll() is None


def test_external_node_keeps_exact_installation_scope(project, spawn, tmp_path):
    root, env = project
    node = tmp_path / "pika_ros/install/agx_arm_ctrl/lib/agx_arm_ctrl/agx_arm_ctrl_single"
    node.parent.mkdir(parents=True)
    node.write_text("import time\ntime.sleep(60)\n")
    child = spawn(env, command=[sys.executable, str(node), "__ns:=/left_arm"])
    assert not processes.owned_driver(processes.process_info(child.pid), root)
    assert processes.stop_project(root, root / "logs", timeout=0.1) == []
    assert child.poll() is None


def test_status_includes_launcher_without_pid_file(project, spawn, monkeypatch, capsys):
    root, env = project
    child = spawn(env)
    monkeypatch.setattr(processes, "__file__", str(root / "scripts/driver_processes.py"))
    monkeypatch.setattr(sys, "argv", ["driver_processes.py", "status"])
    processes.main()
    assert [info["pid"] for info in json.loads(capsys.readouterr().out)] == [child.pid]


def test_post_stop_verification_detects_new_launcher(project, spawn, monkeypatch):
    root, env = project
    child = spawn(env)
    info = processes.process_info(child.pid)
    snapshots = iter(({}, {child.pid: info}))
    monkeypatch.setattr(processes, "process_snapshot", lambda: next(snapshots))
    logs = root / "logs"
    logs.mkdir()
    pid_file = logs / "driver_left.pid"
    pid_file.write_text("99999999")
    with pytest.raises(RuntimeError, match="processes still alive after cleanup"):
        processes.stop_project(root, logs, timeout=0.1)
    assert pid_file.exists()
    assert child.poll() is None


@pytest.mark.parametrize("change", ["reused", "exited", "group", "command"])
def test_group_signal_rechecks_leader_identity(project, monkeypatch, change):
    root, env = project
    leader = {"pid": 900001, "ppid": 1, "pgid": 900001, "start": 123,
              "argv": ["ros2", "launch", "agx_arm_ctrl", "start_single_agx_arm.launch.py",
                       "namespace:=left_arm"]}
    child = {"pid": 900002, "ppid": leader["pid"], "pgid": leader["pgid"], "start": 124,
             "argv": ["synthetic-child"]}
    snapshots = iter(({leader["pid"]: leader, child["pid"]: child}, {}))
    monkeypatch.setattr(processes, "process_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(processes, "environment", lambda pid: env)
    current = dict(leader)
    if change == "reused":
        current["start"] += 1
    elif change == "exited":
        current = None
    elif change == "group":
        current["pgid"] += 10
    else:
        current["argv"] = ["unrelated"]
    monkeypatch.setattr(processes, "process_info", lambda pid: current if pid == leader["pid"] else None)
    monkeypatch.setattr(processes, "same_process", lambda info: info["pid"] == child["pid"])
    monkeypatch.setattr(processes.time, "sleep", lambda seconds: None)
    clock = iter(range(100))
    monkeypatch.setattr(processes.time, "monotonic", lambda: next(clock))
    signals = []
    monkeypatch.setattr(processes.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(processes.os, "killpg", lambda *args: pytest.fail("Unsafe group signal"))
    logs = root / "logs"
    logs.mkdir()
    (logs / "driver_left.identity.json").write_text(json.dumps({**leader, "root": str(root)}))
    processes.stop_project(root, logs, timeout=0)
    assert (child["pid"], processes.signal.SIGTERM) in signals


def test_process_info_excludes_other_users(project, spawn, monkeypatch):
    root, env = project
    child = spawn(env)
    uid = os.getuid()
    monkeypatch.setattr(processes.os, "getuid", lambda: uid + 1)
    assert processes.process_info(child.pid) is None
    assert processes.stop_project(root, root / "logs", timeout=0.1) == []
    assert child.poll() is None


@pytest.mark.parametrize("empty", [False, True])
def test_diagnose_reports_external_nodes_and_parent_chain_read_only(project, monkeypatch, capsys, empty):
    root, env = project
    own = root / "thirdparty/ros_ws/install/agx_arm_ctrl/lib/agx_arm_ctrl/agx_arm_ctrl_single"
    external = root.parent / "pika_ros/install/agx_arm_ctrl/lib/agx_arm_ctrl/agx_arm_ctrl_single"
    rows = [
        {"pid": 101, "ppid": 100, "argv": ["python3", str(own), "__ns:=/left_arm"]},
        {"pid": 102, "ppid": 100, "argv": ["python3", str(external), "__ns:=/right_arm"]},
        {"pid": 103, "ppid": 100, "argv": ["python3", str(external), "__ns:=/other_arm"]},
        {"pid": 104, "ppid": 100, "argv": ["python3", str(external), "__ns:=/left_arm"]},
        {"pid": 100, "ppid": 99, "argv": ["ros2", "launch", "pika_remote_agx_arm", "teleop_double_nero.launch.py"]},
        {"pid": 99, "ppid": 1, "argv": ["bash", "run_offset_teleop.sh"]},
    ]
    monkeypatch.setattr(processes, "process_snapshot", lambda: {} if empty else {row["pid"]: row for row in rows})
    monkeypatch.setattr(processes, "environment", lambda pid: {**env, "ROS_DOMAIN_ID": "74"} if pid == 104 else env)
    monkeypatch.setattr(processes, "__file__", str(root / "scripts/driver_processes.py"))
    monkeypatch.setattr(sys, "argv", ["driver_processes.py", "diagnose"])
    monkeypatch.setattr(processes.os, "kill", lambda *args: pytest.fail("Diagnose must not signal"))
    monkeypatch.setattr(processes.os, "killpg", lambda *args: pytest.fail("Diagnose must not signal groups"))
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Diagnose must not spawn"))
    processes.main()
    report = json.loads(capsys.readouterr().out)
    assert "remote DDS" in report["note"]
    assert report["ROS_DOMAIN_ID"] == "73"
    if empty:
        assert report["nodes"] == []
    else:
        assert [(node["pid"], node["owned"]) for node in report["nodes"]] == [(101, True), (102, False)]
        assert report["nodes"][1]["executable"] == str(external)
        assert report["nodes"][1]["argv"] == rows[1]["argv"]
        assert report["nodes"][1]["ppid"] == 100
        assert report["nodes"][1]["parents"] == rows[4:]


@pytest.mark.parametrize("selection", ["project_src", "project_install", "external_ament", "external_python",
                                        "missing_ament", "missing_python"])
@pytest.mark.parametrize("speed", [None, "15"])
def test_cli_launch_preflight_without_importing_driver(project, monkeypatch, capsys, selection, speed):
    root, env = project
    expected = root / "thirdparty/ros_ws/install/agx_arm_ctrl"
    external = root.parent / "pika_ros/install/agx_arm_ctrl"
    monkeypatch.delenv("SPEED_PERCENT", raising=False)
    if speed is not None:
        monkeypatch.setenv("SPEED_PERCENT", speed)

    def get_package_prefix(name):
        assert name == "agx_arm_ctrl"
        if selection == "missing_ament":
            raise LookupError("package not found")
        return str(external if selection == "external_ament" else expected)

    monkeypatch.setitem(sys.modules, "ament_index_python.packages",
                        SimpleNamespace(get_package_prefix=get_package_prefix))
    package_root = (external if selection == "external_python" else
                    expected if selection == "project_install" else root / "thirdparty/ros_ws/src/agx_arm_ctrl")
    package = package_root / "agx_arm_ctrl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('Driver must not be imported')\n")
    monkeypatch.delitem(sys.modules, "agx_arm_ctrl", raising=False)
    # Isolate module lookup from any actual ROS installation on the test host.
    monkeypatch.setattr(sys, "path", [] if selection == "missing_python" else [str(package_root)])
    monkeypatch.setattr(processes, "__file__", str(root / "scripts/driver_processes.py"))
    monkeypatch.setattr(sys, "argv", ["driver_processes.py", "launch", "--arm", "left", "--can-port", "synthetic"])
    launches = []
    monkeypatch.setattr(processes, "launch", lambda *args: launches.append(args) or 123)
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not launch a driver"))
    if selection.startswith("project_"):
        processes.main()
        assert len(launches) == 1
        command = launches[0][-1]
        assert "auto_enable:=false" in command
        assert "preserve_controller_state:=true" in command
        assert f"speed_percent:={speed or '30'}" in command
    else:
        with pytest.raises(SystemExit) as exc:
            processes.main()
        assert exc.value.code == 2
        message = capsys.readouterr().err
        assert "Refusing driver launch" in message
        assert ("PYTHONPATH" if selection.endswith("python") else "agx_arm_ctrl") in message
        assert launches == []
    assert "agx_arm_ctrl" not in sys.modules


@pytest.mark.parametrize("speed", ["0", "101", "nan", "inf", "3.5", ""])
def test_invalid_handoff_speed_fails_before_package_lookup(monkeypatch, speed):
    monkeypatch.setenv("SPEED_PERCENT", speed)
    monkeypatch.setattr(sys, "argv", ["driver_processes.py", "check-environment"])
    monkeypatch.setattr(processes, "launch", lambda *args: pytest.fail("Must not launch a driver"))
    with pytest.raises(SystemExit) as exc:
        processes.main()
    assert exc.value.code == 2


def test_ament_index_selects_first_overlay_without_loading_driver(project, monkeypatch):
    packages = pytest.importorskip("ament_index_python.packages")
    root, env = project
    external = root.parent / "pika_ros/install/agx_arm_ctrl"
    marker = external / "share/ament_index/resource_index/packages/agx_arm_ctrl"
    marker.parent.mkdir(parents=True)
    marker.touch()
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(external) + os.pathsep + env["AMENT_PREFIX_PATH"])
    assert packages.get_package_prefix("agx_arm_ctrl") == str(external)
