#!/usr/bin/env python3
"""Manage only this deployment's driver processes, including orphaned ROS nodes."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ARMS = ("left", "right")


def process_info(pid):
    try:
        path = Path("/proc") / str(pid)
        if path.stat().st_uid != os.getuid():
            return None
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        argv = [part.decode(errors="replace") for part in (path / "cmdline").read_bytes().split(b"\0") if part]
        return {"pid": int(pid), "ppid": int(fields[1]), "pgid": int(fields[2]),
                "start": int(fields[19]), "argv": argv}
    except (OSError, ValueError, IndexError):
        return None


def environment(pid):
    try:
        return dict(part.decode(errors="replace").split("=", 1)
                    for part in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in part)
    except OSError:
        return {}


def namespace(argv):
    for arm in ARMS:
        if f"__ns:=/{arm}_arm" in argv or f"namespace:={arm}_arm" in argv:
            return arm
    return None


def owned_node(info, root):
    executable = str(root / "thirdparty/ros_ws/install/agx_arm_ctrl/lib/agx_arm_ctrl/agx_arm_ctrl_single")
    return (info is not None and executable in info["argv"] and namespace(info["argv"]) is not None
            and environment(info["pid"]).get("ROS_DOMAIN_ID", "0") == os.environ.get("ROS_DOMAIN_ID", "0"))


def owned_launcher(info, root):
    if info is None or namespace(info["argv"]) is None:
        return False
    argv = info["argv"]
    env = environment(info["pid"])
    pythonpath = env.get("PYTHONPATH", "").split(os.pathsep)
    if (not any(Path(arg).name == "ros2" and argv[index + 1:index + 4] ==
                ["launch", "agx_arm_ctrl", "start_single_agx_arm.launch.py"]
                for index, arg in enumerate(argv))
            or not any(str(path) in pythonpath for path in (root / "thirdparty/pyAgxArm", root / "src"))
            or env.get("ROS_DOMAIN_ID", "0") != os.environ.get("ROS_DOMAIN_ID", "0")):
        return False
    # Match the first package selected by ament, not an external overlay that
    # merely inherited this project's PYTHONPATH. Project src also proves
    # provenance when startup uses a PYAGXARM_ROOT override.
    for prefix in env.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        path = Path(prefix)
        if not path.is_absolute():
            return False
        if (path / "share/ament_index/resource_index/packages/agx_arm_ctrl").is_file():
            return path == root / "thirdparty/ros_ws/install/agx_arm_ctrl"
    return False


def owned_driver(info, root):
    return owned_node(info, root) or owned_launcher(info, root)


def process_snapshot():
    return {info["pid"]: info for path in Path("/proc").iterdir() if path.name.isdigit()
            if (info := process_info(int(path.name))) is not None}


def same_process(info):
    current = process_info(info["pid"])
    return current is not None and current["start"] == info["start"]


def stop_project(root, log_dir, timeout=3.0):
    snapshot = process_snapshot()
    targets = {pid: info for pid, info in snapshot.items() if owned_driver(info, root)}
    groups = {}
    for arm in ARMS:
        pid_file = log_dir / f"driver_{arm}.pid"
        identity_file = log_dir / f"driver_{arm}.identity.json"
        identity = None
        try:
            if identity_file.exists():
                identity = json.loads(identity_file.read_text())
                pid = int(identity["pid"])
            else:
                pid = int(pid_file.read_text().strip())
        except (OSError, ValueError, KeyError):
            continue
        info = snapshot.get(pid)
        valid = (info is not None and owned_driver(info, root)
                 and (identity is None or info["start"] == identity.get("start")))
        if valid:
            targets[pid] = info
        # New launches have an independent session/group. An old shell's process
        # group is NEVER signalled based on a bare legacy PID file.
        if (identity is not None and identity.get("root") == str(root)
                and identity.get("pgid") == pid
                and valid and info["pgid"] == pid):
            groups[pid] = info
    # Capture descendants before a launcher exits and reparents them to PID 1.
    while True:
        children = {pid: info for pid, info in snapshot.items()
                    if info["ppid"] in targets and pid not in targets}
        if not children:
            break
        targets.update(children)
    signalled_groups = set()
    for pgid, leader in groups.items():
        current = process_info(pgid)
        if (current is None or current["start"] != leader["start"]
                or current["pgid"] != pgid or not owned_driver(current, root)):
            continue
        targets.update({pid: info for pid, info in snapshot.items() if info["pgid"] == pgid})
        try:
            os.killpg(pgid, signal.SIGTERM)
            signalled_groups.add(pgid)
        except ProcessLookupError:
            pass
    for info in targets.values():
        if info["pgid"] not in signalled_groups and same_process(info):
            try:
                os.kill(info["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while any(same_process(info) for info in targets.values()) and time.monotonic() < deadline:
        time.sleep(0.05)
    for info in targets.values():
        if same_process(info):
            try:
                os.kill(info["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 2
    while any(same_process(info) for info in targets.values()) and time.monotonic() < deadline:
        time.sleep(0.05)
    remaining = [pid for pid, info in process_snapshot().items() if owned_driver(info, root)]
    if remaining:
        raise RuntimeError(f"Project driver processes still alive after cleanup: {remaining}")
    for arm in ARMS:
        for suffix in ("pid", "identity.json"):
            (log_dir / f"driver_{arm}.{suffix}").unlink(missing_ok=True)
    return sorted(targets)


def launch(root, log_dir, arm, command):
    existing = [pid for pid, info in process_snapshot().items()
                if owned_driver(info, root) and namespace(info["argv"]) == arm]
    if existing:
        raise RuntimeError(f"Refusing duplicate {arm} driver; existing project processes: {existing}")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"driver_{arm}.log").open("w") as log:
        # close_fds prevents drivers from retaining the startup/lifecycle flock.
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    try:
        info = process_info(process.pid)
        if info is None:
            raise RuntimeError(f"{arm} driver exited immediately; inspect its log")
        identity = {key: info[key] for key in ("pid", "pgid", "start")}
        identity["root"] = str(root)
        (log_dir / f"driver_{arm}.identity.json").write_text(json.dumps(identity))
        (log_dir / f"driver_{arm}.pid").write_text(str(process.pid) + "\n")
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        raise
    return process.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "diagnose", "stop", "check-environment", "launch"))
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--can-port")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    log_dir = args.log_dir or root / "outputs/logs"
    if args.action == "status":
        print(json.dumps([info for info in process_snapshot().values() if owned_driver(info, root)], indent=2))
    elif args.action == "diagnose":
        snapshot = process_snapshot()
        domain = os.environ.get("ROS_DOMAIN_ID", "0")
        nodes = []
        for info in snapshot.values():
            executable = next((arg for arg in info["argv"]
                               if Path(arg).name == "agx_arm_ctrl_single"), None)
            if (executable is None or namespace(info["argv"]) is None
                    or environment(info["pid"]).get("ROS_DOMAIN_ID", "0") != domain):
                continue
            parents = []
            seen = {info["pid"]}
            parent = snapshot.get(info["ppid"])
            while parent is not None and parent["pid"] not in seen:
                parents.append(parent)
                seen.add(parent["pid"])
                parent = snapshot.get(parent["ppid"])
            nodes.append({**info, "executable": executable, "owned": owned_node(info, root),
                          "parents": parents})
        print(json.dumps({"note": "Read-only local same-user process snapshot; remote DDS endpoints "
                                 "may not be visible. Parent chains identify owning launches; "
                                 "external processes are not stopped by project cleanup.",
                          "ROS_DOMAIN_ID": domain, "nodes": nodes}, indent=2))
    elif args.action == "stop":
        print("Stopped project driver PIDs:", stop_project(root, log_dir))
    else:
        if args.action == "launch" and (args.arm is None or args.can_port is None):
            parser.error("launch requires --arm and --can-port")
        try:
            speed_percent = int(os.environ.get("SPEED_PERCENT", "30"))
        except ValueError:
            parser.error("SPEED_PERCENT must be an integer in [1, 100]")
        if not 1 <= speed_percent <= 100:
            parser.error("SPEED_PERCENT must be an integer in [1, 100]")
        expected = root / "thirdparty/ros_ws/install/agx_arm_ctrl"
        try:
            from ament_index_python.packages import get_package_prefix

            prefix = get_package_prefix("agx_arm_ctrl")
        except (ImportError, LookupError, OSError, ValueError) as exc:
            parser.error(f"Refusing driver launch: cannot resolve agx_arm_ctrl: {exc}")
        if Path(prefix) != expected:
            parser.error(f"Refusing driver launch: agx_arm_ctrl resolves to {prefix}; expected {expected}. "
                         "Remove the inherited external ROS overlay from AMENT_PREFIX_PATH.")
        # Top-level find_spec locates the package without importing driver code.
        spec = importlib.util.find_spec("agx_arm_ctrl")
        origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
        if origin is None or not any(origin.is_relative_to(path) for path in (
                expected, root / "thirdparty/ros_ws/src/agx_arm_ctrl",
                root / "thirdparty/ros_ws/build/agx_arm_ctrl")):
            parser.error(f"Refusing driver launch: Python agx_arm_ctrl resolves to {origin}; "
                          "expected this project's ROS workspace. Check inherited PYTHONPATH.")
        if args.action == "check-environment":
            return
        command = ["ros2", "launch", "agx_arm_ctrl", "start_single_agx_arm.launch.py",
                   f"namespace:={args.arm}_arm", f"can_port:={args.can_port}", "arm_type:=nero",
                   "effector_type:=agx_gripper", "auto_enable:=false", "control_enabled:=true",
                   "preserve_controller_state:=true", f"speed_percent:={speed_percent}",
                   "tcp_offset:=[0.1755, 0.0, -0.0235, 0.0, 0.0, 0.0]"]
        print(f"Started {args.arm} driver session PID {launch(root, log_dir, args.arm, command)}")


if __name__ == "__main__":
    main()
