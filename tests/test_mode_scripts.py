import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,mode", [("start_rtc.sh", "rtc"), ("start_no_rtc.sh", "sync")])
@pytest.mark.parametrize("recover", [None, "0", "1"])
def test_wrappers_select_mode_and_forward_arguments_without_starting_hardware(tmp_path, name, mode, recover):
    # Intercept the wrapper's exec bash, so start.sh never runs in this test.
    bash = tmp_path / "bash"
    bash.write_text('#!/bin/sh\nprintf "recover=%s\\n" "${RECOVER_BEFORE_START:-0}"\nprintf "%s\\n" "$@"\n')
    bash.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "REPLAN_STEPS": "5"}
    env.pop("RECOVER_BEFORE_START", None)
    if recover is not None:
        env["RECOVER_BEFORE_START"] = recover
    result = subprocess.run(["/bin/bash", str(ROOT / "scripts" / name), "--max-steps", "100"],
                            env=env, capture_output=True, text=True, check=True)
    args = result.stdout.splitlines()
    assert args[0] == f"recover={recover if recover is not None else '1'}"
    assert len([arg for arg in args if arg.endswith(".sh")]) == 1
    assert args.count(str(ROOT / "scripts/start.sh")) == 1
    args = args[args.index(str(ROOT / "scripts/start.sh")):]
    assert args[0] == str(ROOT / "scripts/start.sh")
    assert args[-2:] == ["--inference-mode", mode]
    assert args[args.index("--max-steps") + 1] == "100"
    if mode == "sync":
        assert args[args.index("--replan-steps") + 1] == "5"
        assert "--no-grasp-control" in args


@pytest.mark.parametrize("name", ["start_rtc.sh", "start_no_rtc.sh"])
def test_mode_script_help_does_not_start_drivers(name):
    result = subprocess.run(["/bin/bash", str(ROOT / "scripts" / name), "--help"],
                            capture_output=True, text=True, check=True)
    assert "Usage:" in result.stdout
