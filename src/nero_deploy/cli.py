from __future__ import annotations

import argparse
import logging
import math
import signal
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException

from .cameras import RealSenseCamera
from .config import load_config
from .control import DualArmJointEMA, clamp_action, clamp_action_step, validate_action
from .control.filter import GripperDebouncer
from .control.grasp import GraspCoordinator, resolve_grasp_config
from .control.inference import create_inference_broker, resolve_inference_settings
from .diagnostics.action_trace import ActionTrace
from .policy_client import PolicyClient
from .ros import DualArmBridge


def main() -> None:
    startup_started = time.monotonic()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the Nero three-camera ROS2 policy client.")
    parser.add_argument("--config", type=Path, default=Path("config/nero.yaml"))
    parser.add_argument("--prompt")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--camera-warmup-seconds", type=float,
                        help="Camera exposure settling time; defaults to cameras.warmup_seconds")
    parser.add_argument("--inference-mode", choices=("rtc", "sync", "async"),
                        help="rtc: async with guidance; sync: ordinary blocking inference; async: unguided async")
    parser.add_argument("--replan-steps", type=int,
                        help="Number of leading actions to execute in sync mode (1..16, default 8)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--grasp-control", action=argparse.BooleanOptionalAction, default=None,
                        help="RTC-only grasp coordination; all modes confirm opening twice, closing once")
    args = parser.parse_args()
    cfg = load_config(args.config)
    policy_cfg = cfg["policy"]
    camera_cfg = cfg["cameras"]
    warmup_seconds = (args.camera_warmup_seconds if args.camera_warmup_seconds is not None
                      else float(camera_cfg.get("warmup_seconds", 3)))
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
        parser.error("camera warmup seconds must be finite and non-negative")
    control_cfg = cfg["control"]
    prompt = args.prompt or policy_cfg["prompt"]
    max_steps = args.max_steps or int(control_cfg["max_steps"])
    hz = float(control_cfg["hz"])
    # Zero means no wall-time limit. Synchronous inference adds waiting time,
    # so max_steps/hz cannot predict an episode's duration.
    max_seconds = float(control_cfg.get("max_episode_seconds", 0))
    rtc_cfg = policy_cfg.get("rtc", {})
    inference_mode, replan_steps = resolve_inference_settings(policy_cfg, args.inference_mode, args.replan_steps)
    logging.info("Inference mode: %s; sync replan_steps=%s; policy=%s:%s", inference_mode,
                 replan_steps if inference_mode == "sync" else "n/a", policy_cfg["host"], policy_cfg["port"])
    ema_cfg = control_cfg.get("joint_ema", {})
    ema = DualArmJointEMA(float(ema_cfg.get("alpha", 0.35))) if ema_cfg.get("enabled", False) else None
    grasp_cfg = resolve_grasp_config(inference_mode, control_cfg.get("grasp", {}), args.grasp_control)
    grasp = GraspCoordinator(grasp_cfg) if grasp_cfg.enabled else None
    gripper_filter = GripperDebouncer(confirm_steps=2) if grasp is None else None
    gripper_execution = "coordinated_open2_close1" if grasp is not None else "open2_close1"
    logging.info("Grasp coordination: %s; inference mode=%s",
                 "enabled" if grasp is not None else "disabled", inference_mode)
    logging.info("Gripper execution: %s", gripper_execution)
    trace = None

    stop = threading.Event()
    bridge = None
    spin_thread = None
    cameras = {}
    broker = None

    def spin(node):
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass
        finally:
            stop.set()

    rclpy.init()
    try:
        # rclpy.init installs ROS handlers; our handlers must take precedence.
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        bridge = DualArmBridge()
        spin_thread = threading.Thread(target=spin, args=(bridge,), daemon=True)
        spin_thread.start()
        for name in ("left_wrist", "right_wrist", "third_person"):
            if stop.is_set():
                return
            cameras[name] = RealSenseCamera(
                camera_cfg[name]["serial"],
                camera_cfg["width"],
                camera_cfg["height"],
                camera_cfg["fps"],
                auto_exposure=bool(camera_cfg.get("auto_exposure", True)),
                exposure_us=camera_cfg.get("exposure_us"),
            )
        logging.info("Startup: cameras initialized in %.2fs", time.monotonic() - startup_started)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if stop.is_set():
                return
            state, _ = bridge.state()
            if state is not None and all(camera.snapshot()[0] is not None for camera in cameras.values()):
                break
            time.sleep(0.05)
        else:
            if stop.is_set():
                return
            camera_ready = {name: camera.snapshot()[0] is not None for name, camera in cameras.items()}
            raise RuntimeError(
                "startup readiness timeout: "
                f"arms={bridge.readiness()} cameras={camera_ready}"
            )
        if stop.is_set():
            return
        warmup_deadline = time.monotonic() + warmup_seconds
        logging.info("Startup: inputs ready in %.2fs; camera warmup %.2fs (overlaps policy connection)",
                     time.monotonic() - startup_started, warmup_seconds)
        policy = PolicyClient(
            policy_cfg["host"],
            int(policy_cfg["port"]),
            policy_cfg.get("image_size", 224),
            control_cfg["joint_limit_margin"],
        )
        logging.info("Startup: policy connected in %.2fs; no policy commands published yet",
                     time.monotonic() - startup_started)
        if stop.wait(max(0.0, warmup_deadline - time.monotonic())):
            return
        if args.execute:
            bridge.validate_exclusive_control()
        if stop.is_set():
            return
        logging.info("Startup: ready for inference in %.2fs", time.monotonic() - startup_started)
        broker = create_inference_broker(policy, policy_cfg, hz, inference_mode, replan_steps)
        gripper_open = {"left": True, "right": True}
        trace_cfg = control_cfg.get("action_trace", {})
        if trace_cfg.get("enabled", True):
            output_dir = Path(trace_cfg.get("output_dir", "outputs/action_traces"))
            if not output_dir.is_absolute():
                output_dir = Path(__file__).resolve().parents[2] / output_dir
            trace = ActionTrace(output_dir, {
                "prompt": prompt, "control": control_cfg, "rtc": rtc_cfg,
                "inference_mode": inference_mode,
                "rtc_guidance_enabled": inference_mode == "rtc",
                "sync_replan_steps": replan_steps if inference_mode == "sync" else None,
                "grasp": vars(grasp_cfg), "trace_version": 2,
                "gripper_execution": gripper_execution,
            })
        executed = 0
        started = time.monotonic()
        next_tick = started
        while (not stop.is_set() and executed < max_steps
               and (max_seconds <= 0 or time.monotonic() - started < max_seconds)):
            state, state_age = bridge.state()
            frames = {name: camera.snapshot() for name, camera in cameras.items()}
            if state is None or state_age > control_cfg["max_source_age"]:
                raise RuntimeError("dual-arm feedback is stale")
            if any(frame is None or age > control_cfg["max_source_age"] for frame, age in frames.values()):
                raise RuntimeError("camera frame is stale")
            observation = {
                "images": {name: frames[name][0] for name in ("left_wrist", "right_wrist", "third_person")},
                "state": state,
                "prompt": prompt,
            }
            held_tick = grasp is not None and grasp.active
            if stop.is_set():
                break
            result = broker.infer(observation) if grasp is None else grasp.infer(broker, observation)
            if stop.is_set():
                break
            # Refresh feedback only for the new RTC coordinator. The no-RTC
            # baseline retains its original pre-inference state for EMA/clamping.
            if grasp is not None:
                state, state_age = bridge.state()
                if state is None or state_age > control_cfg["max_source_age"]:
                    raise RuntimeError("dual-arm feedback is stale after inference")
            forces = bridge.gripper_forces()
            action = np.asarray(result["actions"], dtype=np.float32)
            policy_action = action.copy()
            if action.shape != (16,):
                raise RuntimeError(f"broker must return one action with shape (16,), got {action.shape}")
            action, clipped_values, max_clip = clamp_action(
                action, control_cfg["joint_limit_margin"]
            )
            decision = None
            if grasp is not None:
                if executed == 0:
                    grasp.initialize(state, time.monotonic())
                decision = grasp.step(None if held_tick else action, state, time.monotonic(), forces)
                action = decision.action
                gripper_open = decision.gripper_open
            result = dict(result)
            result["action_advanced"] = decision is None or not decision.held
            result["grasp_phase"] = "disabled" if decision is None else decision.phase
            result["grasp_event"] = "" if decision is None else decision.event
            result["gripper_force"] = forces
            result["grasp_target"] = decision.action if decision is not None and decision.held else np.full(16, np.nan)
            if ema is not None:
                if executed == 0:
                    ema.initialize(state)
                action = ema.apply(action)
            ema_action = action.copy()
            action, step_clipped_values, step_max_clip = clamp_action_step(
                action, state, control_cfg["max_joint_step"]
            )
            validate_action(action, state, control_cfg["max_joint_step"], control_cfg["joint_limit_margin"])
            if grasp is None:
                if executed == 0:
                    gripper_filter.initialize(state)
                gripper_open = gripper_filter.update(action)
            if clipped_values or step_clipped_values:
                print(
                    f"action_clamp joint_limits={clipped_values} max_joint={max_clip:.4f} "
                    f"step={step_clipped_values} max_step={step_max_clip:.4f}",
                    flush=True,
                )
            if stop.is_set():
                break
            if args.execute:
                if executed == 0:
                    logging.info("Startup: publishing first policy action at %.2fs",
                                 time.monotonic() - startup_started)
                bridge.publish(action, gripper_open)
                if ema is not None:
                    ema.commit(action)
                if trace is not None:
                    # Record the actual gripper widths published by the bridge.
                    command = action.copy()
                    command[[7, 15]] = [0.1 if gripper_open[arm] else 0.0 for arm in ("left", "right")]
                    trace.record(
                        timestamp=time.monotonic(), state=state, policy_action=policy_action,
                        ema_action=ema_action, command=command, result=result, state_age=state_age,
                    )
                executed += 1
                if decision is not None and decision.replan:
                    # Drop queued AND in-flight old plans; do not consume their
                    # lift/retract suffix after a reach/gripper hold.
                    grasp.after_publish(broker, decision)
            else:
                print("DRY RUN: broker returned one action; no ROS control published", flush=True)
                return
            # Include observation preparation/inference/postprocessing in the tick
            # budget. After a missed deadline, resume without a catch-up burst.
            next_tick += 1 / hz
            now = time.monotonic()
            if next_tick <= now:
                next_tick = now + 1 / hz
            stop.wait(next_tick - now)
        print(f"stopped after {executed}/{max_steps} control steps", flush=True)
    finally:
        stop.set()
        cleanup = []
        if trace is not None:
            cleanup.append(("action trace", trace.save))
        cleanup.extend((f"camera {name}", camera.close) for name, camera in cameras.items())
        if broker is not None:
            cleanup.append(("inference broker", broker.close))
        # Wake the executor and let spin exit before destroying its node.
        cleanup.append(("ROS context", rclpy.try_shutdown))
        if spin_thread is not None:
            cleanup.append(("spin thread", lambda: spin_thread.join(timeout=2)))
        if bridge is not None:
            cleanup.append(("ROS bridge", bridge.destroy_node))
        for name, close in cleanup:
            try:
                close()
            except Exception:
                logging.exception("Could not clean up %s", name)


if __name__ == "__main__":
    main()
