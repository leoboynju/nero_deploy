from __future__ import annotations

import argparse
import math
import struct
import sys
import threading
import time
from typing import NamedTuple

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


TARGETS = {
    "left": [-0.19362683, -1.10344951, -0.10292207, 1.62465719, -0.09833185, -0.07915068, 1.55023635],
    "right": [0.19362683, -1.10344951, 0.10292207, 1.62465719, 0.09833185, 0.07915068, 1.55023635],
}


_FEEDBACK_IDS = (0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A9)


class _Frame(NamedTuple):
    count: int
    timestamp: float
    received_at: float
    values: tuple


class _Feedback:
    def __init__(self):
        self._lock = threading.Lock()
        self._frames = dict.fromkeys(_FEEDBACK_IDS)
        self._counts = dict.fromkeys(_FEEDBACK_IDS, 0)
        self._error = None
        self._commanded_at = None
        self._interval_valid = True
        self._extrema = {}

    def __call__(self, frame):
        can_id = frame.arbitration_id
        if can_id not in self._frames:
            return
        with self._lock:
            self._counts[can_id] += 1
            try:
                data = bytes(frame.data)
                timestamp = float(frame.timestamp)
                if (len(data) != 8 or frame.dlc != 8 or frame.is_extended_id
                        or frame.is_remote_frame or frame.is_error_frame
                        or not math.isfinite(timestamp) or timestamp <= 0):
                    raise ValueError("invalid payload, flags, or timestamp")
                if can_id == 0x2A1:
                    # Raw CAN/MOVE J codes are shared by Nero firmware profiles.
                    values = struct.unpack(">6BH", data)
                    if values[1] != 0 or values[6] != 0:
                        self._error = f"arm fault: arm_status={values[1]} err_code=0x{values[6]:04x}"
                else:
                    angles = struct.unpack(">i", data[:4]) if can_id == 0x2A9 else struct.unpack(">2i", data)
                    values = tuple(math.radians(angle / 1000) for angle in angles)
                if self._commanded_at is not None:
                    previous = self._frames[can_id]
                    if (timestamp <= self._commanded_at
                            or not 0 <= time.time() - timestamp <= 0.25
                            or (previous is not None and timestamp <= previous.timestamp)):
                        self._interval_valid = False
                    elif can_id == 0x2A1:
                        if values[:5] != (1, 0, 1, 0, 0) or values[6] != 0:
                            self._interval_valid = False
                    else:
                        low, high = self._extrema.get(can_id, (values, values))
                        self._extrema[can_id] = (
                            tuple(min(a, b) for a, b in zip(low, values)),
                            tuple(max(a, b) for a, b in zip(high, values)),
                        )
                self._frames[can_id] = _Frame(
                    self._counts[can_id], timestamp, time.monotonic(), values,
                )
            except (AttributeError, TypeError, ValueError, OverflowError, struct.error) as exc:
                self._frames[can_id] = None
                self._error = f"malformed recovery feedback 0x{can_id:x}: {exc}"

    def begin_wait(self):
        with self._lock:
            self._error = None
            self._interval_valid = True
            self._extrema = {}
            # SocketCAN timestamps use wall time; deadlines use monotonic time.
            self._commanded_at = time.time()
            return self._commanded_at, tuple(self._counts.values())

    def snapshot(self):
        """Atomically consume interval evidence while retaining latest frames."""
        with self._lock:
            result = (tuple(self._frames.values()), self._error, self._interval_valid,
                      tuple(self._extrema.get(can_id) for can_id in _FEEDBACK_IDS[1:]))
            self._interval_valid = True
            self._extrema = {}
            return result


def _wait_for_target(robot, feedback, target, timeout):
    """Verify delivered feedback and communication errors visible at each poll.

    CanComm.recv drops CAN error frames before callbacks and normal traffic can
    clear last_error before has_comm_error is polled. This cannot latch all CAN
    faults; callback fault latching covers only feedback actually delivered.
    """
    if len(target) != 7 or not all(math.isfinite(q) for q in target):
        raise ValueError("recovery target must contain exactly seven finite joint angles")
    commanded_at, counts = feedback.begin_wait()
    stamps = (commanded_at,) * len(_FEEDBACK_IDS)
    deadline = time.monotonic() + timeout
    stable_since = None
    last_sample_at = None
    low = high = ()
    samples = 0
    while time.monotonic() < deadline:
        frames, error, interval_valid, extrema = feedback.snapshot()
        now, wall_now = time.monotonic(), time.time()
        if robot.has_comm_error():
            raise RuntimeError("communication error during recovery verification")
        if error is not None:
            raise RuntimeError(error)
        valid = interval_valid and all(frame is not None for frame in frames)
        if valid:
            valid = all(
                frame.timestamp > commanded_at
                and 0 <= wall_now - frame.timestamp <= 0.25
                and 0 <= now - frame.received_at <= 0.25
                for frame in frames
            ) and max(frame.timestamp for frame in frames) - min(frame.timestamp for frame in frames) <= 0.1
            status = frames[0].values
            joints = tuple(q for frame in frames[1:] for q in frame.values)
            valid = valid and status[:5] == (1, 0, 1, 0, 0) and status[6] == 0
            valid = valid and len(joints) == 7 and all(math.isfinite(q) for q in joints)
            errors = tuple(abs(q - desired) for q, desired in zip(joints, target))
            valid = valid and max(errors, default=math.inf) <= 0.01
        if valid:
            interval_low = tuple(q for frame, bounds in zip(frames[1:], extrema)
                                 for q in (bounds[0] if bounds is not None else frame.values))
            interval_high = tuple(q for frame, bounds in zip(frames[1:], extrema)
                                  for q in (bounds[1] if bounds is not None else frame.values))
            valid = all(abs(q - desired) <= 0.01 for bound in (interval_low, interval_high)
                        for q, desired in zip(bound, target))
            valid = valid and max(b - a for a, b in zip(interval_low, interval_high)) <= 0.002
        if not valid:
            stable_since = None
        else:
            # Preserve every received extreme, including intervals in which a
            # slower stream has not yet advanced to form a complete sample.
            if stable_since is not None:
                low = tuple(min(a, b) for a, b in zip(low, interval_low))
                high = tuple(max(a, b) for a, b in zip(high, interval_high))
                if now - last_sample_at > 0.25 or max(b - a for a, b in zip(low, high)) > 0.002:
                    stable_since = None
            if all(frame.count > count and frame.timestamp > stamp
                   for frame, count, stamp in zip(frames, counts, stamps)):
                counts = tuple(frame.count for frame in frames)
                stamps = tuple(frame.timestamp for frame in frames)
                if stable_since is None:
                    stable_since, low, high, samples = now, joints, joints, 0
                samples += 1
                last_sample_at = now
                if samples >= 2 and now - stable_since >= 0.5:
                    return joints, errors, now - stable_since, samples
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    raise RuntimeError("timed out waiting for fresh, target-matching, settled recovery feedback")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move one Nero arm to the initial pose.")
    parser.add_argument("arm", choices=TARGETS)
    parser.add_argument("--can", default=None, help="SocketCAN interface; defaults to can3(left) or can2(right).")
    parser.add_argument("--speed-percent", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--gripper-open-width", type=float, default=0.1)
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--yes", action="store_true", help="Skip the hardware motion confirmation prompt.")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1, 100]")
    if not 0.0 <= args.gripper_open_width <= 0.1:
        parser.error("--gripper-open-width must be in [0, 0.1]")
    if not 0.0 <= args.gripper_force <= 3.0:
        parser.error("--gripper-force must be in [0, 3]")
    channel = args.can or ("can3" if args.arm == "left" else "can2")
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.DEFAULT,
        interface="socketcan",
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(config)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    print(f"WARNING: moving real Nero {args.arm} arm on {channel}")
    if not args.yes and input("Type EXECUTE to continue: ") != "EXECUTE":
        raise SystemExit("recovery cancelled")
    feedback = _Feedback()
    robot.get_context().register_parser_packet_fun(feedback)
    try:
        robot.connect()
        deadline = time.monotonic() + 5
        while not robot.enable():
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out enabling Nero arm")
            time.sleep(0.05)
        robot.set_speed_percent(args.speed_percent)
        print(f"opening {args.arm} gripper to {args.gripper_open_width:.3f} m")
        gripper.move_gripper_m(value=args.gripper_open_width, force=args.gripper_force)
        time.sleep(0.5)
        robot.move_j(TARGETS[args.arm])
        joints, errors, dwell, samples = _wait_for_target(robot, feedback, TARGETS[args.arm], args.timeout)
        print(f"recovery complete: arm={args.arm} measured={list(joints)} "
              f"errors_rad={list(errors)} max_error_rad={max(errors):.6f} "
              f"verified fresh and settled for {dwell:.3f}s across {samples} samples")
    except (Exception, KeyboardInterrupt) as exc:
        print(f"recovery not verified: {exc}; disconnect is not confirmation that the arm stopped", file=sys.stderr)
        raise
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
