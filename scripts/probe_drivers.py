#!/usr/bin/env python3
"""Check live DDS endpoints and feedback without publishing control commands."""

import argparse
import json
import math
import time


def classify(report, mode):
    controls = [entry for arm in report.values() for entry in arm["controls"].values()]
    if any(entry["publishers"] for entry in controls):
        return 3  # An existing controller is active: do not stop/start drivers under it.
    if mode == "ready-or-empty" and classify(report, "empty") == 0:
        return 2  # Stable empty graph: no need to wait out the ready timeout.
    if mode == "empty":
        if classify(report, "ready") == 0:
            return 4  # Live drivers survived cleanup, not merely stale DDS discovery.
        return 0 if all(not entry["subscribers"] for entry in controls) and all(
            not arm["feedback_publishers"] for arm in report.values()) else 1
    return 0 if (all(len(entry["subscribers"]) == 1 for entry in controls)
                 and all(len(arm["feedback_publishers"]) == 1 and arm["messages"] >= 2
                         and arm["feedback_age"] < 0.5 for arm in report.values())) else 1


def main():
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ready", "empty", "ready-or-empty"), default="ready")
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    rclpy.init()
    node = rclpy.create_node("nero_driver_startup_probe")
    received = {arm: [0, float("-inf")] for arm in ("left", "right")}

    def receive(arm, message):
        if len(message.position) >= 8:
            received[arm][0] += 1
            received[arm][1] = time.monotonic()

    for arm in received:
        node.create_subscription(JointState, f"/{arm}_arm/feedback/joint_states",
                                 lambda msg, name=arm: receive(name, msg),
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    started = time.monotonic()
    ready_since = None
    previous_status = None
    report = {}
    try:
        while time.monotonic() - started < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            report = {}
            for arm in received:
                prefix = f"/{arm}_arm"
                names = lambda endpoints: [f"{entry.node_namespace.rstrip('/')}/{entry.node_name}" for entry in endpoints]
                report[arm] = {
                    "messages": received[arm][0], "feedback_age": min(999, now - received[arm][1]),
                    "feedback_publishers": names(node.get_publishers_info_by_topic(prefix + "/feedback/joint_states")),
                    "controls": {topic: {
                        "publishers": names(node.get_publishers_info_by_topic(prefix + "/control/" + topic)),
                        "subscribers": names(node.get_subscriptions_info_by_topic(prefix + "/control/" + topic)),
                    } for topic in ("move_j", "joint_states")},
                }
            status = classify(report, args.mode)
            if status == 3:
                print("Existing control publishers:", json.dumps(report), flush=True)
                return 3
            # Allow discovery to settle; never infer absence from cached ros2 CLI output.
            empty = status == 2 or (args.mode == "empty" and status == 0)
            # Absence needs a longer discovery window than positive readiness.
            discovery_seconds = 3 if empty else 1
            if status in (0, 2, 4) and now - started >= discovery_seconds:
                ready_since = now if ready_since is None or status != previous_status else ready_since
                if now - ready_since >= 0.5:
                    if status == 4:
                        print("Live drivers remain after cleanup:", flush=True)
                    print(json.dumps(report), flush=True)
                    return status
            else:
                ready_since = None
            previous_status = status
        print("Driver graph/feedback not ready:", json.dumps(report), flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
