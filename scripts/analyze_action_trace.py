#!/usr/bin/env python3
"""Summarize control indices and jitter sources from an executed-action trace."""

import argparse
import json

import numpy as np


JOINTS = [*range(7), *range(8, 15)]


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values))}


def distribution(values):
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def summarize(path):
    with np.load(path, allow_pickle=False) as data:
        chunk_ids = data["chunk_id"]
        indices = data["action_index"]
        advanced = data["action_advanced"] if "action_advanced" in data else np.ones(len(indices), dtype=bool)
        changed = chunk_ids[1:] != chunk_ids[:-1]
        dt = np.diff(data["timestamp"])
        segments = np.r_[0, np.flatnonzero(changed) + 1, len(indices)]
        raw_step = np.abs(np.diff(data["policy_action"][:, JOINTS], axis=0)).max(axis=1)
        command_step = np.abs(np.diff(data["command"][:, JOINTS], axis=0)).max(axis=1)
        feedback_step = np.abs(np.diff(data["state"][:, JOINTS], axis=0)).max(axis=1)
        tracking = np.abs(data["command"][:, JOINTS] - data["state"][:, JOINTS]).max(axis=1)
        switch = np.abs(data["switch_delta"][:, JOINTS]).max(axis=1)
        return {
            "steps": len(indices),
            "chunks": len(segments) - 1,
            "chunk_start_index_distribution": distribution(indices[segments[:-1]]),
            "completed_chunk_execution_count_distribution": distribution(
                [int(advanced[start:end].sum()) for start, end in zip(segments[:-2], segments[1:-1])]),
            "within_chunk_index_violations": int(np.sum(
                (~changed) & advanced[1:] & advanced[:-1] & (np.diff(indices) != 1))),
            "held_control_steps": int((~advanced).sum()),
            "grasp_events": ({str(key): int(count) for key, count in
                              zip(*np.unique(data["grasp_event"], return_counts=True)) if key}
                             if "grasp_event" in data else {}),
            "command_interval_ms": stats(dt * 1000),
            "same_tick_plan_change_rad": stats(switch),
            "policy_step_rad_at_switch": stats(raw_step[changed]),
            "policy_step_rad_inside_chunk": stats(raw_step[~changed]),
            "command_step_rad_at_switch": stats(command_step[changed]),
            "command_step_rad_inside_chunk": stats(command_step[~changed]),
            "feedback_step_rad": stats(feedback_step),
            "command_tracking_error_rad": stats(tracking),
            "feedback_age_ms": stats(data["state_age"] * 1000),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="outputs/action_traces/<timestamp>.npz")
    print(json.dumps(summarize(parser.parse_args().trace), indent=2))
