"""Record policy targets, executed commands and feedback without per-tick disk I/O."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path

import numpy as np

from nero_deploy.control.filter import JOINT_INDICES


logger = logging.getLogger(__name__)


class ActionTrace:
    def __init__(self, output_dir: Path, metadata: dict) -> None:
        self.output_dir = output_dir
        self.metadata = metadata
        self.rows: list[dict] = []

    def record(self, *, timestamp, state, policy_action, ema_action, command, result, state_age) -> None:
        switch_delta = result.get("rtc_switch_delta")
        row = {
            "timestamp": float(timestamp),
            "state": np.array(state, dtype=np.float32, copy=True),
            "policy_action": np.array(policy_action, dtype=np.float32, copy=True),
            "ema_action": np.array(ema_action, dtype=np.float32, copy=True),
            "command": np.array(command, dtype=np.float32, copy=True),
            "chunk_id": int(result["rtc_chunk_id"]),
            "action_index": int(result["rtc_action_index"]),
            "chunk_changed": bool(result["rtc_chunk_changed"]),
            "state_age": float(state_age),
            "action_advanced": bool(result.get("action_advanced", True)),
            "grasp_phase": str(result.get("grasp_phase", "disabled")),
            "grasp_event": str(result.get("grasp_event", "")),
            "gripper_force": np.asarray(result.get("gripper_force", [np.nan, np.nan]), dtype=np.float32).copy(),
            "grasp_target": np.asarray(result.get("grasp_target", np.full(16, np.nan)), dtype=np.float32).copy(),
            "rtc_guidance_mask": np.isin(np.arange(16), result.get("rtc_guided_action_indices", [])),
            "switch_delta": np.full(16, np.nan, dtype=np.float32) if switch_delta is None else np.array(switch_delta, dtype=np.float32, copy=True),
        }
        if self.rows and row["chunk_changed"]:
            previous = self.rows[-1]
            raw_jump = np.max(np.abs(row["policy_action"][JOINT_INDICES] - previous["policy_action"][JOINT_INDICES]))
            command_jump = np.max(np.abs(row["command"][JOINT_INDICES] - previous["command"][JOINT_INDICES]))
            tracking = np.max(np.abs(row["command"][JOINT_INDICES] - row["state"][JOINT_INDICES]))
            switch = row["switch_delta"][JOINT_INDICES]
            same_tick_change = float(np.max(np.abs(switch))) if np.isfinite(switch).all() else float("nan")
            logger.info(
                "Action switch: chunk=%d index=%d same_tick_plan_change_rad=%.4f "
                "policy_step_rad=%.4f command_step_rad=%.4f tracking_error_rad=%.4f",
                row["chunk_id"], row["action_index"], same_tick_change, raw_jump, command_jump, tracking,
            )
        self.rows.append(row)

    def save(self) -> Path | None:
        if not self.rows:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".npz")
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
        arrays["metadata_json"] = np.asarray(json.dumps(self.metadata))
        np.savez_compressed(path, **arrays)
        logger.info("Saved %d executed control steps to %s", len(self.rows), path)
        self.rows.clear()
        return path
