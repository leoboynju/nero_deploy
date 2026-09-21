import json
import importlib.util
import logging
from pathlib import Path

import numpy as np

from nero_deploy.diagnostics.action_trace import ActionTrace


def test_trace_records_indices_targets_and_physical_commands(tmp_path, caplog):
    trace = ActionTrace(tmp_path, {"control_hz": 30})
    state = np.zeros(16, dtype=np.float32)
    with caplog.at_level(logging.INFO):
        for index, chunk in [(12, 1), (5, 2), (6, 2)]:
            raw = np.full(16, 0.2)
            command = np.full(16, 0.05)
            command[[7, 15]] = [0.1, 0]
            result = {
                "rtc_chunk_id": chunk,
                "rtc_action_index": index,
                "rtc_chunk_changed": index == 5,
                "rtc_switch_delta": np.full(16, 0.15) if index == 5 else None,
            }
            trace.record(timestamp=len(trace.rows) / 30, state=state, policy_action=raw,
                         ema_action=raw * 0.5, command=command, result=result, state_age=0.01)
            raw[:] = -1  # Buffered samples must not alias the caller's arrays.
    assert "same_tick_plan_change_rad=0.1500" in caplog.text
    path = trace.save()
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["action_index"], [12, 5, 6])
        np.testing.assert_array_equal(data["chunk_id"], [1, 2, 2])
        np.testing.assert_allclose(data["policy_action"], 0.2)
        np.testing.assert_allclose(data["command"][:, 7], 0.1)
        assert data["state"].shape == (3, 16)
        assert json.loads(data["metadata_json"].item()) == {"control_hz": 30}
    assert not trace.rows
    assert trace.save() is None
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_action_trace.py"
    spec = importlib.util.spec_from_file_location("analyze_action_trace", script)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    summary = analyzer.summarize(path)
    assert summary["steps"] == 3
    assert summary["within_chunk_index_violations"] == 0
    assert summary["chunk_start_index_distribution"] == {"5": 1, "12": 1}
    np.testing.assert_allclose(summary["same_tick_plan_change_rad"]["max"], 0.15)


def test_trace_handles_startup_mask_and_held_action_indices(tmp_path):
    trace = ActionTrace(tmp_path, {"trace_version": 2})
    for step, (index, advanced) in enumerate([(0, True), (1, True), (1, False), (1, False)]):
        result = {"rtc_chunk_id": 1, "rtc_action_index": index, "rtc_chunk_changed": step == 0,
                  "action_advanced": advanced, "grasp_phase": "approach" if not advanced else "tracking",
                  "rtc_guided_action_indices": [] if step == 0 else [*range(7), *range(8, 15)]}
        trace.record(timestamp=step / 30, state=np.zeros(16), policy_action=np.zeros(16),
                     ema_action=np.zeros(16), command=np.zeros(16), result=result, state_age=0.01)
    path = trace.save()
    with np.load(path, allow_pickle=False) as data:
        assert data["rtc_guidance_mask"].shape == (4, 16)
        assert not data["rtc_guidance_mask"][:, [7, 15]].any()
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_action_trace.py"
    spec = importlib.util.spec_from_file_location("trace_held_analysis", script)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    summary = analyzer.summarize(path)
    assert summary["held_control_steps"] == 2
    assert summary["within_chunk_index_violations"] == 0
