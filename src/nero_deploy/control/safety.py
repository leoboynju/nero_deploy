from __future__ import annotations

import numpy as np


JOINT_LOWER = np.array([-2.70526, -1.74, -2.75, -1.01, -2.75, -0.73, -1.5707963])
JOINT_UPPER = np.array([2.70526, 1.74, 2.75, 2.14, 2.75, 0.95, 1.5707963])


def validate_action_chunk(actions: np.ndarray, state: np.ndarray, max_joint_step: float, margin: float) -> None:
    if actions.shape != (16, 16):
        raise RuntimeError(f"expected action shape (16, 16), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("policy returned NaN or infinite actions")
    joint_indices = [*range(7), *range(8, 15)]
    joints = actions[:, joint_indices]
    lower = np.tile((JOINT_LOWER + margin).astype(np.float32), 2)
    upper = np.tile((JOINT_UPPER - margin).astype(np.float32), 2)
    if np.any(joints < lower) or np.any(joints > upper):
        raise RuntimeError("policy action violates joint limits")
    current = np.asarray(state)[joint_indices]
    if np.max(np.abs(np.diff(np.vstack([current, joints]), axis=0))) > np.float32(max_joint_step) + 1e-6:
        raise RuntimeError("policy action jump exceeds safety limit")


def clamp_action_chunk(
    actions: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, int, float]:
    """Saturate arm joints at safe limits while preserving gripper outputs."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.shape != (16, 16):
        raise RuntimeError(f"expected action shape (16, 16), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("policy returned NaN or infinite actions")

    joint_indices = [*range(7), *range(8, 15)]
    lower = np.tile((JOINT_LOWER + margin).astype(np.float32), 2)
    upper = np.tile((JOINT_UPPER - margin).astype(np.float32), 2)
    before = actions[:, joint_indices].copy()
    actions[:, joint_indices] = np.clip(before, lower, upper)
    changed = np.abs(actions[:, joint_indices] - before)
    return actions, int(np.count_nonzero(changed)), float(np.max(changed, initial=0.0))


def clamp_action(action: np.ndarray, margin: float) -> tuple[np.ndarray, int, float]:
    """Saturate one physical dual-arm action with shape (16,)."""
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (16,):
        raise RuntimeError(f"expected action shape (16,), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise RuntimeError("policy returned NaN or infinite action")
    joint_indices = [*range(7), *range(8, 15)]
    lower = np.tile((JOINT_LOWER + margin).astype(np.float32), 2)
    upper = np.tile((JOINT_UPPER - margin).astype(np.float32), 2)
    before = action[joint_indices].copy()
    action[joint_indices] = np.clip(before, lower, upper)
    changed = np.abs(action[joint_indices] - before)
    return action, int(np.count_nonzero(changed)), float(np.max(changed, initial=0.0))


def clamp_action_step(
    action: np.ndarray,
    state: np.ndarray,
    max_joint_step: float,
) -> tuple[np.ndarray, int, float]:
    """Limit one action's joint delta relative to the latest measured state."""
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (16,):
        raise RuntimeError(f"expected action shape (16,), got {action.shape}")
    if max_joint_step <= 0:
        raise ValueError("max_joint_step must be positive")
    joint_indices = [*range(7), *range(8, 15)]
    current = np.asarray(state, dtype=np.float32)[joint_indices]
    before = action[joint_indices].copy()
    action[joint_indices] = np.clip(
        before,
        current - np.float32(max_joint_step),
        current + np.float32(max_joint_step),
    )
    changed = np.abs(action[joint_indices] - before)
    return action, int(np.count_nonzero(changed)), float(np.max(changed, initial=0.0))


def validate_action(action: np.ndarray, state: np.ndarray, max_joint_step: float, margin: float) -> None:
    action = np.asarray(action)
    if action.shape != (16,):
        raise RuntimeError(f"expected action shape (16,), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise RuntimeError("policy returned NaN or infinite action")
    joint_indices = [*range(7), *range(8, 15)]
    lower = np.tile((JOINT_LOWER + margin).astype(np.float32), 2)
    upper = np.tile((JOINT_UPPER - margin).astype(np.float32), 2)
    if np.any(action[joint_indices] < lower) or np.any(action[joint_indices] > upper):
        raise RuntimeError("policy action violates joint limits")
    current = np.asarray(state, dtype=np.float32)[joint_indices]
    if np.max(np.abs(action[joint_indices] - current)) > np.float32(max_joint_step) + 1e-6:
        raise RuntimeError("policy action jump exceeds safety limit")
