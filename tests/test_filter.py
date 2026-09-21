import numpy as np
import pytest

from nero_deploy.control import DualArmJointEMA


def test_ema_initializes_without_lagging_to_zero() -> None:
    ema = DualArmJointEMA(0.5)
    first = np.arange(16, dtype=np.float32)
    np.testing.assert_allclose(ema.apply(first), first)


def test_ema_smooths_joints_but_not_grippers() -> None:
    ema = DualArmJointEMA(0.5)
    ema.apply(np.zeros(16, dtype=np.float32))
    next_action = np.ones(16, dtype=np.float32)
    next_action[7] = 0.0
    next_action[15] = 0.0
    filtered = ema.apply(next_action)
    np.testing.assert_allclose(filtered[[*range(7), *range(8, 15)]], 0.5)
    np.testing.assert_allclose(filtered[[7, 15]], 0.0)


def test_ema_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        DualArmJointEMA(0.0)
