from stable_baselines3 import DDPG, SAC, TD3
import pytest

from improved_tds.learning.algorithms import ALGORITHM_CLASSES, algorithm_class


def test_off_policy_algorithm_mapping_includes_sac() -> None:
    assert ALGORITHM_CLASSES == {"ddpg": DDPG, "sac": SAC, "td3": TD3}
    assert algorithm_class("sac") is SAC
    with pytest.raises(ValueError, match="unknown off-policy algorithm"):
        algorithm_class("ppo")
