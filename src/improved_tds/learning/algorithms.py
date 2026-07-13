from __future__ import annotations

from stable_baselines3 import DDPG, SAC, TD3


ALGORITHM_CLASSES = {
    "ddpg": DDPG,
    "sac": SAC,
    "td3": TD3,
}


def algorithm_class(name: str):
    try:
        return ALGORITHM_CLASSES[name]
    except KeyError as error:
        raise ValueError(f"unknown off-policy algorithm {name!r}") from error
