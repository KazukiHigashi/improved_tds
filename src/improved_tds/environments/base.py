from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np


class ArticulatedToolEnv(gym.Env, ABC):
    """Gymnasium contract for a known/initialized grasp acting on one tool DoF."""

    tool_family: str
    tool_instance_id: str

    @property
    @abstractmethod
    def tool_state_limits(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def phase(self) -> str: ...

    @abstractmethod
    def get_tool_state(self) -> np.ndarray: ...

    @abstractmethod
    def get_tool_state_rate(self) -> np.ndarray: ...

    @abstractmethod
    def get_force_observation(self) -> dict[str, np.ndarray | float]: ...

    @abstractmethod
    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def set_tool_parameters(self, params: dict[str, float]) -> None: ...

    @property
    def actuation_force_proxy(self) -> float:
        value = self.get_force_observation().get("synergy_force", 0.0)
        return float(np.asarray(value).reshape(-1)[0])

    @property
    def contact_state(self) -> np.ndarray:
        value: Any = self.get_force_observation().get("contact_flags", np.zeros(0))
        return np.asarray(value, dtype=np.bool_).reshape(-1)

