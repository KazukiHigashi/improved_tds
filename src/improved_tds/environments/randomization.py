from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class DomainRandomizationConfig:
    parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    sensor_noise_std: float = 0.0
    action_latency_steps: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.sensor_noise_std < 0.0:
            raise ValueError("sensor noise must be non-negative")
        if self.action_latency_steps[0] < 0 or self.action_latency_steps[0] > self.action_latency_steps[1]:
            raise ValueError("invalid action latency range")
        for name, bounds in self.parameter_ranges.items():
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"invalid randomization bounds for {name}")


class DomainRandomizationWrapper(gym.Wrapper):
    """Seeded physical-parameter, sensor-noise and action-latency randomization."""

    def __init__(self, env: gym.Env, config: DomainRandomizationConfig):
        super().__init__(env)
        self.config = config
        self._rng = np.random.default_rng()
        self._latency = 0
        self._actions: deque[np.ndarray] = deque()

    def reset(self, *, seed: int | None = None, options=None):
        self._rng = np.random.default_rng(seed)
        parameters = {
            name: float(self._rng.uniform(low, high))
            for name, (low, high) in self.config.parameter_ranges.items()
        }
        if parameters:
            self.env.unwrapped.set_tool_parameters(parameters)
        low, high = self.config.action_latency_steps
        self._latency = int(self._rng.integers(low, high + 1))
        self._actions.clear()
        observation, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["domain_randomization"] = parameters
        info["action_latency_steps"] = self._latency
        return self._noisy(observation), info

    def step(self, action):
        value = np.asarray(action, dtype=np.float32).copy()
        self._actions.append(value)
        if len(self._actions) <= self._latency:
            applied = np.zeros_like(value)
        else:
            applied = self._actions.popleft()
        observation, reward, terminated, truncated, info = self.env.step(applied)
        info = dict(info)
        info["applied_action"] = applied
        return self._noisy(observation), reward, terminated, truncated, info

    def _noisy(self, observation):
        if self.config.sensor_noise_std == 0.0:
            return observation
        result = {key: np.asarray(value).copy() for key, value in observation.items()}
        noise = self._rng.normal(0.0, self.config.sensor_noise_std, size=result["achieved_goal"].shape)
        result["achieved_goal"] = (result["achieved_goal"] + noise).astype(np.float32)
        return result

