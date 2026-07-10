from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ControlObservation:
    tool_state: float | None
    tool_state_rate: float | None = None
    joint_torque: np.ndarray | None = None
    motor_current: np.ndarray | None = None
    synergy_force: float | None = None
    sensor_valid: bool = True
    emergency_stop: bool = False


@dataclass
class ControllerOutput:
    q_command: np.ndarray
    rho_command: float
    mode: str
    saturated: bool = False
    emergency_release: bool = False
    diagnostics: dict[str, float | str | bool] = field(default_factory=dict)


@runtime_checkable
class TDSController(Protocol):
    def reset(self) -> None: ...

    def step(
        self,
        target_tool_state: float,
        observation: ControlObservation,
        dt: float,
        *,
        phase: str | None = None,
    ) -> ControllerOutput: ...

