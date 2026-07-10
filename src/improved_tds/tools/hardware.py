from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


@dataclass
class HardwareState:
    timestamp: float
    q: np.ndarray
    qd: np.ndarray
    tool_state: np.ndarray | None = None
    joint_torque: np.ndarray | None = None
    motor_current: np.ndarray | None = None
    emergency_stop: bool = False


@dataclass
class HardwareCommand:
    q_command: np.ndarray
    emergency_release: bool = False


@runtime_checkable
class HardwareAdapter(Protocol):
    def read_state(self) -> HardwareState: ...

    def write_command(self, command: HardwareCommand) -> None: ...

    def emergency_stop(self) -> None: ...

    def close(self) -> None: ...


class MockHardwareAdapter:
    """Deterministic dependency-injection target for controller and safety tests."""

    def __init__(self, initial_state: HardwareState):
        self.state = initial_state
        self.last_command: HardwareCommand | None = None
        self.stopped = False

    def read_state(self) -> HardwareState:
        return self.state

    def write_command(self, command: HardwareCommand) -> None:
        if self.stopped and not command.emergency_release:
            raise RuntimeError("hardware adapter is stopped")
        self.last_command = command

    def emergency_stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.stopped = True


class OfflineReplayAdapter(MockHardwareAdapter):
    def __init__(self, states: Iterable[HardwareState]):
        self._states = iter(states)
        first = next(self._states, None)
        if first is None:
            raise ValueError("offline replay requires at least one state")
        super().__init__(first)

    def read_state(self) -> HardwareState:
        current = self.state
        following = next(self._states, None)
        if following is not None:
            self.state = following
        return current

