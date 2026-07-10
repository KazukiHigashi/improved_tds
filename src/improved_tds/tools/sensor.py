from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ToolStateSensor(Protocol):
    def read(self) -> tuple[np.ndarray, float] | None:
        """Return `(state, timestamp)` in rad/m, or None on dropout."""

    def reset(self) -> None: ...


class MockToolStateSensor:
    def __init__(self, state: np.ndarray, timestamp: float = 0.0):
        self.state = np.asarray(state, dtype=np.float64).reshape(-1)
        self.timestamp = float(timestamp)
        self.dropout = False

    def read(self) -> tuple[np.ndarray, float] | None:
        if self.dropout:
            return None
        return self.state.copy(), self.timestamp

    def reset(self) -> None:
        self.dropout = False

