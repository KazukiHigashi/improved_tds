from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class ToolModel:
    family: str
    instance_id: str
    state_limits: tuple[float, float]
    state_unit: Literal["rad", "m"]
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        low, high = self.state_limits
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError("tool state limits must be finite and increasing")

