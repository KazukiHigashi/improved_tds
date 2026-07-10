from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilizationConfig:
    gain: float = 0.05
    desired_force: float = 0.0
    phi_max: float = 0.2
    force_limit: float = 10.0
    tool_state_deviation_limit: float = 0.1
    interference_limit: float = 0.05

    def __post_init__(self) -> None:
        if self.gain < 0.0 or self.phi_max <= 0.0 or self.force_limit <= 0.0:
            raise ValueError("stabilization gain/limits are invalid")


class StabilizationSynergy:
    """Independent grasp-stabilization component `b_g * phi`.

    A TDS nullspace is not assumed to be an internal-force subspace. The caller must
    supply or estimate an interference metric and reject directions above the limit.
    """

    def __init__(
        self,
        direction: np.ndarray,
        *,
        config: StabilizationConfig | None = None,
        interference: float = 0.0,
    ):
        vector = np.asarray(direction, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("stabilization direction must be non-zero")
        self.direction = vector / norm
        self.config = config or StabilizationConfig()
        self.interference = float(interference)
        if self.interference > self.config.interference_limit:
            raise ValueError("stabilization direction exceeds tool-state interference limit")
        self.reset()

    def reset(self) -> None:
        self.phi = 0.0

    @classmethod
    def from_pca_component(
        cls,
        component: np.ndarray,
        **kwargs: object,
    ) -> "StabilizationSynergy":
        return cls(component, **kwargs)

    def apply(
        self,
        q_actuation: np.ndarray,
        *,
        force_proxy: float | None,
        tool_state_deviation: float,
        dt: float,
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        q = np.asarray(q_actuation, dtype=np.float64).reshape(-1)
        if q.shape != self.direction.shape:
            raise ValueError("stabilization direction and q command shapes do not match")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if force_proxy is not None and abs(force_proxy) > self.config.force_limit:
            self.phi = 0.0
            return q, {"phi": 0.0, "emergency_release": True}
        force = self.config.desired_force if force_proxy is None else float(force_proxy)
        self.phi += self.config.gain * (self.config.desired_force - force) * dt
        if abs(tool_state_deviation) > self.config.tool_state_deviation_limit:
            scale = self.config.tool_state_deviation_limit / abs(tool_state_deviation)
            self.phi *= scale
        self.phi = float(np.clip(self.phi, -self.config.phi_max, self.config.phi_max))
        return q + self.direction * self.phi, {
            "phi": self.phi,
            "interference": self.interference,
            "emergency_release": False,
        }


def estimate_experimental_direction(
    perturbations: np.ndarray,
    tool_state_changes: np.ndarray,
    force_changes: np.ndarray,
    *,
    actuation_direction: np.ndarray | None = None,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Estimate a direction with high force effect and low tool-state interference."""

    dq = np.asarray(perturbations, dtype=np.float64)
    dc = np.asarray(tool_state_changes, dtype=np.float64).reshape(-1)
    df = np.asarray(force_changes, dtype=np.float64).reshape(-1)
    if dq.ndim != 2 or dq.shape[0] != dc.size or dc.shape != df.shape:
        raise ValueError("perturbation, tool-state, and force sample counts must match")
    force_gradient, *_ = np.linalg.lstsq(dq, df, rcond=None)
    state_gradient, *_ = np.linalg.lstsq(dq, dc, rcond=None)
    direction = force_gradient.copy()
    nuisance = [state_gradient]
    if actuation_direction is not None:
        nuisance.append(np.asarray(actuation_direction, dtype=np.float64).reshape(-1))
    for vector in nuisance:
        denominator = float(vector @ vector) + regularization
        direction -= vector * float(direction @ vector) / denominator
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("no force-sensitive low-interference direction was identified")
    direction /= norm
    interference = float(abs(state_gradient @ direction))
    return direction, interference

