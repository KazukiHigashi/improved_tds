from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from improved_tds.control.base import ControlObservation


@dataclass(frozen=True)
class SafetyLimits:
    rho_min: float
    rho_max: float
    rho_rate_max: float
    rho_acceleration_max: float
    q_min: np.ndarray | None = None
    q_max: np.ndarray | None = None
    torque_limit: np.ndarray | float | None = None
    current_limit: np.ndarray | float | None = None
    force_limit: float | None = None

    def __post_init__(self) -> None:
        if not self.rho_min < self.rho_max:
            raise ValueError("rho_min must be smaller than rho_max")
        if self.rho_rate_max <= 0.0 or self.rho_acceleration_max <= 0.0:
            raise ValueError("rho rate and acceleration limits must be positive")
        if (self.q_min is None) != (self.q_max is None):
            raise ValueError("q_min and q_max must be supplied together")
        if self.q_min is not None:
            low = np.asarray(self.q_min, dtype=np.float64)
            high = np.asarray(self.q_max, dtype=np.float64)
            if low.shape != high.shape or np.any(low >= high):
                raise ValueError("joint limits must have matching shapes with q_min < q_max")


class SafetyLimiter:
    """Stateful rho/rate/acceleration and joint-limit saturation."""

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.reset()

    def reset(self) -> None:
        self.rho: float | None = None
        self.rho_rate = 0.0

    def limit_rho(self, desired: float, dt: float) -> tuple[float, bool]:
        if not np.isfinite(desired) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("desired rho and dt must be finite, with dt > 0")
        bounded = float(np.clip(desired, self.limits.rho_min, self.limits.rho_max))
        saturated = not np.isclose(bounded, desired)
        if self.rho is None:
            self.rho = bounded
            self.rho_rate = 0.0
            return bounded, saturated

        desired_rate = (bounded - self.rho) / dt
        rate_low = max(
            -self.limits.rho_rate_max,
            self.rho_rate - self.limits.rho_acceleration_max * dt,
        )
        rate_high = min(
            self.limits.rho_rate_max,
            self.rho_rate + self.limits.rho_acceleration_max * dt,
        )
        limited_rate = float(np.clip(desired_rate, rate_low, rate_high))
        saturated |= not np.isclose(limited_rate, desired_rate)
        next_rho = float(
            np.clip(self.rho + limited_rate * dt, self.limits.rho_min, self.limits.rho_max)
        )
        actual_rate = (next_rho - self.rho) / dt
        self.rho = next_rho
        self.rho_rate = actual_rate
        return next_rho, saturated

    def limit_q(self, q: np.ndarray) -> tuple[np.ndarray, bool]:
        command = np.asarray(q, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(command)):
            raise ValueError("q command contains NaN or infinity")
        if self.limits.q_min is None:
            return command, False
        low = np.asarray(self.limits.q_min, dtype=np.float64)
        high = np.asarray(self.limits.q_max, dtype=np.float64)
        if command.shape != low.shape:
            raise ValueError("q command does not match configured joint limits")
        limited = np.clip(command, low, high)
        return limited, bool(np.any(limited != command))

    def violation(self, observation: ControlObservation) -> str | None:
        if observation.emergency_stop:
            return "emergency_stop"
        checks = (
            (observation.joint_torque, self.limits.torque_limit, "torque_limit"),
            (observation.motor_current, self.limits.current_limit, "current_limit"),
        )
        for value, limit, reason in checks:
            if value is None or limit is None:
                continue
            if np.any(np.abs(np.asarray(value, dtype=np.float64)) > np.asarray(limit)):
                return reason
        if (
            observation.synergy_force is not None
            and self.limits.force_limit is not None
            and abs(observation.synergy_force) > self.limits.force_limit
        ):
            return "force_limit"
        return None

