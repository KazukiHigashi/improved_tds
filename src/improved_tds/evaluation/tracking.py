from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingMetrics:
    mae: float
    rmse: float
    max_error: float
    tracking_delay_seconds: float
    monotonicity_violations: int
    attainable_range: tuple[float, float]
    peak_force: float | None = None

    def as_dict(self) -> dict[str, float | int | tuple[float, float] | None]:
        return asdict(self)


def compute_tracking_metrics(
    desired: np.ndarray,
    measured: np.ndarray,
    *,
    dt: float,
    force: np.ndarray | None = None,
    delay_threshold: float | None = None,
) -> TrackingMetrics:
    target = np.asarray(desired, dtype=np.float64).reshape(-1)
    actual = np.asarray(measured, dtype=np.float64).reshape(-1)
    if target.shape != actual.shape or target.size == 0:
        raise ValueError("desired and measured must be non-empty arrays of equal length")
    if dt <= 0.0 or not np.all(np.isfinite(target)) or not np.all(np.isfinite(actual)):
        raise ValueError("tracking inputs and dt must be finite, with dt > 0")
    error = target - actual
    threshold = (
        float(delay_threshold)
        if delay_threshold is not None
        else max(0.02 * float(np.ptp(target)), 1e-4)
    )
    settled = np.flatnonzero(np.abs(error) <= threshold)
    delay = float(settled[0] * dt) if settled.size else float(target.size * dt)
    expected_direction = np.sign(target[-1] - target[0])
    differences = np.diff(actual) * expected_direction
    violations = int(np.sum(differences < -1e-8)) if expected_direction != 0.0 else 0
    peak_force = None
    if force is not None:
        force_values = np.asarray(force, dtype=np.float64)
        peak_force = float(np.max(np.abs(force_values)))
    return TrackingMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        max_error=float(np.max(np.abs(error))),
        tracking_delay_seconds=delay,
        monotonicity_violations=violations,
        attainable_range=(float(actual.min()), float(actual.max())),
        peak_force=peak_force,
    )

