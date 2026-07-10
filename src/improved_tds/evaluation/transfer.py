from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from improved_tds.synergy.family import FamilyTDS


def leave_one_instance_out(
    samples: dict[str, tuple[np.ndarray, np.ndarray]],
    estimator_factory: Callable[[], Any],
    *,
    calibration_shots: tuple[int, ...] = (3, 6, 12),
) -> dict[str, dict[str, float]]:
    """Evaluate family-direction transfer while fitting only held-out mean/calibration samples."""

    if len(samples) < 2:
        raise ValueError("leave-one-instance-out requires at least two instances")
    results: dict[str, dict[str, float]] = {}
    individual = {}
    for instance, (q, state) in samples.items():
        individual[instance] = estimator_factory().fit(q, state)
    for held_out, (q, state) in samples.items():
        training_ids = [name for name in samples if name != held_out]
        family = FamilyTDS().fit_directions(
            [individual[name].direction_ for name in training_ids]
        )
        target_direction = individual[held_out].direction_
        cosine = float(abs(family.direction_ @ target_direction))
        metrics = {"direction_cosine": cosine}
        for shots in calibration_shots:
            count = min(int(shots), q.shape[0])
            order = np.argsort(state.reshape(-1))
            indices = order[np.linspace(0, order.size - 1, count).round().astype(int)]
            family.set_instance_mean(np.mean(q[indices], axis=0))
            rho = family.encode(q)[:, 0]
            design = np.column_stack([np.ones(count), rho[indices]])
            coefficients, *_ = np.linalg.lstsq(design, state[indices].reshape(-1), rcond=None)
            predicted = coefficients[0] + coefficients[1] * rho
            metrics[f"state_rmse_{shots}_shots"] = float(
                np.sqrt(np.mean((predicted - state.reshape(-1)) ** 2))
            )
        results[held_out] = metrics
    return results

