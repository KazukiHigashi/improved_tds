from __future__ import annotations

import numpy as np

from improved_tds.evaluation.transfer import leave_one_instance_out
from improved_tds.synergy.supervised import SupervisedLinearTDS


def test_leave_one_instance_out() -> None:
    state = np.linspace(-1.0, 1.0, 30)
    samples = {}
    for index, slope in enumerate((1.0, 1.1, 0.9)):
        q = np.column_stack([slope * state, 0.2 * state]) + index
        samples[f"instance-{index}"] = (q, state[:, None])
    result = leave_one_instance_out(
        samples,
        lambda: SupervisedLinearTDS(method="covariance"),
        calibration_shots=(3, 6),
    )
    assert set(result) == set(samples)
    assert all(metrics["direction_cosine"] > 0.99 for metrics in result.values())
    assert all(metrics["state_rmse_3_shots"] < 0.1 for metrics in result.values())

