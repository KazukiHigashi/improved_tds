from __future__ import annotations

import numpy as np

from improved_tds.synergy.calibration import ToolStateCalibrator


def calibrated_subset(
    calibrator: ToolStateCalibrator,
    rho: np.ndarray,
    tool_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return target/prediction only inside the calibrator's invertible rho domain."""

    coordinate = np.asarray(rho, dtype=np.float64).reshape(-1)
    target = np.asarray(tool_state, dtype=np.float64).reshape(-1)
    if coordinate.shape != target.shape:
        raise ValueError("rho and tool_state sample counts do not match")
    low, high = calibrator.safe_ranges()["rho"]
    mask = (coordinate >= low) & (coordinate <= high)
    if not np.any(mask):
        raise ValueError("no evaluation samples are inside the invertible calibration domain")
    predicted = calibrator.forward(coordinate[mask])
    return target[mask], predicted, float(np.mean(mask))

