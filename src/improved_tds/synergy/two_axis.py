"""Paired operation/grasp synergy axes for two-DoF tool control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression


@dataclass(frozen=True)
class TwoAxisSynergy:
    """One tool-operation axis and one orthogonal candidate grasp axis."""

    method: str
    operation_axis: np.ndarray
    grasp_axis: np.ndarray
    operation_tool_correlation: float
    grasp_tool_correlation: float
    grasp_metric_correlation: float
    operation_variance_fraction: float
    grasp_residual_variance_fraction: float


def fit_orthogonal_grasp_axis(
    q: np.ndarray,
    pose_index: np.ndarray,
    operation_axis: np.ndarray,
    grasp_metric: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    """Fit a unit grasp axis after removing the operation direction.

    ``residual_pca`` maximises residual posture variance. ``force_pls`` uses
    grasp force as a supervised target. Both are explicitly orthogonal to the
    supplied operation axis.
    """
    x = centre_within_pose(np.asarray(q, dtype=np.float64), pose_index)
    operation = _unit(operation_axis)
    residual = x - np.outer(x @ operation, operation)
    if method == "residual_pca":
        _, _, vh = np.linalg.svd(residual, full_matrices=False)
        axis = vh[0]
    elif method == "force_pls":
        force = centre_within_pose(
            np.asarray(grasp_metric, dtype=np.float64).reshape(-1, 1), pose_index
        ).reshape(-1)
        if np.std(force) <= 1e-12:
            raise ValueError("grasp force variation is insufficient for force PLS")
        axis = PLSRegression(n_components=1, scale=False).fit(
            residual, force
        ).x_weights_[:, 0]
    else:
        raise ValueError("grasp-axis method must be 'residual_pca' or 'force_pls'")
    axis = _unit(axis - float(axis @ operation) * operation)
    metric = centre_within_pose(
        np.asarray(grasp_metric, dtype=np.float64).reshape(-1, 1), pose_index
    ).reshape(-1)
    return _orient(axis, residual, metric)


def coordinate_from_normalised(
    reference: np.ndarray,
    axis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    value: float,
) -> float:
    """Map a coordinate in [-1, 1] onto pose-specific kinematic limits."""
    if not -1.0 <= value <= 1.0:
        raise ValueError("normalised coordinate must lie in [-1, 1]")
    reference = np.clip(np.asarray(reference, dtype=np.float64), low, high)
    direction = _unit(axis)
    lower, upper = -np.inf, np.inf
    for position, component, bottom, top in zip(reference, direction, low, high):
        if abs(component) <= 1e-12:
            continue
        endpoints = ((bottom - position) / component, (top - position) / component)
        lower = max(lower, min(endpoints))
        upper = min(upper, max(endpoints))
    return float(value * upper if value >= 0.0 else -value * lower)


def clipped_posture_from_normalised(
    reference: np.ndarray,
    axis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    value: float,
) -> tuple[np.ndarray, float]:
    """Follow a one-scalar synergy path without singleton limit collapse.

    The nominal amplitude is fixed by actuator half-ranges, independent of the
    current reference posture. Each joint is then clipped independently. This
    remains a one-DoF, piecewise-linear path while preventing one joint already
    at a limit from forcing the complete hand synergy to zero motion.
    """
    if not -1.0 <= value <= 1.0:
        raise ValueError("normalised coordinate must lie in [-1, 1]")
    reference = np.clip(np.asarray(reference, dtype=np.float64), low, high)
    direction = _unit(axis)
    halfspan = (np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)) / 2.0
    active = np.abs(direction) > 1e-12
    if not np.any(active):
        raise ValueError("cannot form a synergy from a zero vector")
    scale = float(np.min(halfspan[active] / np.abs(direction[active])))
    coordinate = float(value * scale)
    return np.clip(reference + coordinate * direction, low, high), coordinate


def centre_within_pose(values: np.ndarray, pose_index: np.ndarray) -> np.ndarray:
    """Remove authored-pose offsets without mixing poses into the synergies."""
    values = np.asarray(values, dtype=np.float64)
    pose_index = np.asarray(pose_index, dtype=int).reshape(-1)
    if values.shape[0] != pose_index.size:
        raise ValueError("values and pose_index must contain the same number of rows")
    centred = values.copy()
    for pose in np.unique(pose_index):
        rows = pose_index == pose
        centred[rows] -= centred[rows].mean(axis=0)
    return centred


def balanced_cell_indices(
    pose_index: np.ndarray,
    target_index: np.ndarray,
    valid: np.ndarray,
    *,
    per_cell: int = 1,
) -> np.ndarray:
    """Select at most the same number of valid rows from every pose-target cell."""
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    pose_index = np.asarray(pose_index, dtype=int).reshape(-1)
    target_index = np.asarray(target_index, dtype=int).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if not (pose_index.size == target_index.size == valid.size):
        raise ValueError("cell arrays must have equal length")
    selected: list[int] = []
    for pose in np.unique(pose_index):
        for target in np.unique(target_index):
            rows = np.flatnonzero(valid & (pose_index == pose) & (target_index == target))
            selected.extend(rows[:per_cell].tolist())
    return np.asarray(selected, dtype=int)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot form a synergy from a zero vector")
    return vector / norm


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _orient(axis: np.ndarray, design: np.ndarray, target: np.ndarray) -> np.ndarray:
    correlation = _correlation(design @ axis, target)
    if np.isfinite(correlation) and correlation < 0.0:
        return -axis
    return axis


def fit_two_axis_synergy(
    q: np.ndarray,
    tool_state: np.ndarray,
    pose_index: np.ndarray,
    *,
    method: str,
    grasp_metric: np.ndarray | None = None,
    centering: str = "within_pose",
) -> TwoAxisSynergy:
    """Fit PCA(PC1, PC2) or PLS1 plus residual-PCA grasp direction.

    The candidate grasp direction is deliberately not labelled as confirmed:
    its grasp contribution is reported through ``grasp_metric_correlation`` and
    must subsequently pass the simulated grasp-acquisition experiment.
    """
    q = np.asarray(q, dtype=np.float64)
    tool_state = np.asarray(tool_state, dtype=np.float64).reshape(-1)
    pose_index = np.asarray(pose_index, dtype=int).reshape(-1)
    if q.ndim != 2 or q.shape[0] != tool_state.size or q.shape[0] != pose_index.size:
        raise ValueError("q, tool_state, and pose_index have incompatible shapes")
    if q.shape[1] < 2 or q.shape[0] < 3:
        raise ValueError("two-axis fitting needs at least two joints and three samples")
    if centering not in ("global", "within_pose"):
        raise ValueError("centering must be 'global' or 'within_pose'")
    centre = (lambda values: values - values.mean(axis=0)) if centering == "global" else (
        lambda values: centre_within_pose(values, pose_index)
    )
    x = centre(q)
    y = centre(tool_state[:, None]).reshape(-1)
    if np.linalg.norm(x) <= 1e-12 or np.std(y) <= 1e-12:
        raise ValueError("posture or tool-state variation is insufficient")

    _, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    total_variance = float(np.sum(singular_values**2))
    if method == "pca":
        operation = _unit(vh[0])
        grasp = _unit(vh[1] - float(vh[1] @ operation) * operation)
        operation_variance = float(singular_values[0] ** 2 / total_variance)
    elif method == "pls":
        weights = PLSRegression(n_components=1, scale=False).fit(x, y).x_weights_[:, 0]
        operation = _unit(weights)
        residual = x - np.outer(x @ operation, operation)
        _, residual_singular_values, residual_vh = np.linalg.svd(
            residual, full_matrices=False
        )
        grasp = _unit(
            residual_vh[0] - float(residual_vh[0] @ operation) * operation
        )
        operation_variance = float(np.sum(np.square(x @ operation)) / total_variance)
        singular_values = residual_singular_values
    else:
        raise ValueError("method must be 'pca' or 'pls'")

    operation = _orient(operation, x, y)
    if grasp_metric is not None:
        metric = centre(np.asarray(grasp_metric, dtype=np.float64).reshape(-1, 1)).reshape(-1)
        grasp = _orient(grasp, x, metric)
    else:
        metric = np.zeros_like(y)

    residual_scores = x @ grasp
    residual_total = float(
        np.sum(np.square(x - np.outer(x @ operation, operation)))
    )
    residual_fraction = (
        float(np.sum(np.square(residual_scores)) / residual_total)
        if residual_total > 1e-12
        else 0.0
    )
    return TwoAxisSynergy(
        method=method,
        operation_axis=operation,
        grasp_axis=grasp,
        operation_tool_correlation=_correlation(x @ operation, y),
        grasp_tool_correlation=_correlation(residual_scores, y),
        grasp_metric_correlation=(
            _correlation(residual_scores, metric)
            if grasp_metric is not None
            else float("nan")
        ),
        operation_variance_fraction=operation_variance,
        grasp_residual_variance_fraction=residual_fraction,
    )
