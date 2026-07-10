from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np


EstimatorT = TypeVar("EstimatorT", bound="TDSEstimator")


@runtime_checkable
class TDSEstimator(Protocol):
    """Common one-dimensional TDS contract.

    `rho` always has a final dimension of one. Estimators orient their direction so
    rho has positive correlation with tool state when a supervision signal is supplied.
    Joint positions and decoded outputs are in radians.
    """

    direction_: np.ndarray
    q_mean_: np.ndarray

    def fit(
        self: EstimatorT,
        q: np.ndarray,
        tool_state: np.ndarray | None = None,
        **kwargs: Any,
    ) -> EstimatorT: ...

    def encode(self, q: np.ndarray) -> np.ndarray: ...

    def decode(self, rho: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path | str) -> None: ...

    @classmethod
    def load(cls: type[EstimatorT], path: Path | str) -> EstimatorT: ...


def as_samples(values: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_samples, n_features), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array, single


def scalar_target(values: np.ndarray, n_samples: int) -> np.ndarray:
    target = np.asarray(values, dtype=np.float64)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1 or target.shape[0] != n_samples:
        raise ValueError(f"tool_state must have shape ({n_samples},) or ({n_samples}, 1)")
    if not np.all(np.isfinite(target)):
        raise ValueError("tool_state contains NaN or infinity")
    return target


def orient_to_target(
    direction: np.ndarray,
    scores: np.ndarray,
    target: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    if target is None or np.std(scores) <= 1e-12 or np.std(target) <= 1e-12:
        return direction, float("nan")
    correlation = float(np.corrcoef(scores, target)[0, 1])
    if correlation < 0.0:
        return -direction, -correlation
    return direction, correlation


def validate_fitted(estimator: Any) -> None:
    if not hasattr(estimator, "direction_") or not hasattr(estimator, "q_mean_"):
        raise RuntimeError("estimator has not been fitted")

