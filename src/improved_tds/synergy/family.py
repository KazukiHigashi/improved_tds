from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from improved_tds.synergy.base import as_samples, validate_fitted


class FamilyTDS:
    """Shared TDS direction from sign-aligned instance directions.

    Instance-specific mean posture and tool-state calibration intentionally remain
    outside this model so a held-out instance can be calibrated with a few samples.
    """

    def fit_directions(
        self,
        directions: np.ndarray | Sequence[np.ndarray],
        *,
        weights: np.ndarray | None = None,
        reference_direction: np.ndarray | None = None,
    ) -> "FamilyTDS":
        matrix = np.asarray(directions, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError("directions must have shape (n_instances, n_joints)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("directions contain NaN or infinity")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("directions must be non-zero")
        matrix = matrix / norms[:, None]
        reference = matrix[0] if reference_direction is None else np.asarray(reference_direction)
        reference = reference / np.linalg.norm(reference)
        matrix[np.sum(matrix * reference, axis=1) < 0.0] *= -1.0
        if weights is None:
            normalized_weights = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
        else:
            normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if normalized_weights.shape != (matrix.shape[0],) or np.any(normalized_weights < 0.0):
                raise ValueError("weights must be non-negative and match instance count")
            total = float(normalized_weights.sum())
            if total <= 0.0:
                raise ValueError("at least one weight must be positive")
            normalized_weights /= total
        scatter = np.einsum("i,ij,ik->jk", normalized_weights, matrix, matrix)
        values, vectors = np.linalg.eigh(scatter)
        direction = vectors[:, int(np.argmax(values))]
        if float(direction @ reference) < 0.0:
            direction *= -1.0
        self.direction_ = direction / np.linalg.norm(direction)
        self.instance_directions_ = matrix
        self.weights_ = normalized_weights
        self.confidence_ = float(values[-1] / max(values.sum(), 1e-12))
        return self

    def set_instance_mean(self, q_mean: np.ndarray) -> "FamilyTDS":
        mean = np.asarray(q_mean, dtype=np.float64).reshape(-1)
        if not hasattr(self, "direction_") or mean.shape != self.direction_.shape:
            raise ValueError("instance mean shape must match fitted family direction")
        if not np.all(np.isfinite(mean)):
            raise ValueError("instance mean contains NaN or infinity")
        self.q_mean_ = mean
        return self

    def fit(self, q: np.ndarray, tool_state: np.ndarray | None = None, **_: object) -> "FamilyTDS":
        del tool_state
        samples, _ = as_samples(q, "q")
        if not hasattr(self, "direction_"):
            raise RuntimeError("fit_directions must be called before fitting an instance mean")
        return self.set_instance_mean(samples.mean(axis=0))

    def encode(self, q: np.ndarray) -> np.ndarray:
        validate_fitted(self)
        samples, _ = as_samples(q, "q")
        return ((samples - self.q_mean_) @ self.direction_).reshape(-1, 1)

    def decode(self, rho: np.ndarray) -> np.ndarray:
        validate_fitted(self)
        coordinate = np.asarray(rho, dtype=np.float64).reshape(-1, 1)
        if not np.all(np.isfinite(coordinate)):
            raise ValueError("rho contains NaN or infinity")
        return self.q_mean_ + coordinate * self.direction_

    def save(self, path: Path | str) -> None:
        validate_fitted(self)
        np.savez_compressed(
            Path(path),
            estimator_type=np.asarray("family_tds"),
            direction=self.direction_,
            q_mean=self.q_mean_,
            instance_directions=self.instance_directions_,
            weights=self.weights_,
            confidence=np.asarray(self.confidence_),
        )

    @classmethod
    def load(cls, path: Path | str) -> "FamilyTDS":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["estimator_type"].item()) != "family_tds":
                raise ValueError("file is not a FamilyTDS model")
            model = cls()
            model.direction_ = np.asarray(data["direction"], dtype=np.float64)
            model.q_mean_ = np.asarray(data["q_mean"], dtype=np.float64)
            model.instance_directions_ = np.asarray(data["instance_directions"], dtype=np.float64)
            model.weights_ = np.asarray(data["weights"], dtype=np.float64)
            model.confidence_ = float(data["confidence"].item())
        return model

