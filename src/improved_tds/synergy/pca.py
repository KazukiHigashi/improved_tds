from __future__ import annotations

from pathlib import Path

import numpy as np

from improved_tds.synergy.base import (
    as_samples,
    orient_to_target,
    scalar_target,
    validate_fitted,
)


class PCATDS:
    """One-dimensional PCA baseline compatible with terminal-posture datasets."""

    def __init__(self, *, align_sign: bool = True):
        self.align_sign = bool(align_sign)

    def fit(self, q: np.ndarray, tool_state: np.ndarray | None = None, **_: object) -> "PCATDS":
        samples, _ = as_samples(q, "q")
        if samples.shape[0] < 2:
            raise ValueError("PCA requires at least two samples")
        target = None if tool_state is None else scalar_target(tool_state, samples.shape[0])
        self.q_mean_ = samples.mean(axis=0)
        centered = samples - self.q_mean_
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0].astype(np.float64)
        scores = centered @ direction
        if self.align_sign:
            direction, self.tool_state_correlation_ = orient_to_target(direction, scores, target)
        else:
            self.tool_state_correlation_ = float("nan")
        self.direction_ = direction / np.linalg.norm(direction)
        variance = singular_values**2 / max(samples.shape[0] - 1, 1)
        total = float(variance.sum())
        self.explained_variance_ = float(variance[0])
        self.explained_variance_ratio_ = float(variance[0] / total) if total > 0.0 else 0.0
        return self

    def encode(self, q: np.ndarray) -> np.ndarray:
        validate_fitted(self)
        samples, _ = as_samples(q, "q")
        if samples.shape[1] != self.direction_.shape[0]:
            raise ValueError("q feature count does not match fitted estimator")
        return ((samples - self.q_mean_) @ self.direction_).reshape(-1, 1)

    def decode(self, rho: np.ndarray) -> np.ndarray:
        validate_fitted(self)
        coordinate = np.asarray(rho, dtype=np.float64)
        if coordinate.ndim == 0:
            coordinate = coordinate.reshape(1, 1)
        elif coordinate.ndim == 1:
            coordinate = coordinate.reshape(-1, 1)
        if coordinate.ndim != 2 or coordinate.shape[1] != 1:
            raise ValueError("rho must have shape (n_samples, 1)")
        if not np.all(np.isfinite(coordinate)):
            raise ValueError("rho contains NaN or infinity")
        return self.q_mean_ + coordinate * self.direction_

    def save(self, path: Path | str) -> None:
        validate_fitted(self)
        np.savez_compressed(
            Path(path),
            estimator_type=np.asarray("pca_tds"),
            q_mean=self.q_mean_,
            direction=self.direction_,
            explained_variance=np.asarray(self.explained_variance_),
            explained_variance_ratio=np.asarray(self.explained_variance_ratio_),
            tool_state_correlation=np.asarray(self.tool_state_correlation_),
            align_sign=np.asarray(self.align_sign),
        )

    @classmethod
    def load(cls, path: Path | str) -> "PCATDS":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["estimator_type"].item()) != "pca_tds":
                raise ValueError("file is not a PCATDS model")
            model = cls(align_sign=bool(data["align_sign"].item()))
            model.q_mean_ = np.asarray(data["q_mean"], dtype=np.float64)
            model.direction_ = np.asarray(data["direction"], dtype=np.float64)
            model.explained_variance_ = float(data["explained_variance"].item())
            model.explained_variance_ratio_ = float(data["explained_variance_ratio"].item())
            model.tool_state_correlation_ = float(data["tool_state_correlation"].item())
        return model

