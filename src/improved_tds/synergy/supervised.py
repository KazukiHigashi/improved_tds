from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from improved_tds.synergy.base import as_samples, scalar_target, validate_fitted


class SupervisedLinearTDS:
    """Interpretable one-dimensional TDS supervised by measured tool state.

    The direction is learned in standardized joint space, converted to a unit vector
    in physical joint space, and sign-aligned so `corr(rho, tool_state) >= 0`.
    """

    def __init__(
        self,
        *,
        method: Literal["pls", "covariance"] = "pls",
        standardize: bool = True,
    ):
        if method not in {"pls", "covariance"}:
            raise ValueError("method must be 'pls' or 'covariance'")
        self.method = method
        self.standardize = bool(standardize)

    def fit(
        self,
        q: np.ndarray,
        tool_state: np.ndarray | None = None,
        **_: object,
    ) -> "SupervisedLinearTDS":
        samples, _ = as_samples(q, "q")
        if tool_state is None:
            raise ValueError("SupervisedLinearTDS requires tool_state")
        if samples.shape[0] < 3:
            raise ValueError("supervised TDS requires at least three samples")
        target = scalar_target(tool_state, samples.shape[0])
        self.q_mean_ = samples.mean(axis=0)
        scale = samples.std(axis=0, ddof=1)
        scale[scale <= 1e-12] = 1.0
        self.q_scale_ = scale if self.standardize else np.ones_like(scale)
        x = (samples - self.q_mean_) / self.q_scale_
        y = target - target.mean()
        if np.std(y) <= 1e-12:
            raise ValueError("tool_state has no variation")

        if self.method == "pls":
            pls = PLSRegression(n_components=1, scale=False)
            pls.fit(x, y.reshape(-1, 1))
            standardized_direction = np.asarray(pls.x_weights_[:, 0], dtype=np.float64)
        else:
            standardized_direction = x.T @ y / float(samples.shape[0] - 1)

        physical_direction = self.q_scale_ * standardized_direction
        norm = float(np.linalg.norm(physical_direction))
        if norm <= 1e-12:
            raise ValueError("could not identify a non-zero supervised direction")
        physical_direction /= norm
        scores = (samples - self.q_mean_) @ physical_direction
        correlation = float(np.corrcoef(scores, target)[0, 1])
        if correlation < 0.0:
            physical_direction *= -1.0
            scores *= -1.0
            correlation *= -1.0
        self.direction_ = physical_direction
        self.tool_state_correlation_ = correlation
        design = np.column_stack([np.ones(samples.shape[0]), scores])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        self.tool_state_intercept_ = float(coefficients[0])
        self.tool_state_slope_ = float(coefficients[1])
        predicted = design @ coefficients
        self.tool_state_rmse_ = float(np.sqrt(np.mean((predicted - target) ** 2)))
        reconstructed = self.q_mean_ + scores[:, None] * self.direction_
        self.reconstruction_rmse_ = float(np.sqrt(np.mean((reconstructed - samples) ** 2)))
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

    def predict_tool_state(self, q: np.ndarray) -> np.ndarray:
        rho = self.encode(q)[:, 0]
        return (self.tool_state_intercept_ + self.tool_state_slope_ * rho).reshape(-1, 1)

    def save(self, path: Path | str) -> None:
        validate_fitted(self)
        np.savez_compressed(
            Path(path),
            estimator_type=np.asarray("supervised_linear_tds"),
            method=np.asarray(self.method),
            standardize=np.asarray(self.standardize),
            q_mean=self.q_mean_,
            q_scale=self.q_scale_,
            direction=self.direction_,
            tool_state_correlation=np.asarray(self.tool_state_correlation_),
            tool_state_intercept=np.asarray(self.tool_state_intercept_),
            tool_state_slope=np.asarray(self.tool_state_slope_),
            tool_state_rmse=np.asarray(self.tool_state_rmse_),
            reconstruction_rmse=np.asarray(self.reconstruction_rmse_),
        )

    @classmethod
    def load(cls, path: Path | str) -> "SupervisedLinearTDS":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["estimator_type"].item()) != "supervised_linear_tds":
                raise ValueError("file is not a SupervisedLinearTDS model")
            model = cls(
                method=str(data["method"].item()),
                standardize=bool(data["standardize"].item()),
            )
            for saved, attribute in {
                "q_mean": "q_mean_",
                "q_scale": "q_scale_",
                "direction": "direction_",
                "tool_state_correlation": "tool_state_correlation_",
                "tool_state_intercept": "tool_state_intercept_",
                "tool_state_slope": "tool_state_slope_",
                "tool_state_rmse": "tool_state_rmse_",
                "reconstruction_rmse": "reconstruction_rmse_",
            }.items():
                value = np.asarray(data[saved], dtype=np.float64)
                setattr(model, attribute, float(value.item()) if value.ndim == 0 else value)
        return model

