from __future__ import annotations

from pathlib import Path

import numpy as np

from improved_tds.synergy.base import as_samples, orient_to_target, scalar_target, validate_fitted


class RandomTDS:
    """Seeded normalized random one-dimensional comparison baseline."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def fit(self, q: np.ndarray, tool_state: np.ndarray | None = None, **_: object) -> "RandomTDS":
        samples, _ = as_samples(q, "q")
        self.q_mean_ = samples.mean(axis=0)
        direction = np.random.default_rng(self.seed).normal(size=samples.shape[1])
        direction /= np.linalg.norm(direction)
        target = None if tool_state is None else scalar_target(tool_state, samples.shape[0])
        direction, self.tool_state_correlation_ = orient_to_target(
            direction, (samples - self.q_mean_) @ direction, target
        )
        self.direction_ = direction
        return self

    def encode(self, q: np.ndarray) -> np.ndarray:
        validate_fitted(self)
        samples, _ = as_samples(q, "q")
        return ((samples - self.q_mean_) @ self.direction_).reshape(-1, 1)

    def decode(self, rho: np.ndarray) -> np.ndarray:
        validate_fitted(self)
        coordinate = np.asarray(rho, dtype=np.float64).reshape(-1, 1)
        return self.q_mean_ + coordinate * self.direction_

    def save(self, path: str | Path) -> None:
        validate_fitted(self)
        np.savez_compressed(
            Path(path),
            estimator_type=np.asarray("random_tds"),
            seed=np.asarray(self.seed),
            q_mean=self.q_mean_,
            direction=self.direction_,
            tool_state_correlation=np.asarray(self.tool_state_correlation_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RandomTDS":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["estimator_type"].item()) != "random_tds":
                raise ValueError("file is not a RandomTDS model")
            model = cls(int(data["seed"].item()))
            model.q_mean_ = np.asarray(data["q_mean"], dtype=np.float64)
            model.direction_ = np.asarray(data["direction"], dtype=np.float64)
            model.tool_state_correlation_ = float(data["tool_state_correlation"].item())
        return model

