from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class CorrelationEstimate:
    pearson_r: float
    pearson_ci: tuple[float, float]
    spearman_r: float
    spearman_ci: tuple[float, float]
    null_95: tuple[float, float]
    episodes: int

    def as_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "pearson_r": self.pearson_r,
            "pearson_ci": list(self.pearson_ci),
            "spearman_r": self.spearman_r,
            "spearman_ci": list(self.spearman_ci),
            "null_95": list(self.null_95),
            "episodes": self.episodes,
        }


def correlation_with_episode_bootstrap(
    score: np.ndarray,
    tool_state: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 2_000,
    permutation_samples: int = 2_000,
) -> CorrelationEstimate:
    """Estimate held-out correlation with episode as the resampling unit."""

    x = np.asarray(score, dtype=np.float64).reshape(-1)
    y = np.asarray(tool_state, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("score and tool_state need at least three paired episodes")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("correlation data must be finite")
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        raise ValueError("correlation data must vary")
    if bootstrap_samples < 1 or permutation_samples < 1:
        raise ValueError("resampling counts must be positive")

    rng = np.random.default_rng(seed)
    pearson_samples: list[float] = []
    spearman_samples: list[float] = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, x.size, size=x.size)
        if np.ptp(x[indices]) <= 1e-12 or np.ptp(y[indices]) <= 1e-12:
            continue
        pearson_samples.append(float(pearsonr(x[indices], y[indices]).statistic))
        spearman_samples.append(float(spearmanr(x[indices], y[indices]).statistic))
    if not pearson_samples or not spearman_samples:
        raise ValueError("bootstrap samples have no usable variation")

    null_samples = np.empty(permutation_samples, dtype=np.float64)
    for index in range(permutation_samples):
        null_samples[index] = pearsonr(x, rng.permutation(y)).statistic
    return CorrelationEstimate(
        pearson_r=float(pearsonr(x, y).statistic),
        pearson_ci=tuple(np.quantile(pearson_samples, [0.025, 0.975])),
        spearman_r=float(spearmanr(x, y).statistic),
        spearman_ci=tuple(np.quantile(spearman_samples, [0.025, 0.975])),
        null_95=tuple(np.quantile(null_samples, [0.025, 0.975])),
        episodes=int(x.size),
    )


def paired_bootstrap_difference(
    feedforward: Iterable[float],
    feedback: Iterable[float],
    *,
    seed: int,
    bootstrap_samples: int = 2_000,
) -> dict[str, float | int | list[float]]:
    """Return paired error reduction: feedforward minus feedback."""

    first = np.asarray(list(feedforward), dtype=np.float64).reshape(-1)
    second = np.asarray(list(feedback), dtype=np.float64).reshape(-1)
    if first.shape != second.shape or first.size < 2:
        raise ValueError("paired inputs need at least two values")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("paired inputs must be finite")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    difference = first - second
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        selected = rng.integers(0, difference.size, size=difference.size)
        bootstrap[index] = np.mean(difference[selected])
    scale = float(np.std(difference, ddof=1))
    return {
        "episodes": int(difference.size),
        "mean_error_reduction": float(np.mean(difference)),
        "median_error_reduction": float(np.median(difference)),
        "mean_95_ci": list(np.quantile(bootstrap, [0.025, 0.975])),
        "paired_standardized_effect": float(np.mean(difference) / scale)
        if scale > 1e-12
        else 0.0,
    }
