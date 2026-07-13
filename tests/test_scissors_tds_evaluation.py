from __future__ import annotations

import numpy as np

from improved_tds.evaluation.scissors_tds import (
    correlation_with_episode_bootstrap,
    paired_bootstrap_difference,
)


def test_episode_bootstrap_correlation_is_reproducible() -> None:
    state = np.linspace(0.0, 1.0, 20)
    score = 2.0 * state + 0.01 * np.sin(np.arange(20))
    first = correlation_with_episode_bootstrap(
        score, state, seed=4, bootstrap_samples=100, permutation_samples=100
    )
    second = correlation_with_episode_bootstrap(
        score, state, seed=4, bootstrap_samples=100, permutation_samples=100
    )
    assert first == second
    assert first.pearson_r > 0.99
    assert first.null_95[1] < first.pearson_r


def test_paired_bootstrap_reports_feedback_error_reduction() -> None:
    result = paired_bootstrap_difference(
        [0.3, 0.2, 0.4, 0.5],
        [0.1, 0.1, 0.2, 0.2],
        seed=2,
        bootstrap_samples=100,
    )
    assert result["mean_error_reduction"] == 0.2
    assert result["paired_standardized_effect"] > 0.0
