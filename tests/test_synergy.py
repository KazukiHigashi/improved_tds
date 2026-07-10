from __future__ import annotations

import numpy as np
import pytest

from improved_tds.synergy.family import FamilyTDS
from improved_tds.synergy.pca import PCATDS
from improved_tds.synergy.supervised import SupervisedLinearTDS


@pytest.fixture
def samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    direction = np.array([0.8, -0.4, 0.2, 0.4])
    direction /= np.linalg.norm(direction)
    state = np.linspace(-1.0, 1.0, 200)
    q = np.array([0.2, 0.4, -0.1, 0.7]) + state[:, None] * direction
    q += rng.normal(scale=0.01, size=q.shape)
    return q, state, direction


@pytest.mark.parametrize("kind", ["pca", "pls", "covariance"])
def test_estimators_align_sign_and_round_trip(samples, kind, tmp_path) -> None:
    q, state, expected = samples
    estimator = PCATDS() if kind == "pca" else SupervisedLinearTDS(method=kind)
    estimator.fit(q, state)
    assert np.isclose(np.linalg.norm(estimator.direction_), 1.0)
    assert np.corrcoef(estimator.encode(q)[:, 0], state)[0, 1] > 0.99
    assert abs(estimator.direction_ @ expected) > 0.99
    reconstructed = estimator.decode(estimator.encode(q))
    assert np.sqrt(np.mean((reconstructed - q) ** 2)) < 0.02
    path = tmp_path / f"{kind}.npz"
    estimator.save(path)
    loaded = type(estimator).load(path)
    np.testing.assert_allclose(loaded.direction_, estimator.direction_)


def test_family_tds_aligns_instance_signs(samples) -> None:
    _, _, direction = samples
    perturbed = np.vstack([direction, -direction, direction + np.array([0.01, 0, 0, 0])])
    family = FamilyTDS().fit_directions(perturbed).set_instance_mean(np.zeros(4))
    assert abs(family.direction_ @ direction) > 0.999
    assert family.confidence_ > 0.99

