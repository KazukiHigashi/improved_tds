"""Checks for leakage-free initial-posture PCA and nearest-axis lookup."""

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("evaluate_initial_pca_nn",
                                                   scripts / "evaluate_initial_pca_nn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pca_is_fitted_only_from_given_known_postures():
    m = _module()
    known = np.zeros((8, 13))
    known[:, 0] = np.arange(8) - 3.5
    known[:, 1] = np.array([-1, 1] * 4)
    model = m.fit_pca(known)
    unknown = np.zeros((1, 13))
    unknown[0, 0] = 200.
    projected = m.coordinates(unknown, model, "pca2")
    assert np.allclose(model["mean"], known.mean(axis=0))
    assert projected.shape == (1, 2)
    assert abs(projected[0, 0]) > 100.
    assert np.isclose(np.sum(model["variance"][:2]), 1.)


def test_2d_nn_returns_training_row_and_not_a_task_id():
    m = _module()
    known = np.zeros((6, 13))
    known[:, 0] = [-3, -2, -1, 1, 2, 3]
    known[:, 1] = [0, .2, -.2, .1, -.1, 0]
    model = m.fit_pca(known)
    query = known[[4]] + .01
    index, distance = m.select_nearest(query, known, model, "pca2")
    assert index.tolist() == [4]
    assert distance[0] < .1


def test_unique_initials_does_not_replicate_target_angle_rows():
    m = _module()
    q0 = np.zeros((4, 13))
    q0[2:, 0] = 1.
    block = dict(name="tool", q0=q0, pose=np.array([0, 0, 1, 1]))
    initial, labels, poses = m.unique_initials([block])
    assert initial.shape == (2, 13)
    assert labels.tolist() == ["tool", "tool"]
    assert poses.tolist() == [0, 1]
