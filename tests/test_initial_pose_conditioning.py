"""Synthetic checks of the initial-pose-conditioned rank-one correction."""

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_initial_pose_conditioning.py"
    spec = importlib.util.spec_from_file_location("evaluate_initial_pose_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_initial_predicts_residual_not_visible_on_operation_axis():
    m = _module()
    rng = np.random.default_rng(72)
    operation = np.eye(13)[0]
    entries = []
    for i in range(100):
        q0 = np.zeros(13)
        q0[0] = rng.normal()
        q0[1] = rng.normal()
        residual = np.zeros(13)
        residual[2] = .8 * q0[1] + rng.normal(0, .02)
        entries.append(dict(task="sample", pose=i, q0=q0, residual=residual,
                            sample_count=1))
    projected = m.fit_correction(entries, np.zeros(13), operation, "op_projection")
    full = m.fit_correction(entries, np.zeros(13), operation, "full_initial")
    assert abs(full["correction_axis"] @ operation) < 1e-12
    assert abs(full["initial_loading"][1]) > .98
    assert abs(full["correction_axis"][2]) > .98
    initial = np.stack([e["q0"] for e in entries])
    truth = np.array([e["residual"][2] for e in entries])
    full_score = m.predict_correction(full, initial)
    projected_score = m.predict_correction(projected, initial)
    full_error = np.mean((truth - full_score * full["correction_axis"][2]) ** 2)
    projected_error = np.mean((truth - projected_score * projected["correction_axis"][2]) ** 2)
    assert full_error < .05 * projected_error


def test_pose_means_collapses_targets_from_same_initial_pose():
    m = _module()
    q0 = np.zeros((4, 13))
    q0[2:, 1] = .5
    q = q0.copy()
    q[:, 0] = [0., 1., 0., 1.]
    entries = m.pose_means([dict(name="toy", q=q, q0=q0,
                                      pose=np.array([0, 0, 1, 1]))],
                           np.zeros(13), np.eye(13)[0])
    assert len(entries) == 2
    assert all(entry["sample_count"] == 2 for entry in entries)
    assert np.isclose(entries[1]["residual"][1], .5)
