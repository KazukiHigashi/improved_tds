"""Lightweight checks for the frozen full-RL nine-target replay rules."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nine_targets_include_rl_endpoints():
    ff = load("replay_matched_nine_ff")
    unknown = load("replay_matched_nine_generalization")
    assert len(ff.FRACTIONS) == len(unknown.FRACTIONS) == 9
    assert np.allclose(ff.FRACTIONS, np.linspace(0., 1., 9))
    assert ff.FRACTIONS == unknown.FRACTIONS


def test_frozen_ff_inverse_clips_to_admitted_domain_not_outer_knots():
    ff = load("replay_matched_nine_ff")
    config = dict(selection=dict(map_tool=[-.1, .2, .8, 1.1],
                                 map_u=[-1., -.6, .6, 1.],
                                 rho_start=-2., rho_end=2.),
                  inverse_target_range=[.2, .8])
    below, clipped_below = ff.frozen_score(config, 0.)
    inside, clipped_inside = ff.frozen_score(config, .5)
    above, clipped_above = ff.frozen_score(config, 1.)
    assert np.isclose(below, -.6 * 2.)
    assert np.isclose(inside, 0.)
    assert np.isclose(above, .6 * 2.)
    assert clipped_below and clipped_above and not clipped_inside


def test_unknown_zero_score_uses_normalized_fraction_not_physical_units():
    unknown = load("replay_matched_nine_generalization")
    package = (np.zeros(13), np.ones(13), np.array([.5, .25]),
               np.array([-2., 2.]))
    score, clipped = unknown.command_score(.75, .006, package, None)
    assert np.isclose(score, 1.)
    assert not clipped


def test_mae_illustration_is_first_frozen_pose_and_matches_raw_trials():
    report = load("build_matched_nine_report")
    result = report.RESULT
    source = json.loads((result / "ff" / "scissors_lite_3.json").read_text(encoding="utf-8"))
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    example = summary["mae_illustration"]
    assert example["pose"] == source["test_poses"][0]
    assert example["method"] == "pls"
    assert len(example["rows"]) == 9
    span = source["target_range"][1] - source["target_range"][0]
    for row in example["rows"]:
        assert np.isclose(row["normalized_abs_error"],
                          abs(row["achieved_angle"] - row["target_angle"]) / span)
    assert np.isclose(example["nine_target_mae_percent"],
                      100 * mean(row["normalized_abs_error"] for row in example["rows"]))
    assert "図B" in report.mae_illustration(example)
