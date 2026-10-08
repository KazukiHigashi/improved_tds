"""Unit tests for the leakage boundary and physical-response selection rule."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "few_shot_axis_adaptation", scripts / "few_shot_axis_adaptation.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def probes(angles, *, admitted=True):
    return [dict(score=float(score), achieved_angle=float(angle),
                 admitted=admitted, joint_clipped=False)
            for score, angle in zip((-1., 0., 1.), angles)]


def test_probe_response_handles_negative_physical_slope():
    m = module()
    fit = m.fit_probe_response(probes((.8, .5, .2)), 0., 1.)
    assert fit is not None
    assert fit["slope"] < 0
    assert np.isclose(fit["coverage_fraction"], .6)
    score, clipped = m.inverse_score(.65, fit)
    assert np.isclose(score, -.5)
    assert not clipped


def test_failed_or_nonmonotone_candidate_is_not_selected():
    m = module()
    good = m.fit_probe_response(probes((.1, .5, .9)), 0., 1.)
    assert m.fit_probe_response(probes((.1, .8, .3)), 0., 1.) is None
    assert m.fit_probe_response(probes((.1, .5, .9), admitted=False), 0., 1.) is None
    assert m.fit_probe_response(probes((.50, .51, .52)), 0., 1.) is None
    models = {"common":dict(response=good), "source":dict(response=None)}
    assert m.choose_candidate(models, 0., 1.) == "common"


def test_score_generation_only_uses_known_model_and_clips_to_known_support():
    m = module()
    scores = m.probe_scores(np.array([.5, .2]), np.array([-.5, .5]))
    assert len(scores) == 3
    assert np.all(scores >= -.5) and np.all(scores <= .5)
    assert np.all(np.diff(scores) > 0)


def test_selection_prefers_measured_coverage_not_prior_label():
    m = module()
    common = m.fit_probe_response(probes((.4, .5, .6)), 0., 1.)
    source = m.fit_probe_response(probes((.1, .5, .9)), 0., 1.)
    models = {"common":dict(response=common), "known_source":dict(response=source)}
    assert m.choose_candidate(models, 0., 1.) == "known_source"


def test_guarded_common_first_stops_when_coverage_is_adequate():
    m = module()
    from validate_guarded_few_shot import decide, split_poses
    common = m.fit_probe_response(probes((.1, .5, .9)), 0., 1.)
    models = {"common":dict(response=common), "source":dict(response=None)}
    assert decide(models) == ("common", True)
    cal, test = split_poses(32, 3)
    old = set(np.random.default_rng(59323).choice(32, size=3, replace=False))
    assert cal not in test and cal not in old and not old.intersection(test)
