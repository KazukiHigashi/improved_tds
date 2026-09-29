"""Offline guards for the real-hand trajectory planner (never imports ROS)."""
import importlib.util
from pathlib import Path
import tempfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/real_experiment/operate_tds_jointtrajectory.py"
SPEC = importlib.util.spec_from_file_location("operate_tds_jointtrajectory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    report = {
        "task": "scissors_lite_3", "mean_q": [0.1, 0.2],
        "configurations": {"pca_1d": {
            "dof": 1, "operation_scale": 0.5,
            "fit": {"operation_axis": [1, 0]},
            "selection": {"rho_start": -1, "rho_end": 1},
        }},
    }
    mapping = {"configuration": "pca_1d", "expected_start_rad": [0.1, 0.2],
               "joints": [{"name": name, "sign": 1, "offset_rad": 0,
                            "sim_min_rad": -1, "sim_max_rad": 1,
                            "min_rad": -1, "max_rad": 1, "max_velocity_rad_s": 0.2,
                            "verified": True} for name in ("rh_THJ5", "rh_THJ4")]}
    return report, mapping


def test_valid_offline_plan_and_sha():
    report, mapping = fixture()
    plan = MODULE.make_plan(report, mapping, 0.5, 5, 20, None)
    assert plan["goal_rad"] == [0.35, 0.2]
    assert plan["clipped_sim_goal_rad"] == [0.35, 0.2]
    assert len(plan["sha256"]) == 64
    assert plan["duration_s"] == 5


def test_rejects_unverified_mapping_or_extrapolation():
    report, mapping = fixture()
    mapping["joints"][0]["verified"] = False
    with pytest.raises(ValueError, match="not independently verified"):
        MODULE.make_plan(report, mapping, 0, 5, 20, None)
    mapping["joints"][0]["verified"] = True
    with pytest.raises(ValueError, match="outside selected interval"):
        MODULE.make_plan(report, mapping, 2, 5, 20, None)
    with pytest.raises(ValueError, match="at least five seconds"):
        MODULE.make_plan(report, mapping, 0, 1, 20, None)


def test_simulation_actuator_clipping_precedes_real_mapping():
    report, mapping = fixture()
    mapping["joints"][0]["sim_max_rad"] = 0.4
    plan = MODULE.make_plan(report, mapping, 1, 5, 20, None)
    assert plan["clipped_sim_goal_rad"][0] == 0.4
    assert plan["goal_rad"][0] == 0.4


def test_real_inverse_requires_monotone_measured_branch():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cal.csv"
        path.write_text("score,angle_rad,valid\n-1,0,1\n0,0.5,1\n1,1,1\n")
        assert MODULE.inverse_score(path, 0.25) == -0.5
        path.write_text("score,angle_rad,valid\n-1,0,1\n0,0.4,1\n0,0.6,1\n1,1,1\n")
        assert MODULE.inverse_score(path, 0.25) == -0.5
        with pytest.raises(ValueError, match="outside"):
            MODULE.inverse_score(path, 1.1)
        path.write_text("score,angle_rad,valid\n-1,0,1\n0,0.7,1\n1,0.5,1\n")
        with pytest.raises(ValueError, match="strictly monotone"):
            MODULE.inverse_score(path, 0.6)
