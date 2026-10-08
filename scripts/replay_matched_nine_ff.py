"""Replay frozen PCA/PLS one-DoF controllers on one common nine-goal grid.

This does not refit axes or inverse maps. It uses the paper FF test-pose IDs,
the full RL target span, and every terminal finite observation in the primary
MAE. Existing method-specific-domain reports are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.second_axis_feedback import reset_pristine_simulator
from improved_tds.experiments.simple_synergy import direct_trajectory
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from simulate_initial_pose_conditioning import create_env


FRACTIONS = tuple(float(x) for x in np.linspace(0., 1., 9))


def frozen_score(config: dict, target: float) -> tuple[float, bool]:
    selection = config["selection"]
    domain = config["inverse_target_range"]
    if selection is None or domain is None:
        raise ValueError("frozen FF inverse does not exist")
    knots = np.asarray(selection["map_tool"], float)
    limited_target = float(np.clip(target, *domain))
    u = float(np.interp(limited_target, knots, selection["map_u"]))
    rho = float(selection["rho_start"] + (u + 1.) / 2. *
                (selection["rho_end"] - selection["rho_start"]))
    return rho, bool(target < domain[0] or target > domain[1])


def run_task(task: str, bundle: Path, source: Path, output: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tasks"][task]
    report_path = bundle / entry["selected_report"]
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != entry["selected_report_sha256"]:
        raise ValueError(f"frozen FF report changed: {task}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    category = task.partition("_lite_")[0]
    env, pool, key, lo, hi = create_env(task, bundle, source, report["seed"])
    low, high, _ = _control_bounds(env)
    poses = [int(p) for p in report["test_poses"]]
    if len(poses) != 8 or len(set(poses)) != 8 or pool.pose_count != 32:
        raise ValueError(f"unexpected test-pose split: {task}")
    rows = []
    try:
        reset_pristine_simulator(env)
        for method in ("pca", "pls"):
            config = report["configurations"][method + "_1d"]
            axis = np.asarray(config["fit"]["operation_axis"], float)
            center = np.asarray(report["mean_q"], float)
            for pose in poses:
                for fraction in FRACTIONS:
                    target = lo + fraction * (hi - lo)
                    score, inverse_clipped = frozen_score(config, target)
                    requested = center + score * float(config["operation_scale"]) * axis
                    goal = np.clip(requested, low, high)
                    reset_pristine_simulator(env)
                    outcome = direct_trajectory(
                        env, pose, key, target, report["seed"] + 1000 * pose, [goal],
                        support_criterion="task" if category == "scissors" else "no_fall",
                        require_floor_contact=category == "spray")["outcomes"][0]
                    achieved = float(outcome["achieved"])
                    finite = bool(np.isfinite(achieved) and not outcome["failure_numerical"])
                    rows.append(dict(task=task, method=method, pose=pose,
                        target_fraction=fraction, target_angle=float(target),
                        achieved_angle=achieved if finite else None,
                        normalized_abs_error=(abs(achieved-target)/(hi-lo) if finite else None),
                        held=bool(outcome["valid"]), finite=finite,
                        inverse_clipped=inverse_clipped,
                        joint_clipped=bool(np.max(abs(goal-requested)) > 1e-8),
                        terminated=bool(outcome["terminated"]),
                        truncated=bool(outcome["truncated"])))
    finally:
        env.close()
    result = dict(protocol="frozen_ff_common_full_rl_nine_v1", task=task,
        source_report_sha256=entry["selected_report_sha256"],
        target_range=[lo, hi], target_fractions=FRACTIONS,
        calibration_poses=report["calibration_poses"], test_poses=poses,
        trajectory_seconds=5., terminal_hold_seconds=2.,
        no_retraining=True, no_refitting=True, rows=rows)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{task}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(task, "FF", len(rows), "rows", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "analysis/paper15_paper_bundle_20260928")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "runs/paper15_unified_3m_slew_20260927")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "analysis/matched_nine_goals_20261008/ff")
    args = parser.parse_args()
    _register_tool_manip_gym_environments()
    run_task(args.task, args.bundle, args.source, args.output)


if __name__ == "__main__":
    main()
