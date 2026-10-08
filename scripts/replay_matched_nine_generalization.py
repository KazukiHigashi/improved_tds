"""Matched full-RL nine-goal physical evaluation of frozen unknown-tool axes.

The held-out tool success DB is accessed only by explicitly named oracle arms.
All arms share the FF paper report's eight test pose IDs and nine target angles.
Few-shot uses one of the eight *calibration* pose IDs and three probes per axis.
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
from improved_tds.experiments.collect_posture_database import _actuated_q
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.second_axis_feedback import reset_pristine_simulator
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from evaluate_initial_pca_nn import fit_pca, select_nearest, unique_initials
from evaluate_initial_pose_conditioning import fit_operation, read_category
from few_shot_axis_adaptation import (PROBE_FRACTIONS, choose_candidate,
    evaluate_command, fit_probe_response, inverse_score, probe_scores)
from simulate_initial_pose_conditioning import create_env, fit_angle_inverse, fit_local


FRACTIONS = tuple(float(x) for x in np.linspace(0., 1., 9))
ARMS = ("common_zero", "pca2_nn_zero", "common_three_probe",
        "selected_three_probe", "local_pls_oracle", "local_pls_three_probe_oracle")


def fit_three_probes(env, pose, key, lo, hi, category, package, low, high):
    reference, direction, inverse, bounds = package
    probes = []
    for fraction, score in zip(PROBE_FRACTIONS, probe_scores(inverse, bounds)):
        reset_pristine_simulator(env)
        outcome, joint_clipped = evaluate_command(
            env, pose, key, lo + .5*(hi-lo), 72000 + pose, category,
            reference, direction, float(score), low, high)
        achieved = float(outcome["achieved"])
        admitted = bool(outcome["held"] and not outcome["failed"]
            and not outcome["failure_numerical"] and not outcome["terminated"]
            and not outcome["truncated"] and np.isfinite(achieved))
        probes.append(dict(score=float(score), nominal_fraction=float(fraction),
            achieved_angle=achieved if np.isfinite(achieved) else None,
            admitted=admitted, strict_hold_valid=bool(outcome["valid"]),
            joint_clipped=bool(joint_clipped)))
    # The fitter reads angle only when all probes were admitted. A null failed
    # angle would otherwise raise before rejection, so use NaN internally.
    fit_input = [dict(p, achieved_angle=(p["achieved_angle"] if p["achieved_angle"]
                       is not None else float("nan"))) for p in probes]
    return probes, fit_probe_response(fit_input, lo, hi)


def command_score(fraction, target, package, response):
    inverse, bounds = package[2], package[3]
    raw = (fraction - float(inverse[0])) / float(inverse[1])
    if response is not None:
        return inverse_score(target, response)
    score = float(np.clip(raw, *bounds))
    return score, bool(not np.isclose(score, raw))


def run_task(task: str, bundle: Path, source: Path, output: Path) -> dict:
    category, _, number = task.partition("_lite_")
    variant = int(number)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tasks"][task]
    report_path = bundle / entry["selected_report"]
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != entry["selected_report_sha256"]:
        raise ValueError(f"frozen FF report changed: {task}")
    ff_report = json.loads(report_path.read_text(encoding="utf-8"))
    test_poses = [int(p) for p in ff_report["test_poses"]]
    cal_poses = [int(p) for p in ff_report["calibration_poses"]]
    if len(test_poses) != 8 or len(cal_poses) != 8 or set(test_poses) & set(cal_poses):
        raise ValueError(f"invalid frozen initial-pose split: {task}")
    cal_pose = cal_poses[0]
    blocks = read_category(bundle, category)
    held = blocks[variant-1]
    known = [b for i,b in enumerate(blocks) if i != variant-1]
    center, axis = fit_operation(known)
    common_inverse, common_bounds = fit_angle_inverse(known, center, axis)
    known_packages = {b["name"]: fit_local(b) for b in known}
    common = (center, axis, common_inverse, common_bounds)
    local_oracle = fit_local(held)
    train_initial, train_labels, _ = unique_initials(known)
    pca = fit_pca(train_initial)
    env, pool, key, lo, hi = create_env(task, bundle, source, ff_report["seed"])
    low, high, _ = _control_bounds(env)
    if pool.pose_count != 32 or not np.allclose([lo,hi], ff_report["target_range"]):
        env.close()
        raise ValueError(f"pose pool or target range changed: {task}")
    rows = []
    try:
        reset_pristine_simulator(env)
        candidates = {"common": common, **known_packages}
        candidate_models = {}
        probes_attempted = 0
        for name, package in candidates.items():
            if name != "common":
                observed = candidate_models["common"]["response"]
                if observed is not None and observed["coverage_fraction"] >= .75:
                    break
            probes, response = fit_three_probes(
                env, cal_pose, key, lo, hi, category, package, low, high)
            candidate_models[name] = dict(probes=probes, response=response)
            probes_attempted += len(probes)
        common_response = candidate_models["common"]["response"]
        selected = ("common" if common_response is not None
                    and common_response["coverage_fraction"] >= .75
                    else choose_candidate(candidate_models, 0., 1.))
        oracle_probes, oracle_response = fit_three_probes(
            env, cal_pose, key, lo, hi, category, local_oracle, low, high)
        for pose in test_poses:
            reset_pristine_simulator(env)
            env.reset(seed=ff_report["seed"]+1000*pose,
                      options={"initial_pose_index":pose, key:lo+.5*(hi-lo)})
            initial = _actuated_q(env)
            nearest, _ = select_nearest(initial[None], train_initial, pca, "pca2")
            nearest_name = str(train_labels[nearest[0]])
            arm_specs = {
                "common_zero": (common, None),
                "pca2_nn_zero": (known_packages[nearest_name], None),
                "common_three_probe": (common, common_response),
                "selected_three_probe": (
                    candidates[selected], candidate_models[selected]["response"])
                    if selected is not None else (common, None),
                "local_pls_oracle": (local_oracle, None),
                "local_pls_three_probe_oracle": (local_oracle, oracle_response),
            }
            for fraction in FRACTIONS:
                target = lo + fraction*(hi-lo)
                for arm in ARMS:
                    package, response = arm_specs[arm]
                    score, clipped = command_score(fraction, target, package, response)
                    reset_pristine_simulator(env)
                    outcome, joint_clipped = evaluate_command(
                        env, pose, key, target, ff_report["seed"]+1000*pose,
                        category, package[0], package[1], score, low, high)
                    achieved = float(outcome["achieved"])
                    finite = bool(np.isfinite(achieved) and not outcome["failure_numerical"])
                    rows.append(dict(task=task, arm=arm, pose=pose,
                        target_fraction=fraction, target_angle=float(target),
                        achieved_angle=achieved if finite else None,
                        normalized_abs_error=(abs(achieved-target)/(hi-lo) if finite else None),
                        held=bool(outcome["valid"]), finite=finite,
                        inverse_clipped=clipped, joint_clipped=bool(joint_clipped),
                        selected_source=nearest_name if arm == "pca2_nn_zero" else None,
                        terminated=bool(outcome["terminated"]),
                        truncated=bool(outcome["truncated"])))
    finally:
        env.close()
    result = dict(protocol="unknown_tool_common_full_rl_nine_v1", task=task,
        source_report_sha256=entry["selected_report_sha256"],
        target_range=[lo,hi], target_fractions=FRACTIONS,
        calibration_pose=cal_pose, test_poses=test_poses,
        known_success_postures=800, held_success_postures_oracle_only=200,
        probes_attempted=probes_attempted, selected_candidate=selected,
        candidate_models=candidate_models, oracle_probes=oracle_probes,
        oracle_response=oracle_response, trajectory_seconds=5.,
        terminal_hold_seconds=2., no_retraining=True, rows=rows)
    output.mkdir(parents=True, exist_ok=True)
    (output/f"{task}.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(task, "generalization", len(rows), "rows", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "analysis/paper15_paper_bundle_20260928")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "runs/paper15_unified_3m_slew_20260927")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "analysis/matched_nine_goals_20261008/generalization")
    args = parser.parse_args()
    _register_tool_manip_gym_environments()
    run_task(args.task, args.bundle, args.source, args.output)


if __name__ == "__main__":
    main()
