"""Few-shot, leave-one-tool-out physical calibration of one-DoF TDS.

The held-out tool contributes neither successful final postures nor a policy
to candidate construction. Three independent physical probes on one reset
pose per candidate are the *only* adaptation data. Evaluation uses three
different authored reset poses and five target angles. Every command follows
the existing 5 s ramp + 2 s hold protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_initial_pose_conditioning import CATEGORIES, fit_operation, read_category
from simulate_initial_pose_conditioning import (TARGET_FRACTIONS, create_env,
                                                fit_angle_inverse, fit_local)
from improved_tds.experiments.collect_posture_database import _actuated_q
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.simple_synergy import direct_trajectory
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


PROBE_FRACTIONS = (.15, .5, .85)
ARMS = ("common_db_ff", "common_fewshot", "selected_fewshot", "local_oracle")


def probe_scores(inverse, bounds):
    """Known-tool DB gives only safe-ish candidate scores, never target observations."""
    intercept, slope = map(float, inverse)
    if abs(slope) < 1e-8:
        raise ValueError("degenerate known-tool inverse")
    return np.clip((np.asarray(PROBE_FRACTIONS) - intercept) / slope, *bounds)


def fit_probe_response(probes, lo, hi):
    """Fit a two/three-point monotone response without test-target information.

    A numerically failed or finally unsupported probe makes that package
    ineligible; a transient hold-window failure alone does not discard its
    finite, finally supported angle. Monotone sign may be positive or negative.
    """
    ordered = sorted(probes, key=lambda p: p["score"])
    if len(ordered) != len(PROBE_FRACTIONS):
        raise ValueError("all three probes must be recorded")
    if any(not p["admitted"] for p in ordered):
        return None
    scores = np.asarray([p["score"] for p in ordered], float)
    angles = np.asarray([p["achieved_angle"] for p in ordered], float)
    if np.min(np.diff(scores)) <= 1e-8:
        return None
    delta = np.diff(angles)
    # Allow a small measurement reversal, but reject an effectively flat axis.
    tolerance = .01 * (hi-lo)
    monotone = bool(np.all(delta >= -tolerance) or np.all(delta <= tolerance))
    if not monotone or abs(angles[-1] - angles[0]) <= .05 * (hi-lo):
        return None
    slope, intercept = np.polyfit(scores, angles, 1)
    angle_bounds = [float(np.min(angles)), float(np.max(angles))]
    covered = max(0., min(hi, angle_bounds[1])-max(lo, angle_bounds[0])) / (hi-lo)
    residual = float(np.sqrt(np.mean((angles-(intercept+slope*scores))**2))/(hi-lo))
    return dict(slope=float(slope), intercept=float(intercept),
                score_bounds=[float(scores[0]), float(scores[-1])],
                angle_bounds=angle_bounds, coverage_fraction=float(covered),
                residual_fraction=residual, monotone=monotone,
                joint_clip_count=sum(bool(p["joint_clipped"]) for p in ordered))


def choose_candidate(models, lo, hi):
    """Maximize observed target coverage; never select using evaluation trials."""
    eligible = [(name, model["response"]) for name, model in models.items()
                if model["response"] is not None]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (
        item[1]["coverage_fraction"], -item[1]["joint_clip_count"],
        -item[1]["residual_fraction"], item[0] == "common"))[0]


def inverse_score(target, response):
    raw = (target-response["intercept"])/response["slope"]
    limited = float(np.clip(raw, *response["score_bounds"]))
    return limited, bool(not np.isclose(raw, limited))


def evaluate_command(env, pose, key, target, seed, category, reference, axis, score,
                     low, high):
    requested = reference + score*axis
    goal = np.clip(requested, low, high)
    outcome = direct_trajectory(
        env, pose, key, target, seed, [goal],
        support_criterion="task" if category == "scissors" else "no_fall",
        require_floor_contact=category == "spray")["outcomes"][0]
    return outcome, bool(np.max(np.abs(goal-requested)) > 1e-8)


def evaluate_tool(category, variant, bundle, source, output, *, test_pose_count=3):
    blocks = read_category(bundle, category)
    held = blocks[variant-1]
    known = [b for i,b in enumerate(blocks) if i != variant-1]
    mean, axis = fit_operation(known)
    common_inverse, common_bounds = fit_angle_inverse(known, mean, axis)
    packages = {"common":(mean, axis, common_inverse, common_bounds)}
    packages.update({b["name"]:fit_local(b) for b in known})
    # The held-out local model exists exclusively as a retrospective oracle.
    oracle = fit_local(held)
    env, pool, key, lo, hi = create_env(held["name"], bundle, source, 48200+variant)
    low, high, _ = _control_bounds(env)
    old_test_poses = np.random.default_rng(59320+variant).choice(
        pool.pose_count, size=test_pose_count, replace=False).tolist()
    calibration_pose = next(p for p in range(pool.pose_count) if p not in old_test_poses)
    assert calibration_pose not in old_test_poses
    models = {}
    records = []
    try:
        for name, (reference, direction, inverse, bounds) in packages.items():
            probes = []
            for fraction, score in zip(PROBE_FRACTIONS, probe_scores(inverse, bounds)):
                # Same physical target and seed for each package; this target is
                # not the probe's outcome or a success threshold.
                target = lo+.5*(hi-lo)
                outcome, joint_clipped = evaluate_command(
                    env, calibration_pose, key, target, 62000+calibration_pose,
                    category, reference, direction, float(score), low, high)
                final_supported = bool(outcome["held"] and not outcome["failed"]
                                       and not outcome["failure_numerical"]
                                       and not outcome["terminated"]
                                       and not outcome["truncated"])
                probe = dict(candidate=name, pose=calibration_pose,
                    nominal_fraction=float(fraction), score=float(score),
                    achieved_angle=float(outcome["achieved"]),
                    admitted=bool(final_supported and np.isfinite(outcome["achieved"])),
                    final_supported=final_supported, strict_hold_valid=bool(outcome["valid"]),
                    numerical=bool(outcome["failure_numerical"]),
                    terminated=bool(outcome["terminated"]),
                    joint_clipped=joint_clipped)
                probes.append(probe)
            models[name] = dict(reference=reference.tolist(), axis=direction.tolist(),
                source_inverse=np.asarray(inverse).tolist(),
                source_score_bounds=np.asarray(bounds).tolist(), probes=probes,
                response=fit_probe_response(probes, lo, hi))
        selected = choose_candidate(models, lo, hi)
        for pose in old_test_poses:
            for fraction in TARGET_FRACTIONS:
                target = lo + fraction*(hi-lo)
                for arm in ARMS:
                    if arm == "common_db_ff":
                        name, (reference, direction, inverse, bounds) = "common", packages["common"]
                        raw = (fraction-inverse[0])/inverse[1]
                        score = float(np.clip(raw, *bounds))
                        inverse_clipped = bool(not np.isclose(raw, score))
                    elif arm == "local_oracle":
                        name, (reference, direction, inverse, bounds) = "local_oracle", oracle
                        raw = (fraction-inverse[0])/inverse[1]
                        score = float(np.clip(raw, *bounds))
                        inverse_clipped = bool(not np.isclose(raw, score))
                    else:
                        name = "common" if arm == "common_fewshot" else selected
                        if name is None or models[name]["response"] is None:
                            records.append(dict(task=held["name"], arm=arm, pose=int(pose),
                                target_fraction=float(fraction), target_angle=float(target),
                                available=False, selected_candidate=name))
                            continue
                        reference = np.asarray(models[name]["reference"])
                        direction = np.asarray(models[name]["axis"])
                        score, inverse_clipped = inverse_score(target, models[name]["response"])
                    outcome, joint_clipped = evaluate_command(
                        env, int(pose), key, float(target), 62000+pose,
                        category, reference, direction, score, low, high)
                    achieved = float(outcome["achieved"])
                    records.append(dict(task=held["name"], arm=arm, pose=int(pose),
                        target_fraction=float(fraction), target_angle=float(target),
                        achieved_angle=achieved, available=True,
                        normalized_abs_error=abs(achieved-target)/(hi-lo),
                        supported=bool(outcome["valid"]),
                        final_supported=bool(outcome["held"] and not outcome["failed"]),
                        numerical=bool(outcome["failure_numerical"]),
                        inverse_clipped=inverse_clipped, joint_clipped=joint_clipped,
                        selected_candidate=name, score=float(score)))
        result = dict(protocol="few_shot_axis_selection_v1", task=held["name"],
            new_training=False, known_tools=[b["name"] for b in known],
            heldout_success_db_use="local_oracle_only", target_range=[lo,hi],
            calibration_pose=calibration_pose, test_poses=old_test_poses,
            probe_fractions=list(PROBE_FRACTIONS), test_fractions=list(TARGET_FRACTIONS),
            operation_ramp_seconds=5., terminal_hold_seconds=2.,
            candidate_count=len(models), selected_candidate=selected,
            models=models, physical=records)
        output.mkdir(parents=True, exist_ok=True)
        (output/f"{held['name']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(held["name"], "selected", selected, "calibratable",
              sum(m["response"] is not None for m in models.values()),
              "test_rows", len(records), flush=True)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=Path("analysis/paper15_paper_bundle_20260928"))
    parser.add_argument("--source", type=Path,
                        default=Path("runs/paper15_unified_3m_slew_20260927"))
    parser.add_argument("--output", type=Path,
                        default=Path("analysis/few_shot_axis_adaptation_20261008"))
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--variant", type=int, choices=range(1,6))
    args = parser.parse_args()
    _register_tool_manip_gym_environments()
    for category in args.categories:
        for variant in ([args.variant] if args.variant else range(1,6)):
            evaluate_tool(category, variant, args.bundle, args.source, args.output)


if __name__ == "__main__":
    main()
