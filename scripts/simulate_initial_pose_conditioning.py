"""Physical leave-one-tool-out FF control with an initial-pose-conditioned offset.

All shared arms use the *same* operation axis and the same DB-linear inverse.
They differ only in whether/how the authored initial joint posture determines a
fixed second-axis offset. No held-out tool's policy or successful final poses
are used in those arms. A local per-tool PLS model is a labelled reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from evaluate_initial_pose_conditioning import (CATEGORIES, fit_correction,
                                                fit_operation, pose_means,
                                                predict_correction, read_category,
                                                unit)
from improved_tds.environments.scissors_smooth import SmoothReachConfig
from improved_tds.experiments.collect_posture_database import _actuated_q, _build_env
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.reach_tds_direct import configure_scissors_evaluation_palm_radius
from improved_tds.experiments.simple_synergy import direct_trajectory
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


ARMS = ("global_no_offset", "initial_projection", "initial_pls",
        "task_specific_reference")
TARGET_FRACTIONS = (.1, .3, .5, .7, .9)


def create_env(name, bundle, source, seed):
    path = bundle / name / "database" / "success_postures.npz"
    with np.load(path, allow_pickle=False) as db:
        meta = json.loads(str(db["metadata_json"].item()))
    contract = meta.get("contract", {})
    version = meta.get("scissors_control_version", "simple")
    smooth = (SmoothReachConfig(float(contract.get("action_delta_fraction", .05)),
                               float(contract.get("request_gap_penalty_weight", 1.)))
              if version == "smooth" else None)
    env, pool, _, key, lo, hi = _build_env(
        meta["env_id"], source / "pose_pool", seed,
        research_task=True, scissors_control_version=version,
        scissors_smooth_config=smooth,
        tool_action_slew_rate=float(contract.get("tool_action_slew_rate", 0.)),
        tool_request_gap_penalty_weight=float(contract.get("tool_request_gap_penalty_weight", 1.)),
        object_position_noise=0., max_episode_steps=10000,
        force_floor_contact=name.startswith("spray"), render_mode=None)
    if name.startswith("scissors"):
        configure_scissors_evaluation_palm_radius(env, 1.5, contract.get("palm_sphere_radius"))
    return env, pool, key, float(lo), float(hi)


def fit_angle_inverse(blocks, center, operation):
    q = np.concatenate([b["q"] for b in blocks])
    y = np.concatenate([b["y"] for b in blocks])
    score = (q - center) @ operation
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(score)), score]), y,
                           rcond=None)[0]
    if coef[1] < 1e-8:
        raise ValueError("angle calibration slope must be positive")
    score_bounds = np.array([1.5 * score.min(), 1.5 * score.max()])
    return coef, score_bounds


def fit_local(block):
    q, y = block["q"], block["y"]
    center = q.mean(0)
    axis = unit(PLSRegression(n_components=1, scale=False)
                .fit(q - center, y - y.mean()).x_weights_[:, 0])
    if np.corrcoef((q - center) @ axis, y)[0, 1] < 0:
        axis = -axis
    inverse, bounds = fit_angle_inverse([block], center, axis)
    return center, axis, inverse, bounds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path,
                   default=Path("analysis/paper15_paper_bundle_20260928"))
    p.add_argument("--source", type=Path,
                   default=Path("runs/paper15_unified_3m_slew_20260927"))
    p.add_argument("--output", type=Path,
                   default=Path("analysis/initial_pose_conditioned_tds_20261007"))
    p.add_argument("--poses", type=int, default=3)
    p.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    p.add_argument("--variant", type=int, choices=range(1, 6))
    args = p.parse_args()
    if args.poses < 1:
        p.error("at least one reset pose required")
    args.output.mkdir(parents=True, exist_ok=True)
    _register_tool_manip_gym_environments()
    for category in args.categories:
        blocks = read_category(args.bundle, category)
        for variant in ([args.variant] if args.variant else range(1, 6)):
            held = blocks[variant - 1]
            train = [b for i, b in enumerate(blocks) if i != variant - 1]
            mean, op = fit_operation(train)
            grouped = pose_means(train, mean, op)
            models = {
                "initial_projection": fit_correction(grouped, mean, op, "op_projection"),
                "initial_pls": fit_correction(grouped, mean, op, "full_initial"),
            }
            inverse, bounds = fit_angle_inverse(train, mean, op)
            local_mean, local_op, local_inverse, local_bounds = fit_local(held)
            env, pool, key, lo, hi = create_env(held["name"], args.bundle,
                                                  args.source, 48200 + variant)
            low, high, _ = _control_bounds(env)
            rng = np.random.default_rng(59320 + variant)
            poses = rng.choice(pool.pose_count, size=args.poses, replace=False).tolist()
            results = []
            try:
                for pose in poses:
                    for fraction in TARGET_FRACTIONS:
                        target = lo + fraction * (hi - lo)
                        seed = 62000 + pose
                        env.reset(seed=seed,
                                  options={"initial_pose_index": int(pose), key: float(target)})
                        initial_q = _actuated_q(env)
                        for arm in ARMS:
                            correction_raw = correction = 0.
                            correction_clipped = False
                            if arm == "task_specific_reference":
                                ref, axis, coefficients, score_bounds = (
                                    local_mean, local_op, local_inverse, local_bounds)
                                second = np.zeros_like(op)
                            else:
                                ref, axis, coefficients, score_bounds = mean, op, inverse, bounds
                                if arm != "global_no_offset":
                                    model = models[arm]
                                    correction_raw = float(predict_correction(model, initial_q))
                                    correction = float(np.clip(correction_raw, *model["g_limits"]))
                                    correction_clipped = not np.isclose(correction, correction_raw)
                                    second = model["correction_axis"]
                                else:
                                    second = np.zeros_like(op)
                            raw_score = (fraction - coefficients[0]) / coefficients[1]
                            score = float(np.clip(raw_score, *score_bounds))
                            requested = ref + score * axis + correction * second
                            goal = np.clip(requested, low, high)
                            outcome = direct_trajectory(
                                env, int(pose), key, target, seed, [goal],
                                support_criterion=("task" if category == "scissors" else "no_fall"),
                                require_floor_contact=category == "spray")["outcomes"][0]
                            achieved = float(outcome["achieved"])
                            results.append(dict(task=held["name"], arm=arm, pose=int(pose),
                                                target_fraction=fraction, target_angle=target,
                                                achieved_angle=achieved,
                                                normalized_abs_error=abs(achieved - target) / (hi - lo),
                                                supported=bool(outcome["valid"]),
                                                numerical=bool(outcome["failure_numerical"]),
                                                score=float(score), inverse_clipped=bool(
                                                    not np.isclose(score, raw_score)),
                                                correction=float(correction),
                                                raw_correction=float(correction_raw),
                                                correction_clipped=correction_clipped,
                                                joint_clipped=bool(np.max(np.abs(goal - requested)) > 1e-8)))
                    (args.output / f"{held['name']}_physical.json").write_text(
                        json.dumps(results, indent=2), encoding="utf-8")
                for arm in ARMS:
                    selected = [r for r in results if r["arm"] == arm]
                    print(held["name"], arm, "MAE", round(float(np.mean(
                        [r["normalized_abs_error"] for r in selected])), 4),
                          "held", sum(r["supported"] for r in selected),
                          "jointclip", sum(r["joint_clipped"] for r in selected), flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()
