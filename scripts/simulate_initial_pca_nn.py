"""Leave-one-tool-out MuJoCo test of PCA2 initial-pose nearest-axis transfer.

All zero-shot conditions fit only the other four tool variants. The target
tool's successful postures enter only the clearly labelled local reference
and retrospective axis-cosine diagnostic. Every arm uses identical reset
poses, target angles, five-second ramps, and two-second holds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_initial_pca_nn import (fit_pca, select_nearest, unique_initials)
from evaluate_initial_pose_conditioning import CATEGORIES, fit_operation, read_category
from simulate_initial_pose_conditioning import (TARGET_FRACTIONS, create_env,
                                                fit_angle_inverse, fit_local)
from improved_tds.experiments.collect_posture_database import _actuated_q
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.simple_synergy import direct_trajectory
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


ARMS = ("common_axis", "raw13_nn_package", "pca2_nn_package",
        "pca2_selected_axis_only", "task_specific_reference")


def pooled_inverse_for_selected_axis(blocks, center, axis):
    q = np.concatenate([b["q"] for b in blocks])
    y = np.concatenate([b["y"] for b in blocks])
    score = (q - center) @ axis
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(score)), score]),
                                    y, rcond=None)[0]
    if abs(coefficients[1]) < 1e-8:
        raise ValueError("selected direction has no pooled angle covariance")
    return coefficients, np.array([1.5 * score.min(), 1.5 * score.max()])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path,
                   default=Path("analysis/paper15_paper_bundle_20260928"))
    p.add_argument("--source", type=Path,
                   default=Path("runs/paper15_unified_3m_slew_20260927"))
    p.add_argument("--output", type=Path,
                   default=Path("analysis/initial_pca2_nn_20261008"))
    p.add_argument("--poses", type=int, default=3)
    p.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    p.add_argument("--variant", type=int, choices=range(1, 6))
    args = p.parse_args()
    if args.poses < 1:
        p.error("at least one authored reset pose required")
    args.output.mkdir(parents=True, exist_ok=True)
    _register_tool_manip_gym_environments()
    for category in args.categories:
        blocks = read_category(args.bundle, category)
        for variant in ([args.variant] if args.variant else range(1, 6)):
            target_block = blocks[variant - 1]
            sources = [b for i, b in enumerate(blocks) if i != variant - 1]
            source_models = {b["name"]:fit_local(b) for b in sources}
            local_model = fit_local(target_block)
            train_initial, train_labels, _ = unique_initials(sources)
            pca = fit_pca(train_initial)
            mean, op = fit_operation(sources)
            common_inverse, common_bounds = fit_angle_inverse(sources, mean, op)
            selected_axis_models = {name:pooled_inverse_for_selected_axis(
                sources, mean, model[1]) for name,model in source_models.items()}
            env, pool, key, lo, hi = create_env(target_block["name"], args.bundle,
                                                  args.source, 48200 + variant)
            low, high, _ = _control_bounds(env)
            poses = np.random.default_rng(59320 + variant).choice(
                pool.pose_count, size=args.poses, replace=False).tolist()
            records = []
            try:
                for pose in poses:
                    for fraction in TARGET_FRACTIONS:
                        target = lo + fraction * (hi - lo)
                        seed = 62000 + pose
                        env.reset(seed=seed, options={"initial_pose_index":int(pose),
                                                           key:float(target)})
                        initial = _actuated_q(env)
                        raw_idx, raw_distance = select_nearest(
                            initial[None], train_initial, pca, "raw13")
                        two_idx, two_distance = select_nearest(
                            initial[None], train_initial, pca, "pca2")
                        raw_source = str(train_labels[raw_idx[0]])
                        two_source = str(train_labels[two_idx[0]])
                        arms = {
                            "common_axis":(mean, op, common_inverse, common_bounds, None),
                            "raw13_nn_package":(*source_models[raw_source],raw_source),
                            "pca2_nn_package":(*source_models[two_source],two_source),
                            "pca2_selected_axis_only":(
                                mean, source_models[two_source][1],
                                *selected_axis_models[two_source], two_source),
                            "task_specific_reference":(*local_model,target_block["name"]),
                        }
                        for arm in ARMS:
                            ref, axis, inverse, bounds, selected_source = arms[arm]
                            raw_score = (fraction - inverse[0]) / inverse[1]
                            score = float(np.clip(raw_score, *bounds))
                            requested = ref + score * axis
                            goal = np.clip(requested, low, high)
                            outcome = direct_trajectory(
                                env, int(pose), key, float(target), seed, [goal],
                                support_criterion=("task" if category == "scissors" else "no_fall"),
                                require_floor_contact=category == "spray")["outcomes"][0]
                            achieved = float(outcome["achieved"])
                            records.append(dict(task=target_block["name"],arm=arm,
                                pose=int(pose),target_fraction=fraction,
                                target_angle=float(target),achieved_angle=achieved,
                                normalized_abs_error=abs(achieved-target)/(hi-lo),
                                supported=bool(outcome["valid"]),
                                numerical=bool(outcome["failure_numerical"]),
                                inverse_clipped=bool(not np.isclose(score,raw_score)),
                                joint_clipped=bool(np.max(np.abs(goal-requested))>1e-8),
                                score=score,selected_source=selected_source,
                                selected_axis_cosine_to_target=float(axis@local_model[1]),
                                raw13_nearest_source=raw_source,
                                pca2_nearest_source=two_source,
                                raw13_nearest_distance=float(raw_distance[0]),
                                pca2_nearest_distance=float(two_distance[0]),
                                pca2_variance_fraction=float(np.sum(pca["variance"][:2]))))
                    (args.output/f"{target_block['name']}_physical.json").write_text(
                        json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
                for arm in ARMS:
                    chosen = [r for r in records if r["arm"]==arm]
                    print(target_block["name"],arm,
                          "MAE",round(float(np.mean([r["normalized_abs_error"]
                                                  for r in chosen])),4),
                          "held",sum(r["supported"] for r in chosen),
                          "jointclip",sum(r["joint_clipped"] for r in chosen),flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()
