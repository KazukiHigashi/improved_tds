"""One-DoF alternative: select a known tool's operation axis by initial pose.

For each held-out tool reset, select the nearest authored initial posture from
the other four tools. Reuse that source tool's PLS operation axis, success-pose
mean and DB-linear inverse. This uses physical initial joint coordinates, not
an ordinal or one-hot task ID, and requires no second controlled axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_initial_pose_conditioning import CATEGORIES, read_category
from simulate_initial_pose_conditioning import (TARGET_FRACTIONS, create_env,
                                                fit_local)
from improved_tds.experiments.collect_posture_database import _actuated_q
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.simple_synergy import direct_trajectory
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def closest_source(initial_q, source_blocks):
    distances = []
    for source in source_blocks:
        unique = np.stack([source["q0"][np.flatnonzero(source["pose"] == pose)[0]]
                           for pose in np.unique(source["pose"])])
        distances.append(float(np.min(np.linalg.norm(unique - initial_q, axis=1))))
    index = int(np.argmin(distances))
    return index, distances[index], distances


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
            target_block = blocks[variant - 1]
            sources = [b for i, b in enumerate(blocks) if i != variant - 1]
            local_models = [fit_local(b) for b in sources]
            env, pool, key, lo, hi = create_env(target_block["name"], args.bundle,
                                                  args.source, 48200 + variant)
            low, high, _ = _control_bounds(env)
            poses = np.random.default_rng(59320 + variant).choice(
                pool.pose_count, size=args.poses, replace=False).tolist()
            rows = []
            try:
                for pose in poses:
                    for fraction in TARGET_FRACTIONS:
                        target = lo + fraction * (hi - lo)
                        seed = 62000 + pose
                        env.reset(seed=seed,
                                  options={"initial_pose_index": int(pose), key:float(target)})
                        initial = _actuated_q(env)
                        selected, distance, source_distances = closest_source(initial, sources)
                        center, axis, inverse, bounds = local_models[selected]
                        raw_score = (fraction - inverse[0]) / inverse[1]
                        score = float(np.clip(raw_score, *bounds))
                        raw_goal = center + score * axis
                        goal = np.clip(raw_goal, low, high)
                        outcome = direct_trajectory(
                            env, int(pose), key, float(target), seed, [goal],
                            support_criterion=("task" if category == "scissors" else "no_fall"),
                            require_floor_contact=category == "spray")["outcomes"][0]
                        achieved = float(outcome["achieved"])
                        rows.append(dict(task=target_block["name"], arm="initial_nearest_axis",
                                         pose=int(pose), target_fraction=fraction,
                                         target_angle=target, achieved_angle=achieved,
                                         normalized_abs_error=abs(achieved - target) / (hi - lo),
                                         supported=bool(outcome["valid"]),
                                         numerical=bool(outcome["failure_numerical"]),
                                         score=score, inverse_clipped=bool(
                                             not np.isclose(score, raw_score)),
                                         correction=0., raw_correction=0.,
                                         correction_clipped=False,
                                         joint_clipped=bool(np.max(np.abs(goal - raw_goal)) > 1e-8),
                                         selected_source=sources[selected]["name"],
                                         nearest_initial_distance=distance,
                                         source_distances=source_distances))
                    (args.output / f"{target_block['name']}_nearest.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf-8")
                print(target_block["name"], "nearest-axis MAE",
                      round(float(np.mean([r["normalized_abs_error"] for r in rows])),4),
                      "selected", sorted({r["selected_source"] for r in rows}), flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()
