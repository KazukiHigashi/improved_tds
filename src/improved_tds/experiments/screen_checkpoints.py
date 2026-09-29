"""Pick the policy to analyse by screening every checkpoint on one fixed grid.

SAC can collapse late in training, and what gets saved as the final model is
whatever it collapsed into: in the v5 spray run, spray2 tracked its target to
6 mrad at 700k steps and by 1M ignored the target entirely, driving the trigger
to a fixed 0.24 rad every time. The training log cannot be used to catch this,
because its rolling success rate is over whatever pose and target the sampler
happened to draw.

Every checkpoint is therefore replayed over the same grid of authored poses and
target fractions with the deterministic policy, and the winner is chosen by a
rule fixed before the numbers are seen:

1. most successful episodes;
2. among ties, the smallest median tool-state tracking error -- success is a
   binary test against a 0.03 rad tolerance, so equally successful policies
   still differ in how precisely they track, and a synergy analysis reads the
   joint-to-tool mapping straight off that;
3. among ties still, the most trained.

    python -m improved_tds.experiments.screen_checkpoints \
        --run-dir runs/spray_lite_1m_graspfix_v5_20260816 \
        --pose-pool-dir ../tool-pose-pipeline/grasps_v3 \
        --output analysis/checkpoint_screen.json
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from improved_tds.environments.pose_pool import (
    InitialPosePoolWrapper,
    PushGoalWrapper,
    SprayGoalWrapper,
    pool_filename_for_environment,
)
from improved_tds.evaluation.reports import write_report
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.learning.algorithms import algorithm_class

_STEPS = re.compile(r"_(\d+)_steps$")


def _fractions(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.replace(",", " ").split())
    if not values:
        raise argparse.ArgumentTypeError("give at least one target fraction")
    if any(not 0.0 < value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("target fractions must lie in (0, 1]")
    return values


def candidate_policies(task_dir: Path) -> list[tuple[int, Path]]:
    """Every saved policy for a variant, as ``(steps, path)`` oldest first.

    The final ``model.zip`` is included as its own candidate; it is not assumed
    to be the same as the last checkpoint, and it is not assumed to be the best.
    """
    candidates: list[tuple[int, Path]] = []
    for checkpoint in (task_dir / "checkpoints").glob("*.zip"):
        match = _STEPS.search(checkpoint.stem)
        if match is not None:
            candidates.append((int(match.group(1)), checkpoint))
    final = task_dir / "model.zip"
    if final.is_file():
        trained = max((steps for steps, _ in candidates), default=0)
        candidates.append((trained, final))
    return sorted(candidates, key=lambda item: (item[0], item[1].name))


def score_policy(goal_env, policy, poses, fractions, limit, option_key) -> dict[str, Any]:
    successes = 0
    errors: list[float] = []
    for pose in poses:
        for fraction in fractions:
            target = fraction * limit
            observation, _ = goal_env.reset(
                seed=pose,
                options={"initial_pose_index": int(pose), option_key: target},
            )
            succeeded, achieved = False, 0.0
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = goal_env.step(action)
                succeeded = succeeded or bool(info["is_stable_success"])
                achieved = float(observation["achieved_goal"].reshape(-1)[0])
                if terminated or truncated:
                    break
            successes += int(succeeded)
            errors.append(abs(achieved - target))
    return {
        "episodes": len(poses) * len(fractions),
        "successes": successes,
        "median_abs_error": float(np.median(errors)),
        "mean_abs_error": float(np.mean(errors)),
    }


def screen_variant(
    env_id: str,
    task_dir: Path,
    pose_pool_dir: Path,
    fractions: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    base_env = gym.make(env_id)
    pool_path = pose_pool_dir / pool_filename_for_environment(base_env)
    if not pool_path.is_file():
        base_env.close()
        raise FileNotFoundError(f"initial-pose pool not found: {pool_path}")
    pool_env = InitialPosePoolWrapper(base_env, pool_path, seed=seed)
    task_name = type(pool_env.unwrapped).__name__
    if task_name == "ShadowHandSprayBottleEnv":
        goal_env = SprayGoalWrapper(pool_env, seed=seed)
        option_key = "target_trigger_angle"
        limit = float(goal_env.unwrapped.trigger_limit)
    elif task_name == "PushBottleEnv":
        goal_env = PushGoalWrapper(pool_env, seed=seed)
        option_key = "target_push_depth"
        limit = float(goal_env.unwrapped.push_limit)
    else:
        pool_env.close()
        raise TypeError(f"no goal wrapper for {task_name}")

    poses = list(range(pool_env.pose_count))
    results: list[dict[str, Any]] = []
    try:
        for steps, path in candidate_policies(task_dir):
            policy = algorithm_class("sac").load(path, env=goal_env, device="cpu")
            score = score_policy(goal_env, policy, poses, fractions, limit, option_key)
            score.update({"steps": steps, "policy": path.name, "path": str(path)})
            results.append(score)
            print(
                f"  {path.name:<28} {score['successes']:3d}/{score['episodes']} "
                f"median abs err {score['median_abs_error']:.4f}",
                flush=True,
            )
    finally:
        goal_env.close()

    # The rule, fixed before the numbers were seen: most successes, then the
    # tightest tracking, then the most trained.
    best = max(
        results,
        key=lambda item: (item["successes"], -item["median_abs_error"], item["steps"]),
    )
    return {
        "env_id": env_id,
        "pose_pool": pool_path.name,
        "poses": len(poses),
        "target_fractions": list(fractions),
        "tool_limit": limit,
        "candidates": results,
        "selected": best,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--env-id-template",
        default="ShadowHandLiteSpray{variant}-v0",
        help="env id per variant, with a {variant} placeholder",
    )
    parser.add_argument(
        "--task-dir-template",
        default="spray_lite_{variant}",
        help="run subdirectory per variant, with a {variant} placeholder",
    )
    parser.add_argument("--variants", default="1 2 3 4 5")
    parser.add_argument("--target-fractions", type=_fractions, default=(0.1, 0.3, 0.5))
    parser.add_argument("--seed", type=int, default=11000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _register_tool_manip_gym_environments()
    variants = [part for part in args.variants.replace(",", " ").split() if part]

    report: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "pose_pool_dir": str(args.pose_pool_dir),
        "target_fractions": list(args.target_fractions),
        "selection_rule": (
            "most successes; ties broken by smallest median absolute tracking "
            "error, then by most training steps"
        ),
        "variants": {},
    }
    for variant in variants:
        task_dir = args.run_dir / args.task_dir_template.format(variant=variant)
        if not task_dir.is_dir():
            print(f"=== {variant}: no run directory, skipping ===", flush=True)
            continue
        print(f"=== variant {variant} ===", flush=True)
        result = screen_variant(
            args.env_id_template.format(variant=variant),
            task_dir,
            args.pose_pool_dir,
            args.target_fractions,
            args.seed,
        )
        report["variants"][variant] = result
        selected = result["selected"]
        print(
            f"  -> selected {selected['policy']} ({selected['steps']} steps): "
            f"{selected['successes']}/{selected['episodes']}, "
            f"median abs err {selected['median_abs_error']:.4f}",
            flush=True,
        )
    write_report(args.output, report)
    print(f"\nwrote {args.output}", flush=True)
    for variant, result in report["variants"].items():
        print(f"  variant {variant}: {result['selected']['path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
