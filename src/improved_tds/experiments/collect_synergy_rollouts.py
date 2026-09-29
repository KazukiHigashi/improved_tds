"""Collect a balanced rollout set from a trained policy, for synergy analysis.

The trajectories recorded alongside training mix policies from every stage of
learning, sample goals at random, and -- because the tool-manip-gym models carry
no MuJoCo sensors -- hold no contact information at all. None of that suits the
analysis in ``docs/synergy_analysis_plan.md``, which needs a fixed policy, an
even spread over the authored initial poses and over the target angle, and a
per-digit record of what the hand is gripping.

This script rolls the final policy out deterministically over the full
``initial pose x target fraction`` grid and writes the result as a trajectory
dataset plus a summary of where the successes fell.

    python -m improved_tds.experiments.collect_synergy_rollouts \
        --env-id ShadowHandLiteSpray1-v0 \
        --model runs/.../spray_lite_1/model.zip \
        --pose-pool-dir ../tool-pose-pipeline/grasps \
        --output analysis/spray_lite_1_rollouts.npz
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

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
from improved_tds.learning.collector import SuccessfulTrajectoryCollector


def _fractions(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.replace(",", " ").split())
    if not values:
        raise argparse.ArgumentTypeError("give at least one target fraction")
    if any(not 0.0 < value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("target fractions must lie in (0, 1]")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default="sac")
    parser.add_argument(
        "--target-fractions",
        type=_fractions,
        default=(0.1, 0.2, 0.3, 0.4, 0.5),
        help="target tool angle as a fraction of the tool limit",
    )
    parser.add_argument(
        "--episodes-per-cell",
        type=int,
        default=3,
        help="repeats per (initial pose, target) cell; repeats differ because "
        "the bottle placement is randomized on reset",
    )
    parser.add_argument(
        "--bottle-xy-noise",
        type=float,
        default=None,
        help="override the object placement noise (0 fixes the object). Training "
        "randomises placement for robustness, but a randomly placed object moves "
        "the tool DoF by an amount no hand synergy could encode, which caps how "
        "well any synergy can explain the achieved tool state",
    )
    parser.add_argument("--required-grasp-contacts", type=int, default=2)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="also record unsuccessful episodes (the analysis uses successes only)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes_per_cell < 1:
        parser.error("episodes-per-cell must be positive")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output} (pass --overwrite)")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model_path = args.model
    if not model_path.exists() and model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    if not model_path.is_file():
        parser.error(f"model does not exist: {args.model}")
    model_path = model_path.resolve()
    with model_path.open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    _register_tool_manip_gym_environments()
    env_kwargs = {}
    if args.bottle_xy_noise is not None:
        env_kwargs["bottle_xy_noise"] = float(args.bottle_xy_noise)
    base_env = gym.make(args.env_id, **env_kwargs)
    pool_path = args.pose_pool_dir / pool_filename_for_environment(base_env)
    if not pool_path.is_file():
        base_env.close()
        parser.error(f"initial-pose pool not found: {pool_path}")
    pool_env = InitialPosePoolWrapper(base_env, pool_path, seed=args.seed)
    pose_count = pool_env.pose_count

    task_name = type(pool_env.unwrapped).__name__
    goal_kwargs = {
        "required_grasp_contacts": args.required_grasp_contacts,
        "success_hold_steps": args.success_hold_steps,
        "seed": args.seed,
    }
    if task_name == "ShadowHandSprayBottleEnv":
        goal_env = SprayGoalWrapper(pool_env, **goal_kwargs)
    elif task_name == "PushBottleEnv":
        goal_env = PushGoalWrapper(pool_env, **goal_kwargs)
    else:
        pool_env.close()
        parser.error(f"no goal wrapper for {task_name}")

    if task_name == "ShadowHandSprayBottleEnv":
        option_key, tool_limit = (
            "target_trigger_angle",
            float(goal_env.unwrapped.trigger_limit),
        )
    else:
        option_key, tool_limit = (
            "target_push_depth",
            float(goal_env.unwrapped.push_limit),
        )

    # One probe reset names the contact channels, so the recorded vectors can be
    # read back without the analysis having to guess their layout.
    _, probe_info = goal_env.reset(seed=args.seed)
    digit_names = [str(name) for name in probe_info.get("grasp_digit_names", ())]

    env = SuccessfulTrajectoryCollector(
        goal_env,
        output_path=args.output,
        controller_name=f"{args.algorithm}_her_deterministic",
        policy_checkpoint=str(model_path),
        simulator_parameters={
            "env_id": args.env_id,
            "policy_sha256": model_sha256,
            "pose_pool": pool_path.name,
            "pose_count": pose_count,
            "bottle_xy_noise": args.bottle_xy_noise,
            "target_fractions": list(args.target_fractions),
            "tool_limit": tool_limit,
            "episodes_per_cell": args.episodes_per_cell,
            "required_grasp_contacts": args.required_grasp_contacts,
            "success_hold_steps": args.success_hold_steps,
            "seed": args.seed,
            "contact_channels": list(SuccessfulTrajectoryCollector.CONTACT_CHANNELS),
            "contact_digits": digit_names,
        },
        save_failures=args.save_failures,
        dataset_role="synergy_analysis",
        save_every_trajectories=50,
    )
    model = algorithm_class(args.algorithm).load(model_path, env=env, device="cpu")

    pose_attempts: Counter = Counter()
    pose_successes: Counter = Counter()
    target_attempts: Counter = Counter()
    target_successes: Counter = Counter()
    episode = 0
    try:
        for pose_index in range(pose_count):
            for fraction in args.target_fractions:
                for _ in range(args.episodes_per_cell):
                    before = env.successful_trajectory_count
                    observation, _ = env.reset(
                        seed=args.seed + episode,
                        options={
                            "initial_pose_index": pose_index,
                            option_key: fraction * tool_limit,
                        },
                    )
                    done = False
                    while not done:
                        action, _ = model.predict(observation, deterministic=True)
                        observation, _, terminated, truncated, _ = env.step(action)
                        done = bool(terminated or truncated)
                    succeeded = env.successful_trajectory_count > before
                    pose_attempts[pose_index] += 1
                    target_attempts[fraction] += 1
                    if succeeded:
                        pose_successes[pose_index] += 1
                        target_successes[fraction] += 1
                    episode += 1
            print(
                f"pose {pose_index + 1}/{pose_count}: "
                f"{pose_successes[pose_index]}/{pose_attempts[pose_index]} successful",
                flush=True,
            )
    finally:
        env.close()

    summary = {
        "env_id": args.env_id,
        "dataset_role": "synergy_analysis",
        "model": str(model_path),
        "model_sha256": model_sha256,
        "pose_pool": pool_path.name,
        "pose_count": pose_count,
        "target_fractions": list(args.target_fractions),
        "tool_limit": tool_limit,
        "episodes_per_cell": args.episodes_per_cell,
        "attempted_episodes": episode,
        "successful_trajectories": env.successful_trajectory_count,
        "success_rate": (
            env.successful_trajectory_count / episode if episode else 0.0
        ),
        "per_pose_attempts": {str(k): int(v) for k, v in sorted(pose_attempts.items())},
        "per_pose_successes": {
            str(k): int(pose_successes[k]) for k in sorted(pose_attempts)
        },
        "poses_without_success": [
            int(k) for k in sorted(pose_attempts) if pose_successes[k] == 0
        ],
        "per_target_attempts": {
            f"{k:g}": int(v) for k, v in sorted(target_attempts.items())
        },
        "per_target_successes": {
            f"{k:g}": int(target_successes[k]) for k in sorted(target_attempts)
        },
        "rejected_trajectories": env.rejected_trajectories,
        "rejection_reasons": dict(env.rejection_reasons),
    }
    write_report(args.output.with_suffix(".summary.json"), summary)
    print(
        f"{args.output.name}: {summary['successful_trajectories']} successful "
        f"of {episode} episodes "
        f"({100.0 * summary['success_rate']:.1f}%), "
        f"{len(summary['poses_without_success'])} poses never succeeded",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
