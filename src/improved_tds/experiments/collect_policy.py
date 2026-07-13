from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np

import improved_tds
from improved_tds.evaluation.reports import write_report
from improved_tds.learning.collector import SuccessfulTrajectoryCollector
from improved_tds.learning.algorithms import algorithm_class


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect a balanced formal trajectory database from a fixed policy"
    )
    parser.add_argument("--env-id", default="ExpScissor1-v0")
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default="sac")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-min", type=float, default=0.05)
    parser.add_argument("--target-max", type=float, default=0.75)
    parser.add_argument("--target-bins", type=int, default=8)
    parser.add_argument("--successes-per-target", type=int, default=20)
    parser.add_argument("--max-attempts-per-target", type=int, default=200)
    parser.add_argument(
        "--split", choices=("fit", "calibration", "test"), required=True
    )
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--hold-initial-steps", type=int, default=30)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    args = parser.parse_args(argv)
    if args.target_min >= args.target_max:
        parser.error("target-min must be smaller than target-max")
    if args.target_bins < 2 or args.successes_per_target < 1:
        parser.error("target-bins must be at least 2 and successes-per-target must be positive")
    if args.max_attempts_per_target < args.successes_per_target:
        parser.error("max-attempts-per-target must be at least successes-per-target")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    model_path = args.model
    if not model_path.exists() and model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    if not model_path.is_file():
        parser.error(f"model does not exist: {args.model}")
    model_path = model_path.resolve()
    with model_path.open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    improved_tds.register_environments()
    base_env = gym.make(
        args.env_id,
        hold_initial_steps=args.hold_initial_steps,
        success_hold_steps=args.success_hold_steps,
    )
    env = SuccessfulTrajectoryCollector(
        base_env,
        output_path=args.output,
        controller_name=f"{args.algorithm}_her_deterministic",
        policy_checkpoint=str(model_path),
        simulator_parameters={
            "target_min": args.target_min,
            "target_max": args.target_max,
            "target_bins": args.target_bins,
            "successes_per_target": args.successes_per_target,
            "max_attempts_per_target": args.max_attempts_per_target,
            "split": args.split,
            "policy_sha256": model_sha256,
            "success_hold_steps": args.success_hold_steps,
            "hold_initial_steps": args.hold_initial_steps,
        },
        dataset_role=f"formal_balanced_{args.split}",
        save_every_trajectories=args.successes_per_target,
    )
    model = algorithm_class(args.algorithm).load(model_path, env=env, device="cpu")
    bin_edges = np.linspace(args.target_min, args.target_max, args.target_bins + 1)
    attempts = np.zeros(args.target_bins, dtype=np.int64)
    success_counts = np.zeros(args.target_bins, dtype=np.int64)
    split_seed_offset = {"fit": 0, "calibration": 1_000_000, "test": 2_000_000}[
        args.split
    ]
    rng = np.random.default_rng(args.seed + split_seed_offset)
    try:
        episode = 0
        for bin_index, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            while (
                success_counts[bin_index] < args.successes_per_target
                and attempts[bin_index] < args.max_attempts_per_target
            ):
                target = float(rng.uniform(low, high))
                before = env.successful_trajectory_count
                attempts[bin_index] += 1
                observation, _ = env.reset(
                    seed=args.seed + split_seed_offset + episode,
                    options={"target": float(target)},
                )
                done = False
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                if env.successful_trajectory_count > before:
                    success_counts[bin_index] += 1
                episode += 1
    finally:
        env.close()

    quota_met = bool(np.all(success_counts >= args.successes_per_target))
    bin_labels = [
        f"[{low:.8g}, {high:.8g}{']' if index == args.target_bins - 1 else ')'}"
        for index, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:]))
    ]
    summary = {
        "dataset_role": f"formal_balanced_{args.split}",
        "split": args.split,
        "model": str(model_path),
        "model_sha256": model_sha256,
        "algorithm": args.algorithm,
        "attempted_episodes": int(np.sum(attempts)),
        "successful_trajectories": env.successful_trajectory_count,
        "successes_per_target_required": args.successes_per_target,
        "max_attempts_per_target": args.max_attempts_per_target,
        "target_attempts": {
            label: int(value) for label, value in zip(bin_labels, attempts)
        },
        "target_successes": {
            label: int(value) for label, value in zip(bin_labels, success_counts)
        },
        "covered_target_bins": int(
            np.sum(success_counts >= args.successes_per_target)
        ),
        "total_target_bins": args.target_bins,
        "quota_met": quota_met,
    }
    write_report(args.output.with_suffix(".summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
