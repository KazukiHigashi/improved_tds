from __future__ import annotations

import argparse

import gymnasium as gym

import improved_tds
from improved_tds.learning.collector import SuccessfulTrajectoryCollector


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect successful interaction trajectories")
    parser.add_argument("--env-id", default="TDS-Trigger-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-failures", action="store_true")
    args = parser.parse_args(argv)
    improved_tds.register_environments()
    env = SuccessfulTrajectoryCollector(
        gym.make(args.env_id),
        output_path=args.output,
        controller_name="random_policy",
        save_failures=args.save_failures,
    )
    for episode in range(args.episodes):
        _, _ = env.reset(seed=args.seed + episode)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
    env.close()
    print(f"saved {len(env.dataset.trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    main()

