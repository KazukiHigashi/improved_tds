from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import numpy as np

import improved_tds


def main(argv: list[str] | None = None) -> None:
    """Optional SB3 DDPG+HER baseline; the TDS core does not depend on SB3."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="ExpScissor1-v0")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/ddpg_her")
    args = parser.parse_args(argv)
    improved_tds.register_environments()
    env = gym.make(args.env_id)
    noise = NormalActionNoise(np.zeros(env.action_space.shape[0]), 0.2 * np.ones(env.action_space.shape[0]))
    model = DDPG(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future"},
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=10_000,
        action_noise=noise,
        policy_kwargs={"net_arch": [256, 256, 256]},
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.learn(args.total_timesteps)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    env.close()


if __name__ == "__main__":
    main()

