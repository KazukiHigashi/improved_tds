from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import yaml

import improved_tds
from improved_tds.environments.button import ButtonParameters
from improved_tds.environments.trigger import TriggerParameters
from improved_tds.learning.collector import SuccessfulTrajectoryCollector


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect trajectories from a YAML config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    environment = dict(config["environment"])
    env_id = environment.pop("id")
    environment.pop("tool_family", None)
    parameters = environment.pop("parameters", {})
    if env_id == "TDS-Trigger-v0":
        environment["parameters"] = TriggerParameters(**parameters)
    elif env_id == "TDS-Button-v0":
        environment["parameters"] = ButtonParameters(**parameters)
    else:
        environment.update(parameters)
        environment.pop("tool_instance_id", None)
    improved_tds.register_environments()
    collection = config.get("collection", {})
    output = collection.get("output", f"runs/{env_id}/trajectories.npz")
    env = SuccessfulTrajectoryCollector(gym.make(env_id, **environment), output_path=output)
    seed = int(config.get("seed", 0))
    for episode in range(args.episodes):
        env.reset(seed=seed + episode)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
    env.close()
    print(f"saved {len(env.dataset.trajectories)} successful trajectories to {output}")


if __name__ == "__main__":
    main()

