from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import numpy as np

import improved_tds
from improved_tds.evaluation.reports import write_report
from improved_tds.learning.collector import SuccessfulTrajectoryCollector
from improved_tds.learning.algorithms import algorithm_class


class TrainingDiagnosticsCallback(BaseCallback):
    def __init__(self, collector: SuccessfulTrajectoryCollector):
        super().__init__(verbose=0)
        self.collector = collector
        self.action_components = 0
        self.saturated_action_components = 0

    @property
    def action_saturation_fraction(self) -> float:
        if self.action_components == 0:
            return 0.0
        return self.saturated_action_components / self.action_components

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", []), dtype=np.float64)
        if actions.size:
            self.action_components += int(actions.size)
            self.saturated_action_components += int(np.sum(np.abs(actions) > 0.95))
        self.logger.record(
            "rollout/stable_success_trajectories",
            self.collector.successful_trajectory_count,
        )
        self.logger.record(
            "rollout/action_saturation_fraction",
            self.action_saturation_fraction,
        )
        return True


def main(argv: list[str] | None = None) -> None:
    """SB3 SAC/TD3/DDPG+HER学習と同期した候補姿勢DB収集を行う。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="ExpScissor1-v0")
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default="sac")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/sac_her")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hold-initial-steps", type=int, default=30)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--dataset-save-frequency", type=int, default=100)
    args = parser.parse_args(argv)
    positive_values = {
        "total_timesteps": args.total_timesteps,
        "checkpoint_freq": args.checkpoint_freq,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "success_hold_steps": args.success_hold_steps,
        "dataset_save_frequency": args.dataset_save_frequency,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid or args.action_noise < 0.0 or args.learning_starts < 0:
        parser.error(f"invalid non-positive training values: {invalid}")

    improved_tds.register_environments()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = (
        Path(args.dataset)
        if args.dataset is not None
        else destination.with_name(f"{destination.name}_training_candidates.npz")
    )
    checkpoints_dir = (
        Path(args.checkpoints_dir)
        if args.checkpoints_dir is not None
        else destination.with_name(f"{destination.name}_checkpoints")
    )
    env_kwargs = {}
    if args.env_id.startswith(("ExpScissor", "ImprovedScissor")):
        env_kwargs = {
            "hold_initial_steps": args.hold_initial_steps,
            "success_hold_steps": args.success_hold_steps,
        }
    base_env = gym.make(args.env_id, **env_kwargs)
    env = SuccessfulTrajectoryCollector(
        base_env,
        output_path=dataset_path,
        controller_name=f"{args.algorithm}_her",
        policy_checkpoint=None,
        simulator_parameters={
            "algorithm": args.algorithm,
            "learning_rate": args.learning_rate,
            "action_noise": args.action_noise if args.algorithm != "sac" else None,
            "learning_starts": args.learning_starts,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "replay_buffer": {
                "n_sampled_goal": 4,
                "goal_selection_strategy": "future",
                "copy_info_dict": True,
            },
            "ent_coef": "auto" if args.algorithm == "sac" else None,
            "success_hold_steps": args.success_hold_steps,
            "hold_initial_steps": args.hold_initial_steps,
        },
        save_failures=args.save_failures,
        dataset_role="training_candidate",
        save_every_trajectories=args.dataset_save_frequency,
    )
    action_noise = None
    if args.algorithm in {"td3", "ddpg"}:
        action_noise = NormalActionNoise(
            np.zeros(env.action_space.shape[0]),
            args.action_noise * np.ones(env.action_space.shape[0]),
        )
    algorithm_kwargs = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "replay_buffer_class": HerReplayBuffer,
        "replay_buffer_kwargs": {
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
            "copy_info_dict": True,
        },
        "seed": args.seed,
        "device": args.device,
        "verbose": 1,
    }
    if args.algorithm == "sac":
        algorithm_kwargs.update(
            {
                "ent_coef": "auto",
                "policy_kwargs": {"net_arch": [256, 256]},
            }
        )
    else:
        algorithm_kwargs.update(
            {
                "action_noise": action_noise,
                "policy_kwargs": {"net_arch": [256, 256, 256]},
            }
        )
    model = algorithm_class(args.algorithm)(
        "MultiInputPolicy",
        env,
        **algorithm_kwargs,
    )
    checkpoint = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoints_dir),
        name_prefix=f"{args.algorithm}_her",
    )
    diagnostics = TrainingDiagnosticsCallback(env)
    model.learn(
        args.total_timesteps,
        callback=CallbackList([checkpoint, diagnostics]),
    )
    model.save(destination)
    summary = {
        "algorithm": args.algorithm,
        "env_id": args.env_id,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "successful_candidate_trajectories": env.successful_trajectory_count,
        "action_saturation_fraction": diagnostics.action_saturation_fraction,
        "dataset": str(dataset_path),
        "model": str(destination.with_suffix(".zip")),
        "checkpoints_dir": str(checkpoints_dir),
        "learning_starts": args.learning_starts,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "action_noise": args.action_noise if args.algorithm != "sac" else None,
        "ent_coef": "auto" if args.algorithm == "sac" else None,
    }
    write_report(destination.with_name(f"{destination.name}_training_summary.json"), summary)
    print(summary)
    env.close()


if __name__ == "__main__":
    main()
