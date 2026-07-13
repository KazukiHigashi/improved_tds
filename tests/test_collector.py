from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import improved_tds
from improved_tds.data import TrajectoryDataset
from improved_tds.learning import SuccessfulTrajectoryCollector


@pytest.mark.simulation
def test_successful_full_trajectory_is_saved(tmp_path) -> None:
    improved_tds.register_environments()
    path = tmp_path / "button_trajectory.npz"
    env = SuccessfulTrajectoryCollector(gym.make("TDS-Button-v0"), output_path=path)
    env.reset(seed=2, options={"target": 0.005})
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(np.array([0.03], dtype=np.float32))
        if terminated or truncated:
            break
    env.close()
    dataset = TrajectoryDataset.load(path)
    assert len(dataset.trajectories) == 1
    assert len(dataset.trajectories[0].steps) > 1
    assert dataset.trajectories[0].metadata.success
    assert dataset.trajectories[0].steps[-1].terminated


@pytest.mark.simulation
def test_scissors_collector_commits_only_stable_success_with_goal(tmp_path) -> None:
    improved_tds.register_environments()
    success_path = tmp_path / "scissors_success.npz"
    env = SuccessfulTrajectoryCollector(
        gym.make(
            "ExpScissor1-v0",
            angle_threshold=2.0,
            success_hold_steps=2,
            max_episode_steps=3,
        ),
        output_path=success_path,
    )
    env.reset(seed=3, options={"target": 0.0})
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        if terminated or truncated:
            break
    env.close()
    dataset = TrajectoryDataset.load(success_path)
    trajectory = dataset.trajectories[0]
    assert trajectory.metadata.success
    assert trajectory.metadata.termination_reason == "time_limit"
    assert trajectory.metadata.stable_success_steps == 2
    assert trajectory.steps[-1].truncated
    assert trajectory.steps[-1].desired_tool_state is not None
    assert any(step.is_stable_success for step in trajectory.steps)

    failure_path = tmp_path / "scissors_failure.npz"
    env = SuccessfulTrajectoryCollector(
        gym.make(
            "ExpScissor1-v0",
            angle_threshold=2.0,
            success_hold_steps=5,
            max_episode_steps=3,
        ),
        output_path=failure_path,
    )
    env.reset(seed=4, options={"target": 0.0})
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        if terminated or truncated:
            break
    env.close()
    assert not failure_path.exists()
