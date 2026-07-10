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

