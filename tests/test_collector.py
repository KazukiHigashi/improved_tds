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


def test_recording_failure_warns_and_training_continues():
    """A malformed episode must cost that episode, not the training run.

    One non-monotone trajectory used to raise out of `step`, which killed a job
    835k steps in. The episode is dropped with a warning and counted instead.
    """
    import warnings

    import gymnasium as gym
    import numpy as np

    from improved_tds.learning.collector import SuccessfulTrajectoryCollector

    class _Clock(gym.Env):
        """Minimal env whose clock can be wound backwards on demand."""

        observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                "achieved_goal": gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
                "desired_goal": gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
            }
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        dt = 0.02

        def __init__(self):
            self.data = type("D", (), {"time": 0.0, "qpos": np.zeros(2), "qvel": np.zeros(2),
                                       "actuator_force": np.zeros(2), "sensordata": np.zeros(1)})()
            self._t = 0

        def _obs(self):
            return {
                "observation": np.zeros(2, dtype=np.float32),
                "achieved_goal": np.zeros(1, dtype=np.float32),
                "desired_goal": np.zeros(1, dtype=np.float32),
            }

        def reset(self, *, seed=None, options=None):
            self.data.time, self._t = 0.0, 0
            return self._obs(), {}

        def step(self, action):
            self._t += 1
            self.data.time = 0.0 if self._t == 3 else self._t * self.dt  # clock restarts
            done = self._t >= 5
            return self._obs(), 0.0, done, False, {"is_success": 1.0, "is_stable_success": 1.0}

    collector = SuccessfulTrajectoryCollector(_Clock(), save_failures=True)
    collector.reset(seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            collector.step(collector.action_space.sample())

    assert collector.rejected_trajectories == 1
    assert "monotone" in next(iter(collector.rejection_reasons))
    assert any("not recorded" in str(w.message) for w in caught)
    assert collector.dataset.trajectories == []   # the bad episode was not kept


def test_reported_contacts_and_pose_index_are_recorded():
    """Grasp and driven-part contacts land in the record as separate blocks.

    The tool-manip-gym models define no MuJoCo sensors, so without this the
    contact channels would be recorded empty and the synergy analysis would have
    nothing to witness the grasp with.
    """
    import gymnasium as gym
    import numpy as np

    from improved_tds.learning.collector import SuccessfulTrajectoryCollector

    class _Contacting(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                "achieved_goal": gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
                "desired_goal": gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
            }
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        dt = 0.02

        def __init__(self):
            self.data = type("D", (), {"time": 0.0, "qpos": np.zeros(2), "qvel": np.zeros(2),
                                       "actuator_force": np.zeros(2), "sensordata": np.zeros(0)})()
            self._t = 0

        def _obs(self):
            return {
                "observation": np.zeros(2, dtype=np.float32),
                "achieved_goal": np.zeros(1, dtype=np.float32),
                "desired_goal": np.zeros(1, dtype=np.float32),
            }

        def reset(self, *, seed=None, options=None):
            self.data.time, self._t = 0.0, 0
            return self._obs(), {"initial_pose_index": 7, "initial_pose_pool": "pool.npz"}

        def step(self, action):
            self._t += 1
            self.data.time = self._t * self.dt
            info = {
                "is_success": 1.0,
                "is_stable_success": 1.0,
                "grasp_digit_contacts": np.array([True, False, False, True, False]),
                "grasp_digit_normal_forces": np.array([2.0, 0.0, 0.0, 3.0, 0.0]),
                "trigger_digit_contacts": np.array([False, True, False, False, False]),
                "trigger_digit_normal_forces": np.array([0.0, 1.5, 0.0, 0.0, 0.0]),
            }
            return self._obs(), 0.0, self._t >= 3, False, info

    collector = SuccessfulTrajectoryCollector(_Contacting())
    collector.reset(seed=0)
    for _ in range(3):
        collector.step(collector.action_space.sample())

    trajectory = collector.dataset.trajectories[0]
    step = trajectory.steps[0]
    assert step.contact_flags.shape == (10,)
    assert step.contact_flags[:5].sum() == 2      # bottle is held by two digits
    assert step.contact_flags[5:].sum() == 1      # one digit works the trigger
    assert step.contact_forces[8] == 0.0
    assert step.contact_forces[6] == 1.5
    assert trajectory.metadata.simulator_parameters["initial_pose_index"] == 7
    assert trajectory.metadata.simulator_parameters["initial_pose_pool"] == "pool.npz"
