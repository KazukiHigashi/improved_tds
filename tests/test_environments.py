from __future__ import annotations

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import improved_tds


@pytest.mark.simulation
@pytest.mark.parametrize(
    "env_id",
    [
        "ExpScissor1-v0",
        "ExpScissor2-v0",
        "ExpScissor3-v0",
        "ExpScissor4-v0",
        "ExpScissor5-v0",
        "TDS-Trigger-v0",
        "TDS-Button-v0",
    ],
)
def test_environment_reset_step_and_articulated_api(env_id) -> None:
    improved_tds.register_environments()
    env = gym.make(env_id)
    obs, _ = env.reset(seed=5)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    result = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert len(result) == 5
    unwrapped = env.unwrapped
    assert unwrapped.get_tool_state().shape == (1,)
    assert unwrapped.get_tool_state_rate().shape == (1,)
    assert "joint_torque" in unwrapped.get_force_observation()
    check_env(unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.simulation
def test_scissors_returned_observation_matches_internal_state() -> None:
    improved_tds.register_environments()
    env = gym.make("ExpScissor1-v0")
    env.reset(seed=8)
    returned, _, _, _, _ = env.step(
        np.full(env.action_space.shape, 0.7, dtype=np.float32)
    )
    internal = env.unwrapped._get_obs()
    for key in returned:
        np.testing.assert_allclose(returned[key], internal[key], atol=1e-7)
    env.close()


@pytest.mark.simulation
def test_scissors_stable_success_is_observable_but_not_her_terminal() -> None:
    improved_tds.register_environments()
    env = gym.make(
        "ExpScissor1-v0",
        angle_threshold=2.0,
        success_hold_steps=3,
        max_episode_steps=4,
    )
    env.reset(seed=9, options={"target": 0.0})
    infos = []
    results = []
    for _ in range(4):
        result = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        results.append(result)
        infos.append(result[-1])
    assert [info["success_streak"] for info in infos] == [1, 2, 3, 4]
    assert [info["is_stable_success"] for info in infos] == [0.0, 0.0, 1.0, 1.0]
    assert all(not result[2] for result in results)
    assert results[-1][3]
    env.close()


@pytest.mark.simulation
def test_scissors_reset_restores_complete_dynamic_state() -> None:
    improved_tds.register_environments()
    env = gym.make("ExpScissor1-v0", hold_initial_steps=2)
    env.reset(seed=11, options={"target": 0.3})
    initial = env.unwrapped.data
    expected = {
        "qpos": initial.qpos.copy(),
        "qvel": initial.qvel.copy(),
        "ctrl": initial.ctrl.copy(),
        "time": float(initial.time),
    }
    for _ in range(3):
        env.step(np.full(env.action_space.shape, 0.8, dtype=np.float32))
    env.reset(seed=11, options={"target": 0.3})
    np.testing.assert_allclose(env.unwrapped.data.qpos, expected["qpos"], atol=1e-10)
    np.testing.assert_allclose(env.unwrapped.data.qvel, expected["qvel"], atol=1e-10)
    np.testing.assert_allclose(env.unwrapped.data.ctrl, expected["ctrl"], atol=1e-10)
    np.testing.assert_allclose(env.unwrapped.data.qvel, 0.0, atol=1e-12)
    assert float(env.unwrapped.data.time) == pytest.approx(expected["time"])
    env.close()


@pytest.mark.simulation
def test_scissors_her_reward_preserves_grasp_condition() -> None:
    improved_tds.register_environments()
    env = gym.make("ExpScissor1-v0")
    unwrapped = env.unwrapped
    achieved = np.array([[0.2]], dtype=np.float32)
    desired = np.array([[0.2]], dtype=np.float32)
    in_grasp = unwrapped.compute_reward(
        achieved, desired, [{"is_in_grasp_space": 1.0}]
    )
    out_of_grasp = unwrapped.compute_reward(
        achieved, desired, [{"is_in_grasp_space": 0.0}]
    )
    np.testing.assert_array_equal(in_grasp, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(out_of_grasp, np.array([-1.0], dtype=np.float32))
    env.close()
