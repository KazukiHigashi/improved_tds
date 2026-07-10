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

