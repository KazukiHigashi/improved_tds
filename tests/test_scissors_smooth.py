import gymnasium as gym
import numpy as np
import pytest

from improved_tds.environments.control_refinement import ControlRefinement
from improved_tds.environments.scissors_smooth import ScissorsSmoothReachWrapper
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


@pytest.mark.parametrize('variant', [1, 2, 3, 4, 5])
def test_smooth_setpoint_bound_and_reward(variant):
    _register_tool_manip_gym_environments()
    env = ControlRefinement(ScissorsSmoothReachWrapper(
        gym.make(f'ScissorsVariant{variant}-v0', hold_initial_steps=0)))
    try:
        obs, _ = env.reset(seed=31)
        assert env.observation_space.contains(obs)
        for value in (1., -1., 1.):
            action = np.full(env.action_space.shape, value, dtype=np.float32)
            obs, reward, _, _, info = env.step(action)
            assert info['applied_action_delta_max'] <= .050001
            assert np.max(np.abs(info['applied_action'] - env.previous_action)) == 0
            assert np.max(np.abs(env.unwrapped.data.ctrl)) <= np.max(np.abs(env.unwrapped.model.actuator_ctrlrange))
            assert reward == pytest.approx(float(env.compute_reward(
                obs['achieved_goal'], obs['desired_goal'], info)[0]))
            assert env.observation_space.contains(obs)
            if info.get('numerical_failure'):
                break
    finally:
        env.close()


def test_contacts_are_diagnostic_not_a_grasp_or_success_gate(monkeypatch):
    _register_tool_manip_gym_environments()
    env = ScissorsSmoothReachWrapper(gym.make('ScissorsVariant4-v0', hold_initial_steps=0))
    try:
        env.reset(seed=7)
        env._contacts[:2] = 1  # Parent would have counted two fingers.
        monkeypatch.setattr(env, '_handle_fingers', lambda: (set(), set()))
        info = env._success_info({'angle_error': 0.}, env._start_pos, env._start_rotation)
        assert info['is_on_palm'] and info['is_success'] and info['grasp_maintained']
        assert not info['grasp_supports_both_handles']
        monkeypatch.setattr(env, '_handle_fingers', lambda: ({0, 1}, set()))
        info = env._success_info({'angle_error': 0.}, env._start_pos, env._start_rotation)
        assert info['is_success']
        monkeypatch.setattr(env, '_handle_fingers', lambda: ({0}, {1}))
        info = env._success_info({'angle_error': 0.}, env._start_pos, env._start_rotation)
        assert info['is_success'] and info['grasp_supports_both_handles']
    finally:
        env.close()
