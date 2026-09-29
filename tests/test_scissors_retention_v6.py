import gymnasium as gym
import numpy as np
import pytest

from improved_tds.environments.scissors_control import ScissorsControlWrapper
from improved_tds.environments.scissors_research import PostureCriteria
from improved_tds.environments.scissors_retention_v6 import ScissorsRetentionV6Wrapper
from improved_tds.environments.control_refinement import ControlRefinement
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


GOOD = dict(is_success=True, root_linear_speed=.1, root_angular_speed=1.)


def make(variant=3, training=False):
    _register_tool_manip_gym_environments()
    return ScissorsRetentionV6Wrapper(
        gym.make(f'ScissorsVariant{variant}-v0', hold_initial_steps=0, max_episode_steps=400),
        criteria=PostureCriteria(hold_seconds=2.), retention_training_steps=750000 if training else 0)


@pytest.mark.parametrize('variant', [3, 4, 5])
def test_shared_strict_criteria_and_rewards(variant):
    env = ControlRefinement(make(variant))
    try:
        obs, info = env.reset(seed=8)
        assert env.observation_space.contains(obs)
        assert info['success_hold_steps'] == 50
        assert info['task_version'] == 'scissors_retention_v6'
        obs, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert reward == pytest.approx(float(env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)[0]))
    finally:
        env.close()


def test_no_action_delta_gate_and_no_prehold_frames_counted():
    env = make()
    try:
        env.reset(seed=0)
        for _ in range(3):
            info = env._advance_phase(GOOD, .1, False)
        assert env.freeze_remaining == 60 and env.probe_attempts == 1
        assert env.freeze_good == 0 and not info['is_stable_success']
        for _ in range(49):
            assert not env._advance_phase(GOOD, 0., False)['is_stable_success']
        assert env._advance_phase(GOOD, 0., False)['is_stable_success']
    finally:
        env.close()


def test_settling_tolerance_never_relaxes_final_streak():
    env = make()
    try:
        env.reset(seed=0)
        for _ in range(3): env._advance_phase(GOOD, .1, False)
        for _ in range(10):
            info = env._advance_phase({'is_success': False}, 0., False)
            assert not info['is_stable_success'] and not info['retention_aborted']
        for _ in range(49): assert not env._advance_phase(GOOD, 0., False)['is_stable_success']
        assert env._advance_phase(GOOD, 0., False)['is_stable_success']
    finally:
        env.close()


def test_timeout_recovery_and_physical_failure_immediate():
    env = make()
    try:
        env.reset(seed=0)
        for _ in range(3): env._advance_phase(GOOD, .1, False)
        for i in range(60):
            info = env._advance_phase({'is_success': i % 15 != 0}, 0., False)
            assert not info['is_stable_success']
        assert info['retention_aborted'] and env.cooldown == 10
        assert env.probe_completions == 1 and env.probe_successes == 0
        env.freeze_remaining = 60
        info = env._advance_phase(GOOD, 0., True)
        assert info['retention_aborted'] and info['reward_retention'] == 0.
        assert not info['is_stable_success'] and env.freeze_remaining == 0
    finally:
        env.close()


def test_curriculum_cannot_stall_and_validation_never_shortens():
    env = make(training=True)
    try:
        env.retention_steps = 450000
        env.reset(seed=0)
        assert env.stage == 3 and env.episode_seconds == 2.
        env.training_curriculum = False
        env.stage = 0
        for seed in range(10):
            env.reset(seed=seed)
            assert env.episode_seconds == 2.
    finally:
        env.close()


def test_short_probe_not_success_and_numeric_failure_not_fake_probe():
    env = make(training=True)
    try:
        env.reset(seed=0)
        env.episode_seconds = .4
        env.freeze_remaining = 20
        for _ in range(10): info = env._advance_phase(GOOD, 0., False)
        assert info['retention_curriculum_success'] and not info['is_stable_success']
        assert info['reward_retention'] == 3. + .8 * env.success_bonus and not env.certified
        count = env.probe_completions
        obs, _, done, _, info = env._numerical_failure()
        assert done and info['numerical_failure'] and env.probe_completions == count
        assert all(np.isfinite(x).all() for x in obs.values())
    finally:
        env.close()


def test_command_is_fixed_during_settling_hold_and_after_success(monkeypatch):
    env = make()
    try:
        obs, _ = env.reset(seed=0)
        base = {**obs, 'observation': obs['observation'][:-9]}
        actions = []
        def fake(self, action):
            actions.append(np.array(action))
            return base, 1., False, False, {**GOOD, 'is_success': len(actions) > 10}
        monkeypatch.setattr(ScissorsControlWrapper, 'step', fake)
        env.freeze_remaining = 60
        for i in range(70):
            _, _, done, _, info = env.step(np.ones(env.action_space.shape))
            assert not done and info['is_stable_success'] == (i >= 59)
            np.testing.assert_equal(actions[-1], 0.)
        assert env.certified
    finally:
        env.close()


def test_feedback_bonus_split_and_fixed_reward_recomputation():
    env = make()
    try:
        _, reset_info = env.reset(seed=0)
        info = {**reset_info, **GOOD, 'grasp_maintained': True, 'tool_speed': 0., 'task_failure': False}
        goal = np.array([.5])
        terms = env._reward_terms(goal, goal, info)
        assert terms['success_bonus'][0] == pytest.approx(.2 * env.success_bonus)
        env.freeze_remaining = 60
        phase = env._advance_phase(info, 0., False)
        assert phase['reward_retention'] == pytest.approx(2. + .8 * env.success_bonus)
        expected = terms['total'] + phase['reward_retention']
        np.testing.assert_allclose(env.compute_reward(goal, goal, {**info, **phase}), expected)
        failure_terms = env._reward_terms(goal, goal, {**info, 'task_failure': True})
        assert failure_terms['total'][0] == -env.failure_penalty
    finally:
        env.close()
