"""Shared stability shaping and explicit active/ignored action semantics."""
import gymnasium as gym
import numpy as np
from .scissors_control import ScissorsControlConfig
from .scissors_retention_v6 import ScissorsRetentionV6Wrapper


class ScissorsRetentionV7Wrapper(ScissorsRetentionV6Wrapper):
    servo_margin = .15  # rad, a soft reward margin, NOT an action constraint
    servo_scale = .3

    def __init__(self, env, **kwargs):
        self._last_tracking_error = np.zeros(env.action_space.shape, dtype=np.float32)
        kwargs.setdefault('control_config', ScissorsControlConfig(safety_weight=2.))
        super().__init__(env, **kwargs)
        if self.unwrapped.relative_control:
            raise ValueError('fixed-command certification requires absolute control')
        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces,
            'control_enabled': gym.spaces.Box(0., 1., (1,), dtype=np.float32),
            'servo_tracking_error': gym.spaces.Box(-np.inf, np.inf, env.action_space.shape, dtype=np.float32)})

    def _augment_info(self, info):
        info = super()._augment_info(info)
        raw = self.unwrapped
        gap = raw.data.ctrl[raw._actuator_ids] - raw.data.actuator_length[raw._actuator_ids]
        if np.isfinite(gap).all():
            self._last_tracking_error = np.asarray(np.clip(gap / self.servo_scale, -1e6, 1e6), dtype=np.float32)
        excess = np.maximum(np.abs(gap) - self.servo_margin, 0.) / self.servo_scale
        info['servo_gap_cost'] = float(np.mean(np.minimum(excess**2, 4.)))
        info['servo_gap_max_rad'] = float(np.max(np.abs(gap)))
        return info

    def _obs(self, obs):
        return {**super()._obs(obs),
            'control_enabled': np.array([not (self.freeze_remaining > 0 or self.certified)], dtype=np.float32),
            'servo_tracking_error': self._last_tracking_error.copy()}

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info['task_version'] = 'scissors_retention_v7'
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info['task_version'] = 'scissors_retention_v7'
        # The outer observation wrapper must not inject ignored random requests
        # into the next state. The applied fixed command remains observable.
        info['canonicalize_ignored_action'] = True
        return obs, reward, done, truncated, info

    def _reward_terms(self, achieved_goal, desired_goal, info):
        terms = super()._reward_terms(achieved_goal, desired_goal, info)
        n = len(np.asarray(achieved_goal).reshape(-1))
        infos = [info] * n if isinstance(info, dict) else list(info)
        failed = np.array([x.get('task_failure', False) for x in infos])
        support = np.array([x.get('support_fraction', 0.) for x in infos])
        gap_cost = np.array([x.get('servo_gap_cost', 0.) for x in infos])
        terms['support_margin'] = np.where(failed, 0., 1.5 * support).astype(np.float32)
        terms['servo_gap_penalty'] = np.where(failed, 0., -.5 * gap_cost).astype(np.float32)
        terms['total'] = terms['total'] + terms['support_margin'] + terms['servo_gap_penalty']
        return terms

    def goal_sampling_report(self):
        return {**super().goal_sampling_report(), 'task_version': 'scissors_retention_v7',
            'phase_aware_sac': True, 'support_margin_weight': 1.5,
            'servo_gap_penalty_weight': .5, 'servo_margin_rad': self.servo_margin,
            'servo_scale_rad': self.servo_scale, 'hard_servo_limit': False}
