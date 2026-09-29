"""v4: common Scissors fixed-command retention curriculum (no HER)."""
import gymnasium as gym
import numpy as np
from .scissors_control import ScissorsControlWrapper


def retention_seconds(progress):
    """Common schedule; the last 40% of training uses the final two-second task."""
    return .4 if progress < .2 else .8 if progress < .4 else 1.2 if progress < .6 else 2.


class ScissorsRetentionWrapper(ScissorsControlWrapper):
    def __init__(self, env, *, retention_training_steps=0, **kwargs):
        self.retention_training_steps = int(retention_training_steps)
        self.retention_steps = 0
        self.freeze_remaining = 0
        self.freeze_good = 0
        self.acquire_good = 0
        self.episode_seconds = 2.
        super().__init__(env, **kwargs)
        size = self.observation_space['observation'].shape[0] + 4
        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces,
            'observation': gym.spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)})

    def _retention_obs(self, obs):
        final_steps = int(np.ceil(2./self.unwrapped.dt))
        extra = np.array([self.freeze_remaining/final_steps, self.freeze_good/final_steps,
                          min(self.acquire_good/10., 1.), self.episode_seconds/2.], dtype=np.float32)
        return {**obs, 'observation': np.concatenate([obs['observation'], extra])}

    def reset(self, **kwargs):
        self.freeze_remaining = self.freeze_good = self.acquire_good = 0
        progress = self.retention_steps/max(1,self.retention_training_steps)
        self.episode_seconds = retention_seconds(progress) if self.retention_training_steps else 2.
        obs, info = super().reset(**kwargs)
        info.update(is_stable_success=0., success_hold_steps=int(np.ceil(2./self.unwrapped.dt)),
                    retention_seconds=self.episode_seconds, task_version='scissors_retention_v4')
        return self._retention_obs(obs), info

    def step(self, action):
        self.retention_steps += 1
        was_frozen = self.freeze_remaining > 0
        # The APPLIED command includes the v3 rate limiter. Hold it exactly.
        requested = self.applied_action.copy() if was_frozen else action
        obs, reward, done, truncated, info = super().step(requested)
        good = bool(info.get('is_success', False)) and not info.get('task_failure',False)
        if was_frozen:
            self.freeze_remaining -= 1
            self.freeze_good = self.freeze_good + 1 if good else 0
        else:
            self.acquire_good = self.acquire_good + 1 if good else 0
        required = int(np.ceil(self.episode_seconds/self.unwrapped.dt - 1e-10))
        final_required = int(np.ceil(2./self.unwrapped.dt - 1e-10))
        completed = was_frozen and self.freeze_remaining == 0 and self.freeze_good >= required
        strict_success = completed and self.freeze_good >= final_required and not(done or truncated)
        # Dense near-goal/settling/grasp rewards are retained during acquisition;
        # short successful holds give shaping, never the final success label.
        bonus = (2. if was_frozen and good else 0.) + (5. if completed else 0.)
        if info.get('task_failure',False): bonus = 0.
        reward = float(reward) + bonus
        if was_frozen and self.freeze_remaining == 0:
            self.acquire_good = 0
        if not was_frozen and self.acquire_good >= int(np.ceil(.4/self.unwrapped.dt-1e-10)):
            self.freeze_remaining = required
            self.freeze_good = 0
            self.acquire_good = 0
        info = dict(info, is_stable_success=float(strict_success),
            success_hold_steps=final_required, retention_seconds=self.episode_seconds,
            retention_frozen=was_frozen, retention_success_steps=self.freeze_good,
            retention_curriculum_success=bool(completed), reward_retention=bonus,
            reward_total=reward, task_version='scissors_retention_v4')
        return self._retention_obs(obs), reward, bool(done or strict_success), truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Reward depends on temporal, goal-dependent phase: forbid HER in trainer.
        base = super().compute_reward(achieved_goal, desired_goal, info)
        infos = [info] if isinstance(info,dict) else list(info)
        return np.asarray(base) + np.array([x.get('reward_retention',0.) for x in infos],dtype=np.float32)

    def goal_sampling_report(self):
        return {**super().goal_sampling_report(), 'task_version':'scissors_retention_v4',
                'final_fixed_hold_seconds':2., 'training_retention_seconds':self.episode_seconds,
                'retention_training_steps':self.retention_steps,
                'retention_schedule':[[0,.4],[.2,.8],[.4,1.2],[.6,2.]]}
