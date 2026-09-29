"""Task-independent bounded actuator-setpoint slew for Push and Spray.

The Scissors smooth task implements the same requested-to-applied rule inside
its task wrapper.  A rate in normalized action units per *simulated second*
keeps the physical slew equal when simulator step durations differ.
"""
import gymnasium as gym
import numpy as np


class ToolActionSlewWrapper(gym.Wrapper):
    @property
    def criteria(self):
        return self.env.criteria

    @property
    def success_hold_steps(self):
        return self.env.success_hold_steps

    def __init__(self, env, *, rate_per_second=1.25, request_gap_penalty_weight=1.0):
        super().__init__(env)
        if (not np.isfinite(rate_per_second) or rate_per_second <= 0 or
                not np.isfinite(request_gap_penalty_weight) or request_gap_penalty_weight < 0):
            raise ValueError("positive finite slew rate and nonnegative gap penalty required")
        raw = env.unwrapped
        dt = float(getattr(raw, "dt", float(raw.model.opt.timestep) * int(raw.frame_skip)))
        if not np.isfinite(dt) or dt <= 0 or rate_per_second * dt > 1:
            raise ValueError("invalid simulator step duration or slew rate")
        self.rate_per_second = float(rate_per_second)
        self.max_delta = float(rate_per_second * dt)
        self.request_gap_penalty_weight = float(request_gap_penalty_weight)
        self.applied_action = np.zeros(env.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        self.applied_action[:] = 0
        obs, info = self.env.reset(**kwargs)
        return obs, {**info, "applied_action": self.applied_action.copy()}

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != self.action_space.shape or not np.isfinite(requested).all():
            raise ValueError("invalid action shape or nonfinite command")
        requested = np.clip(requested, -1, 1)
        previous = self.applied_action.copy()
        applied = np.clip(previous + np.clip(requested - previous,
            -self.max_delta, self.max_delta), -1, 1).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(applied)
        gap = float(np.mean(np.square(requested - applied)))
        penalty = -self.request_gap_penalty_weight * gap
        self.applied_action = applied
        info = dict(info)
        info.update(applied_action=applied.copy(), requested_action=requested.copy(),
            applied_action_delta_max=float(np.max(np.abs(applied - previous))),
            action_rate_limited_fraction=float(np.mean(np.abs(requested - applied) > 1e-6)),
            requested_applied_gap_cost=gap, reward_request_gap_penalty=penalty,
            reward_total=float(reward) + penalty)
        return obs, float(reward) + penalty, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        base = np.asarray(self.env.compute_reward(achieved_goal, desired_goal, info))
        if isinstance(info, dict):
            gap = float(info.get("requested_applied_gap_cost", 0.0))
        else:
            gap = np.asarray([item.get("requested_applied_gap_cost", 0.0) for item in info])
        return base - self.request_gap_penalty_weight * gap

    def goal_sampling_report(self):
        return {**self.env.goal_sampling_report(),
            "tool_action_slew_rate_per_second": self.rate_per_second,
            "tool_action_slew_max_delta": self.max_delta,
            "tool_request_gap_penalty_weight": self.request_gap_penalty_weight}

    def set_target_max_fraction(self, maximum):
        return self.env.set_target_max_fraction(maximum)
