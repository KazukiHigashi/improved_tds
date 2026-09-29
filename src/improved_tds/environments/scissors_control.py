"""Opt-in v3 control task. Shared constants for every Scissors geometry."""
from dataclasses import asdict, dataclass
import gymnasium as gym
import numpy as np
from .scissors_research import ScissorsPostureWrapper


@dataclass(frozen=True)
class ScissorsControlConfig:
    # Normalized absolute commands still span [-1,1]; only their rate is limited.
    action_rate_per_second: float = 2.5
    settling_weight: float = 2.0
    settling_angle_sigma: float = 0.10
    settling_speed_sigma: float = 0.30
    safety_weight: float = 0.5


class ScissorsControlWrapper(ScissorsPostureWrapper):
    def __init__(self, env, *, control_config=None, **kwargs):
        self.control_config = control_config or ScissorsControlConfig()
        if any(not np.isfinite(x) or x <= 0 for x in asdict(self.control_config).values()):
            raise ValueError('control parameters must be finite and positive')
        super().__init__(env, **kwargs)
        self.applied_action = np.zeros(self.action_space.shape, dtype=np.float32)
        space = self.observation_space['observation']
        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces,
            'observation': gym.spaces.Box(-np.inf, np.inf,
                (space.shape[0] + 2*self.action_space.shape[0],), dtype=np.float32)})

    def _control_observation(self, obs):
        u = self.unwrapped
        ranges = u.model.actuator_ctrlrange[u._actuator_ids]
        neutral = 2*(u._neutral_ctrl-ranges[:,0])/np.maximum(np.diff(ranges,axis=1).ravel(),1e-8)-1
        return {**obs, 'observation': np.concatenate([
            obs['observation'], self.applied_action, neutral]).astype(np.float32)}

    def reset(self, **kwargs):
        self.applied_action[:] = 0
        obs, info = super().reset(**kwargs)
        return self._control_observation(obs), info

    def step(self, action):
        requested = np.clip(np.asarray(action,dtype=np.float32), -1, 1)
        if requested.shape != self.action_space.shape:
            raise ValueError('wrong action shape')
        limit = self.control_config.action_rate_per_second*self.unwrapped.dt
        self.applied_action = np.clip(self.applied_action + np.clip(
            requested-self.applied_action, -limit, limit), -1, 1).astype(np.float32)
        obs, reward, done, truncated, info = super().step(self.applied_action)
        info['action_rate_limited_fraction'] = float(np.mean(abs(requested-self.applied_action)>1e-6))
        return self._control_observation(obs), reward, done, truncated, info

    def goal_sampling_report(self):
        return {**super().goal_sampling_report(), 'task_version': 'scissors_control_v3',
                'control_config': asdict(self.control_config)}

    def _reward_terms(self, achieved_goal, desired_goal, info):
        terms = super()._reward_terms(achieved_goal, desired_goal, info)
        a = np.asarray(achieved_goal).reshape(-1)
        g = np.asarray(desired_goal).reshape(-1)
        infos = [info]*len(a) if isinstance(info,dict) else list(info)
        cfg, criteria = self.control_config, self.criteria
        failed = np.array([i.get('task_failure',False) for i in infos], dtype=bool)
        supported = np.array([i.get('grasp_maintained',0) for i in infos]) > .5
        speed = np.array([i.get('tool_speed',np.inf) for i in infos])
        settling = cfg.settling_weight*np.exp(-.5*((a-g)/cfg.settling_angle_sigma)**2
                     -.5*(speed/cfg.settling_speed_sigma)**2)*supported
        displacement = np.array([i.get('object_displacement',0.) for i in infos])/criteria.max_displacement
        rotation = np.array([i.get('object_rotation_error',0.) for i in infos])/criteria.max_rotation
        contact = np.array([i.get('grasp_contact_count',criteria.min_support_bodies) for i in infos])
        risk = .5*np.minimum(displacement**2,1.) + .5*np.minimum(rotation**2,1.)
        risk += np.clip(1-contact/criteria.min_support_bodies,0.,1.)
        terms['settling'] = np.where(failed,0.,settling).astype(np.float32)
        terms['safety_penalty'] = np.where(failed,0.,-cfg.safety_weight*risk).astype(np.float32)
        terms['total'] = terms['total']+terms['settling']+terms['safety_penalty']
        return terms
