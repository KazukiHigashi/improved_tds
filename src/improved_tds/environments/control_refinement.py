"""Common v2 observation and goal-local braking for all tool categories."""
import gymnasium as gym
import numpy as np


class ControlRefinement(gym.Wrapper):
    @property
    def criteria(self):
        return self.env.criteria

    @property
    def success_hold_steps(self):
        return self.env.success_hold_steps

    def goal_sampling_report(self):
        return self.env.goal_sampling_report()

    def __init__(self, env):
        super().__init__(env)
        size=env.observation_space['observation'].shape[0]+env.action_space.shape[0]
        self.observation_space=gym.spaces.Dict({**env.observation_space.spaces,
            'observation':gym.spaces.Box(-np.inf,np.inf,(size,),dtype=np.float32)})
        self.previous_action=np.zeros(env.action_space.shape,dtype=np.float32)
        u=env.unwrapped
        if hasattr(u,'target_angle_range'):
            self.span=float(np.diff(u.target_angle_range)[0])
        else:
            self.span=float(getattr(u,'push_limit',getattr(u,'trigger_limit',1.)))
        self.dt=float(getattr(u,'dt',u.model.opt.timestep*getattr(u,'frame_skip',1)))

    def _obs(self,obs):
        return {**obs,'observation':np.concatenate([obs['observation'],self.previous_action])}

    def reset(self,**kwargs):
        self.previous_action[:]=0
        obs,info=self.env.reset(**kwargs)
        self.previous_achieved=float(obs['achieved_goal'][0])
        return self._obs(obs),info

    def compute_reward(self,achieved,desired,info):
        reward=self.env.compute_reward(achieved,desired,info)
        return np.asarray(reward)+self._braking(achieved,desired,info)

    def _braking(self,achieved,desired,info):
        a=np.asarray(achieved).reshape(-1);g=np.asarray(desired).reshape(-1)
        infos=[info]*len(a) if isinstance(info,dict) else list(info)
        speed=np.asarray([i.get('normalized_tool_speed',0.) for i in infos])
        failed=np.asarray([i.get('task_failure',False) or i.get('object_dropped',False) or i.get('object_tipped',False) for i in infos])
        proximity=np.exp(-.5*((a-g)/(.10*self.span))**2)
        # Bounded and active near the relabelled goal only; moving far away is unpenalized.
        return np.where(failed,0.,-.5*proximity*np.minimum(speed**2,1.)).astype(np.float32)

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        achieved=float(obs['achieved_goal'][0])
        info=dict(info)
        info['normalized_tool_speed']=(achieved-self.previous_achieved)/(self.dt*self.span)
        self.previous_achieved=achieved
        # A slew-limited task must expose the setpoint actually applied to the
        # plant; otherwise the augmented observation hides controller state.
        self.previous_action=np.asarray(info.get('applied_action',action),dtype=np.float32).copy()
        if info.get('numerical_failure', False):
            # Preserve the finite terminal sentinel supplied by the environment.
            # Invalid requests are not observations and this row is discarded.
            self.previous_action[:] = 0.
        if info.get('canonicalize_ignored_action', False) and info.get('retention_frozen', False):
            self.previous_action[:] = 0.
        braking=float(self._braking(obs['achieved_goal'],obs['desired_goal'],info)[0])
        info['reward_braking']=braking
        info['reward_total']=float(reward)+braking
        return self._obs(obs),info['reward_total'],terminated,truncated,info
