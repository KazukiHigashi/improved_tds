from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from improved_tds.environments.action_slew import ToolActionSlewWrapper


class DummyTool(gym.Env):
    def __init__(self):
        self.action_space=gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-1.,1.,(1,),dtype=np.float32)
        self.model=SimpleNamespace(opt=SimpleNamespace(timestep=.01))
        self.frame_skip=2
        self.received=None

    def reset(self, *, seed=None, options=None):
        return np.zeros(1,dtype=np.float32),{}

    def step(self, action):
        self.received=np.asarray(action).copy()
        return np.zeros(1,dtype=np.float32),1.,False,False,{}

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.asarray(1.)


def test_slew_uses_simulated_seconds_and_penalizes_request_gap():
    env=ToolActionSlewWrapper(DummyTool(),rate_per_second=1.25,
        request_gap_penalty_weight=1.)
    env.reset()
    _,reward,_,_,info=env.step(np.ones(2,dtype=np.float32))
    assert np.allclose(env.unwrapped.received,[.025,.025])
    assert np.isclose(reward,1.-.975**2)
    assert np.isclose(env.compute_reward(0,0,info),reward)
    env.close()
