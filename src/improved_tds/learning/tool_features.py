"""Scale goal channels inside the network; replay/evaluation keep physical units."""
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ToolFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space,goal_span=1.):
        super().__init__(observation_space,sum(int(np.prod(s.shape)) for s in observation_space.spaces.values()))
        if goal_span<=0: raise ValueError('goal_span must be positive')
        self.goal_span=float(goal_span)

    def forward(self,observations):
        return torch.cat([observations[k].flatten(start_dim=1)/(self.goal_span if k in ('achieved_goal','desired_goal') else 1.)
                          for k in sorted(observations)],dim=1)


class ScissorsControlFeatures(BaseFeaturesExtractor):
    """Compress force/velocity outliers and expose HER-consistent signed error."""
    def __init__(self, observation_space, goal_span=1.):
        super().__init__(observation_space,
            sum(int(np.prod(s.shape)) for s in observation_space.spaces.values())+1)
        if goal_span <= 0:
            raise ValueError('goal_span must be positive')
        self.goal_span = float(goal_span)

    def forward(self, observations):
        channels = []
        for key in sorted(observations):
            x = observations[key].flatten(start_dim=1)
            channels.append(x/self.goal_span if key in ('achieved_goal','desired_goal')
                            else torch.sign(x)*torch.log1p(torch.abs(x)))
        channels.append((observations['desired_goal']-observations['achieved_goal'])/self.goal_span)
        return torch.cat(channels,dim=1)
