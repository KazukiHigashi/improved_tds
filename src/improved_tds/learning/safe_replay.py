"""Exclude numerical-reset transitions instead of training on simulator recovery."""
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer


class NumericalSafeReplayBuffer(DictReplayBuffer):
    dropped_vector_transitions = 0

    def add(self, obs, next_obs, action, reward, done, infos):
        bad = any(x.get('numerical_failure',False) for x in infos)
        bad |= not all(np.isfinite(x).all() for d in (obs,next_obs) for x in d.values())
        bad |= not np.isfinite(action).all() or not np.isfinite(reward).all()
        if bad:
            # VecEnv writes one row for all environments. Drop the whole row
            # (including safe siblings) rather than invent a replacement sample.
            self.dropped_vector_transitions+=1
            return
        return super().add(obs,next_obs,action,reward,done,infos)
