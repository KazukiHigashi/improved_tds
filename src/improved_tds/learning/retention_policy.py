"""SAC with one dummy action and zero entropy in fixed-command states.

Physics ignores policy requests during retention. Enforce Q(s,a)=V(s) there,
instead of learning an arbitrary dependence on requests which were not applied.
The environment still observes its actual applied command and retention phase.
"""
import torch
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import Actor, SACPolicy


def active_action_mask(obs):
    return obs['control_enabled'].reshape(-1, 1) > .5


def canonical_action(obs, action):
    return torch.where(active_action_mask(obs), action, torch.zeros_like(action))


class RetentionActor(Actor):
    def forward(self, obs, deterministic=False):
        return canonical_action(obs, super().forward(obs, deterministic))

    def action_log_prob(self, obs):
        action, log_prob = super().action_log_prob(obs)
        return canonical_action(obs, action), torch.where(
            active_action_mask(obs).flatten(), log_prob, torch.zeros_like(log_prob))


class RetentionCritic(ContinuousCritic):
    def forward(self, obs, actions):
        return super().forward(obs, canonical_action(obs, actions))

    def q1_forward(self, obs, actions):
        return super().q1_forward(obs, canonical_action(obs, actions))


class RetentionSACPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        return RetentionActor(**self._update_features_extractor(self.actor_kwargs, features_extractor)).to(self.device)

    def make_critic(self, features_extractor=None):
        return RetentionCritic(**self._update_features_extractor(self.critic_kwargs, features_extractor)).to(self.device)
