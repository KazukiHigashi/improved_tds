import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from improved_tds.environments.scissors_retention_v7 import ScissorsRetentionV7Wrapper
from improved_tds.environments.scissors_research import PostureCriteria
from improved_tds.environments.control_refinement import ControlRefinement
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.learning.retention_policy import RetentionSACPolicy, RetentionActor, RetentionCritic
from improved_tds.learning.tool_features import ScissorsControlFeatures


def make(variant=3):
    _register_tool_manip_gym_environments()
    return ControlRefinement(ScissorsRetentionV7Wrapper(
        gym.make(f'ScissorsVariant{variant}-v0', hold_initial_steps=0, max_episode_steps=400),
        criteria=PostureCriteria(hold_seconds=2.)))


@pytest.mark.parametrize('variant', [3,4,5])
def test_shared_spaces_reward_and_unchanged_criteria(variant):
    env=make(variant)
    try:
        obs,info=env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert obs['control_enabled'][0]==1 and info['success_hold_steps']==50
        assert env.criteria.angle_tolerance==.03 and env.criteria.max_tool_speed==.1
        assert env.env.control_config.safety_weight==2.
        for _ in range(3):
            obs,reward,done,trunc,info=env.step(np.zeros(env.action_space.shape,dtype=np.float32))
            assert env.observation_space.contains(obs)
            assert reward==pytest.approx(float(env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)[0]))
            if done or trunc:break
        assert info['task_version']=='scissors_retention_v7'
    finally:env.close()


def test_ignored_actions_do_not_change_physics_or_observations():
    envs=[make(),make()]
    try:
        for env in envs:
            env.reset(seed=12)
            env.env.freeze_remaining=60
        left=envs[0].step(np.ones(envs[0].action_space.shape,dtype=np.float32))
        right=envs[1].step(-np.ones(envs[1].action_space.shape,dtype=np.float32))
        for key in left[0]:np.testing.assert_array_equal(left[0][key],right[0][key])
        np.testing.assert_array_equal(envs[0].unwrapped.data.ctrl,envs[1].unwrapped.data.ctrl)
        assert left[1:4]==right[1:4]
        assert left[0]['control_enabled'][0]==0
        np.testing.assert_equal(envs[0].previous_action,0.)
    finally:
        for env in envs:env.close()


def test_no_hard_servo_limit_and_failure_penalty_unchanged():
    env=make()
    try:
        env.reset(seed=1)
        task=env.env
        assert not task.goal_sampling_report()['hard_servo_limit']
        goal=np.array([.5])
        base=dict(task_failure=False,grasp_maintained=True,support_fraction=1.,tool_speed=0.,servo_gap_cost=0.)
        t=task._reward_terms(goal,goal,base)
        large=task._reward_terms(goal,goal,{**base,'servo_gap_cost':4.})
        assert t['support_margin'][0]==1.5
        assert large['total'][0]==pytest.approx(t['total'][0]-2.)
        failed=task._reward_terms(goal,goal,{**base,'task_failure':True})
        assert failed['total'][0]==-task.failure_penalty
    finally:env.close()


def test_phase_actor_entropy_critic_invariance_and_gradients(tmp_path):
    torch.set_num_threads(1)
    env=make()
    try:
        model=SAC(RetentionSACPolicy,env,ent_coef=.02,device='cpu',buffer_size=32,
            policy_kwargs=dict(net_arch=[16,16],features_extractor_class=ScissorsControlFeatures))
        obs,_=env.reset(seed=2)
        batch={k:torch.as_tensor(np.stack([v,v]),dtype=torch.float32) for k,v in obs.items()}
        batch['control_enabled']=torch.tensor([[0.],[1.]])
        actor=model.actor;critic=model.critic
        assert isinstance(actor,RetentionActor) and isinstance(critic,RetentionCritic)
        action,logp=actor.action_log_prob(batch)
        assert torch.count_nonzero(action[0])==0 and logp[0]==0
        assert torch.isfinite(logp).all() and torch.count_nonzero(action[1])>0
        x=torch.randn_like(action,requires_grad=True)
        q=critic(batch,x)[0]
        other=critic(batch,-x)[0]
        assert torch.equal(q[0],other[0])
        q.sum().backward()
        assert torch.count_nonzero(x.grad[0])==0
        assert torch.count_nonzero(x.grad[1])>0
        for deterministic in (True,False):
            assert torch.count_nonzero(actor(batch,deterministic)[0])==0
        # Serialized policy class must survive normal SAC.load used by DB tools.
        path=tmp_path/'policy.zip';model.save(path)
        loaded=SAC.load(path,env=env,device='cpu')
        assert isinstance(loaded.actor,RetentionActor) and isinstance(loaded.critic,RetentionCritic)
        obs['control_enabled'][:]=0
        np.testing.assert_equal(loaded.predict(obs,deterministic=True)[0],0.)
    finally:env.close()


def test_numeric_sentinel_stays_finite_and_failure_is_not_success():
    env=make()
    try:
        env.reset(seed=0)
        obs,_,done,_,info=env.env._numerical_failure()
        assert done and info['numerical_failure'] and not info['is_stable_success']
        assert all(np.isfinite(x).all() for x in obs.values())
    finally:env.close()


def test_sac_optimizer_accepts_an_all_frozen_batch():
    torch.set_num_threads(1)
    env=make()
    try:
        model=SAC(RetentionSACPolicy,env,ent_coef=.02,device='cpu',buffer_size=32,
            batch_size=8,learning_starts=0,
            policy_kwargs=dict(net_arch=[16,16],features_extractor_class=ScissorsControlFeatures))
        obs,_=env.reset(seed=0)
        obs['control_enabled'][:]=0
        batch={k:v[None] for k,v in obs.items()}
        for _ in range(16):
            model.replay_buffer.add(batch,batch,np.ones((1,env.action_space.shape[0]),dtype=np.float32),
                np.array([1.]),np.array([0.]),[{}])
        model._setup_learn(8)
        model.train(gradient_steps=4,batch_size=8)
        assert model._n_updates==4
        assert all(torch.isfinite(x).all() for x in model.policy.parameters())
    finally:env.close()
