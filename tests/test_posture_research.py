from dataclasses import replace
from pathlib import Path
import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv
from improved_tds.environments.pose_pool import PushGoalWrapper, SprayGoalWrapper
from improved_tds.environments.scissors_research import ScissorsPostureWrapper, PostureCriteria
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.learning.collector import SuccessfulTrajectoryCollector
from improved_tds.environments.control_refinement import ControlRefinement


def make_worker():
    _register_tool_manip_gym_environments()
    return SuccessfulTrajectoryCollector(ScissorsPostureWrapper(gym.make('ScissorsVariant1-v0'),target_max_fraction=.3))


def test_curriculum_reaches_inner_sampler_in_spawn_worker():
    env=SubprocVecEnv([make_worker],start_method='spawn')
    try:
        env.env_method('set_target_max_fraction',1.0)
        assert env.env_method('goal_sampling_report')[0]['effective_curriculum_max']==1.0
        goals=[float(env.reset()['desired_goal'][0,0]) for _ in range(60)]
        assert max(goals)>.8
        assert all(n>0 for n in env.env_method('goal_sampling_report')[0]['sampled_goal_bins'])
    finally:
        env.close()


@pytest.mark.parametrize('name,wrapper,key,target',[
    ('ShadowHandLitePush2-v0',PushGoalWrapper,'target_push_depth',.005),
    ('ShadowHandLiteSpray2-v0',SprayGoalWrapper,'target_trigger_angle',.2)])
def test_observation_has_no_conflicting_embedded_goal(name,wrapper,key,target):
    _register_tool_manip_gym_environments()
    env=wrapper(gym.make(name))
    try:
        a,_=env.reset(seed=42,options={key:target})
        b,_=env.reset(seed=42,options={key:0.0})
        np.testing.assert_allclose(a['observation'],b['observation'])
        assert a['desired_goal'][0]!=b['desired_goal'][0]
    finally:
        env.close()


def test_relabelled_success_requires_low_speed_and_contact():
    _register_tool_manip_gym_environments()
    env=ScissorsPostureWrapper(gym.make('ScissorsVariant1-v0'))
    try:
        assert env.success_hold_steps==10
        info=dict(is_in_grasp_space=1,grasp_maintained=1,tool_speed=.2,task_failure=False)
        terms=env._reward_terms([.5],[.5],info)
        assert terms['success_bonus'][0]==0
        info['tool_speed']=0
        assert env._reward_terms([.5],[.5],info)['success_bonus'][0]==5
        info.update(is_in_grasp_space=0,grasp_maintained=0)
        assert env._reward_terms([.5],[.5],info)['failure_penalty'][0]==0
        info['task_failure']=True
        assert env.compute_reward([.5],[.5],info)[0]==-25
    finally:
        env.close()


@pytest.mark.parametrize('name,wrapper',[('ScissorsVariant1-v0',ScissorsPostureWrapper),
    ('ShadowHandLitePush1-v0',PushGoalWrapper),('ShadowHandLiteSpray1-v0',SprayGoalWrapper)])
def test_v2_observation_and_relabelled_braking(name,wrapper):
    _register_tool_manip_gym_environments()
    env=ControlRefinement(wrapper(gym.make(name)))
    try:
        obs,_=env.reset(seed=3)
        assert env.observation_space.contains(obs)
        action=np.zeros(env.action_space.shape,dtype=np.float32)
        obs,reward,_,_,info=env.step(action)
        assert env.observation_space.contains(obs)
        assert reward==pytest.approx(float(env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)[0]))
        synthetic=dict(normalized_tool_speed=1.)
        assert env._braking([0.],[0.],synthetic)[0]==pytest.approx(-.5)
        assert abs(env._braking([0.],[env.span],synthetic)[0])<1e-10
        assert env._braking([0.],[0.],dict(normalized_tool_speed=1.,task_failure=True))[0]==0.
    finally:env.close()
