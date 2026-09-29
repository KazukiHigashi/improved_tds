import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import pytest
import torch
from improved_tds.environments.scissors_control import ScissorsControlWrapper
from improved_tds.environments.control_refinement import ControlRefinement
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.learning.tool_features import ScissorsControlFeatures


@pytest.mark.parametrize('variant', [1,2,3,4,5])
def test_v3_shared_criteria_rate_limit_and_her(variant):
    _register_tool_manip_gym_environments()
    task=ScissorsControlWrapper(gym.make(f'ScissorsVariant{variant}-v0',hold_initial_steps=0))
    env=ControlRefinement(task)
    try:
        obs,_=env.reset(seed=3)
        assert env.observation_space.contains(obs)
        assert task.criteria.angle_tolerance==.03
        assert task.criteria.max_tool_speed==.1
        assert task.success_hold_steps==10
        for sign in [1,-1,1]:
            before=task.applied_action.copy()
            obs,reward,_,_,info=env.step(np.full(env.action_space.shape,sign,dtype=np.float32))
            assert np.max(abs(task.applied_action-before))<=.100001
            assert env.observation_space.contains(obs)
            assert reward==pytest.approx(float(env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)[0]))
        env.reset()
        assert not np.any(task.applied_action)
        info=dict(grasp_maintained=1,is_in_grasp_space=1,tool_speed=0.,grasp_contact_count=2)
        still=task._reward_terms([.5],[.5],info)
        fast=task._reward_terms([.5],[.5],{**info,'tool_speed':1.})
        far=task._reward_terms([.5],[1.],info)
        assert still['settling'][0]>fast['settling'][0]
        assert still['settling'][0]>far['settling'][0]
        assert task.compute_reward([.5],[.5],{**info,'task_failure':True})[0]==-25
        assert task._reward_terms([.5],[.5],{**info,'grasp_maintained':0})['settling'][0]==0
        features=ScissorsControlFeatures(env.observation_space)
        batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}
        result=features(batch)
        assert result.shape==(1,features.features_dim)
        assert torch.isfinite(result).all()
        assert result[0,-1].item()==pytest.approx(float(obs['desired_goal'][0]-obs['achieved_goal'][0]))
    finally: env.close()


def test_scalar_acceptance_is_json_serializable():
    # A full bin short-circuits before the final Python bool expression.
    counts=np.array([1])
    accepted=bool(True and counts[0]<1 and True)
    assert json.loads(json.dumps({'accepted':accepted}))=={'accepted':False}
