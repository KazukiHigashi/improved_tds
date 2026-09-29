import gymnasium as gym
import numpy as np
import pytest
from improved_tds.environments.scissors_control import ScissorsControlWrapper
from improved_tds.environments.scissors_retention import ScissorsRetentionWrapper, retention_seconds
from improved_tds.environments.scissors_research import PostureCriteria
from improved_tds.environments.control_refinement import ControlRefinement
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def test_schedule():
    assert [retention_seconds(x) for x in (0,.199,.2,.399,.4,.599,.6,1)]==[.4,.4,.8,.8,1.2,1.2,2.,2.]


@pytest.mark.parametrize('variant',[3,4,5])
def test_spaces_rewards_and_common_final_criteria(variant):
    _register_tool_manip_gym_environments()
    task=ScissorsRetentionWrapper(gym.make(f'ScissorsVariant{variant}-v0',hold_initial_steps=0),
        criteria=PostureCriteria(hold_seconds=2.))
    env=ControlRefinement(task)
    try:
        obs,info=env.reset(seed=7)
        assert env.observation_space.contains(obs)
        assert info['success_hold_steps']==50
        assert task.episode_seconds==2.
        obs,reward,_,_,info=env.step(np.zeros(env.action_space.shape,dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert not info['is_stable_success']
        assert reward==pytest.approx(float(env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)[0]))
    finally:env.close()


@pytest.mark.parametrize('budget,expected_hold,expect_final',[(0,50,True),(100000,10,False)])
def test_frozen_actual_command_and_strict_label(monkeypatch,budget,expected_hold,expect_final):
    _register_tool_manip_gym_environments()
    env=ScissorsRetentionWrapper(gym.make('ScissorsVariant3-v0',hold_initial_steps=0),
        criteria=PostureCriteria(hold_seconds=2.),retention_training_steps=budget)
    try:
        obs,_=env.reset(seed=7)
        base_obs={**obs,'observation':obs['observation'][:-4]}
        applied=[]
        def fake_step(self,action):
            applied.append(np.asarray(action).copy())
            self.applied_action=np.asarray(action).copy()
            return base_obs,1.,False,False,dict(is_success=True,task_failure=False,is_stable_success=True)
        monkeypatch.setattr(ScissorsControlWrapper,'step',fake_step)
        for _ in range(10):
            _,_,done,_,info=env.step(np.full(env.action_space.shape,.2,dtype=np.float32))
            assert not info['is_stable_success'] and not done
        assert env.freeze_remaining==expected_hold
        for i in range(expected_hold):
            _,_,done,_,info=env.step(np.full(env.action_space.shape,-.9,dtype=np.float32))
            np.testing.assert_allclose(applied[-1],.2)
            assert bool(info['is_stable_success'])==(expect_final and i==expected_hold-1)
        assert bool(done)==expect_final
        assert info['retention_curriculum_success']
    finally:env.close()


def test_v4_rejects_her():
    from improved_tds.experiments.train_sb3 import main
    with pytest.raises(SystemExit):
        main(['--scissors-control-version','v4','--her-sampled-goals','1'])
