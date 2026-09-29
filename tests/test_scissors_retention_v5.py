import gymnasium as gym
import numpy as np
import pytest
from improved_tds.environments.scissors_control import ScissorsControlWrapper
from improved_tds.environments.scissors_research import PostureCriteria
from improved_tds.environments.scissors_retention_v5 import ScissorsRetentionV5Wrapper
from improved_tds.environments.control_refinement import ControlRefinement
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.learning.safe_replay import NumericalSafeReplayBuffer


def make(variant=3,training=False):
    _register_tool_manip_gym_environments()
    return ScissorsRetentionV5Wrapper(gym.make(f'ScissorsVariant{variant}-v0',hold_initial_steps=0,max_episode_steps=400),
        criteria=PostureCriteria(hold_seconds=2.),retention_training_steps=750000 if training else 0)


GOOD=dict(is_success=True,root_linear_speed=0.,root_angular_speed=0.)


@pytest.mark.parametrize('variant',[3,4,5])
def test_shared_spaces_rewards_and_validation_duration(variant):
    task=make(variant);env=ControlRefinement(task)
    try:
        obs,info=env.reset(seed=8)
        assert env.observation_space.contains(obs)
        assert task.episode_seconds==2. and info['success_hold_steps']==50
        obs,reward,_,_,info=env.step(np.zeros(env.action_space.shape,dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert reward==pytest.approx(float(env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)[0]))
    finally:env.close()


def test_two_second_success_does_not_terminate_or_release(monkeypatch):
    env=make()
    try:
        obs,_=env.reset(seed=4);base_obs={**obs,'observation':obs['observation'][:-9]}
        actions=[]
        def fake(self,action):
            actions.append(np.array(action));self.applied_action=np.array(action)
            return base_obs,1.,False,False,{**GOOD,'task_failure':False,'grasp_maintained':True}
        monkeypatch.setattr(ScissorsControlWrapper,'step',fake)
        env.freeze_remaining=50
        for i in range(60):
            _,reward,done,_,info=env.step(np.ones(env.action_space.shape,dtype=np.float32))
            assert not done
            assert info['is_stable_success']==(i>=49)
            np.testing.assert_equal(actions[-1],0.)
        assert env.certified and reward>1.
    finally:env.close()


def test_probe_failure_releases_and_readiness_gates():
    env=make()
    try:
        env.reset(seed=0)
        for _ in range(20):env._advance_phase(GOOD,.1,False)
        assert env.freeze_remaining==0
        for _ in range(10):env._advance_phase(GOOD,0.,False)
        assert env.freeze_remaining==50
        info=env._advance_phase({**GOOD,'is_success':False},0.,False)
        assert info['retention_aborted'] and not info['is_stable_success']
        assert env.freeze_remaining==0 and env.cooldown==10
    finally:env.close()


def test_competence_promotion_and_no_easy_validation():
    env=make(training=True)
    try:
        env.reset(seed=0);env.episode_seconds=.4
        for _ in range(20):env._record_probe(False)
        assert env.stage==0
        for _ in range(14):env._record_probe(True)
        assert env.stage==1
        env.training_curriculum=False
        env.reset(seed=0)
        assert env.episode_seconds==2.
    finally:env.close()


def test_acquisition_window_never_masks_pose_failure(monkeypatch):
    env=make()
    try:
        env.reset(seed=0)
        base=dict(grasp_maintained=False,failure_contact=True,task_failure=True,
                  failure_rotation=False,failure_displacement=False,object_dropped=False,grasp_contact_count=1)
        monkeypatch.setattr(ScissorsControlWrapper,'_augment_info',lambda self,info:{**base,**info})
        env.acquired=False;env.episode_step=3
        assert not env._augment_info({})['task_failure']
        assert env._augment_info({'failure_rotation':True})['task_failure']
        env.acquired=True
        assert env._augment_info({})['task_failure']
        env.acquired=False;env.episode_step=10
        assert env._augment_info({})['task_failure']
    finally:env.close()


def test_clock_rewind_is_finite_terminal_failure(monkeypatch):
    env=make()
    try:
        obs,_=env.reset(seed=0);base={**obs,'observation':obs['observation'][:-9]}
        def fake(self,action):
            self.unwrapped.data.time-=1.
            return base,1.,False,False,GOOD
        monkeypatch.setattr(ScissorsControlWrapper,'step',fake)
        obs,_,done,_,info=env.step(np.zeros(env.action_space.shape))
        assert done and info['numerical_failure'] and not info['is_stable_success']
        assert all(np.isfinite(x).all() for x in obs.values())
    finally:env.close()


def test_invalid_vector_replay_row_is_not_stored():
    space=gym.spaces.Dict({'observation':gym.spaces.Box(-1,1,(2,),dtype=np.float32)})
    action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    b=NumericalSafeReplayBuffer(16,space,action_space,device='cpu',n_envs=2)
    obs={'observation':np.zeros((2,2),dtype=np.float32)}
    action=np.zeros((2,1),dtype=np.float32)
    b.add(obs,obs,action,np.zeros(2),np.zeros(2),[{}, {'numerical_failure':True}])
    assert b.pos==0 and b.dropped_vector_transitions==1
    b.add(obs,obs,action,np.zeros(2),np.zeros(2),[{},{}])
    assert b.pos==1


def test_invalid_episode_is_not_recorded():
    from improved_tds.learning.collector import SuccessfulTrajectoryCollector
    class Invalid(gym.Env):
        observation_space=gym.spaces.Dict({'achieved_goal':gym.spaces.Box(-1,1,(1,),dtype=np.float32)})
        action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
        def reset(self,**kwargs):return {'achieved_goal':np.array([0.],dtype=np.float32)},{}
        def step(self,action):return {'achieved_goal':np.array([0.],dtype=np.float32)},-25.,True,False,{'numerical_failure':True}
    env=SuccessfulTrajectoryCollector(Invalid())
    env.reset()
    env.step(np.zeros(1))
    assert env.rejected_trajectories==1
    assert env.rejection_reasons['numerical_failure']==1
    assert not env._steps and not env._success
