from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
from improved_tds.environments.scissors_reach import ScissorsReachWrapper
from improved_tds.environments.control_refinement import ControlRefinement
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.experiments.collect_reach_successes import successful_state
from improved_tds.experiments.reach_tds_direct import configure_scissors_evaluation_palm_radius


def make(variant=3):
    _register_tool_manip_gym_environments()
    return ControlRefinement(ScissorsReachWrapper(gym.make(f'ScissorsVariant{variant}-v0',hold_initial_steps=0)))


@pytest.mark.parametrize('variant',[1,2,3,4,5])
def test_shared_simple_rule_and_reward(variant):
    env=make(variant)
    try:
        obs,info=env.reset(seed=1)
        assert env.observation_space.contains(obs)
        assert env.success_hold_steps==5 and env.unwrapped.angle_threshold==.03
        assert not info['is_stable_success'] and info['success_streak']==0
        assert 'control_enabled' not in obs
        assert not hasattr(env.env,'freeze_remaining')
        obs,reward,_,_,info=env.step(np.zeros(env.action_space.shape,dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert reward==pytest.approx(float(env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)[0]))
    finally:env.close()


def test_five_feedback_steps_not_fifty_and_no_velocity_limit():
    env=make()
    try:
        env.reset(seed=2);task=env.env
        task._contacts[:]=[1,1,0,0,0]
        task.unwrapped.data.qvel[task._hinge_dof]=10.
        for i in range(5):
            info=task._success_info(dict(angle_error=.02,tool_speed=10.),task._start_pos,task._start_rotation)
            assert info['is_stable_success']==(i==4)
            assert info['tool_speed']==10. and not info['speed_success_criterion']
        # Success has not frozen the actuator command.
        before=env.unwrapped.data.ctrl.copy()
        env.step(np.full(env.action_space.shape,.2,dtype=np.float32))
        assert not np.allclose(before,env.unwrapped.data.ctrl)
        info=task._success_info(dict(angle_error=.04),task._start_pos,task._start_rotation)
        assert not info['is_success'] and info['success_streak']==0
    finally:env.close()


def test_numeric_failure_does_not_create_a_success():
    env=make()
    try:
        env.reset(seed=0)
        obs,_,done,_,info=env.step(np.full(env.action_space.shape,np.nan))
        assert done and info['numerical_failure'] and not info['is_stable_success']
        assert all(np.isfinite(v).all() for v in obs.values())
    finally:env.close()


def test_palm_sphere_alone_gates_grasp_and_angle_success(monkeypatch):
    env=make()
    try:
        env.reset(seed=0);task=env.env;task._contacts[:]=0
        info=task._success_info(dict(angle_error=0.),task._start_pos,task._start_rotation)
        assert info['is_on_palm'] and info['is_success'] and not info['task_failure']
        reward=task.compute_reward(np.array([0.]),np.array([0.]),info)[0]
        assert reward > -task.failure_penalty
        dropped=task._success_info(dict(angle_error=0.),task._start_pos-np.array([0.,0.,.09]),task._start_rotation)
        assert dropped['object_dropped'] and not dropped['task_failure'] and dropped['is_success']
        monkeypatch.setattr(task.unwrapped,'_scissors_joint_pos',
                            lambda: task.unwrapped._grasp_center_space()+np.array([.06,0,0]))
        outside=task._success_info(dict(angle_error=0.),task._start_pos,task._start_rotation)
        assert not outside['is_on_palm'] and not outside['grasp_maintained'] and not outside['is_success']
        assert not outside['task_failure']
    finally:env.close()


def test_offline_tds_scaled_palm_sphere_accepts_more_postures_without_contact_gate(monkeypatch):
    env=make()
    try:
        env.reset(seed=0);task=env.env;raw=env.unwrapped
        original=float(raw.grasp_threshold)
        center=raw._grasp_center_space().copy()
        task._contacts[:]=0
        monkeypatch.setattr(raw,'_scissors_joint_pos',
            lambda:center+np.array([1.2*original,0.,0.]))
        before=task._success_info(dict(angle_error=0.),task._start_pos,task._start_rotation)
        assert not before['is_on_palm']
        source_radius,evaluation_radius=configure_scissors_evaluation_palm_radius(
            env,1.5,original)
        after=task._success_info(dict(angle_error=0.),task._start_pos,task._start_rotation)
        assert source_radius==original
        assert evaluation_radius==1.5*original
        assert after['is_on_palm'] and after['grasp_maintained'] and after['is_success']
        assert after['grasp_contact_count']==0
        np.testing.assert_allclose(raw._grasp_center_space(),center)
    finally:env.close()


@pytest.mark.parametrize('variant',[1,2,3,4,5])
def test_original_green_sphere_is_rigidly_attached_to_palm(variant):
    _register_tool_manip_gym_environments()
    raw=gym.make(f'ScissorsVariant{variant}-v0',hold_initial_steps=0)
    initial_center=raw.unwrapped._grasp_center_space().copy()
    initial_radius=raw.unwrapped.grasp_threshold
    env=ScissorsReachWrapper(raw)
    try:
        np.testing.assert_allclose(raw.unwrapped._grasp_center_space(),initial_center,atol=1e-9)
        assert raw.unwrapped.grasp_threshold==initial_radius
        env.reset(seed=variant)
        u=env.unwrapped; palm=u._tds_palm_body_id
        expected=u.data.xpos[palm]+u.data.xmat[palm].reshape(3,3)@u._tds_palm_sphere_local_center
        np.testing.assert_allclose(u._grasp_center_space(),expected,atol=1e-9)
        u.data.xpos[palm]+=np.array([.01,-.02,.03])
        np.testing.assert_allclose(u._grasp_center_space(),expected+np.array([.01,-.02,.03]),atol=1e-9)
        quarter_turn=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        u.data.xmat[palm][:]=quarter_turn.ravel()
        np.testing.assert_allclose(u._grasp_center_space(),
            u.data.xpos[palm]+quarter_turn@u._tds_palm_sphere_local_center,atol=1e-9)
        assert env.goal_sampling_report()['grasp_criterion']=='hinge_center_inside_palm_attached_original_sphere'
    finally:env.close()


def test_contact_auxiliary_changes_reward_only():
    _register_tool_manip_gym_environments()
    base=ScissorsReachWrapper(gym.make('ScissorsVariant3-v0',hold_initial_steps=0),
                             contact_auxiliary_reward_weight=0.)
    auxiliary=ScissorsReachWrapper(gym.make('ScissorsVariant3-v0',hold_initial_steps=0),
                                  contact_auxiliary_reward_weight=.25)
    try:
        base.reset(seed=1);auxiliary.reset(seed=1)
        info=base._success_info(dict(angle_error=0.),base._start_pos,base._start_rotation)
        info['grasp_contact_fraction']=.6
        a=base._reward_terms(np.array([0.]),np.array([0.]),info)
        b=auxiliary._reward_terms(np.array([0.]),np.array([0.]),info)
        assert b['contact_auxiliary'][0]==pytest.approx(.15)
        assert b['total'][0]-a['total'][0]==pytest.approx(.15,abs=1e-5)
        assert info['is_success'] and info['is_on_palm']
    finally:base.close();auxiliary.close()


def test_collection_filters_failures_and_nonfinite_states():
    good=dict(is_success=True,is_stable_success=True,grasp_maintained=True)
    assert successful_state(good,np.array([0.,1.]),.5,0.,1.)
    for key in ('is_success','is_stable_success','grasp_maintained'):
        assert not successful_state({**good,key:False},np.array([0.,1.]),.5,0.,1.)
    for key in ('task_failure','numerical_failure'):
        assert not successful_state({**good,key:True},np.array([0.,1.]),.5,0.,1.)
    assert not successful_state(good,np.array([np.nan]),.5,0.,1.)
    assert not successful_state(good,np.array([0.]),1.01,0.,1.)
    # A successful endpoint posture is not rejected for a within-tolerance
    # soft-joint-limit overshoot. Save the actual angle, not a clipped angle.
    assert successful_state({**good,'angle_tolerance':.03},np.array([0.]),-.001,0.,1.)
    assert not successful_state({**good,'angle_tolerance':.03},np.array([0.]),-.04,0.,1.)


def test_collector_resumes_partial_and_only_saves_balanced_successes(tmp_path,monkeypatch):
    from improved_tds.experiments import collect_reach_successes as c
    model=tmp_path/'best.zip';model.write_bytes(b'test policy')
    pool=tmp_path/'pool.npz';pool.write_bytes(b'test pool')
    class Env:
        action_space=SimpleNamespace(shape=(2,))
        spec=SimpleNamespace(max_episode_steps=5)
        def __init__(self):
            self.unwrapped=self;self._actuator_ids=np.array([0,1]);self.data=SimpleNamespace(ctrl=np.zeros(2))
            self.grasp_threshold=.05;self._tds_palm_sphere_local_center=np.array([.1,0,0])
            self.angle_tolerance=.03
        def reset(self,seed,options):
            self.target=options['target'];self.n=0;self.fail=seed==9301
            self.data.ctrl[:]=self.target
            return dict(achieved_goal=np.array([self.target])),{}
        def step(self,action):
            self.n+=1
            success=self.n>=5 and not self.fail
            return dict(achieved_goal=np.array([self.target])),0.,False,self.n>=5,dict(
                is_success=success,is_stable_success=success,grasp_maintained=True,success_hold_steps=5,
                grasp_contact_count=2)
        def close(self):pass
    monkeypatch.setattr(c,'_register_tool_manip_gym_environments',lambda:None)
    monkeypatch.setattr(c,'_build_env',lambda *a,**kw:(Env(),SimpleNamespace(pose_count=32),pool,'target',0.,1.))
    monkeypatch.setattr(c,'_actuated_q',lambda e:np.full(2,e.target))
    monkeypatch.setattr(c.SAC,'load',lambda *a,**kw:SimpleNamespace(predict=lambda *a,**kw:(np.zeros(2),None)))
    output=tmp_path/'db.npz'
    args=['--env-id','ScissorsVariant3-v0','--model',str(model),'--pose-pool-dir',str(tmp_path),
        '--output',str(output),'--per-bin','2']
    assert c.main([*args,'--max-episodes','1'])==2
    assert not output.exists()
    assert c.main([*args,'--max-episodes','100'])==0
    with np.load(output,allow_pickle=False) as z:
        assert len(z['q'])==20 and z['success'].all() and (z['hold_steps']==5).all()
        b=np.minimum(9,(z['tool_state']*10).astype(int))
        assert np.bincount(b,minlength=10).tolist()==[2]*10
        assert 0. in z['target'] and 1. in z['target']
    expanded=tmp_path/'db_30.npz'
    larger=[*args[:args.index('--output')+1],str(expanded),'--per-bin','3',
            '--extend-from',str(output)]
    assert c.main([*larger,'--max-episodes','100'])==0
    with np.load(expanded,allow_pickle=False) as z:
        assert len(z['q'])==30 and z['success'].all()
    with np.load(output,allow_pickle=False) as z:
        assert len(z['q'])==20
