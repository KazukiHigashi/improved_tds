import numpy as np
from improved_tds.experiments.simple_synergy import (
    preparation_candidates, steps, state, coverage, matrices, trajectory, direct_trajectory,
    simulation_step_seconds,
)


def test_candidates_cal_only_actual_offsets():
    data=dict(q=np.zeros((3,2)),pose_index=np.array([1,2,3]),success=np.array([True,True,False]),
        target_fraction=np.array([.3,.3,.3]),initial_q=np.zeros((3,2)),
        command_q=np.array([[1,2],[99,99],[9,9]]))
    offsets,rows=preparation_candidates(data,[1,3])
    assert rows==[0]
    assert len(offsets)==2
    np.testing.assert_equal(offsets[1],[1,2])


def test_holds_in_seconds_and_failure_grace():
    assert steps(.4,.04)==10
    assert steps(2.,.04)==50
    assert not state({'grasp_maintained':False,'task_failure':False})['failed']
    assert state({'task_failure':True})['failed']
    assert not state({'is_on_palm':1.,'grasp_maintained':1.,
                      'object_dropped':True,'object_tipped':True})['failed']
    assert state({'is_on_palm':0.,'task_failure':True})['failed']


def test_control_period_does_not_evaluate_missing_fallback():
    from types import SimpleNamespace
    scissors=SimpleNamespace(dt=.04,model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)))
    scissors.unwrapped=scissors
    assert simulation_step_seconds(scissors)==.04
    push=SimpleNamespace(frame_skip=10,model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)))
    push.unwrapped=push
    assert simulation_step_seconds(push)==.02


def test_missing_not_success_or_zero_error():
    record={'outcomes':[None,dict(achieved=.4,valid=True),dict(achieved=.9,valid=False)]}
    y,valid=matrices([record],3)
    assert np.isnan(y[0,0]) and not valid[0,0]
    assert coverage(record['outcomes'],0,1)==0
    assert coverage([dict(achieved=.1,valid=True),dict(achieved=.9,valid=True)],0,1)==.8


class Dummy:
    dt=.04
    _neutral_ctrl=np.array([0.])
    _ctrl_low=np.array([-1.])
    _ctrl_high=np.array([1.])
    _ctrl_halfspan=np.array([1.])
    _actuator_ids=np.array([0])
    def __init__(self, fail_at=None):
        from types import SimpleNamespace
        self.unwrapped=self
        self.model=SimpleNamespace(actuator_ctrlrange=np.array([[-1.,1.]]),
                                   actuator_trnid=np.array([[0,0]]),jnt_qposadr=np.array([0]))
        self.data=SimpleNamespace(qpos=np.array([0.]))
        self.fail_at=fail_at
        self.resets=0
    def reset(self, **kwargs):
        self.resets+=1;self.n=0
        return {'achieved_goal':np.array([0.])},{'grasp_maintained':True}
    def step(self, action):
        self.n+=1
        failed=self.n==self.fail_at
        return {'achieved_goal':np.array([.2])},0,failed,False,dict(grasp_maintained=not failed,task_failure=failed)


def test_continuous_single_reset_and_full_window():
    env=Dummy()
    out=trajectory(env,0,'target',.2,1,np.array([.1]),.2,[np.array([.1]),np.array([-.1])])
    assert env.resets==1
    assert out['preparation']['valid']
    assert all(x['valid'] for x in out['outcomes'])
    assert env.n==5+25+2*(125+50)  # 5 s ramp + 2 s hold at dt=0.04


def test_preparation_failure_does_not_retry_or_evaluate():
    env=Dummy(fail_at=3)
    out=trajectory(env,0,'target',.2,1,np.array([.1]),.2,[np.array([.1])])
    assert env.resets==1
    assert not out['preparation']['valid']
    assert out['outcomes']==[None]


def test_direct_trajectory_uses_one_absolute_target_without_preparation():
    env=Dummy()
    out=direct_trajectory(env,0,'target',.2,1,[np.array([.35])])
    assert env.resets==1
    assert env.n==125+50
    assert out['initial_q']==[0.]
    assert out['outcomes'][0]['command_q']==[.35]
    assert out['outcomes'][0]['valid']
    assert 'preparation' not in out


def test_direct_trajectory_never_retries_after_physical_failure():
    env=Dummy(fail_at=3)
    out=direct_trajectory(env,0,'target',.2,1,[np.array([.35]),np.array([.5])])
    assert env.resets==1 and env.n==3
    assert not out['outcomes'][0]['valid']
    assert out['outcomes'][1] is None


def test_direct_capture_can_use_25_step_intervals_and_include_reset():
    env=Dummy()
    captured=[]
    direct_trajectory(env,0,'target',.2,1,[np.array([.35])],
        capture=lambda step,obs,info: captured.append(step),
        capture_every=25,capture_initial=True)
    assert captured==list(range(0,176,25))


def test_no_fall_criterion_does_not_require_finger_contacts():
    from improved_tds.experiments.simple_synergy import state
    info=dict(grasp_maintained=False,grasp_contact_count=0,
        object_dropped=False,object_tipped=False)
    assert not state(info)['held']
    assert state(info,support_criterion='no_fall')['held']
    assert not state({**info,'object_tipped':True},support_criterion='no_fall')['held']
    assert state({**info,'object_tipped':True},support_criterion='no_fall')['failed']
