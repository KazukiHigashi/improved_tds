import numpy as np

from improved_tds.experiments.reach_tds_direct import reconstructed_posture, summarise
from improved_tds.experiments.reach_tds import physical_levels


def test_direct_reconstruction_has_no_initial_pose_or_preparation_offset():
    mean=np.array([.2,-.1,.4])
    op=np.array([1.,0.,0.])
    grasp=np.array([0.,1.,0.])
    low=np.full(3,-2.)
    high=np.full(3,2.)
    target=reconstructed_posture(mean,op,.5,.4,grasp,.8,-.25,low,high)
    np.testing.assert_allclose(target,[.4,-.3,.4])
    assert len(reconstructed_posture(mean,op,.5,0.,grasp,.8,0.,low,high))==3


def test_global_component_score_and_expanded_endpoints():
    q=np.array([[-1.,.1],[0.,.1],[1.,.1]])
    mean=q.mean(axis=0)
    op=np.array([1.,0.])
    scores=(q-mean)@op
    np.testing.assert_allclose(physical_levels(scores,3),[-1.5,0.,1.5])
    np.testing.assert_allclose(mean+scores[:,None]*op,q)


def test_direct_summary_does_not_gate_operation_on_preparation():
    ok=dict(achieved=.25,valid=True,short_window=dict(valid=True))
    config=dict(selection=None,test=[dict(coverage=.25,diagnostic_coverage=.25,
        diagnostic=dict(outcomes=[ok]),target_trials=[dict(controller='database_linear_baseline',
            abs_error=.1,outcome=ok)])])
    result=summarise(config,[0.,1.])
    assert result['held_fraction']==1.
    assert result['controller_comparison']['database_linear_baseline']['held_mae']==.1
