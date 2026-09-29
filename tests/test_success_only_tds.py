import numpy as np
from improved_tds.experiments.collect_until_tds_ready import successful_source_rows,inspect_splits
from improved_tds.experiments.evaluate_strict_synergy_control import balanced_angle_success_indices


def test_failures_outside_range_and_short_holds_not_saved():
    data=dict(success=np.array([True,False,True,True,True]),tool_state=np.array([.5,.5,-.01,1.01,.5]),
        hold_steps=np.array([10,10,10,10,9]),q=np.arange(5)[:,None])
    filtered=successful_source_rows(data,dict(target_range=[0,1],env_id='ScissorsVariant5-v0'))
    assert len(filtered['q'])==1
    assert filtered['success'].all()
    assert filtered['q'][0,0]==0


def test_partial_selection_reports_deficit_instead_of_padding():
    b=np.repeat(np.arange(10),10);fraction=(b+.5)/10
    data=dict(q=np.arange(100)[:,None]*.03,pose_index=np.zeros(100,int),success=np.ones(100,bool),
        target_fraction=fraction,tool_state=fraction.copy())
    data['success'][-2:]=False
    ids=balanced_angle_success_indices(data,[0],100,0,[0,1],allow_partial=True)
    assert len(ids)==98 and data['success'][ids].all()
    assert np.bincount(b[ids],minlength=10).tolist()==[10]*9+[8]


def test_all_fixed_splits_must_be_satisfied():
    rng=np.random.default_rng(1)
    pose=np.repeat(np.arange(32),20);fraction=np.tile((np.arange(20)+.5)/20,32)
    data=dict(q=rng.normal(size=(640,13)),initial_q=np.zeros((640,13)),command_q=np.zeros((640,13)),
        pose_index=pose,target_fraction=fraction,tool_state=fraction.copy(),success=np.ones(640,bool))
    results,deficits=inspect_splits(data,dict(pose_count=32,target_range=[0,1]))
    assert not deficits
    assert len(results)==5 and all(r['complete'] for r in results.values())
    data['success'][fraction>.9]=False
    results,deficits=inspect_splits(data,dict(pose_count=32,target_range=[0,1]))
    assert deficits and all(not r['complete'] for r in results.values())
