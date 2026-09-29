import numpy as np
import pytest
from improved_tds.experiments.reach_tds import physical_levels, errors, load_database, summarise


def test_physical_range_uses_1_5_times_observed_extrema():
    x=physical_levels(np.array([-3.,.5]))
    assert len(x)==17 and x[0]==-4.5 and x[-1]==.75
    with pytest.raises(ValueError,match='straddle zero'):
        physical_levels(np.array([.1,.5]))


def test_errors_missing_not_zero_and_held_separate():
    t=[dict(abs_error=None,outcome=None),dict(abs_error=.2,outcome=dict(valid=False)),
       dict(abs_error=.1,outcome=dict(valid=True))]
    assert errors(t)==[.2,.1]
    assert errors(t,True)==[.1]


def test_previous_100_row_databases_are_not_silently_reused():
    from pathlib import Path
    root=Path('runs/scissors_reach_20260923')
    if not root.exists():pytest.skip('integration artifacts absent')
    for i in (3,4,5):
        with pytest.raises(ValueError,match='200-row'):
            load_database(root/f'scissors_lite_{i}'/'success_postures.npz')


def test_short_coverage_not_substituted_for_held_coverage():
    points=[dict(achieved=v,valid=False,short_window=dict(achieved=v,valid=True)) for v in (0.,1.)]
    c=dict(selection=None,test=[dict(coverage=0.,diagnostic_coverage=0.,diagnostic=dict(outcomes=points),target_trials=[])])
    s=summarise(c,[0,1])
    assert s['continuous_coverage']==0 and s['independent_coverage']==0
    assert s['short_window_coverage']==1 and s['mae'] is None


def test_load_database_uses_category_target_units(tmp_path):
    import json
    low, high, tolerance = 0., .008, .0006
    achieved=np.repeat((np.arange(10)+.5)/10*(high-low),20)
    target=achieved.copy()
    target[0]=low
    target[-1]=high
    path=tmp_path/'push_successes.npz'
    np.savez_compressed(path,q=np.zeros((200,2)),initial_q=np.zeros((200,2)),
        command_q=np.zeros((200,2)),tool_state=achieved,target=target,
        success=np.ones(200,dtype=bool),hold_steps=np.full(200,5),
        metadata_json=json.dumps(dict(complete=True,target_range=[low,high],
            contract=dict(success_tolerance=tolerance))))
    data,meta=load_database(path)
    assert len(data['q'])==200 and meta['target_range']==[low,high]
