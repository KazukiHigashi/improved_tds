import numpy as np

from improved_tds.experiments.evaluate_scissors4_fullscore_fit import fit_full_span, summarise


def test_full_span_includes_low_island_and_interpolates_missing_gap():
    levels=np.array([-2.,-1.,0.,1.])
    records=[{'outcomes':[{'achieved':.2,'valid':True},
                          {'achieved':.4,'valid':False},
                          {'achieved':.6,'valid':True},
                          {'achieved':1.,'valid':True}]}]
    predict,meta=fit_full_span(levels,records)
    assert meta['level_counts']==[1,0,1,1]
    assert meta['calibration_held']==3
    assert np.isclose(predict(-1.)['linear'],.4)
    assert np.isfinite(list(predict(-.5).values())).all()


def test_unseen_score_mae_separates_observed_and_held():
    trials=[dict(predicted=dict(linear=.2,isotonic=.2,ramp=.2),achieved=.3,valid=True),
            dict(predicted=dict(linear=.2,isotonic=.2,ramp=.2),achieved=.5,valid=False)]
    summary=summarise(trials)
    assert summary['linear']['trials']==2
    assert summary['linear']['held']==1
    assert np.isclose(summary['linear']['observed_mae'],.2)
    assert np.isclose(summary['linear']['held_mae'],.1)
