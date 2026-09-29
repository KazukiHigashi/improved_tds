import numpy as np
import pytest
from improved_tds.experiments.evaluate_strict_synergy_control import balanced_angle_success_indices


def dataset():
    b=np.repeat(np.arange(10),12)
    fractions=(b+.5)/10
    return dict(q=np.arange(120,dtype=float)[:,None]*np.ones((1,13))*.02,
        pose_index=np.zeros(120,int),success=np.ones(120,bool),
        target_fraction=fractions,tool_state=fractions.copy())


def test_exact_success_only_quotas_reproducible_and_no_test_rows():
    d=dataset()
    d['pose_index'][-2:]=31
    ids=balanced_angle_success_indices(d,[0],100,42,[0,1])
    assert len(ids)==len(set(ids))==100
    assert np.bincount((d['target_fraction'][ids]*10).astype(int)).tolist()==[10]*10
    assert np.all(d['pose_index'][ids]==0)
    np.testing.assert_equal(ids,balanced_angle_success_indices(d,[0],100,42,[0,1]))


@pytest.mark.parametrize('failure',['success','range','bin','duplicate'])
def test_no_silent_padding_or_failed_samples(failure):
    d=dataset()
    if failure=='success':d['success'][:3]=False
    if failure=='range':d['tool_state'][:3]=-.01
    if failure=='bin':d['tool_state'][:3]=.99
    if failure=='duplicate':d['q'][:4]=d['q'][0]
    with pytest.raises(ValueError,match='Insufficient'):
        balanced_angle_success_indices(d,[0],100,42,[0,1])


def test_physical_units_are_normalized_for_push():
    d=dataset();d['tool_state']*=.01
    assert len(balanced_angle_success_indices(d,[0],100,42,[0,.01]))==100
