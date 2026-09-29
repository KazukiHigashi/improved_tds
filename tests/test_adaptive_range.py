import numpy as np

from improved_tds.synergy.adaptive_range import common_inverse_domain


def test_common_inverse_domain_excludes_both_clamped_ends():
    selection={'map_tool':[.30,.60,1.05],
               'ramp':{'tool_low':.37,'tool_high':.98}}
    assert common_inverse_domain(selection,[0.,1.])==[.37,.98]
    assert common_inverse_domain(None,[0.,1.]) is None
from improved_tds.synergy.adaptive_range import select_interval, posture, fit_ramp, invert_ramp


def test_negative_response_gets_reversed_using_calibration():
    x = np.linspace(-1, 1, 9)
    y = np.tile((1 - x) / 2, (8, 1))
    result = select_interval(x, y, np.ones_like(y, bool), [0, 1], full=True)
    assert result['direction'] == -1
    assert result['rho_start'] > result['rho_end']
    assert result['map_tool'][0] < result['map_tool'][-1]


def test_u_shape_requires_branch_selection():
    x = np.linspace(-2, 2, 17)
    y = np.tile(x*x/4, (8, 1))
    assert select_interval(x, y, np.ones_like(y, bool), [0, 1], full=True) is None
    result = select_interval(x, y, np.ones_like(y, bool), [0, 1])
    assert result is not None
    assert result['rho_start'] * result['rho_end'] >= 0


def test_flat_or_inconsistent_or_unheld_is_rejected():
    x = np.linspace(-1, 1, 9)
    y = np.tile((x+1)/2, (8, 1))
    assert select_interval(x, y*1e-5, np.ones_like(y, bool), [0, 1]) is None
    y[4:] = 1-y[4:]
    assert select_interval(x, y, np.ones_like(y, bool), [0, 1]) is None
    y = np.tile((x+1)/2, (8, 1))
    valid = np.ones_like(y, bool); valid[:3] = False
    assert select_interval(x, y, valid, [0, 1]) is None


def test_expansion_preserves_joint_bounds():
    assert np.allclose(posture(np.zeros(2), np.array([1., .25]), 1., 2.,
                               np.array([-1., -1.]), np.ones(2)), [1., .5])


def test_nonuniform_grid_maps_actual_coordinates():
    x = np.array([-2., -.5, 0., .25, 1.])
    y = np.tile((x+2)/3, (8, 1))
    result = select_interval(x, y, np.ones_like(y, bool), [0, 1], full=True)
    assert np.allclose(result['map_u'], -1+2*(x+2)/3)
    assert result['calibration_interpolated_normalised_mae'] < 1e-12


def test_ramp_and_isotonic_use_same_selected_calibration_grid():
    x=np.linspace(-1.,1.,9)
    y=np.tile(np.clip((x+.5)/1.,0.,1.),(8,1))
    selection=select_interval(x,y,np.ones_like(y,bool),[0,1],full=True)
    assert selection is not None
    ramp=selection['ramp']
    assert ramp is not None and ramp['calibration_rmse']<1e-10
    assert abs(invert_ramp(ramp,.25)-(-.25))<.05
    assert np.all(np.diff(ramp['fitted'])>=-1e-12)
