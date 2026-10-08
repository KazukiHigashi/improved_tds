import numpy as np
import pytest

from improved_tds.control.axis_feedback import AxisFeedback, AxisFeedbackConfig, estimate_sensitivity
from improved_tds.experiments.second_axis_feedback import (
    additional_clipping, common_targets, inverse_score, summarize_trials, paired_comparisons)


def point(y, **kwargs):
    return dict(valid=True, complete=True, clipped=False, terminal_mean=y,
                terminal_std=0., **kwargs)


def sensitivity(sign=1):
    return estimate_sensitivity([[point(.5-sign*.1),point(.5),point(.5+sign*.1)]]*2,.1,1.)


@pytest.mark.parametrize('sign',[-1,1])
def test_controller_corrects_with_signed_calibration_and_time_scaled_rate(sign):
    controller=AxisFeedback(sensitivity(sign),1.)
    candidate=controller.propose(.9,.1,.02)
    assert candidate*sign>0
    assert abs(candidate)<=.25*.02
    controller.accept(candidate)
    controller.stop('contact_state_changed')
    assert controller.propose(.0,.8,.02)==candidate


def test_controller_does_not_extrapolate_beyond_local_calibration():
    controller=AxisFeedback(sensitivity(),1.)
    for _ in range(1000):
        controller.accept(controller.propose(1.,0.,.02))
    assert controller.offset==pytest.approx(.1)


def test_turning_point_uses_only_a_consistent_side():
    s=estimate_sensitivity([[point(.6),point(.3),point(.7)]]*2,.1,1.)
    assert s['enabled'] and s['reason']=='calibrated_one_side'
    assert s['offset_low']==0. and s['offset_high']==.1


def test_noise_and_pose_sign_disagreement_disable_control():
    assert not estimate_sensitivity([[point(.5),point(.50001),point(.50002)]]*2,.1,1.)['enabled']
    assert not estimate_sensitivity([[point(.4),point(.5),point(.6)],
                                     [point(.6),point(.5),point(.4)]],.1,1.)['enabled']


def test_invalid_calibration_pose_is_excluded_with_provenance():
    bad=point(.4)
    bad['valid']=False
    s=estimate_sensitivity([[bad,point(.5),point(.6)],[point(.4),point(.5),point(.6)]],.1,1.)
    assert s['total_poses']==2 and s['usable_poses']==2 and s['enabled']
    assert s['offset_low']==0. and s['offset_high']==.1
    assert s['excluded_reasons'][0]['reason']=='invalid_or_clipped_probe'


def test_two_good_calibration_poses_remain_visible_among_bad_ones():
    good=[point(.4),point(.5),point(.6)]
    bad=[point(.4),dict(point(.5),valid=False),point(.6)]
    s=estimate_sensitivity([good]*2+[bad]*6,.1,1.)
    assert s['enabled'] and s['usable_poses']==2 and s['total_poses']==8
    assert len(s['excluded_reasons'])==12


def test_minority_sign_is_excluded_and_reported():
    positive=[point(.4),point(.5),point(.6)]
    negative=[point(.6),point(.5),point(.4)]
    s=estimate_sensitivity([positive]*3+[negative],.1,1.)
    assert s['enabled'] and s['usable_poses']==3
    assert sum(r['reason']=='opposite_response_sign' for r in s['excluded_reasons'])==2


def test_equal_positive_negative_votes_do_not_enable_a_side():
    positive=[point(.4),point(.5),point(.6)]
    negative=[point(.6),point(.5),point(.4)]
    s=estimate_sensitivity([positive]*4+[negative]*4,.1,1.)
    assert not s['enabled']
    assert any(r['reason']=='response_sign_tie' for r in s['excluded_reasons'])


def test_only_additional_joint_or_actuator_saturation_blocks_feedback():
    lo,hi=np.array([-1.,-1.]),np.array([1.,1.])
    neutral=np.array([.4,0.]); half=np.array([.5,1.])
    reference=np.array([1.2,.2])
    assert not additional_clipping(reference,reference,lo,hi,neutral,half)
    assert not additional_clipping(np.array([.9,.2]),reference,lo,hi,neutral,half)
    assert additional_clipping(np.array([1.3,.2]),reference,lo,hi,neutral,half)
    assert additional_clipping(np.array([1.2,1.1]),reference,lo,hi,neutral,half)
    assert additional_clipping(np.array([1.2,.95]),reference,lo,hi,neutral,np.array([.5,.5]))


def test_contact_mode_change_is_logged_but_does_not_reject_tool_response():
    probes=[dict(point(.4),terminal_contact_signature=((1,3),)),
            dict(point(.5),terminal_contact_signature=((1,3),)),
            dict(point(.6),terminal_contact_signature=((2,3),))]
    assert estimate_sensitivity([probes]*2,.1,1.)['enabled']


def test_clipped_positive_probe_leaves_negative_feedback_range():
    positive=dict(point(.6),clipped=True)
    s=estimate_sensitivity([[point(.4),point(.5),positive]]*2,.1,1.)
    assert s['enabled'] and s['offset_low']==-.1 and s['offset_high']==0.
    c=AxisFeedback(s,1.)
    assert c.propose(.8,.2,.04)==0.
    assert c.saturated


def test_paired_failure_penalty_prevents_missing_trial_advantage():
    rows=[dict(pose=0,target=.5,condition='ff',valid=True,abs_error=.1),
          dict(pose=0,target=.5,condition='axis1_fb',valid=True,abs_error=.2),
          dict(pose=0,target=.5,condition='axis2_fb',valid=False,abs_error=None)]
    s=paired_comparisons(rows,1.)['ff']
    assert s['jointly_valid']==0 and s['validity_losses']==1
    assert s['all_attempt_penalized_delta_percent']==pytest.approx(90.)


@pytest.mark.parametrize('sign',[-1,1])
def test_synthetic_first_order_plant_reduces_error_with_fixed_gain(sign):
    controller=AxisFeedback(sensitivity(sign),1.)
    y=0.
    for _ in range(100):
        controller.accept(controller.propose(.05,y,.02))
        y += .2*(sign*controller.offset-y)
    assert abs(.05-y)<.015
    assert abs(controller.offset)<=.1


def test_nonfinite_sensor_latches_without_changing_command():
    controller=AxisFeedback(sensitivity(),1.)
    assert controller.propose(.5,np.nan,.02)==0.
    assert controller.stop_reason=='nonfinite_sensor'


def test_pairing_separates_unavailable_axis_from_enabled_comparison():
    rows=[]
    for pose in (0,1):
        for c in ('ff','axis1_fb','axis2_fb'):
            rows.append(dict(pose=pose,target=.5,condition=c,valid=True,abs_error=.1,
                             sensitivity_enabled=c!='ff' and (pose==0 or c=='axis1_fb')))
    strata=paired_comparisons(rows,1.)['axis1_fb']['by_calibration_eligibility']
    assert strata['both_enabled']['pairs']==1 and strata['axis1_only']['pairs']==1


def test_same_targets_and_no_inverse_extrapolation():
    c1=dict(inverse_target_range=[.1,.8],selection=dict(map_tool=[.1,.8],map_u=[-1,1],rho_start=-.4,rho_end=.7))
    c2=dict(inverse_target_range=[.2,.9])
    grid,domain=common_targets([c1,c2],3)
    assert domain==[.2,.8]
    np.testing.assert_allclose(grid,[.35,.5,.65])
    assert inverse_score(c1,.8)==pytest.approx(.7)
    with pytest.raises(ValueError): inverse_score(c1,1.)


def test_invalid_and_missing_trials_remain_in_denominator():
    base=dict(condition='ff',complete=True,stop_reason=None,score_total_variation=0.)
    rows=[dict(base,valid=True,abs_error=.1,achieved=.2),
          dict(base,valid=False,abs_error=.3,achieved=.9),
          dict(base,valid=False,abs_error=None,achieved=None)]
    s=summarize_trials(rows,[0.,1.])['ff']
    assert s['trials']==3 and s['finite']==2 and s['valid']==1
    assert s['all_finite_mae_percent']==pytest.approx(20.)
    assert s['valid_mae_percent']==pytest.approx(10.)
    assert s['valid_fraction_percent']==pytest.approx(100/3)


@pytest.mark.parametrize('kwargs',[{'rate_per_second':0},{'probe_offset':.6},{'gain_per_second':np.nan}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError): AxisFeedbackConfig(**kwargs)
