"""Local integration checks; skipped if the frozen 200-pose bundle is absent."""
import json
from pathlib import Path

import numpy as np
import pytest

from improved_tds.control.axis_feedback import AxisFeedbackConfig
from improved_tds.experiments.second_axis_feedback import inverse_score, run_trial, reset_pristine_simulator


@pytest.mark.simulation
@pytest.mark.parametrize('task',['scissors_lite_3','scissors_lite_4','push_lite_3','spray_lite_3'])
def test_disabled_feedback_matches_ff_and_legacy_trajectory(task):
    import mujoco
    from improved_tds.environments.scissors_smooth import SmoothReachConfig
    from improved_tds.experiments.collect_posture_database import _build_env
    from improved_tds.experiments.reach_tds_direct import configure_scissors_evaluation_palm_radius
    from improved_tds.experiments.simple_synergy import direct_trajectory
    from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
    from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
    bundle=Path(__file__).resolve().parents[1]/'analysis/paper15_paper_bundle_20260928'
    if not (bundle/task/'report.json').exists(): pytest.skip('local frozen bundle unavailable')
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    source=json.loads((bundle/manifest['tasks'][task]['selected_report']).read_text(encoding='utf-8'))
    meta=source['database_metadata']; contract=meta['contract']
    _register_tool_manip_gym_environments()
    env,_,_,key,lo,hi=_build_env(meta['env_id'],bundle/'pose_pool',source['seed'],
        research_task=True,scissors_control_version=meta['scissors_control_version'],
        scissors_smooth_config=SmoothReachConfig(float(contract.get('action_delta_fraction',.05)),
                                                float(contract.get('request_gap_penalty_weight',1.))),
        tool_action_slew_rate=float(contract.get('tool_action_slew_rate',0.)),
        tool_request_gap_penalty_weight=float(contract.get('tool_request_gap_penalty_weight',1.)),
        object_position_noise=0.,max_episode_steps=10000,force_floor_contact=task.startswith('spray'))
    try:
        reset_pristine_simulator(env)
        if task.startswith('scissors'):
            configure_scissors_evaluation_palm_radius(env,source['scissors_evaluation_palm_radius_multiplier'],contract.get('palm_sphere_radius'))
        config=source['configurations']['pca_1d']
        target=float(np.mean(config['inverse_target_range']))
        rho=inverse_score(config,target)
        axes=[np.array(config['fit']['operation_axis'])*config['operation_scale'],
              np.array(config['fit']['grasp_axis'])*config['grasp_scale']]
        pose=source['test_poses'][0]; seed=source['seed']+1000*pose
        mean=np.array(source['mean_q'])
        support='task' if task.startswith('scissors') else 'no_fall'
        floor=task.startswith('spray')
        disabled=dict(enabled=False,reason='disabled_for_test',slope=None,noise=0.)
        settings=dict(tool_span=hi-lo,feedback=AxisFeedbackConfig())
        rows=[run_trial(env,key,pose,seed,target,mean,axes,rho,c,disabled,settings,support,floor)
              for c in ('ff','axis1_fb','axis2_fb')]
        for row in rows[1:]:
            np.testing.assert_allclose([x['tool'] for x in row['trace']],
                                       [x['tool'] for x in rows[0]['trace']],rtol=0,atol=1e-10)
            assert row['elapsed_seconds']==rows[0]['elapsed_seconds']==pytest.approx(7.)
        low,high,_=_control_bounds(env)
        reset_pristine_simulator(env)
        legacy=direct_trajectory(env,pose,key,target,seed,[np.clip(mean+rho*axes[0],low,high)],
                                 support_criterion=support,require_floor_contact=floor)['outcomes'][0]
        assert rows[0]['achieved']==pytest.approx(legacy['achieved'],abs=1e-9)
        assert rows[0]['valid']==legacy['valid']
    finally: env.close()
