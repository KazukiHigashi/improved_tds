"""Direct reconstructed-posture PCA/PLS control, with no preparation offset.

Fit200 is success-only and angle-balanced. PCA/PLS see the global mean-centred
measured final joint postures. Every sweep and target trial resets an authored
initial pose and directly moves to mean_q + rho*scale*operation_axis (+ the
fixed second-axis coordinate for a two-DoF trial) over five seconds.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from improved_tds.environments.scissors_smooth import SmoothReachConfig
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds, _jsonable
from improved_tds.experiments.reach_tds import load_database, physical_levels
from improved_tds.experiments.simple_synergy import direct_trajectory, matrices, coverage
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.adaptive_range import amplitude, select_interval, invert_ramp, common_inverse_domain
from improved_tds.synergy.two_axis import fit_two_axis_synergy


def reconstructed_posture(mean_q, operation, op_scale, rho, grasp, grasp_scale, g, low, high):
    """Absolute posture in the success-pose coordinate system, then joint limits."""
    return np.clip(mean_q + float(rho)*op_scale*operation + float(g)*grasp_scale*grasp,
                   low, high)


def configure_scissors_evaluation_palm_radius(env, multiplier, source_radius=None):
    """Scale only the offline TDS sphere; leave training and DB rules unchanged."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Scissors evaluation palm radius multiplier must be positive and finite')
    raw=env.unwrapped
    if not hasattr(raw,'_tds_palm_sphere_local_center'):
        raise TypeError('Scissors evaluation requires the palm-attached sphere')
    original=float(raw.grasp_threshold)
    if source_radius is not None and not np.isclose(original,float(source_radius),rtol=0,atol=1e-8):
        raise ValueError('source DB palm radius differs from reconstructed evaluation environment')
    raw.grasp_threshold=original*float(multiplier)
    return original,float(raw.grasp_threshold)


def summarise(config, target_range):
    tests=config['test']
    outcomes=[o for p in tests for o in p['diagnostic']['outcomes']]
    trials=[t for p in tests for t in p['target_trials']]
    by_controller={}
    for name in sorted({t['controller'] for t in trials}):
        group=[t for t in trials if t['controller']==name]
        observed=[t['abs_error'] for t in group if t['abs_error'] is not None]
        held=[t['abs_error'] for t in group if t['abs_error'] is not None and t['outcome']['valid']]
        by_controller[name]=dict(trials=len(group), observed=len(observed), held=len(held),
            mae=float(np.mean(observed)) if observed else None,
            held_mae=float(np.mean(held)) if held else None)
    return dict(independent_coverage=float(np.mean([p['diagnostic_coverage'] for p in tests])),
        continuous_coverage=float(np.mean([p['coverage'] for p in tests])),
        held_fraction=float(np.mean([o is not None and o['valid'] for o in outcomes])),
        short_window_fraction=float(np.mean([o is not None and o['short_window'] is not None
            and o['short_window']['valid'] for o in outcomes])),
        inverse_available=config.get('inverse_target_range') is not None,
        inverse_domain_fraction=(float(np.diff(config['inverse_target_range'])[0]
            /np.diff(target_range)[0]) if config.get('inverse_target_range') is not None else 0.),
        controller_comparison=by_controller)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--task',choices=[f'{category}_lite_{i}' for category in ('push','spray','scissors')
        for i in range(1,6)],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--poses',type=int,default=8)
    p.add_argument('--points',type=int,default=17)
    p.add_argument('--render',action='store_true')
    p.add_argument('--scissors-evaluation-palm-radius-multiplier',type=float,default=1.,
        help='Offline Scissors Calibration/Test palm-sphere radius scale; RL and DB are unchanged')
    a=p.parse_args(argv)
    if a.poses<1 or a.points<3:
        p.error('positive pose count and at least three sweep points required')
    if (not np.isfinite(a.scissors_evaluation_palm_radius_multiplier)
            or a.scissors_evaluation_palm_radius_multiplier<=0
            or (not a.task.startswith('scissors') and a.scissors_evaluation_palm_radius_multiplier!=1.)):
        p.error('Scissors-only palm radius multiplier must be positive and finite')
    dbpath=a.source_root/a.task/'success_postures.npz'
    data,meta=load_database(dbpath)
    a.output.mkdir(parents=True,exist_ok=True)
    contract=meta.get('contract',{})
    version=meta.get('scissors_control_version','simple')
    smooth=(SmoothReachConfig(float(contract.get('action_delta_fraction',.05)),
        float(contract.get('request_gap_penalty_weight',1.))) if version=='smooth' else None)
    _register_tool_manip_gym_environments()
    env,pool,_,key,lo,hi=_build_env(meta['env_id'],a.source_root/'pose_pool',a.seed,
        research_task=True,scissors_control_version=version,scissors_smooth_config=smooth,
        tool_action_slew_rate=float(contract.get('tool_action_slew_rate',0.)),
        tool_request_gap_penalty_weight=float(contract.get('tool_request_gap_penalty_weight',1.)),
        object_position_noise=0.,max_episode_steps=10000,
        force_floor_contact=a.task.startswith('spray'),
        render_mode='rgb_array' if a.render else None)
    source_palm_radius=evaluation_palm_radius=None
    if a.task.startswith('scissors'):
        source_palm_radius,evaluation_palm_radius=configure_scissors_evaluation_palm_radius(
            env,a.scissors_evaluation_palm_radius_multiplier,contract.get('palm_sphere_radius'))
    support_criterion=('task' if a.task.startswith('scissors') else 'no_fall')
    require_floor_contact=a.task.startswith('spray')
    def run_direct(pose, target, destinations, *, capture=None):
        return direct_trajectory(env,pose,key,target,a.seed+1000*pose,destinations,
            capture=capture,support_criterion=support_criterion,
            require_floor_contact=require_floor_contact)
    order=np.random.default_rng(a.seed).permutation(pool.pose_count)
    if 2*a.poses>len(order):
        env.close()
        raise ValueError('calibration and test initial poses must be disjoint')
    calposes=order[:a.poses].tolist()
    testposes=order[a.poses:2*a.poses].tolist()
    low,high,_=_control_bounds(env)
    mean_q=np.mean(data['q'],axis=0)
    axes={method:fit_two_axis_synergy(data['q'],data['tool_state'],data['pose_index'],
        method=method,grasp_metric=data['grasp_metric'],centering='global')
        for method in ('pca','pls')}
    result=dict(protocol=('reach_tds_direct_v3_scaled_palm_evaluation'
        if a.task.startswith('scissors') and a.scissors_evaluation_palm_radius_multiplier!=1.
        else 'reach_tds_direct_v2_no_contact_gate'),task=a.task,seed=a.seed,
        database_sha256=hashlib.sha256(dbpath.read_bytes()).hexdigest(),
        database_metadata=meta,fit_rows=len(data['q']),mean_q=mean_q.tolist(),
        evaluation_grasp_criterion=('no_drop_or_tip_for_2s' if support_criterion=='no_fall'
            else ('hinge_center_inside_scaled_palm_attached_sphere'
                if a.scissors_evaluation_palm_radius_multiplier!=1.
                else contract.get('grasp_criterion'))),
        source_policy_grasp_criterion=contract.get('grasp_criterion'),
        evaluation_support_criterion=support_criterion,
        floor_contact_forced_enabled=require_floor_contact,
        fit_database_grasp_criterion=contract.get('grasp_criterion','historical_contact_gated'),
        source_policy_palm_sphere_radius=source_palm_radius,
        palm_sphere_radius=evaluation_palm_radius,
        scissors_evaluation_palm_radius_multiplier=(a.scissors_evaluation_palm_radius_multiplier
            if a.task.startswith('scissors') else None),
        target_range=[lo,hi],calibration_poses=calposes,test_poses=testposes,
        axis_cosine=float(axes['pca'].operation_axis@axes['pls'].operation_axis),
        trajectory_seconds=5.,operation_hold_seconds=2.,
        fixed_preparation=False,online_rl=False,state_restoration=False,
        unseen_pose_evaluation=False,configurations={})
    def progress(phase):
        (a.output/'progress.json').write_text(json.dumps({'phase':phase}),encoding='utf-8')
        print(phase,flush=True)
    try:
        for method,fit in axes.items():
            op,grasp=fit.operation_axis,fit.grasp_axis
            op_scale,g_scale=amplitude(op,low,high),amplitude(grasp,low,high)
            score=(data['q']-mean_q)@op/op_scale
            levels=physical_levels(score,a.points)
            for dof in (1,2):
                label=f'{method}_{dof}d'
                progress(label)
                def destination(rho,g):
                    return reconstructed_posture(mean_q,op,op_scale,rho,grasp,g_scale,g,low,high)
                gcandidates=[]
                for g in ([0.] if dof==1 else [-1.,-.5,0.,.5,1.]):
                    records=[run_direct(pose,lo+.3*(hi-lo),
                        [destination(0.,g)])['outcomes'][0] for pose in calposes]
                    gcandidates.append(dict(g=g,valid_count=sum(o['valid'] for o in records),records=records))
                chosen=max(gcandidates,key=lambda c:(c['valid_count'],-abs(c['g'])))
                g=chosen['g']
                def independent(pose,capture=None):
                    trials=[]
                    for j,rho in enumerate(levels):
                        rec=run_direct(pose,lo+.3*(hi-lo),[destination(rho,g)])
                        trials.append(rec['outcomes'][0])
                        if capture is not None:
                            capture(j,rho,rec)
                    return dict(outcomes=trials)
                cal=[independent(pose) for pose in calposes]
                yy,valid=matrices(cal,len(levels))
                selection=select_interval(levels,yy,valid,[lo,hi])
                inverse_range=common_inverse_domain(selection,[lo,hi])
                sweep=(np.linspace(selection['rho_start'],selection['rho_end'],13)
                    if selection else levels)
                path=np.r_[sweep,sweep[::-1]]
                coef=np.polyfit(data['tool_state'],score,1)
                config=dict(method=method,dof=dof,fit=asdict(fit),operation_scale=op_scale,
                    grasp_scale=g_scale,grasp_candidates=gcandidates,grasp_score=g,
                    fit_rho=score.tolist(),fit_tool=data['tool_state'].tolist(),
                    diagnostic_levels=levels.tolist(),levels=path.tolist(),
                    calibration=cal,selection=selection,
                    inverse_target_range=inverse_range,
                    excluded_target_ranges=([lo,inverse_range[0]],[inverse_range[1],hi])
                    if inverse_range is not None else None,
                    database_linear_coefficients=coef.tolist(),test=[],images={})
                frames=[]
                if a.render:
                    from PIL import Image
                    from improved_tds.experiments.evaluate_synergy_sweep_coverage import _caption,_contact_sheet
                    def frame(title,subtitle):
                        raw=Image.fromarray(env.render().copy())
                        raw.thumbnail((360,270))
                        return _caption(np.asarray(raw),[f'{a.task} / {label} / direct',title,subtitle])
                    def sheets(name,items):
                        names=[]
                        for k,start in enumerate(range(0,len(items),24)):
                            fname=f'{label}_{name}_{k+1}.png'
                            _contact_sheet(items[start:start+24],f'{label}: {name} / page {k+1}').save(a.output/fname)
                            names.append(fname)
                        config['images'][name]=names
                for ip,pose in enumerate(testposes):
                    render=a.render and ip==0
                    def capture(step,obs,info):
                        frames.append(frame(f'Continuous direct sweep: step {step}',
                            f'tool={float(obs["achieved_goal"][0]):.3f}; '
                            f'contact_gate={bool(info.get("grasp_maintained"))}; '
                            f'tip={bool(info.get("object_tipped",info.get("bottle_tipped",False)))}'))
                    tr=run_direct(pose,lo+.3*(hi-lo),
                        [destination(rho,g) for rho in path],capture=capture if render else None)
                    sweepframes=[]
                    def endpoint(j,rho,rec):
                        o=rec['outcomes'][0]
                        sweepframes.append(frame(f'Direct score={rho:+.3f}',
                            f'tool={o["achieved"]:.3f}; held2s={o["valid"]}'))
                    diag=independent(pose,endpoint if render else None)
                    targets=[]
                    targetframes=[]
                    target_grid=np.linspace(*inverse_range,9) if inverse_range else np.linspace(lo,hi,9)
                    for target in target_grid:
                        for controller in (('isotonic','ramp') if inverse_range else ('database_linear_baseline',)):
                            if controller=='isotonic':
                                u=float(np.interp(target,selection['map_tool'],selection['map_u']))
                            elif controller=='ramp':
                                u=invert_ramp(selection['ramp'],target)
                            else:
                                u=None
                            rho=(selection['rho_start']+(u+1)/2*(selection['rho_end']-selection['rho_start'])
                                if controller!='database_linear_baseline' else
                                float(np.clip(np.polyval(coef,target),levels[0],levels[-1])))
                            def target_capture(step,obs,info):
                                targetframes.append(frame(f'{controller}: target={target:.3f}; step {step}',
                                    f'tool={float(obs["achieved_goal"][0]):.3f}; score={rho:+.3f}'))
                            rec=run_direct(pose,float(target),
                                [destination(rho,g)],capture=target_capture if render and any(
                                    np.isclose(target,x) for x in (lo,(lo+hi)/2,hi)) else None)
                            o=rec['outcomes'][0]
                            observed=not o['failure_numerical'] and np.isfinite(o['achieved'])
                            bounds=(selection['map_tool'] if controller=='isotonic' else
                                [selection['ramp']['tool_low'],selection['ramp']['tool_high']]
                                if controller=='ramp' else None)
                            targets.append(dict(target=float(target),rho=float(rho),u=u,controller=controller,
                                outcome=o,abs_error=abs(o['achieved']-target) if observed else None,
                                outside_calibrated_range=bool(target<bounds[0] or target>bounds[-1])
                                if bounds else None))
                    tr.update(pose=pose,diagnostic=diag,target_trials=targets,
                        coverage=coverage(tr['outcomes'],lo,hi),
                        diagnostic_coverage=coverage(diag['outcomes'],lo,hi))
                    config['test'].append(tr)
                    if render:
                        sheets('independent_postures',sweepframes)
                        sheets('target_every20',targetframes)
                if a.render:
                    sheets('continuous_every20',frames)
                config['summary']=summarise(config,[lo,hi])
                result['configurations'][label]=config
                (a.output/'report.partial.json').write_text(json.dumps(result,default=_jsonable),encoding='utf-8')
        (a.output/'report.json').write_text(json.dumps(result,default=_jsonable),encoding='utf-8')
        progress('completed')
    finally:
        env.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
