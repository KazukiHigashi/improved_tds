"""PCA/PLS 1/2-DoF FF evaluation using the completed reach-success databases.

Uses the previous fixed-preparation/independent and continuous sweep protocol.
RL is never invoked during control. Calibration/Test resets are disjoint, but
Fit200 was collected from the whole authored pool: this is NOT unseen-pose CV.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.environments.scissors_smooth import SmoothReachConfig
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds, _jsonable
from improved_tds.experiments.simple_synergy import (
    trajectory, preparation_candidates, matrices, coverage,
)
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.adaptive_range import amplitude, select_interval, correlation, invert_ramp
from improved_tds.synergy.two_axis import fit_two_axis_synergy


def load_database(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files if k != 'metadata_json'}
        meta = json.loads(z['metadata_json'].item())
    if not meta.get('complete') or len(data['q']) != 200:
        raise ValueError('A completed 200-row database is required')
    if not data['success'].all() or np.min(data['hold_steps']) < 5:
        raise ValueError('All rows must be five-step successes')
    for name in ('q', 'initial_q', 'command_q', 'tool_state', 'target'):
        if not np.isfinite(data[name]).all():
            raise ValueError('Nonfinite database: ' + name)
    contract=meta.get('contract',{})
    tolerance=float(contract.get('success_tolerance',.03))
    if np.max(abs(data['tool_state'] - data['target'])) > tolerance+1e-6:
        raise ValueError('Database contains unsuccessful target errors')
    low,high=(float(x) for x in meta.get('target_range',[0.,1.]))
    if not low<high:
        raise ValueError('Invalid database target range')
    bins=np.minimum((np.clip((data['tool_state']-low)/(high-low),0,1)*10).astype(int),9)
    if not np.all(np.bincount(bins,minlength=10)==20):
        raise ValueError('Expected twenty successful rows in each achieved-angle bin')
    if not np.any(np.isclose(data['target'],low)) or not np.any(np.isclose(data['target'],high)):
        raise ValueError('Both endpoint targets must be represented')
    return data, meta


def physical_levels(scores, count=17):
    """Expand both observed score endpoints by 1.5, without fixed constants."""
    scores=np.asarray(scores,dtype=float)
    if scores.size<2 or not np.isfinite(scores).all() or count<3:
        raise ValueError('finite scores and at least three sweep points required')
    low,high=float(np.min(scores)),float(np.max(scores))
    if not low<0<high:
        raise ValueError('score endpoints must straddle zero for the requested 1.5x expansion')
    return np.linspace(1.5*low,1.5*high,count)


def errors(trials, valid_only=False):
    return [t['abs_error'] for t in trials if t['abs_error'] is not None
            and (not valid_only or t['outcome']['valid'])]


def summarise(config, target_range):
    tests = config['test']
    trials = [t for p in tests for t in p['target_trials']]
    ee, vv = errors(trials), errors(trials, True)
    independent = [x for p in tests for x in p['diagnostic']['outcomes']]
    def short_cov(p):
        return coverage([x['short_window'] for x in p['diagnostic']['outcomes']
                         if x is not None and x['short_window'] is not None], *target_range)
    by_controller={}
    for controller in sorted({t['controller'] for t in trials}):
        group=[t for t in trials if t['controller']==controller]
        all_errors=errors(group)
        held_errors=errors(group,True)
        by_controller[controller]=dict(trials=len(group),observed=len(all_errors),held=len(held_errors),
            mae=float(np.mean(all_errors)) if all_errors else None,
            held_mae=float(np.mean(held_errors)) if held_errors else None)
    return dict(continuous_coverage=float(np.mean([p['coverage'] for p in tests])),
                independent_coverage=float(np.mean([p['diagnostic_coverage'] for p in tests])),
                short_window_coverage=float(np.mean([short_cov(p) for p in tests])),
                held_fraction=sum(x is not None and x['valid'] for x in independent)/len(independent),
                target_trials=len(trials), observed_errors=len(ee), held_errors=len(vv),
                mae=float(np.mean(ee)) if ee else None,
                held_mae=float(np.mean(vv)) if vv else None,
                inverse_available=config['selection'] is not None,
                controller_comparison=by_controller)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--task', required=True, choices=['scissors_lite_'+str(i) for i in (3,4,5)])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--poses', type=int, default=8)
    p.add_argument('--points', type=int, default=17)
    p.add_argument('--render', action='store_true')
    a = p.parse_args(argv)
    if a.poses < 1 or a.points < 3:
        raise ValueError('positive poses and >=3 sweep points required')
    root = a.output
    root.mkdir(parents=True, exist_ok=True)
    dbpath = a.source_root/a.task/'success_postures.npz'
    data, meta = load_database(dbpath)
    env_id = meta['env_id']
    control_version = meta.get('scissors_control_version', 'simple')
    if control_version not in ('simple', 'smooth'):
        raise ValueError(f'Unsupported reach database control version: {control_version}')
    contract = meta.get('contract', {})
    smooth_config = (SmoothReachConfig(
        action_delta_fraction=float(contract.get('action_delta_fraction', 0.05)),
        request_gap_penalty_weight=float(contract.get('request_gap_penalty_weight', 1.0)),
    ) if control_version == 'smooth' else None)
    _register_tool_manip_gym_environments()
    env, pool, _, key, lo, hi = _build_env(
        env_id, a.source_root/'pose_pool', a.seed, research_task=True,
        scissors_control_version=control_version, scissors_smooth_config=smooth_config,
        object_position_noise=0.,
        max_episode_steps=10000, render_mode='rgb_array' if a.render else None)
    order = np.random.default_rng(a.seed).permutation(pool.pose_count)
    if 2*a.poses > len(order):
        raise ValueError('Calibration and Test must be disjoint')
    calposes = order[:a.poses].tolist()
    testposes = order[a.poses:2*a.poses].tolist()
    result = dict(protocol='reach_tds_v1', task=a.task, seed=a.seed,
        scissors_control_version=control_version,
        smooth_config=asdict(smooth_config) if smooth_config is not None else None,
        env_id=env_id, target_range=[lo,hi], fit_rows=200,
        database_sha256=hashlib.sha256(dbpath.read_bytes()).hexdigest(),
        database_metadata=meta, calibration_poses=calposes, test_poses=testposes,
        unseen_pose_evaluation=False, online_rl=False, state_restoration=False,
        test_retries=0, preparation_ramp_seconds=5., operation_ramp_seconds=5.,
        preparation_hold_seconds=1., operation_hold_seconds=2.,
        short_window_steps=5, configurations={})
    def progress(phase):
        (root/'progress.json').write_text(json.dumps({'phase':phase}), encoding='utf-8')
        print(phase, flush=True)
    try:
        progress('fixed_preparation_calibration')
        offsets, rows = preparation_candidates(data, calposes)
        candidates = []
        for index, offset in enumerate(offsets):
            for duration in (5.,):
                rr = [trajectory(env, pose, key, lo+.3*(hi-lo), a.seed+1000*pose,
                      offset, duration, [])['preparation'] for pose in calposes]
                candidates.append(dict(index=index, duration=duration, offset=offset.tolist(),
                    valid_count=sum(x['valid'] for x in rr), records=rr))
        prep = max(candidates, key=lambda c:(c['valid_count'], c['duration'], -np.linalg.norm(c['offset'])))
        base_offset = np.array(prep['offset'])
        result.update(preparation=prep, preparation_candidates=candidates,
                      preparation_source_rows=rows)
        low, high, _ = _control_bounds(env)
        axes = {m:fit_two_axis_synergy(data['q'], data['tool_state'], data['pose_index'],
                method=m, grasp_metric=data['grasp_metric'], centering='global') for m in ('pca','pls')}
        result['axis_cosine'] = float(axes['pca'].operation_axis @ axes['pls'].operation_axis)
        for method, fit in axes.items():
            op, grasp = fit.operation_axis, fit.grasp_axis
            op_scale, g_scale = amplitude(op, low, high), amplitude(grasp, low, high)
            for dof in (1,2):
                label = f'{method}_{dof}d'
                progress(label)
                gcandidates = []
                for g in ([0.] if dof == 1 else [-1.,-.5,0.,.5,1.]):
                    offset = base_offset+g*g_scale*grasp
                    rr = [trajectory(env, pose, key, lo+.3*(hi-lo), a.seed+1000*pose,
                          offset, prep['duration'], [])['preparation'] for pose in calposes]
                    gcandidates.append(dict(g=g, valid_count=sum(x['valid'] for x in rr), records=rr))
                gg = max(gcandidates, key=lambda c:(c['valid_count'], -abs(c['g'])))
                offset = base_offset+gg['g']*g_scale*grasp
                fit_rho = (data['q']-data['initial_q']-offset) @ op/op_scale
                levels = physical_levels(fit_rho, a.points)
                def independent(pose, capture=None):
                    trials=[]
                    for j,rho in enumerate(levels):
                        tr=trajectory(env,pose,key,lo+.3*(hi-lo),a.seed+1000*pose,
                            offset,prep['duration'],[rho*op_scale*op])
                        trials.append(tr['outcomes'][0])
                        if capture is not None:
                            capture(j,rho,tr)
                    return dict(outcomes=trials)
                cal=[independent(pose) for pose in calposes]
                yy,valid=matrices(cal,len(levels))
                selection=select_interval(levels,yy,valid,[lo,hi])
                sweep=np.linspace(selection['rho_start'],selection['rho_end'],13) if selection else levels
                path=np.r_[sweep,sweep[::-1]]
                config=dict(method=method,dof=dof,fit=asdict(fit),operation_scale=op_scale,
                    grasp_scale=g_scale,grasp_candidates=gcandidates,grasp_score=gg['g'],
                    offset=offset.tolist(),fit_rho=fit_rho.tolist(),fit_tool=data['tool_state'].tolist(),
                    diagnostic_levels=levels.tolist(),levels=path.tolist(),
                    calibration=cal,selection=selection,test=[],images={})
                # An explicitly labelled DB-only linear FF baseline ensures target
                # errors are measured even when physical Calibration rejects an inverse.
                coef=np.polyfit(data['tool_state'],fit_rho,1)
                config['target_controller']='isotonic_vs_ramp' if selection else 'database_linear_baseline'
                config['database_linear_coefficients']=coef.tolist()
                frames=[]
                from improved_tds.experiments.evaluate_synergy_sweep_coverage import _caption, _contact_sheet
                def frame(title,subtitle):
                    from PIL import Image
                    raw=Image.fromarray(env.render().copy())
                    raw.thumbnail((360,270))
                    return _caption(np.asarray(raw),[f'{a.task} / {method.upper()} / {dof} DoF',title,subtitle])
                def sheets(name,items):
                    names=[]
                    for k,start in enumerate(range(0,len(items),24)):
                        fname=f'{label}_{name}_{k+1}.png'
                        _contact_sheet(items[start:start+24],f'{label}: {name} / page {k+1}').save(root/fname)
                        names.append(fname)
                    config['images'][name]=names
                for ip,pose in enumerate(testposes):
                    render=a.render and ip==0
                    def capture(step,obs,info):
                        frames.append(frame(f'Continuous sweep: step {step}',
                            f'tool={float(obs["achieved_goal"][0]):.3f} rad; grasp={bool(info.get("grasp_maintained"))}'))
                    tr=trajectory(env,pose,key,lo+.3*(hi-lo),a.seed+1000*pose,offset,
                        prep['duration'],[rho*op_scale*op for rho in path],capture=capture if render else None)
                    sweepframes=[]
                    def endpoint(j,rho,trial):
                        item=trial['outcomes'][0]
                        text='not reached (preparation terminated)' if item is None else f'tool={item["achieved"]:.3f}; held2s={item["valid"]}'
                        sweepframes.append(frame(f'Independent operation sweep: rho={rho:+.3f}',text))
                    diag=independent(pose,endpoint if render else None)
                    targets=[]
                    targetframes=[]
                    for target in np.linspace(lo,hi,9):
                        controllers=('isotonic','ramp') if selection else ('database_linear_baseline',)
                        for controller in controllers:
                            if controller=='isotonic':
                                u=float(np.interp(target,selection['map_tool'],selection['map_u']))
                            elif controller=='ramp':
                                u=invert_ramp(selection['ramp'],target)
                            else:
                                u=None
                            rho=(selection['rho_start']+(u+1)/2*(selection['rho_end']-selection['rho_start'])
                                if selection else float(np.clip(np.polyval(coef,target),levels[0],levels[-1])))
                            def target_capture(step,obs,info):
                                targetframes.append(frame(f'Target={target:.3f} rad; rho={rho:+.3f}; step {step}',
                                    f'tool={float(obs["achieved_goal"][0]):.3f}; {controller}'))
                            tt=trajectory(env,pose,key,float(target),a.seed+1000*pose,offset,
                                prep['duration'],[rho*op_scale*op],
                                capture=target_capture if render and any(np.isclose(target,x) for x in (lo,(lo+hi)/2,hi)) else None)
                            item=tt['outcomes'][0]
                            observed=item is not None and not item['failure_numerical'] and np.isfinite(item['achieved'])
                            target_range=(selection['map_tool'] if controller=='isotonic' else
                                [selection['ramp']['tool_low'],selection['ramp']['tool_high']] if controller=='ramp' else None)
                            targets.append(dict(target=float(target),rho=float(rho),u=u,outcome=item,
                                controller=controller,abs_error=abs(item['achieved']-target) if observed else None,
                                outside_calibrated_range=bool(target<target_range[0] or target>target_range[-1]) if target_range else None))
                    tr.update(pose=pose,diagnostic=diag,target_trials=targets,
                        coverage=coverage(tr['outcomes'],lo,hi),diagnostic_coverage=coverage(diag['outcomes'],lo,hi))
                    config['test'].append(tr)
                    if render:
                        sheets('operation_sweep',sweepframes);sheets('target_every20',targetframes)
                sheets('continuous_every20',frames)
                if a.render and dof==2:
                    gf=[]
                    for g in np.linspace(-1,1,9):
                        rr=trajectory(env,testposes[0],key,lo+.3*(hi-lo),a.seed+1000*testposes[0],
                            base_offset+g*g_scale*grasp,prep['duration'],[])
                        gf.append(frame(f'Residual grasp-axis sweep: g={g:+.2f}',
                            f'tool={rr["preparation"]["achieved"]:.3f}; held1s={rr["preparation"]["valid"]}'))
                    sheets('grasp_sweep',gf)
                config['summary']=summarise(config,[lo,hi])
                for cohort,records in [('calibration',cal),('test',[p['diagnostic'] for p in config['test']])]:
                    yy,valid=matrices(records,len(levels))
                    mean=np.array([np.mean(yy[valid[:,j],j]) if valid[:,j].any() else np.nan for j in range(len(levels))])
                    ok=np.isfinite(mean)
                    config[cohort+'_held_correlation']=correlation(levels[ok],mean[ok])
                result['configurations'][label]=config
                (root/'report.partial.json').write_text(json.dumps(result,default=_jsonable),encoding='utf-8')
        (root/'report.json').write_text(json.dumps(result,default=_jsonable),encoding='utf-8')
        progress('completed')
    finally:
        env.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
