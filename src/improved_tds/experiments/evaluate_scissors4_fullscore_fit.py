"""Fit the entire Scissors-4 score span, then test unseen midpoint scores.

Only held calibration observations contribute to a fit. No contiguous-region
selection is performed: the stable low-score island is retained, and the
unstable gap is interpolated. Test scores are midpoints between calibration
levels, evaluated by direct five-second trajectories from disjoint test poses.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from improved_tds.environments.scissors_smooth import SmoothReachConfig
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds, _jsonable
from improved_tds.experiments.simple_synergy import direct_trajectory
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.adaptive_range import fit_ramp


def fit_full_span(levels, records):
    """Return three forward maps fitted to every *held* calibration sample."""
    pairs=[(float(levels[j]),float(outcome['achieved']))
           for record in records for j,outcome in enumerate(record['outcomes'])
           if outcome is not None and outcome['valid'] and np.isfinite(outcome['achieved'])]
    if len(pairs)<3:
        raise ValueError('at least three held calibration observations required')
    pairs.sort()
    x,y=np.asarray(pairs,dtype=float).T
    unique,inv,counts=np.unique(x,return_inverse=True,return_counts=True)
    means=np.bincount(inv,weights=y)/counts
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x,y)
    u=-1.+2.*(x-levels[0])/(levels[-1]-levels[0])
    ramp=fit_ramp(u,y)
    if ramp is None:
        raise ValueError('ramp could not be fitted to held full-span observations')

    def predict(rho):
        rho=float(rho)
        urho=-1.+2.*(rho-levels[0])/(levels[-1]-levels[0])
        linear=float(np.interp(rho,unique,means))
        isotonic=float(iso.predict([rho])[0])
        ramp_y=float(ramp['tool_low']+(ramp['tool_high']-ramp['tool_low'])*
                     np.clip((urho-ramp['start'])/(ramp['end']-ramp['start']),0.,1.))
        return dict(linear=linear,isotonic=isotonic,ramp=ramp_y)

    metadata=dict(calibration_held=len(pairs),calibration_total=len(records)*len(levels),
        level_counts=[int(np.sum(x==level)) for level in levels],
        isotonic_at_levels=[predict(level)['isotonic'] for level in levels],
        ramp=ramp)
    metadata['raw_means']=[float(means[np.flatnonzero(unique==level)[0]])
                           if np.any(unique==level) else None for level in levels]
    return predict,metadata


def summarise(trials):
    output={}
    for name in ('linear','isotonic','ramp'):
        observed=[abs(t['predicted'][name]-t['achieved']) for t in trials
                  if t['achieved'] is not None]
        held=[abs(t['predicted'][name]-t['achieved']) for t in trials
              if t['achieved'] is not None and t['valid']]
        output[name]=dict(trials=len(trials),observed=len(observed),held=len(held),
            observed_mae=float(np.mean(observed)) if observed else None,
            held_mae=float(np.mean(held)) if held else None)
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-analysis',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError(a.output)
    source=json.loads((a.source_analysis/'results/scissors_lite_4'/str(a.seed)/'report.json')
                      .read_text(encoding='utf-8'))
    dbmeta=source['database_metadata']
    _register_tool_manip_gym_environments()
    contract=dbmeta['contract']
    smooth=SmoothReachConfig(float(contract['action_delta_fraction']),
                             float(contract['request_gap_penalty_weight']))
    env,pool,_,key,lo,hi=_build_env(dbmeta['env_id'],a.source_run/'pose_pool',a.seed,
        research_task=True,scissors_control_version='smooth',scissors_smooth_config=smooth,
        object_position_noise=0.,max_episode_steps=10000)
    low,high,_=_control_bounds(env)
    output=dict(protocol='scissors4_fullscore_unseen_midpoints_v1',seed=a.seed,
        source_report=str(a.source_analysis/'results/scissors_lite_4'/str(a.seed)/'report.json'),
        fit_rows=source['fit_rows'],calibration_poses=source['calibration_poses'],
        test_poses=source['test_poses'],trajectory_seconds=5.,hold_seconds=2.,
        task_target_range=[lo,hi],methods={})
    try:
        for method in ('pca','pls'):
            c=source['configurations'][method+'_1d']
            levels=np.asarray(c['diagnostic_levels'],dtype=float)
            fit,fit_meta=fit_full_span(levels,c['calibration'])
            operation=np.asarray(c['fit']['operation_axis'],dtype=float)
            mean_q=np.asarray(source['mean_q'],dtype=float)
            scale=float(c['operation_scale'])
            unseen=(levels[:-1]+levels[1:])/2
            trials=[]
            for pose in source['test_poses']:
                for j,rho in enumerate(unseen):
                    destination=np.clip(mean_q+float(rho)*scale*operation,low,high)
                    record=direct_trajectory(env,pose,key,lo+.3*(hi-lo),
                        a.seed+1000*pose,[destination])['outcomes'][0]
                    achieved=(float(record['achieved']) if record is not None
                        and not record['failure_numerical'] and np.isfinite(record['achieved']) else None)
                    trials.append(dict(pose=int(pose),midpoint_index=j,score=float(rho),
                        predicted=fit(rho),achieved=achieved,
                        valid=bool(record is not None and record['valid']),
                        held=bool(record is not None and record['held']),
                        failure_numerical=bool(record is not None and record['failure_numerical'])))
            output['methods'][method]=dict(calibration_levels=levels.tolist(),
                unseen_scores=unseen.tolist(),fit=fit_meta,trials=trials,
                summary=summarise(trials))
    finally:
        env.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,default=_jsonable,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
