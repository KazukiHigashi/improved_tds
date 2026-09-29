"""Matched replay of exact successful postures versus PCA/PLS reconstructions.

This diagnostic never changes training or the TDS reports. For each of the
eight TDS test initial poses, it selects one success-DB row from that same pose
closest to midrange target, then replays its measured q, command q, and four
low-dimensional reconstructions with identical reset seed and target.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.reach_tds import load_database
from improved_tds.experiments.reach_tds_direct import reconstructed_posture
from improved_tds.experiments.simple_synergy import direct_trajectory
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.environments.scissors_smooth import SmoothReachConfig


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--analysis-root',type=Path,required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError(a.output)
    dbpath=a.source_root/a.task/'success_postures.npz'
    data,meta=load_database(dbpath)
    report=json.loads((a.analysis_root/a.task/'report.json').read_text(encoding='utf-8'))
    contract=meta['contract']
    _register_tool_manip_gym_environments()
    version=meta.get('scissors_control_version','simple')
    smooth=(SmoothReachConfig(float(contract.get('action_delta_fraction',.05)),
        float(contract.get('request_gap_penalty_weight',1.))) if version=='smooth' else None)
    env,pool,_,key,lo,hi=_build_env(meta['env_id'],a.source_root/'pose_pool',report['seed'],
        research_task=True,scissors_control_version=version,scissors_smooth_config=smooth,
        tool_action_slew_rate=float(contract.get('tool_action_slew_rate',0.)),
        tool_request_gap_penalty_weight=float(contract.get('tool_request_gap_penalty_weight',1.)),
        object_position_noise=0.,max_episode_steps=10000)
    lower,upper,_=_control_bounds(env)
    mean=np.asarray(report['mean_q'])
    results=[]
    try:
        for pose in report['test_poses']:
            ids=np.flatnonzero(data['pose_index']==pose)
            if not len(ids):
                results.append(dict(pose=pose,missing_database_row=True))
                continue
            index=int(ids[np.argmin(abs(data['target_fraction'][ids]-.5))])
            row=data['q'][index]
            target=float(data['target'][index])
            destinations=dict(exact_measured=row,exact_command=data['command_q'][index])
            for label,config in report['configurations'].items():
                fit=config['fit']
                op=np.asarray(fit['operation_axis'])
                grasp=np.asarray(fit['grasp_axis'])
                rho=float((row-mean)@op/config['operation_scale'])
                destinations[label]=reconstructed_posture(mean,op,config['operation_scale'],rho,
                    grasp,config['grasp_scale'],config['grasp_score'],lower,upper)
            trials={}
            for label,destination in destinations.items():
                outcome=direct_trajectory(env,int(pose),key,target,
                    report['seed']+1000*int(pose),[destination])['outcomes'][0]
                trials[label]=dict(valid=bool(outcome['valid']),held=bool(outcome['held']),
                    achieved=float(outcome['achieved']),tool_error_from_database=float(
                        abs(outcome['achieved']-data['tool_state'][index])),
                    target_error=float(abs(outcome['achieved']-target)),
                    hold_fraction=float(outcome['hold_fraction']),
                    ramp_held_fraction=float(outcome['ramp_held_fraction']),
                    terminated=bool(outcome['terminated']),truncated=bool(outcome['truncated']),
                    numerical=bool(outcome['failure_numerical']),
                    reconstructed_q_rms=float(np.sqrt(np.mean((destination-row)**2))))
            results.append(dict(pose=int(pose),database_row=index,
                database_tool=float(data['tool_state'][index]),target=target,trials=trials))
            print(f'{a.task} pose={pose} exact={trials["exact_measured"]["valid"]} '
                  f'pca1={trials["pca_1d"]["valid"]} pls1={trials["pls_1d"]["valid"]}',flush=True)
    finally:
        env.close()
    with dbpath.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    result=dict(protocol='matched_direct_replay_v1',task=a.task,
        database_sha256=digest,tds_seed=report['seed'],test_poses=report['test_poses'],
        selection='one same-pose success row nearest 0.5 target fraction per TDS test pose',
        full_policy_trajectory_replayed=False,
        target_equals_database_row=True,trajectory_seconds=5.,hold_seconds=2.,rows=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
