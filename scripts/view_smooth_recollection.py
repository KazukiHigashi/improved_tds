"""Replay the completed Scissors smooth success-posture collection in Viewer.

From the improved_tds directory:
    python scripts/view_smooth_recollection.py --variant 4 --episodes 10

The script verifies the frozen Best Policy and pose-pool hashes against the
completed database, then calls the very same preview loop as collection.
Nothing is written to the database. Optional --hold-seconds freezes the
accepted command *after* a successful collection event for diagnosis only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from view_reach_collection import main as view_collection


def view_direct(a, parser):
    """Replay one direct-PCA/PLS trial, using the recorded split and fitted axis."""
    from improved_tds.environments.scissors_smooth import SmoothReachConfig
    from improved_tds.experiments.collect_posture_database import _build_env
    from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
    from improved_tds.experiments.reach_tds_direct import reconstructed_posture
    from improved_tds.experiments.simple_synergy import direct_trajectory
    from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments

    task=f'scissors_lite_{a.variant}'
    report_path=a.direct_root/'results'/task/str(a.split_seed)/'report.json'
    database=a.source_root/task/'success_postures.npz'
    if not report_path.is_file() or not database.is_file():
        parser.error(f'missing direct report or database: {report_path}, {database}')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('protocol')!='reach_tds_direct_v1' or report['database_sha256']!=sha256(database):
        parser.error('direct report does not match the success-posture database')
    if report['database_metadata'].get('contract',{}).get('grasp_criterion') != \
            'hinge_center_inside_palm_attached_original_sphere':
        print('NOTE: PCA/PLS axes come from the archived contact-gated database; '
              'this replay evaluates them with the new palm-sphere rule.',flush=True)
    label=f'{a.method}_1d'
    config=report['configurations'][label]
    levels=np.asarray(config['diagnostic_levels'],dtype=float)
    rho=float(a.score if a.score is not None else levels[-1]*a.score_multiplier)
    pose=int(a.pose if a.pose is not None else report['test_poses'][0])
    if pose not in report['test_poses'] and not a.allow_other_pose:
        parser.error('pose is not in recorded test split; use --allow-other-pose to override')
    contract=report['database_metadata'].get('contract',{})
    smooth=SmoothReachConfig(float(contract.get('action_delta_fraction',.05)),
                             float(contract.get('request_gap_penalty_weight',1.)))
    _register_tool_manip_gym_environments()
    env,_,_,key,lo,hi=_build_env(report['database_metadata']['env_id'],
        a.source_root/'pose_pool',a.split_seed,research_task=True,
        scissors_control_version='smooth',scissors_smooth_config=smooth,
        object_position_noise=0.,max_episode_steps=10000,
        render_mode=None if a.no_render else 'human')
    try:
        low,high,_=_control_bounds(env)
        fit=config['fit']
        goal=reconstructed_posture(np.asarray(report['mean_q']),
            np.asarray(fit['operation_axis']),config['operation_scale'],rho,
            np.asarray(fit['grasp_axis']),config['grasp_scale'],0.,low,high)
        print(f'{task} {label}, seed={a.split_seed}, test pose={pose}, score={rho:+.4f}; '
              f'original max={levels[-1]:+.4f}; 5 s move + 2 s hold',flush=True)
        started=time.monotonic()
        def capture(step,obs,info):
            flags=', '.join(f'{k}={bool(info.get(k,False))}' for k in
                ('grasp_maintained','failure_contact','failure_displacement','failure_rotation',
                 'object_dropped','object_tipped','task_failure'))
            moving=int(info.get('grasp_moving_finger_count',0))
            mirror=int(info.get('grasp_mirror_finger_count',0))
            print(f'step={step:4d}, tool={float(obs["achieved_goal"][0]):+.3f} rad, '
                  f'hinge-to-palm-sphere={float(info.get("palm_sphere_distance",float("nan"))):.3f} m '
                  f'(radius={float(info.get("palm_sphere_radius",float("nan"))):.3f} m), '
                  f'contacts(moving/mirror; diagnostic)={moving}/{mirror}, {flags}',flush=True)
            if not a.no_render:
                delay=step*env.unwrapped.dt/a.speed-(time.monotonic()-started)
                if delay>0:
                    time.sleep(delay)
        outcome=direct_trajectory(env,pose,key,lo+.3*(hi-lo),a.split_seed+1000*pose,
                                  [goal],capture=capture)['outcomes'][0]
        print(f'final: angle={outcome["achieved"]:+.3f} rad, '
              f'held2s={outcome["valid"]}, hold_fraction={outcome["hold_fraction"]:.3f}, '
              f'ramp_held_fraction={outcome["ramp_held_fraction"]:.3f}, '
              f'terminated={outcome["terminated"]}, truncated={outcome["truncated"]}',flush=True)
    finally:
        env.close()
    return 0


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',type=int,choices=(3,4,5),default=4)
    p.add_argument('--run-root',type=Path,default=Path('runs/scissors_smooth_control_v1_20260924'))
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seed',type=int,help='override the original collection seed; omit for exact replay order')
    p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--print-every',type=int,default=10)
    p.add_argument('--bin',type=int,choices=range(10),help='focus one target bin; changes the original collection order')
    p.add_argument('--pose',type=int,help='fix an initial pose; changes the original collection order')
    p.add_argument('--hold-seconds',type=float,default=0.,
                   help='after success, freeze the last applied command; diagnostic, not collection')
    p.add_argument('--pause-on-success',action='store_true')
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--direct',action='store_true',help='view direct reconstructed-posture control instead of RL recollection')
    p.add_argument('--source-root',type=Path,default=Path('runs/scissors_smooth_tds_200_20260925'))
    p.add_argument('--direct-root',type=Path,default=Path('analysis/scissors_smooth_tds_direct_20260925'))
    p.add_argument('--split-seed',type=int,default=20260925)
    p.add_argument('--method',choices=('pca','pls'),default='pca')
    p.add_argument('--score',type=float,help='physical score; overrides --score-multiplier')
    p.add_argument('--score-multiplier',type=float,default=1.,help='multiply the original positive score endpoint')
    p.add_argument('--allow-other-pose',action='store_true')
    a=p.parse_args(argv)
    if a.episodes<1 or a.speed<=0 or a.print_every<1 or not np.isfinite(a.hold_seconds) or a.hold_seconds<0 \
            or not np.isfinite(a.score_multiplier) or a.score_multiplier<=0 \
            or (a.score is not None and not np.isfinite(a.score)):
        p.error('invalid episodes, speed, print interval or hold duration')
    if a.direct:
        return view_direct(a,p)
    task=f'scissors_lite_{a.variant}'
    model=a.run_root/task/'best_success_policy.zip'
    pool=a.run_root/'pose_pool'/f'{task}.npz'
    database=a.run_root/task/'success_postures.npz'
    for path in (model,pool,database):
        if not path.is_file():p.error(f'missing frozen artifact: {path}')
    with np.load(database,allow_pickle=False) as data:
        meta=json.loads(data['metadata_json'].item())
    contract=meta.get('contract',{})
    if (not meta.get('complete') or meta.get('scissors_control_version')!='smooth'
            or contract.get('model_sha256')!=sha256(model)
            or contract.get('pool_sha256')!=sha256(pool)):
        p.error('database does not match this smooth Best Policy and pose pool')
    if contract.get('grasp_criterion') != 'hinge_center_inside_palm_attached_original_sphere':
        print('NOTE: frozen model/database used the historical contact-gated success rule; '
              'this Viewer now shows palm-sphere success and is not an exact label replay.',flush=True)
    print(f'Verified smooth database: {meta["rows"]} successful rows, '
          f'{meta["episodes"]} collection episodes, bins={meta["counts"]}',flush=True)
    args=['--variant',str(a.variant),'--run-root',str(a.run_root),
          '--control-version','smooth','--episodes',str(a.episodes),
          '--speed',str(a.speed),'--print-every',str(a.print_every)]
    if a.seed is not None:args+=['--seed',str(a.seed)]
    if a.bin is not None:args+=['--bin',str(a.bin)]
    if a.pose is not None:args+=['--pose',str(a.pose)]
    if a.hold_seconds:args+=['--hold-seconds',str(a.hold_seconds)]
    if a.pause_on_success:args.append('--pause-on-success')
    if a.no_render:args.append('--no-render')
    return view_collection(args)


if __name__=='__main__':
    raise SystemExit(main())
