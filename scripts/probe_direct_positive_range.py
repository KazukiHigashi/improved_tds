"""Probe positive PCA/PLS scores beyond the completed direct-TDS sweep.

Each trial resets the same held-out pose/seed as the archived experiment. The
reported angle-only span is diagnostic; valid coverage requires a full 2 s
grasp hold. The success-posture database and archived reports are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from improved_tds.environments.scissors_smooth import SmoothReachConfig
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.reach_tds_direct import reconstructed_posture
from improved_tds.experiments.simple_synergy import direct_trajectory, coverage
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def span(values, lo, hi):
    finite=[x for x in values if np.isfinite(x)]
    return coverage([dict(achieved=x,valid=True) for x in finite],lo,hi)


def plot_task(task, methods, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for method, color in (('pca','#2877b9'),('pls','#cf6d20')):
        rows=methods[method]['rows']
        x=[r['factor'] for r in rows]
        axes[0].plot(x,[r['new_angle_max'] for r in rows],'o-',label=method.upper(),color=color)
        axes[1].plot(x,[r['combined_angle_only_coverage'] for r in rows],'o--',
                     label=f'{method.upper()} angle only',color=color)
        axes[1].plot(x,[r['combined_held_coverage'] for r in rows],'s-',
                     label=f'{method.upper()} held 2 s',color=color)
    axes[0].axhline(1.,color='k',ls=':',lw=1,label='target upper limit')
    axes[0].set(ylabel='Maximum observed angle [rad]',xlabel='Positive score / original maximum')
    axes[1].set(ylabel='Pooled endpoint span / 0–1 rad (not continuous coverage)',
                xlabel='Positive score / original maximum',ylim=(-.03,1.03))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'{task}: reset-independent direct posture, 5 s + 2 s')
    fig.tight_layout()
    fig.savefig(output/f'{task}.png',dpi=150)
    plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('runs/scissors_smooth_tds_200_20260925'))
    p.add_argument('--direct-root',type=Path,default=Path('analysis/scissors_smooth_tds_direct_20260925'))
    p.add_argument('--output',type=Path,default=Path('analysis/direct_positive_range_probe_20260925'))
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--factors',nargs='+',type=float,default=[1.,1.5,2.,3.,4.])
    p.add_argument('--tasks',nargs='+',choices=[f'scissors_lite_{i}' for i in (3,4,5)],
                   default=[f'scissors_lite_{i}' for i in (3,4,5)])
    p.add_argument('--plots-only',action='store_true',help='regenerate PNGs from existing results.json')
    a=p.parse_args(argv)
    if a.plots_only:
        result=json.loads((a.output/'results.json').read_text(encoding='utf-8'))
        for task,methods in result['tasks'].items():
            plot_task(task,methods,a.output)
        return 0
    if any(not np.isfinite(f) or f<1 for f in a.factors):
        p.error('factors must be finite and >=1')
    _register_tool_manip_gym_environments()
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(protocol='direct_positive_range_probe_v1',seed=a.seed,factors=a.factors,
                grasp_criterion='hinge_center_inside_palm_attached_original_sphere',
                note='test split is reused for diagnosis, not model selection',tasks={})
    for task in a.tasks:
        db=a.source_root/task/'success_postures.npz'
        report_path=a.direct_root/'results'/task/str(a.seed)/'report.json'
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if report['protocol']!='reach_tds_direct_v1' or report['database_sha256']!=hashlib.sha256(db.read_bytes()).hexdigest():
            raise ValueError(f'archived direct report and database mismatch: {task}')
        contract=report['database_metadata'].get('contract',{})
        smooth=SmoothReachConfig(float(contract.get('action_delta_fraction',.05)),
                                 float(contract.get('request_gap_penalty_weight',1.)))
        env,_,_,key,lo,hi=_build_env(report['database_metadata']['env_id'],
            a.source_root/'pose_pool',a.seed,research_task=True,
            scissors_control_version='smooth',scissors_smooth_config=smooth,
            object_position_noise=0.,max_episode_steps=10000)
        by_method={}
        try:
            low,high,_=_control_bounds(env)
            for method in ('pca','pls'):
                config=report['configurations'][f'{method}_1d']
                fit=config['fit']
                op=np.asarray(fit['operation_axis'])
                grasp=np.asarray(fit['grasp_axis'])
                max_rho=float(config['diagnostic_levels'][-1])
                poses=report['test_poses']
                # Old reports were scored with contact-gated grasp and cannot
                # contribute held points to a palm-sphere result.
                archived_compatible=(report['database_metadata'].get('contract',{})
                    .get('grasp_criterion')==result['grasp_criterion'])
                baseline=[o for trial in config['test'] for o in trial['diagnostic']['outcomes']
                    if o is not None] if archived_compatible else []
                baseline_held=[o['achieved'] for o in baseline if o['valid']]
                baseline_finite=[o['achieved'] for o in baseline if not o['failure_numerical'] and np.isfinite(o['achieved'])]
                cumulative_held=list(baseline_held)
                cumulative_finite=list(baseline_finite)
                rows=[]
                previous_goal=None
                for factor in a.factors:
                    rho=factor*max_rho
                    goal=reconstructed_posture(np.asarray(report['mean_q']),op,
                        config['operation_scale'],rho,grasp,config['grasp_scale'],0.,low,high)
                    outcomes=[]
                    for pose in poses:
                        rec=direct_trajectory(env,pose,key,lo+.3*(hi-lo),a.seed+1000*pose,[goal])
                        o=rec['outcomes'][0]
                        outcomes.append(dict(pose=pose,achieved=o['achieved'],valid=o['valid'],
                            hold_fraction=o['hold_fraction'],ramp_held_fraction=o['ramp_held_fraction'],
                            failure_contact=o['failure_contact'],
                            failure_displacement=o['failure_displacement'],
                            failure_rotation=o['failure_rotation'],
                            failure_numerical=o['failure_numerical'],
                            terminated=o['terminated']))
                    cumulative_held.extend(o['achieved'] for o in outcomes if o['valid'])
                    cumulative_finite.extend(o['achieved'] for o in outcomes if not o['failure_numerical']
                                             and np.isfinite(o['achieved']))
                    rows.append(dict(factor=factor,score=rho,
                        new_held=sum(o['valid'] for o in outcomes),trials=len(outcomes),
                        new_angle_max=max(o['achieved'] for o in outcomes),
                        new_angle_min=min(o['achieved'] for o in outcomes),
                        combined_held_coverage=span(cumulative_held,lo,hi),
                        combined_angle_only_coverage=span(cumulative_finite,lo,hi),
                        clipped_joints=int(np.count_nonzero(np.isclose(goal,low)|np.isclose(goal,high))),
                        command_change_from_previous=None if previous_goal is None else
                            float(np.linalg.norm(goal-previous_goal)),outcomes=outcomes))
                    previous_goal=goal
                    print(f'{task} {method} x{factor:g}: held {rows[-1]["new_held"]}/{len(poses)}, '
                          f'angle max {rows[-1]["new_angle_max"]:.3f}, '
                          f'held coverage {rows[-1]["combined_held_coverage"]:.3f}',flush=True)
                by_method[method]=dict(original_positive_max=max_rho,
                    archived_baseline_compatible=archived_compatible,
                    original_held_coverage=span(baseline_held,lo,hi) if archived_compatible else None,
                    original_angle_only_coverage=span(baseline_finite,lo,hi) if archived_compatible else None,
                    original_held_count=len(baseline_held),rows=rows)
        finally:
            env.close()
        result['tasks'][task]=by_method
        (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        plot_task(task,by_method,a.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
