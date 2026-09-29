"""Paired Scissors smooth SAC pilot: palm-only reward versus contact auxiliary.

The two runs for each variant share the seed, pose pool, reward weights, and
validation grid; only --scissors-contact-auxiliary-weight differs. Outputs are
isolated from all previous models. This pilot does not replace a long-horizon
multi-seed training comparison.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/scissors_palm_aux_ablation_20260926'))
    p.add_argument('--pose-pool-dir',type=Path,default=Path('runs/scissors_smooth_tds_200_20260925/pose_pool'))
    p.add_argument('--variants',nargs='+',type=int,choices=(3,4,5),default=[3,4,5])
    p.add_argument('--timesteps',type=int,default=100_000)
    p.add_argument('--n-envs',type=int,default=4)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--weights',nargs='+',type=float,default=[0.,.25])
    a=p.parse_args(argv)
    if a.output.exists():
        p.error(f'output already exists: {a.output}; choose a new run directory')
    if a.timesteps<1000 or a.n_envs<1 or a.workers<1 or len(a.weights)!=2 \
            or any(not 0<=w<=1 for w in a.weights) or a.weights[0]==a.weights[1]:
        p.error('invalid pilot configuration')
    project=Path(__file__).resolve().parents[1]
    source=project/'src'
    pool_dir=a.pose_pool_dir.resolve()
    for v in a.variants:
        if not (pool_dir/f'scissors_lite_{v}.npz').is_file():
            p.error(f'missing pose pool for Scissors {v}')
    root=a.output.resolve()
    root.mkdir(parents=True)
    manifest=dict(protocol='palm_sphere_contact_auxiliary_ablation_v1',
        started_utc=datetime.now(timezone.utc).isoformat(),variants=a.variants,
        paired_seeds={str(v):4000+v-1 for v in a.variants},weights=a.weights,
        timesteps=a.timesteps,n_envs=a.n_envs,workers=a.workers,
        pose_pool_dir=str(pool_dir),
        pool_sha256={str(v):digest(pool_dir/f'scissors_lite_{v}.npz') for v in a.variants},
        source_sha256={name:digest(project/name) for name in (
            'src/improved_tds/environments/scissors_reach.py',
            'src/improved_tds/environments/scissors_smooth.py',
            'src/improved_tds/experiments/train_sb3.py')},
        fixed=dict(grasp_criterion='hinge_center_inside_palm_attached_original_sphere',
                   interaction_reward_weight=0.,success_hold_steps=5,
                   validation_target_fractions=[0.,.25,.5,.75,1.]))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    jobs=[(v,w) for v in a.variants for w in a.weights]
    statuses={f'{v}:{w:g}':'queued' for v,w in jobs}
    def write_status():
        (root/'status.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                                     jobs=statuses),indent=2),encoding='utf-8')
    write_status()
    child_env=os.environ.copy()
    child_env.update(PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def run(v,w):
        label=f'contact_{str(w).replace(".","p")}'
        folder=root/f'scissors_lite_{v}'/label
        folder.mkdir(parents=True)
        cmd=[sys.executable,'-m','improved_tds.experiments.train_sb3',
            '--env-id',f'ScissorsVariant{v}-v0','--total-timesteps',str(a.timesteps),
            '--seed',str(4000+v-1),'--n-envs',str(a.n_envs),
            '--scissors-control-version','smooth','--her-sampled-goals','0',
            '--learning-rate','0.00005','--gradient-steps','2','--sac-ent-coef','0.02',
            '--pose-pool-dir',str(pool_dir),'--require-drivable-pool',
            '--target-min-fraction','0','--target-max-fraction','1',
            '--scissors-reward-type','dense','--scissors-progress-weight','1',
            '--scissors-grasp-weight','1','--scissors-interaction-weight','0',
            '--scissors-contact-auxiliary-weight',str(w),
            '--scissors-success-bonus','5','--scissors-failure-penalty','25',
            '--success-hold-steps','5','--output',str(folder/'model'),
            '--dataset',str(folder/'training_candidates.npz'),
            '--checkpoints-dir',str(folder/'checkpoints'),
            '--checkpoint-freq',str(a.timesteps),'--validation-freq',str(a.timesteps),
            '--validation-target-fractions','0 0.25 0.5 0.75 1',
            '--validation-success-threshold','0.95','--validation-consecutive','2',
            '--best-policy-output',str(folder/'best_policy'),
            '--validation-report',str(folder/'validation_history.json')]
        (folder/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        with (folder/'train.log').open('w',encoding='utf-8') as out, \
             (folder/'train.err.log').open('w',encoding='utf-8') as err:
            completed=subprocess.run(cmd,cwd=project,env=child_env,stdout=out,stderr=err)
        return completed.returncode
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures={executor.submit(run,v,w):(v,w) for v,w in jobs}
        for future in as_completed(futures):
            v,w=futures[future]
            code=future.result()
            statuses[f'{v}:{w:g}']='completed' if code==0 else f'failed: exit {code}'
            write_status()
            print(f'Scissors {v}, contact auxiliary {w:g}: {statuses[f"{v}:{w:g}"]}',flush=True)
    return 0 if all(v=='completed' for v in statuses.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
