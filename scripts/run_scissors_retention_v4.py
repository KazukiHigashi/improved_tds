"""Fresh Scissors 1--5, up to 3M steps, at most four learners x four environments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def wait_for_previous_run(path, notify, timeout_seconds=43200, interval_seconds=60):
    """Background launcher wait, not interactive agent polling; fail closed."""
    deadline=time.monotonic()+timeout_seconds
    while True:
        try:
            prior=json.loads((path/'status.json').read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            prior={'tasks':{}}  # tolerate an in-progress status write
        states=prior.get('tasks',{})
        if states and all(x=='completed' for x in states.values()):
            notify('previous run completed; starting queued learners')
            return
        if any(str(x).startswith('failed') for x in states.values()):
            raise RuntimeError('previous run failed; queued run not started')
        if time.monotonic()>=deadline:
            raise TimeoutError('previous run did not finish within 12 hours')
        time.sleep(interval_seconds)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/scissors_retention_v4_20260920'))
    parser.add_argument('--timesteps',type=int,default=3000000)
    parser.add_argument('--variants',default='3 4 5')
    parser.add_argument('--n-envs',type=int,default=4)
    parser.add_argument('--version',choices=('simple','smooth','v4','v5','v6','v7'),default='v4')
    parser.add_argument('--early-pool-source',type=Path,
        help='drivable initial-pose pool directory for Scissors 1 and 2')
    parser.add_argument('--after-run',type=Path,help='launch only after this run completes; snapshot now')
    parser.add_argument('--collect-successes',action='store_true',help='simple/smooth mode: collect 200 balanced successes from frozen Best Policy after training')
    args=parser.parse_args(argv)
    if args.collect_successes and args.version not in ('simple','smooth'):raise ValueError('success collection requires simple/smooth training')
    variants=[int(x) for x in args.variants.replace(',', ' ').split()]
    if not variants or any(x not in (1,2,3,4,5) for x in variants) or len(set(variants))!=len(variants):
        raise ValueError('variants must be distinct Scissors 1--5')
    if args.timesteps<=0 or args.n_envs<=0:raise ValueError('positive steps/env count required')
    root=args.output.resolve()
    if root.exists():raise FileExistsError(root)
    project=Path(__file__).resolve().parents[1]
    if args.after_run is not None:
        args.after_run=args.after_run.resolve()
        if not (args.after_run/'status.json').is_file():raise FileNotFoundError(args.after_run/'status.json')
    source=project/'runs/scissors_control_v3_20260918/pose_pool'
    early_source=(args.early_pool_source.resolve() if args.early_pool_source else
                  project.parent/'tool-pose-pipeline/grasps_v4_scissors_drive')
    pool_sources={v:(early_source if v in (1,2) else source) for v in variants}
    for v in variants:
        if not (pool_sources[v]/f'scissors_lite_{v}.npz').is_file():
            raise FileNotFoundError(pool_sources[v]/f'scissors_lite_{v}.npz')
    root.mkdir(parents=True);(root/'pose_pool').mkdir()
    shutil.copytree(project/'src',root/'source_snapshot')
    shutil.copy2(__file__,root/'launcher_snapshot.py')
    hashes={}
    for v in variants:
        name=f'scissors_lite_{v}.npz';shutil.copy2(pool_sources[v]/name,root/'pose_pool'/name)
        with (pool_sources[v]/name).open('rb') as f:hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),version='scissors_retention_'+args.version,
        tasks=variants,total_timesteps=args.timesteps,n_envs=args.n_envs,learners=min(4,len(variants)),fresh_training=True,
        source_pool={str(v):str(pool_sources[v]) for v in variants},pool_sha256=hashes,her=False,full_target_range=True,
        final_fixed_command_hold_seconds=2.,freeze_acquisition_seconds=.4,
        hold_schedule=[[0,.4],[.2,.8],[.4,1.2],[.6,2.]],
        validation_fixed_final_criteria=True,validation_interval=100000,checkpoint_interval=100000,
        early_stop=dict(threshold=.95,consecutive=2),category_reward_identical=True)
    manifest['after_run']=str(args.after_run) if args.after_run else None
    if args.version in ('simple','smooth'):
        manifest.update(version='scissors_reach_'+args.version,final_fixed_command_hold_seconds=None,
            freeze_acquisition_seconds=None,hold_schedule=None,success_hold_steps=5,
            speed_success_criterion=False,phase_aware_sac=False,collect_successes=args.collect_successes,
            collection_rows=200,collection_bins=10,collection_requires_endpoints=True,
            reward_progress_weight=1.,reward_grasp_weight=1.,
            grasp_criterion='hinge_center_inside_palm_attached_original_sphere',
            contact_auxiliary_reward_weight=0.,
            contact_unit='distinct_fingers_diagnostic_only')
        if args.version=='smooth':
            manifest.update(action_delta_fraction=.05,request_gap_penalty_weight=1.,
                contact_unit='distinct_fingers_on_both_handles_diagnostic_only')
    if args.version in ('v5','v6','v7'):
        manifest.update(hold_schedule='competence >=70% of last20 probes promotes .4/.8/1.2/2s',
                        final_duration_probability=.2,acquisition_seconds=.4,
                        acquisition_reference_seconds=.12,success_terminates=False,
                        numerical_transition_filter=True,matched_reference_evaluation_at_end=True)
    if args.version in ('v6','v7'):
        manifest.update(hold_schedule='competence or budget .2/.4/.6 promotes .4/.8/1.2/2s',
                        freeze_acquisition_seconds=.12,settling_budget_seconds=.4,
                        hard_command_change_gate=False,hard_root_speed_gate=False,
                        final_40_percent_full_duration=True,completion_bonus_duration_scaled=True,
                        feedback_success_bonus_fraction=.2,fixed_success_bonus_fraction=.8)
    if args.version=='v7':
        manifest.update(phase_aware_sac=True,masked_hold_action_and_entropy=True,
                        canonical_ignored_action_observation=True,servo_tracking_observation=True,
                        support_margin_weight=1.5,safety_weight=2.,servo_gap_penalty_weight=.5,
                        servo_margin_rad=.15,servo_scale_rad=.3,hard_servo_limit=False)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'source_snapshot'),OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    statuses={f'scissors_lite_{v}':'queued' for v in variants}
    def status():
        (root/'status.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),tasks=statuses),indent=2),encoding='utf-8')
    def run(v):
        task=root/f'scissors_lite_{v}';task.mkdir()
        def phase(name):
            (task/'phase.json').write_text(json.dumps(dict(phase=name,utc=datetime.now(timezone.utc).isoformat())),encoding='utf-8')
        phase('training')
        cmd=[sys.executable,'-m','improved_tds.experiments.train_sb3',
             '--env-id',f'ScissorsVariant{v}-v0','--total-timesteps',str(args.timesteps),'--seed',str(4000+v-1),
             '--n-envs',str(args.n_envs),'--scissors-control-version',args.version,'--her-sampled-goals','0',
             '--learning-rate','0.00005','--gradient-steps','2','--sac-ent-coef','0.02',
             '--pose-pool-dir',str(root/'pose_pool'),'--require-drivable-pool',
             '--target-min-fraction','0','--target-max-fraction','1','--scissors-reward-type','dense',
             '--scissors-coarse-tracking-weight','1','--scissors-fine-tracking-weight','1',
             '--scissors-coarse-tracking-sigma','0.35','--scissors-progress-weight','5',
             '--scissors-grasp-weight','0.25','--scissors-interaction-weight','0.25',
             '--scissors-contact-auxiliary-weight','0',
             '--scissors-success-bonus','5','--scissors-failure-penalty','25','--scissors-grasp-gate-floor','0.25',
             '--control-delta-penalty-weight','0.1','--control-magnitude-penalty-weight','0.002',
             '--output',str(task/'model'),'--dataset',str(task/'training_candidates.npz'),
             '--checkpoints-dir',str(task/'checkpoints'),'--checkpoint-freq','100000','--validation-freq','100000',
             '--validation-target-fractions','0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0',
             '--validation-success-threshold','0.95','--validation-consecutive','2',
             '--best-policy-output',str(task/'best_policy'),'--validation-report',str(task/'validation_history.json')]
        if args.version in ('simple','smooth'):
            cmd[cmd.index('--scissors-progress-weight')+1]='1'
            cmd[cmd.index('--scissors-grasp-weight')+1]='1'
            cmd.extend(['--success-hold-steps','5'])
        (task/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        with (task/'train.log').open('w',encoding='utf-8') as out,(task/'train.err.log').open('w',encoding='utf-8') as err:
            process=subprocess.Popen(cmd,cwd=project,env=env,stdout=out,stderr=err)
            (task/'pid.txt').write_text(str(process.pid))
            code=process.wait()
        if code:raise RuntimeError(f'exit {code}; see {task}/train.err.log')
        if args.collect_successes:
            phase('collecting_success_postures')
            with (task/'collect.log').open('w',encoding='utf-8') as out,(task/'collect.err.log').open('w',encoding='utf-8') as err:
                subprocess.run([sys.executable,'-m','improved_tds.experiments.collect_reach_successes',
                    '--env-id',f'ScissorsVariant{v}-v0','--model',str(task/'best_success_policy.zip'),
                    '--pose-pool-dir',str(root/'pose_pool'),'--output',str(task/'success_postures.npz'),
                    '--per-bin','20',
                    '--seed',str(9300+v),'--scissors-control-version',args.version],cwd=project,env=env,stdout=out,stderr=err,check=True)
        if args.version in ('v5','v6','v7') and args.timesteps>=100000:
            with (task/'reference_eval.log').open('w',encoding='utf-8') as out,(task/'reference_eval.err.log').open('w',encoding='utf-8') as err:
                subprocess.run([sys.executable,'-m','improved_tds.experiments.evaluate_retention_v5',
                    '--run-dir',str(task),'--pool-dir',str(root/'pose_pool'),'--variant',str(v),
                    '--version',args.version],
                    cwd=project,env=env,stdout=out,stderr=err,check=True)
        phase('completed')
    status()
    if args.after_run is not None:
        statuses={k:'waiting_for_previous_run' for k in statuses};status()
        print(f'Queued with frozen source; waiting for {args.after_run}',flush=True)
        try:wait_for_previous_run(args.after_run,lambda message:print(message,flush=True))
        except Exception as exc:
            statuses={k:'failed: '+str(exc) for k in statuses};status()
            raise
        manifest['learners_started_utc']=datetime.now(timezone.utc).isoformat()
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with ThreadPoolExecutor(max_workers=min(4,len(variants))) as executor:
        futures={executor.submit(run,v):v for v in variants}
        statuses={k:'running' for k in statuses};status()
        for future in as_completed(futures):
            v=futures[future]
            try:future.result();statuses[f'scissors_lite_{v}']='completed'
            except Exception as exc:statuses[f'scissors_lite_{v}']='failed: '+str(exc)
            status()


if __name__=='__main__':main()
