"""Frozen, four-worker 15-task pipeline: RL collection -> strict -> adaptive FF -> HTML."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


SEEDS=(20260911,20260912,20260913,20260914,20260915)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('analysis/tds_mixed_v3_20260918'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--grid-points',type=int,default=101)
    p.add_argument('--success-source-root',type=Path)
    p.add_argument('--adaptive-grasp-method',choices=('force_pls','residual_pca'),default='force_pls')
    a=p.parse_args()
    if a.workers<1 or a.grid_points<2: raise ValueError('invalid workers/grid')
    project=Path(__file__).resolve().parents[1];workspace=project.parent
    root=a.output.resolve()
    if root.exists(): raise FileExistsError(f'Fresh output required: {root}')
    tasks=[]
    for category in ('push','spray','scissors'):
        for v in range(1,6):
            name=f'{category}_lite_{v}'
            revised=category=='scissors' and v>=3
            run=project/'runs'/('scissors_control_v3_20260918' if revised else 'tool_control_v2_20260917')
            model=run/name/'best_pose_coverage_policy.zip'
            pool=(run/'pose_pool' if revised else workspace/'tool-pose-pipeline'/
                  ('grasps_posture_dynamic_v2_mild' if category=='scissors' else 'grasps_v3'))
            poolfile=pool/(f'spray_lite_spray{v}.npz' if category=='spray' else f'{name}.npz')
            envid=f'ScissorsVariant{v}-v0' if category=='scissors' else f'ShadowHandLite{category.title()}{v}-v0'
            if not model.is_file() or not poolfile.is_file(): raise FileNotFoundError((model,poolfile))
            tasks.append(dict(name=name,env_id=envid,category=category,version='v3' if revised else 'v2',
                source_model=str(model),source_pool=str(poolfile),model_sha256=digest(model),pool_sha256=digest(poolfile)))
    root.mkdir(parents=True)
    for folder in ('databases','strict_reports','reports','inputs','fit_databases'):
        (root/folder).mkdir()
    shutil.copytree(project/'src',root/'source_snapshot')
    shutil.copy2(__file__,root/'runner_snapshot.py')
    for t in tasks:
        inputs=root/'inputs'/t['name'];inputs.mkdir()
        shutil.copy2(t['source_model'],inputs/'policy.zip')
        shutil.copy2(t['source_pool'],inputs/Path(t['source_pool']).name)
        t.update(model=str(inputs/'policy.zip'),pool_dir=str(inputs))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),tasks=tasks,seeds=SEEDS,
        grid_points=a.grid_points,raw_rows_per_task=32*a.grid_points,fit_rows=100,angle_bins=10,
        fit_poses=16,calibration_poses=8,test_poses=8,workers=a.workers,
        policy_selection='best_pose_coverage_policy',success_only_fit=True,
        success_only_database=bool(a.success_source_root),
        adaptive_grasp_method=a.adaptive_grasp_method,
        success_source_root=str(a.success_source_root.resolve()) if a.success_source_root else None,
        collect_until_all_split_quotas_met=bool(a.success_source_root),
        ff_control='direct absolute commands; common research criteria, no RL v3 rate limiter',
        arms=['legacy','direction','range','joint'])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    env=os.environ.copy()
    packages=[str(root/'source_snapshot')]+[str(workspace/'tool-manip-gym'/k) for k in ('gym-scissors','gym-slider','gym-push','gym-spray')]
    env.update(PYTHONPATH=os.pathsep.join(packages),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    status_rows={t['name']:{'phase':'queued'} for t in tasks}
    def status(state,**extra):
        result=dict(state=state,utc=datetime.now(timezone.utc).isoformat(),tasks=status_rows,**extra)
        temporary=root/'status.tmp';temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');temporary.replace(root/'status.json')
        table=''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(v["phase"])}</td><td>{html.escape(v.get("error",""))}</td></tr>' for k,v in status_rows.items())
        (root/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>TDS experiment status</title>'
            f'<h1>15-task PCA/PLS TDS experiment</h1><p>{html.escape(state)} / {result["utc"]}</p>'
            '<p>Fit: 100 success-only postures, 10 per angle bin; 16/8/8 reset-pose split, five seeds. '
            'Scissors: 1 DoF; Push/Spray: grasp + operation. Incomplete tasks are not counted as completed experiments.</p>'
            '<p><a href="comparison/index.html">PCA / PLS side-by-side comparison (available after evaluation)</a> | '
            '<a href="summary/index.html">All four conditions (available after all 75 reports finish)</a></p>'
            f'<table border="1"><tr><th>Task</th><th>Phase</th><th>Error</th></tr>{table}</table></html>',encoding='utf-8')
    # Only the main thread writes status; workers report via a queue.
    from queue import Queue, Empty
    updates=Queue()
    def run_command(cmd,log):
        with log.open('w',encoding='utf-8') as out,log.with_suffix('.err.log').open('w',encoding='utf-8') as err:
            subprocess.run([sys.executable,*cmd],env=env,cwd=project,stdout=out,stderr=err,check=True)
    def collect_task(t):
        name=t['name'];database=root/'databases'/f'{name}.npz'
        updates.put((name,'collecting'))
        fractions=' '.join(f'{i/(a.grid_points-1):.12g}' for i in range(a.grid_points))
        if a.success_source_root:
            run_command(['-m','improved_tds.experiments.collect_until_tds_ready',
                '--source-database',str(a.success_source_root.resolve()/'databases'/f'{name}.npz'),
                '--model',t['model'],'--pose-pool-dir',t['pool_dir'],'--output',str(database)],database.with_suffix('.log'))
        else:
            run_command(['-m','improved_tds.experiments.collect_posture_database','--env-id',t['env_id'],
                '--model',t['model'],'--pose-pool-dir',t['pool_dir'],'--output',str(database),
                '--research-task','--scissors-control-version',t['version'],'--representative','last',
                '--target-fractions',fractions,'--object-position-noise','0','--strict-success-window',
                '--success-window-steps','10' if t['category']=='scissors' else '5'],database.with_suffix('.log'))
        updates.put((name,'checking_fit_quotas'))
        run_command(['-m','improved_tds.experiments.audit_tds_fit_database','--database',str(database),
            '--output-dir',str(root/'fit_databases'/name)],root/'databases'/f'{name}_fit_audit.log')
        return name

    def evaluate_task(t):
        name=t['name'];database=root/'databases'/f'{name}.npz'
        for seed in SEEDS:
            stem=f'{name}_seed{seed}';strict=root/'strict_reports'/f'{stem}.json'
            updates.put((name,f'strict/{seed}'))
            run_command(['-m','improved_tds.experiments.evaluate_strict_synergy_control','--database',str(database),
                '--pose-pool-dir',t['pool_dir'],'--output',str(strict),'--split-seed',str(seed),
                '--common-fit-rows','100','--balanced-angle-bins','10'],strict.with_suffix('.log'))
            updates.put((name,f'adaptive/{seed}'))
            cmd=['-m','improved_tds.experiments.evaluate_adaptive_synergy','--strict-report',str(strict),
                 '--pose-pool-dir',t['pool_dir'],'--output',str(root/'reports'/f'{stem}.json'),
                 '--grasp-method',a.adaptive_grasp_method]
            if seed==SEEDS[0]:cmd+=['--render-dir',str(root/'images'/name)]
            run_command(cmd,root/'reports'/f'{stem}.log')
        return name
    status('running')
    failures=[]
    from concurrent.futures import wait,FIRST_COMPLETED
    for function,completed_phase in ((collect_task,'database_ready'),(evaluate_task,'completed')):
        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            pending={executor.submit(function,t):t['name'] for t in tasks if t['name'] not in failures}
            while pending:
                done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                while True:
                    try:name,phase=updates.get_nowait();status_rows[name]['phase']=phase
                    except Empty:break
                for future in done:
                    name=pending.pop(future)
                    try:future.result();status_rows[name]['phase']=completed_phase
                    except Exception as exc:
                        status_rows[name]=dict(phase='failed',error=str(exc));failures.append(name)
                status('running',failures=failures)
    if failures:
        run_command(['-m','improved_tds.experiments.compare_tds_methods','--root',str(root)],root/'comparison.log')
        status('incomplete',failures=failures)
        return 2
    status('summarizing')
    try:
        run_command(['-m','improved_tds.experiments.summarize_adaptive_synergy','--report-dir',str(root/'reports'),
            '--output-dir',str(root/'summary'),'--expected-reports','75'],root/'summary.log')
        run_command(['-m','improved_tds.experiments.compare_tds_methods','--root',str(root)],root/'comparison.log')
    except Exception:
        status('summary_failed',error=traceback.format_exc());raise
    status('completed',html=str(root/'comparison/index.html'))
    return 0


if __name__=='__main__':raise SystemExit(main())
