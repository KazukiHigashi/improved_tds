"""Rerun only direct PCA/PLS TDS for the frozen 15-task paper campaign.

The old training models, success-posture databases, and failed TDS reports are
read-only inputs. This launcher snapshots corrected evaluation source and
writes all new results to a separate output directory.
"""
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
import threading


TASKS = [f'{category}_lite_{variant}'
         for category in ('scissors', 'push', 'spray') for variant in range(1, 6)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
        default=Path('runs/paper15_unified_3m_slew_20260927'))
    parser.add_argument('--output', type=Path,
        default=Path('analysis/paper15_unified_tds_corrected_20260928'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--poses', type=int, default=8)
    parser.add_argument('--points', type=int, default=17)
    parser.add_argument('--categories', nargs='+', choices=('scissors','push','spray'),
        default=('scissors','push','spray'))
    parser.add_argument('--scissors-evaluation-palm-radius-multiplier',type=float,default=1.,
        help='Apply only to Scissors Calibration/Test, not RL training or success-posture DB')
    args = parser.parse_args(argv)
    tasks=[task for task in TASKS if task.split('_')[0] in args.categories]
    if (not 1 <= args.workers <= 4 or args.poses < 1 or args.points < 3
            or not 0 < args.scissors_evaluation_palm_radius_multiplier < float('inf')):
        parser.error('expected 1--4 workers, positive poses and at least 3 sweep points')
    project = Path(__file__).resolve().parents[1]
    source = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not (source / 'manifest.json').is_file():
        raise FileNotFoundError(source / 'manifest.json')
    database_hashes = {}
    for task in tasks:
        database = source / task / 'success_postures.npz'
        if not database.is_file():
            raise FileNotFoundError(database)
        with database.open('rb') as stream:
            database_hashes[task] = hashlib.file_digest(stream, 'sha256').hexdigest()
    output.mkdir(parents=True)
    shutil.copytree(project / 'src', output / 'evaluation_source_snapshot')
    shutil.copy2(__file__, output / 'launcher_snapshot.py')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol='paper15_direct_tds_no_contact_gate_v3', source_root=str(source),
        training_and_databases_reused=True, database_sha256=database_hashes,
        tasks=tasks, workers=args.workers, calibration_poses=args.poses,
        test_poses=args.poses, diagnostic_score_points=args.points,
        score_range='1.5x each fitted component score min/max',
        scissors_evaluation_palm_radius_multiplier=args.scissors_evaluation_palm_radius_multiplier,
        scissors_training_and_database_palm_radius_unchanged=True,
        trajectory_seconds=5., hold_seconds=2.,
        fixes=['read task dt without eager fallback evaluation',
               'set Gymnasium and task-internal episode horizons to 10000',
               'Push/Spray calibration and TDS require no drop or tip, not contact count',
               'Spray floor contact is explicitly kept enabled'],
        previous_reports_preserved=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(output / 'evaluation_source_snapshot'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    lock = threading.Lock()
    status = {task: 'queued' for task in tasks}

    def update(task, value):
        with lock:
            status[task] = value
            (output / 'status.json').write_text(json.dumps(dict(
                updated_utc=datetime.now(timezone.utc).isoformat(), tasks=status), indent=2),
                encoding='utf-8')

    def evaluate(task):
        directory = output / task
        directory.mkdir()
        command = [sys.executable, '-m', 'improved_tds.experiments.reach_tds_direct',
            '--source-root', str(source), '--task', task,
            '--output', str(directory), '--seed',
            str(20260927+int(task.rsplit('_', 1)[1])),
            '--poses', str(args.poses), '--points', str(args.points)]
        if task.startswith('scissors'):
            command.extend(['--scissors-evaluation-palm-radius-multiplier',
                str(args.scissors_evaluation_palm_radius_multiplier)])
        (directory / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        update(task, 'running')
        with (directory / 'run.log').open('w', encoding='utf-8') as out, \
             (directory / 'run.err.log').open('w', encoding='utf-8') as err:
            process = subprocess.Popen(command, cwd=project, env=environment,
                stdout=out, stderr=err)
            (directory / 'pid.txt').write_text(str(process.pid), encoding='utf-8')
            return_code = process.wait()
        if return_code:
            raise RuntimeError(f'exit {return_code}; see {directory / "run.err.log"}')
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        if len(report['configurations']) != 4:
            raise RuntimeError('incomplete PCA/PLS x 1/2 DoF report')
        for config in report['configurations'].values():
            trials = [trial['outcome'] for pose in config['test']
                      for trial in pose['target_trials']]
            if any(trial['truncated'] and trial['executed_steps'] == 200 for trial in trials):
                raise RuntimeError('internal 200-step truncation recurred')
        update(task, 'completed')

    update(tasks[0], 'queued')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as exc:
                update(task, f'failed: {exc}')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
