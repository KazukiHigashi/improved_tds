"""Fresh, frozen-source 15-task paper campaign: train, collect, then direct TDS.

Four concurrent learners with four environments each. No old model is resumed.
Every artifact is written below a new run root; failures remain visible in status.json.
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


TASKS = [(category, variant, f'{category}_lite_{variant}',
          f'ScissorsVariant{variant}-v0' if category == 'scissors' else
          f'ShadowHandLitePush{variant}-v0' if category == 'push' else
          f'ShadowHandLiteSpray{variant}-v0')
         for category in ('scissors', 'push', 'spray') for variant in range(1, 6)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('runs/paper15_unified_3m_slew_20260927'))
    parser.add_argument('--timesteps', type=int, default=3_000_000)
    parser.add_argument('--learners', type=int, default=4)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--collection-max-episodes', type=int, default=50_000)
    parser.add_argument('--skip-tds', action='store_true')
    args = parser.parse_args(argv)
    if args.timesteps <= 0 or not 1 <= args.learners <= 4 or args.n_envs <= 0:
        parser.error('positive timesteps and envs, and 1--4 concurrent learners, required')
    if args.collection_max_episodes < 0:
        parser.error('collection episode budget must be nonnegative')
    project = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    pool_sources = {
        'scissors': project / 'runs/scissors_palm_all5_3m_20260926/pose_pool',
        'push': project.parent / 'tool-pose-pipeline/grasps_v3',
        'spray': project.parent / 'tool-pose-pipeline/grasps_v3',
    }
    files = {}
    for category, variant, name, _ in TASKS:
        filename = f'spray_lite_spray{variant}.npz' if category == 'spray' else f'{name}.npz'
        source = pool_sources[category] / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        files[name] = (filename, source)
    root.mkdir(parents=True)
    (root / 'pose_pool').mkdir()
    shutil.copytree(project / 'src', root / 'source_snapshot')
    shutil.copy2(__file__, root / 'launcher_snapshot.py')
    hashes = {}
    for name, (filename, source) in files.items():
        shutil.copy2(source, root / 'pose_pool' / filename)
        with source.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        protocol='paper15_unified_3m_slew_direct_tds_v1',
        branch='feat/unified-paper-15task-20260927',
        tasks=[name for _, _, name, _ in TASKS],
        training=dict(cap_steps=args.timesteps, concurrent_learners=args.learners,
            environments_per_learner=args.n_envs, algorithm='SAC', HER=False,
            learning_rate=5e-5, fixed_entropy_coefficient=.02, gradient_steps=2,
            target_fraction_range=[0, 1], validation_every_steps=100000,
            checkpoint_every_steps=100000, early_stop=dict(success_rate=.95, consecutive=2),
            success_hold_steps=5, action_slew_normalized_units_per_simulated_second=1.25,
            action_request_gap_penalty_weight=1.,
            common_reward_coefficients=dict(progress=1., grasp=1., success_bonus=5.,
                failure_penalty=25., control_delta_penalty=.1,
                control_magnitude_penalty=.002),
            category_specific_terms=dict(scissors='coarse+fine tracking; palm-only grasp; interaction .25; no contact auxiliary',
                push='tracking 2; thumb contact .25; two-finger grasp and drop/tip failure',
                spray='tracking 2; index proximity .25; two-finger grasp and drop/tip failure'),
            caveat='Equal simulator steps do not imply equal simulated seconds across categories.'),
        collection=dict(success_only=True, rows=200, achieved_angle_bins=10,
            per_bin=20, endpoint_targets=True, frozen_best_policy=True,
            max_episodes=args.collection_max_episodes),
        tds=dict(global_centering=True, fit_rows=200, methods=['pca','pls'],
            dofs=[1,2], score_levels=17, score_range='1.5x observed min/max',
            calibration_poses=8, test_poses=8, trajectory_seconds=5,
            terminal_hold_seconds=2, independent_and_continuous_sweeps=True,
            inverse_methods=['isotonic','ramp']), pool_sha256=hashes)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / 'source_snapshot'), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONUNBUFFERED='1')
    lock = threading.Lock()
    status = {name: 'queued_training' for _, _, name, _ in TASKS}

    def set_status(name, value):
        with lock:
            status[name] = value
            (root / 'status.json').write_text(json.dumps(dict(
                updated_utc=datetime.now(timezone.utc).isoformat(), tasks=status), indent=2),
                encoding='utf-8')

    def call(name, stage, cmd):
        task = root / name
        task.mkdir(exist_ok=True)
        (task / f'{stage}_command.json').write_text(json.dumps(cmd, indent=2), encoding='utf-8')
        with (task / f'{stage}.log').open('w', encoding='utf-8') as out, \
             (task / f'{stage}.err.log').open('w', encoding='utf-8') as err:
            process = subprocess.Popen(cmd, cwd=project, env=env, stdout=out, stderr=err)
            (task / f'{stage}.pid').write_text(str(process.pid), encoding='utf-8')
            code = process.wait()
        if code:
            raise RuntimeError(f'{stage} exit {code}; inspect {task / (stage + ".err.log")}')

    def train(item):
        category, variant, name, env_id = item
        task = root / name
        set_status(name, 'training')
        cmd = [sys.executable, '-m', 'improved_tds.experiments.train_sb3',
            '--env-id', env_id, '--total-timesteps', str(args.timesteps),
            '--seed', str({'scissors':4000,'push':5000,'spray':6000}[category]+variant),
            '--n-envs', str(args.n_envs), '--scissors-control-version', 'smooth',
            '--scissors-action-delta-fraction', '.05',
            '--scissors-request-gap-penalty-weight', '1',
            '--tool-action-slew-rate', '1.25', '--tool-request-gap-penalty-weight', '1',
            '--her-sampled-goals', '0', '--learning-rate', '.00005',
            '--gradient-steps', '2', '--sac-ent-coef', '.02',
            '--pose-pool-dir', str(root / 'pose_pool'),
            '--target-min-fraction', '0', '--target-max-fraction', '1',
            '--success-hold-steps', '5',
            '--spray-tracking-reward-weight', '2', '--spray-progress-reward-weight', '1',
            '--spray-grasp-reward-weight', '1', '--spray-index-proximity-reward-weight', '.25',
            '--spray-success-bonus', '5', '--spray-failure-penalty', '25',
            '--push-tracking-reward-weight', '2', '--push-progress-reward-weight', '1',
            '--push-grasp-reward-weight', '1', '--push-thumb-contact-reward-weight', '.25',
            '--push-success-bonus', '5', '--push-failure-penalty', '25',
            '--scissors-coarse-tracking-weight', '1', '--scissors-fine-tracking-weight', '1',
            '--scissors-progress-weight', '1', '--scissors-grasp-weight', '1',
            '--scissors-interaction-weight', '.25',
            '--scissors-contact-auxiliary-weight', '0',
            '--scissors-success-bonus', '5', '--scissors-failure-penalty', '25',
            '--control-delta-penalty-weight', '.1',
            '--control-magnitude-penalty-weight', '.002',
            '--output', str(task / 'model'),
            '--dataset', str(task / 'training_candidates.npz'),
            '--checkpoints-dir', str(task / 'checkpoints'),
            '--checkpoint-freq', '100000', '--validation-freq', '100000',
            '--validation-target-fractions', '0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0',
            '--validation-success-threshold', '.95', '--validation-consecutive', '2',
            '--best-policy-output', str(task / 'best_policy'),
            '--validation-report', str(task / 'validation_history.json')]
        if category == 'scissors':
            cmd.append('--require-drivable-pool')
        call(name, 'train', cmd)
        set_status(name, 'trained')

    def collect(item):
        category, variant, name, env_id = item
        task = root / name
        set_status(name, 'collecting')
        cmd = [sys.executable, '-m', 'improved_tds.experiments.collect_reach_successes',
            '--env-id', env_id, '--model', str(task / 'best_success_policy.zip'),
            '--pose-pool-dir', str(root / 'pose_pool'),
            '--output', str(task / 'success_postures.npz'), '--per-bin', '20',
            '--max-episodes', str(args.collection_max_episodes),
            '--seed', str(9300+{'scissors':0,'push':100,'spray':200}[category]+variant),
            '--scissors-control-version', 'smooth',
            '--tool-action-slew-rate', '1.25', '--tool-request-gap-penalty-weight', '1']
        call(name, 'collect', cmd)
        set_status(name, 'collected')

    def evaluate(item):
        _, variant, name, _ = item
        task = root / name
        set_status(name, 'evaluating_tds')
        cmd = [sys.executable, '-m', 'improved_tds.experiments.reach_tds_direct',
            '--source-root', str(root), '--task', name,
            '--output', str(task / 'tds_direct'),
            '--seed', str(20260927+variant), '--poses', '8', '--points', '17']
        call(name, 'tds', cmd)
        set_status(name, 'completed')

    def phase(items, action, workers):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(action, item): item for item in items}
            good = []
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                    good.append(item)
                except Exception as exc:
                    set_status(item[2], f'failed_{action.__name__}: {exc}')
            return good

    set_status(TASKS[0][2], 'queued_training')
    trained = phase(TASKS, train, args.learners)
    collected = phase(trained, collect, args.learners)
    if not args.skip_tds:
        phase(collected, evaluate, args.learners)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
