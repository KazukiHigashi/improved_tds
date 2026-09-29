"""Render the recorded 17 independent calibration scores every 25 control steps.

One representative calibration initial pose is replayed per task and method.
The physics, score, reset seed and controller are the same as in the frozen
reports; only RGB capture is enabled. The 2-DoF runs have grasp score zero and
are therefore identical to the corresponding 1-DoF runs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from improved_tds.environments.scissors_smooth import SmoothReachConfig
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.evaluate_synergy_sweep_coverage import _caption, _contact_sheet
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.reach_tds_direct import (
    reconstructed_posture, configure_scissors_evaluation_palm_radius)
from improved_tds.experiments.simple_synergy import direct_trajectory, simulation_step_seconds
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


TASKS = [f'{category}_lite_{variant}' for category in ('scissors', 'push', 'spray')
         for variant in range(1, 6)]


def report_path(task, output, report_root=None):
    root = (report_root if report_root is not None else
            output if not task.startswith('scissors') else
            output.parent/'paper15_unified_tds_corrected_20260928')
    return root/task/'report.json'


def render_one(task, output, source, report_root=None):
    report = json.loads(report_path(task, output, report_root).read_text(encoding='utf-8'))
    contract = report['database_metadata']['contract']
    version = report['database_metadata'].get('scissors_control_version', 'simple')
    smooth = (SmoothReachConfig(float(contract.get('action_delta_fraction', .05)),
        float(contract.get('request_gap_penalty_weight', 1.))) if version == 'smooth' else None)
    _register_tool_manip_gym_environments()
    env, _, _, key, lo, hi = _build_env(report['database_metadata']['env_id'],
        source/'pose_pool', report['seed'], research_task=True,
        scissors_control_version=version, scissors_smooth_config=smooth,
        tool_action_slew_rate=float(contract.get('tool_action_slew_rate', 0.)),
        tool_request_gap_penalty_weight=float(contract.get('tool_request_gap_penalty_weight', 1.)),
        object_position_noise=0., max_episode_steps=10000,
        force_floor_contact=task.startswith('spray'), render_mode='rgb_array')
    if task.startswith('scissors'):
        configure_scissors_evaluation_palm_radius(env,
            float(report.get('scissors_evaluation_palm_radius_multiplier',1.)),
            report.get('source_policy_palm_sphere_radius'))
    task_output = output/'trajectory_frames'/task
    task_output.mkdir(parents=True, exist_ok=True)
    pose = int(report['calibration_poses'][0])
    low, high, _ = _control_bounds(env)
    mean_q = np.asarray(report['mean_q'])
    dt = simulation_step_seconds(env)
    entries = {}
    try:
        for method in ('pca', 'pls'):
            label = f'{method}_1d'
            config = report['configurations'][label]
            two_d = report['configurations'][f'{method}_2d']
            if config['grasp_score'] != 0 or two_d['grasp_score'] != 0:
                raise ValueError(f'{task}/{method}: 1D and 2D are not identical')
            operation = np.asarray(config['fit']['operation_axis'])
            grasp = np.asarray(config['fit']['grasp_axis'])
            levels = np.asarray(config['diagnostic_levels'])
            frames_by_score = []
            for index, rho in enumerate(levels):
                goal = reconstructed_posture(mean_q, operation,
                    config['operation_scale'], rho, grasp, config['grasp_scale'],
                    config['grasp_score'], low, high)
                frames = []
                def capture(step, obs, info):
                    image = Image.fromarray(env.render().copy())
                    image.thumbnail((256, 192))
                    support=(f'palm={bool(info.get("is_on_palm",False))} '
                        f'tip={bool(info.get("object_tipped",False))}'
                        if task.startswith('scissors') else
                        f'contact_gate={bool(info.get("grasp_maintained",False))} '
                        f'tip={bool(info.get("object_tipped",info.get("bottle_tipped",False)))} '
                        f'floor={info.get("floor_contact_enabled","static")}')
                    fields = [f'{task} {method.upper()} score {index+1}/17 rho={rho:+.3f}',
                        f'step={step} t={step*dt:.2f}s phase={"ramp" if step<=round(5/dt) else "hold"}',
                        f'tool={float(obs["achieved_goal"][0]):.4f} contacts={int(info.get("grasp_contact_count",0))}',
                        support]
                    frames.append(_caption(np.asarray(image), fields))
                result = direct_trajectory(env, pose, key, lo+.3*(hi-lo),
                    report['seed']+1000*pose, [goal], capture=capture,
                    support_criterion='task' if task.startswith('scissors') else 'no_fall',
                    require_floor_contact=task.startswith('spray'),
                    capture_every=25, capture_initial=True)['outcomes'][0]
                expected = config['calibration'][0]['outcomes'][index]
                angle_diff = abs(result['achieved']-expected['achieved'])
                tolerance = 1e-4 if task.startswith('push') else 5e-3
                if angle_diff > tolerance:
                    raise RuntimeError(f'{task}/{method}/{index}: replay mismatch '
                        f'valid {result["valid"]}/{expected["valid"]}, tool diff {angle_diff}')
                validity_matches_report=bool(result['valid']==expected['valid'])
                filename = f'{method}_score_{index+1:02d}.jpg'
                _contact_sheet(frames, f'{task} {method.upper()} score {index+1}/17 '
                    f'rho={rho:+.3f} report valid={expected["valid"]}; '
                    f'replay valid={result["valid"]}').save(task_output/filename,
                    quality=87, subsampling=0)
                frames_by_score.append(dict(score_index=index+1, rho=float(rho),
                    image=f'trajectory_frames/{task}/{filename}',
                    frames=len(frames), valid=bool(expected['valid']),
                    replay_valid=bool(result['valid']),
                    validity_matches_report=validity_matches_report,
                    achieved=float(result['achieved']),
                    replay_abs_tool_difference=float(angle_diff),
                    contact_gate_at_endpoint=bool(result['contact_held'])
                    if not task.startswith('scissors') else None))
            entries[method] = frames_by_score
    finally:
        env.close()
    manifest=dict(task=task, calibration_pose=pose, simulator_step_seconds=dt,
        source_report=str(report_path(task,output,report_root)),
        scissors_evaluation_palm_radius=report.get('palm_sphere_radius') if task.startswith('scissors') else None,
        ramp_seconds=5., hold_seconds=2., frame_interval_steps=25,
        screenshot_scope='one fixed calibration initial pose; all 17 scores; PCA/PLS 1D',
        two_dof_duplicate_of_one_dof=True, methods=entries,
        replay_validity_mismatches=sum(not item['validity_matches_report']
            for group in entries.values() for item in group),
        max_replay_abs_tool_difference=max(item['replay_abs_tool_difference']
            for group in entries.values() for item in group))
    (task_output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=Path('analysis/paper15_no_contact_floor_on_20260928'))
    parser.add_argument('--source', type=Path,
        default=Path('runs/paper15_unified_3m_slew_20260927'))
    parser.add_argument('--report-root',type=Path,
        help='read selected TDS report.json from this directory instead of the historical default')
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--categories',nargs='+',choices=('scissors','push','spray'),
        default=('scissors','push','spray'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--force', action='store_true',
        help='regenerate existing contact sheets and replay manifests')
    args = parser.parse_args()
    if args.task:
        render_one(args.task, args.output, args.source,args.report_root)
        return
    if not 1 <= args.workers <= 3:
        parser.error('workers must be 1..3')
    output = args.output.resolve()
    source = args.source.resolve()
    project = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(project/'src'), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    selected=[task for task in TASKS if task.split('_')[0] in args.categories]
    status = {task: 'queued' for task in selected}
    status_path = output/'trajectory_frames'/'status.json'
    status_path.parent.mkdir(parents=True, exist_ok=True)
    def update(task, value):
        status[task] = value
        status_path.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
            tasks=status),indent=2),encoding='utf-8')
    def call(task):
        task_dir = output/'trajectory_frames'/task
        if (task_dir/'manifest.json').is_file() and not args.force:
            return 'completed (existing)'
        task_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()),
            '--task', task, '--output', str(output), '--source', str(source)]
        if args.report_root is not None:
            command.extend(['--report-root',str(args.report_root.resolve())])
        with (task_dir/'render.log').open('w',encoding='utf-8') as out, \
             (task_dir/'render.err.log').open('w',encoding='utf-8') as err:
            result = subprocess.run(command,cwd=project,env=env,stdout=out,stderr=err)
        if result.returncode:
            raise RuntimeError(f'exit={result.returncode}; see {task_dir/"render.err.log"}')
        return 'completed'
    update(selected[0], 'queued')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(call,task):task for task in selected}
        for future in as_completed(futures):
            task=futures[future]
            try:
                update(task,future.result())
            except Exception as exc:
                update(task,f'failed: {exc}')
    print(status_path)


if __name__ == '__main__':
    main()
