"""Four-arm FF experiment: legacy, direction, range, joint grasp/range selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from improved_tds.synergy.adaptive_range import amplitude, correlation, posture, select_interval
from improved_tds.experiments.collect_posture_database import _actuated_q, _build_env
from improved_tds.experiments.evaluate_strict_synergy_control import _reference_for_pose, _posture_on_axis, _jsonable
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds, _execute
from improved_tds.experiments.evaluate_synergy_sweep_coverage import _caption, _contact_sheet, interval_coverage
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


ARMS = ('legacy', 'direction', 'range', 'joint')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-report', type=Path, required=True)
    parser.add_argument('--pose-pool-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-dir', type=Path)
    parser.add_argument('--pilot', action='store_true', help='One method and two extra grasp candidates; not a production result')
    parser.add_argument('--grasp-method',choices=('force_pls','residual_pca'),default='force_pls')
    args = parser.parse_args(argv)
    source = json.loads(args.strict_report.read_text(encoding='utf-8'))
    with np.load(source['database'], allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files if k != 'metadata_json'}
    assert not set(source['calibration_poses']) & set(source['test_poses'])
    assert not set(source['fit_poses']) & set(source['test_poses'])
    _register_tool_manip_gym_environments()
    env, pool, _, option, lo, hi = _build_env(
        source['env_id'], args.pose_pool_dir, source['split_seed'],
        max_episode_steps=210, object_position_noise=0.,
        render_mode='rgb_array' if args.render_dir else None,
        research_task=source.get('research_task',False),
    )
    low, high, _ = _control_bounds(env)
    nominal = lo + .3 * (hi-lo)
    targets = np.linspace(lo, hi, 9)
    output = dict(schema_version=1, pilot=args.pilot, source_report=str(args.strict_report.resolve()),
                  grasp_method='one_dof' if source['env_id'].startswith('Scissors') else args.grasp_method,
                  research_task=source.get('research_task',False),
                  rl_scissors_control_version=source.get('rl_scissors_control_version','legacy'),
                  ff_control_version=source.get('ff_control_version','legacy'),
                  grasp_signal_definition=source.get('grasp_signal_definition','legacy grasp metric'),
                  env_id=source['env_id'], split_seed=source['split_seed'],
                  fit_poses=source['fit_poses'], calibration_poses=source['calibration_poses'],
                  test_poses=source['test_poses'], target_range=[lo, hi], configurations={})
    try:
        for method in (('pls',) if args.pilot else ('pca', 'pls')):
            key = method + ('__one_dof' if source['env_id'].startswith('Scissors') else '__'+args.grasp_method)
            original = source['configurations'][key]
            if original.get('available') is False:
                raise ValueError(f'Grasp method {key} unavailable: {original["reason"]}')
            axis = np.asarray(original['operation_axis'])
            grasp_axis = None if original['grasp_axis'] is None else np.asarray(original['grasp_axis'])
            scale = amplitude(axis, low, high)
            measured = np.asarray(source['measured_reference_delta'])
            command = np.asarray(source['command_reference_delta'])
            initial = (float(original['reference_preload']), float(original['normalised_grasp']))
            cache = {}

            def grasp_for(pose, candidate, seed):
                preload, g = candidate
                reference = _reference_for_pose(env, int(pose), option, nominal,
                    measured + preload*(command-measured), low, high, seed)
                return reference if grasp_axis is None else _posture_on_axis(reference, grasp_axis, g, low, high)[0]

            def trial(pose, candidate, rho, *, target=nominal, render=False, timeline=False):
                # Matched reset seed across scores, candidates, arms and targets.
                seed = int(source['split_seed']) + 1000*int(pose)
                grasp = grasp_for(pose, candidate, seed)
                q = posture(grasp, axis, scale, rho, low, high)
                timeline_frames=[]
                def capture(step,observation,info):
                    timeline_frames.append(_caption(env.render().copy(),[
                        f'{source["env_id"]} | {method.upper()} | joint | step {step}',
                        f'target={target:.5f} actual={float(observation["achieved_goal"][0]):.5f}',
                        f'phase={"grasp" if step<=100 else "operation ramp" if step<=150 else "hold"} held={bool(info.get("grasp_maintained",False))}']))
                result = _execute(env, int(pose), option, float(target), grasp, q,
                                  seed=seed, grasp_steps=100, ramp_steps=50, hold_steps=50,
                                  frame_callback=capture if timeline else None)
                if timeline_frames:
                    timeline_path=args.render_dir/f'{method}__joint_target_timeline.png'
                    timeline_path.parent.mkdir(parents=True,exist_ok=True)
                    _contact_sheet(timeline_frames,f'{method.upper()} / joint / target={target:.5f}: every 20 steps (terminal frame included)').save(timeline_path)
                    result['timeline_image']=str(timeline_path.resolve())
                measured_q = _actuated_q(env)
                result.update(rho=float(rho), physical_coordinate=float(rho*scale),
                              command_delta=float(np.linalg.norm(q-grasp)),
                              measured_delta=float(np.linalg.norm(measured_q-grasp)))
                frame = env.render().copy() if render else None
                return result, frame

            def calibrate(candidate, levels):
                records = []
                for pose in source['calibration_poses']:
                    row = []
                    for rho in levels:
                        ck = (int(pose), *candidate, round(float(rho), 8))
                        if ck not in cache:
                            cache[ck] = trial(pose, candidate, rho)[0]
                        row.append(cache[ck])
                    records.append(row)
                y = np.asarray([[r['achieved'] for r in row] for row in records])
                valid = np.asarray([[r['held'] and not r['failed'] for r in row] for row in records])
                return records, y, valid

            fixed_levels = np.linspace(-1., 1., 13)
            fixed_records, fixed_y, fixed_valid = calibrate(initial, fixed_levels)
            # Reconstruct the legacy cumulative-max map on matched reset trials.
            usable = fixed_valid.sum(axis=0) > 0
            legacy_y = [float(fixed_y[fixed_valid[:, i], i].mean()) for i in np.flatnonzero(usable)]
            legacy_tool = np.maximum.accumulate(legacy_y) if legacy_y else np.array([])
            distinct = np.r_[True, np.diff(legacy_tool)>1e-9] if len(legacy_tool) else np.array([], bool)
            legacy = dict(rho_start=-1., rho_end=1., direction=1,
                          map_u=fixed_levels[usable][distinct].tolist(), map_tool=legacy_tool[distinct].tolist(),
                          raw_rho=fixed_levels.tolist(), raw_tool=fixed_y.mean(axis=0).tolist())
            direction = select_interval(fixed_levels, fixed_y, fixed_valid, [lo, hi], full=True)
            # Grid spacing .25 includes the endpoints for amplitudes .5, 1, 1.5, 2.
            # Stop extending when every pose has identical clipped commands.
            extended = np.linspace(-2., 2., 17)

            def effective_levels(candidate):
                grasps = [grasp_for(p, candidate, source['split_seed']+1000*int(p)) for p in source['calibration_poses']]
                paths = np.asarray([[posture(g, axis, scale, r, low, high) for r in extended] for g in grasps])
                keep = np.r_[True, np.max(np.linalg.norm(np.diff(paths, axis=1), axis=2), axis=0)>1e-8]
                return extended[keep]

            candidates = [initial]
            if grasp_axis is not None:
                gs = (0., -.5) if args.pilot else (-1., -.5, 0., .5, 1.)
                candidates += [(initial[0], g) for g in gs]
            if not args.pilot:
                candidates += [(p, 0.) for p in (0., .5, 1.)]
            candidates = list(dict.fromkeys(candidates))
            summaries = []
            best = None
            range_choice = None
            cal_records = []
            for candidate in candidates:
                levels = effective_levels(candidate)
                records, y, valid = calibrate(candidate, levels)
                selection = select_interval(levels, y, valid, [lo, hi])
                summaries.append(dict(candidate=list(candidate), selection=selection))
                cal_records.append(dict(candidate=list(candidate), levels=levels.tolist(), outcomes=records))
                if candidate == initial:
                    range_choice = selection
                if selection is not None:
                    quality = (selection['coverage'], selection['held_rate'], selection['direction_agreement'],
                               -selection['calibration_interpolated_normalised_mae'], -selection['reverse_travel'])
                    if best is None or quality > best[0]:
                        best = (quality, candidate, selection)
            choices = dict(legacy=(initial, legacy), direction=(initial, direction),
                           range=(initial, range_choice), joint=(best[1], best[2]) if best else (initial, None))
            output['configurations'][method] = dict(fit_correlation=original['operation_tool_correlation'],
                operation_axis=axis, nominal_amplitude=scale, candidates=summaries,
                fixed_calibration=dict(levels=fixed_levels.tolist(), outcomes=fixed_records),
                calibration_records=cal_records, arms={})
            for arm, (candidate, choice) in choices.items():
                result = dict(candidate=list(candidate), selection=choice, calibrated=choice is not None,
                              poses=[], fit_samples={}, images=[])
                output['configurations'][method]['arms'][arm] = result
                if choice is None:
                    result['reason'] = 'No interval met held-cohort, direction, range and coverage criteria on calibration poses.'
                    continue
                a, b = choice['rho_start'], choice['rho_end']
                result['grasp_score'] = float(-1 - 2*a/(b-a))
                # Unclipped projection: outside-domain samples remain visible.
                rows = np.asarray(source['fit_rows'], dtype=int)
                scores, tools, distances = [], [], []
                for row in rows:
                    reference = np.clip(data['initial_q'][row] + measured + candidate[0]*(command-measured), low, high)
                    grasp = reference if grasp_axis is None else _posture_on_axis(reference, grasp_axis, candidate[1], low, high)[0]
                    rho = float((data['q'][row]-grasp) @ axis / scale)
                    scores.append(-1 + 2*(rho-a)/(b-a))
                    tools.append(float(data['tool_state'][row]))
                    distances.append(float(np.linalg.norm(data['q'][row] - (grasp+rho*scale*axis))))
                result['fit_samples'] = dict(score=scores, tool=tools, off_axis_distance=distances,
                    definition='Unclipped linear projection relative to candidate grasp; not nearest-point projection on clipped path.')
                render_pose = int(source['test_poses'][0]) # predetermined; no Test-based selection
                frames = []
                for pose in source['test_poses']:
                    outcomes = []
                    for u in np.linspace(-1., 1., 13):
                        rho = a + (u+1)/2*(b-a)
                        do_render = bool(args.render_dir) and int(pose) == render_pose
                        item, frame = trial(pose, candidate, rho, render=do_render)
                        item['score'] = float(u)
                        outcomes.append(item)
                        if frame is not None:
                            frames.append(_caption(frame, [f'{source["env_id"]} | {method.upper()} | {arm} | pose {pose}',
                                f'u={u:+.2f} rho={rho:+.3f} tool={item["achieved"]:.5f} held={item["held"]}',
                                f'command delta={item["command_delta"]:.3f} measured delta={item["measured_delta"]:.3f}']))
                    errors = []
                    if len(choice['map_tool']) >= 2:
                        for target in targets:
                            u = float(np.interp(target, choice['map_tool'], choice['map_u']))
                            rho = a + (u+1)/2*(b-a)
                            item, _ = trial(pose, candidate, rho, target=target,
                                timeline=bool(args.render_dir) and arm=='joint' and int(pose)==render_pose
                                    and np.isclose(target,targets[len(targets)//2]))
                            if 'timeline_image' in item:
                                result['images'].append(item['timeline_image'])
                            item.update(target=float(target), score=u, abs_error=abs(item['achieved']-target),
                                calibration_reachable=bool(choice['map_tool'][0] <= target <= choice['map_tool'][-1]))
                            errors.append(item)
                    else:
                        result['inverse_map_failure'] = True
                    valid_y = [o['achieved'] for o in outcomes if o['held'] and not o['failed']]
                    result['poses'].append(dict(pose=int(pose), outcomes=outcomes, target_trials=errors,
                        held_coverage=interval_coverage(np.asarray(valid_y), lo, hi)))
                if frames:
                    image_path = args.render_dir / f'{method}__{arm}.png'
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    _contact_sheet(frames, f'{method.upper()} / {arm}: selected u range; 13 independent reset trials').save(image_path)
                    result['images'].append(str(image_path.resolve()))
                result['test_mean_correlation'] = correlation(np.linspace(-1, 1, 13), np.mean(
                    [[o['achieved'] for o in p['outcomes']] for p in result['poses']], axis=0))
            print(f'completed {source["env_id"]} {method}', flush=True)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(output, indent=2, default=_jsonable), encoding='utf-8')
    temp.replace(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
