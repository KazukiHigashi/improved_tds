"""Fixed open-loop preparation and scalar tool sweeps; no online policy or retries.

Calibration selects ONE command offset and transition duration, shared by both
methods and every held-out reset. Test poses never select their own reference.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from improved_tds.experiments.collect_posture_database import _build_env, _actuated_q
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.evaluate_strict_synergy_control import _jsonable
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.adaptive_range import amplitude, posture, select_interval


def preparation_candidates(data, cal_poses):
    """Actual successful command offsets, not a coordinate-wise synthetic median."""
    offsets = [np.zeros(data['q'].shape[1])]
    rows = []
    for pose in cal_poses:
        ids = np.flatnonzero((data['pose_index'] == pose) & data['success'].astype(bool))
        if not len(ids):
            continue
        row = int(ids[np.argmin(abs(data['target_fraction'][ids] - .3))])
        delta = data['command_q'][row] - data['initial_q'][row]
        if not any(np.allclose(delta, x) for x in offsets):
            offsets.append(delta)
            rows.append(row)
    return offsets, rows


def steps(seconds, dt):
    return max(1, int(np.ceil(seconds / dt - 1e-10)))


def simulation_step_seconds(env):
    """Read task control period without evaluating an unavailable fallback."""
    raw = env.unwrapped
    dt = getattr(raw, 'dt', None)
    if dt is None:
        dt = float(raw.model.opt.timestep) * int(raw.frame_skip)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('invalid simulator control period')
    return dt


def state(info, *, support_criterion='task'):
    if support_criterion not in ('task', 'no_fall'):
        raise ValueError(f'unknown support criterion: {support_criterion}')
    palm_only='is_on_palm' in info
    fallen=bool(info.get('object_dropped', False) or info.get('object_tipped', False)
                or info.get('bottle_dropped', False) or info.get('bottle_tipped', False))
    if support_criterion == 'no_fall' and palm_only:
        raise ValueError('no_fall is restricted to Push and Spray')
    if support_criterion == 'no_fall' and not (
        {'object_dropped', 'object_tipped'} <= info.keys() or
        {'bottle_dropped', 'bottle_tipped'} <= info.keys()
    ):
        raise KeyError('no_fall requires explicit drop and tip observations')
    failed=bool(info.get('task_failure', False) or (not palm_only and fallen))
    held=(not failed if support_criterion == 'no_fall' else
          bool(info.get('grasp_maintained', False)))
    return dict(held=held, contact_held=bool(info.get('grasp_maintained', False)),
                failed=failed,
                **{k: bool(info.get(k, False)) for k in
                   ('failure_contact', 'failure_displacement', 'failure_rotation')})


def trajectory(env, pose, key, target, seed, offset, seconds, destinations, *, capture=None):
    """One reset, fixed preparation, then continuous FF destinations.

    destinations are offsets relative to the prepared reference. No state-based
    correction or retry. Failure stops the trajectory and leaves explicit missing
    records. A valid point requires every frame in a two-second terminal window.
    """
    obs, info = env.reset(seed=seed, options={'initial_pose_index': int(pose), key: float(target)})
    low, high, half = _control_bounds(env)
    neutral = np.asarray(env.unwrapped._neutral_ctrl).copy()
    reference = np.clip(_actuated_q(env) + offset, low, high)
    dt = simulation_step_seconds(env)
    done = False
    elapsed = 0

    def move(origin, goal, duration, hold):
        nonlocal obs, info, done, elapsed
        holding = []
        ramp_held = []
        short = None
        n = steps(duration, dt)
        for i in range(n + steps(hold, dt)):
            q = origin + min(1., (i + 1) / n) * (goal - origin)
            action = np.clip((q - neutral) / half, -1., 1.).astype(np.float32)
            warnings_before = numerical_warnings(env)
            obs, _, terminated, truncated, info = env.step(action)
            numerical = (bool(info.get('numerical_failure', False) or info.get('failure_numerical', False))
                         or np.any(numerical_warnings(env) > warnings_before)
                         or not np.isfinite(_actuated_q(env)).all()
                         or not np.isfinite(obs['achieved_goal']).all())
            if numerical:
                info = dict(info, task_failure=True, grasp_maintained=False)
                terminated = True
            elapsed += 1
            now = state(info)
            (holding if i >= n else ramp_held).append(now['held'] and not now['failed'])
            done = bool(terminated or truncated)
            if len(holding) == 5:
                short = dict(achieved=float(obs['achieved_goal'][0]),
                             valid=bool(all(holding) and not done),
                             measured_q=_actuated_q(env).tolist())
            if capture is not None and (elapsed % 20 == 0 or done):
                capture(elapsed, obs, info)
            if done:
                break
        valid = not done and len(holding) == steps(hold, dt) and all(holding)
        return dict(**state(info), valid=valid, achieved=float(obs['achieved_goal'][0]),
                    short_window=short,
                    executed_steps=elapsed, terminated=bool(terminated), truncated=bool(truncated),
                    failure_numerical=bool(numerical),
                    hold_fraction=float(np.mean(holding)) if holding else 0.,
                    ramp_held_fraction=float(np.mean(ramp_held)) if ramp_held else 0.,
                    measured_q=_actuated_q(env).tolist(), command_q=goal.tolist())

    prep = move(neutral, reference, seconds, 1.)
    outcomes = []
    origin = reference
    for delta in destinations:
        if done:
            outcomes.append(None)
            continue
        goal = np.clip(reference + delta, low, high)
        item = move(origin, goal, 5., 2.)
        # Never silently accept a tool trial whose fixed preparation failed.
        item['valid'] = bool(prep['valid'] and item['valid'])
        if item['short_window'] is not None:
            item['short_window']['valid'] = bool(
                prep['short_window'] is not None and prep['short_window']['valid']
                and item['short_window']['valid'])
        outcomes.append(item)
        origin = goal
    return dict(preparation=prep, outcomes=outcomes, reference=reference.tolist())


def direct_trajectory(env, pose, key, target, seed, destinations, *, capture=None,
                      support_criterion='task', require_floor_contact=False,
                      capture_every=20, capture_initial=False):
    """Move from the measured reset posture directly to absolute hand postures.

    Each destination is a reconstructed joint posture, never an offset from a
    learned preparation. Independent calls reset once per score; a multi-point
    call makes a continuous path between reconstructed postures. Every segment
    has a five-second linear command ramp and a two-second terminal hold.
    """
    obs, info = env.reset(seed=seed, options={'initial_pose_index': int(pose), key: float(target)})
    if require_floor_contact and info.get('floor_contact_enabled') is not True:
        raise RuntimeError('floor contact must be enabled for this experiment')
    if capture_every < 1:
        raise ValueError('capture_every must be positive')
    if capture is not None and capture_initial:
        capture(0, obs, info)
    low, high, half = _control_bounds(env)
    neutral = np.asarray(env.unwrapped._neutral_ctrl, dtype=np.float64)
    origin = _actuated_q(env)
    initial = origin.copy()
    dt = simulation_step_seconds(env)
    done = False
    elapsed = 0
    outcomes = []
    for destination in destinations:
        if done:
            outcomes.append(None)
            continue
        goal = np.clip(np.asarray(destination, dtype=np.float64), low, high)
        if goal.shape != initial.shape or not np.isfinite(goal).all():
            raise ValueError('absolute reconstructed posture has invalid shape or values')
        held = []
        ramp_held = []
        short = None
        numerical = False
        terminated = truncated = False
        ramp_steps = steps(5., dt)
        hold_steps = steps(2., dt)
        for i in range(ramp_steps + hold_steps):
            q = origin + min(1., (i + 1) / ramp_steps) * (goal - origin)
            action = np.clip((q - neutral) / half, -1., 1.).astype(np.float32)
            warnings_before = numerical_warnings(env)
            obs, _, terminated, truncated, info = env.step(action)
            if require_floor_contact and info.get('floor_contact_enabled') is not True:
                raise RuntimeError('floor contact became disabled during this experiment')
            numerical = (bool(info.get('numerical_failure', False) or info.get('failure_numerical', False))
                         or np.any(numerical_warnings(env) > warnings_before)
                         or not np.isfinite(_actuated_q(env)).all()
                         or not np.isfinite(obs['achieved_goal']).all())
            if numerical:
                info = dict(info, task_failure=True, grasp_maintained=False)
                terminated = True
            elapsed += 1
            current = state(info, support_criterion=support_criterion)
            (held if i >= ramp_steps else ramp_held).append(current['held'] and not current['failed'])
            done = bool(terminated or truncated)
            if len(held) == 5:
                short = dict(achieved=float(obs['achieved_goal'][0]),
                             valid=bool(all(held) and not done),
                             measured_q=_actuated_q(env).tolist())
            if capture is not None and (elapsed % capture_every == 0 or done):
                capture(elapsed, obs, info)
            if done:
                break
        valid = not done and len(held) == hold_steps and all(held)
        outcomes.append(dict(**state(info, support_criterion=support_criterion), valid=valid,
            achieved=float(obs['achieved_goal'][0]),
            short_window=short, executed_steps=elapsed, terminated=bool(terminated),
            truncated=bool(truncated), failure_numerical=bool(numerical),
            hold_fraction=float(np.mean(held)) if held else 0.,
            ramp_held_fraction=float(np.mean(ramp_held)) if ramp_held else 0.,
            measured_q=_actuated_q(env).tolist(), command_q=goal.tolist()))
        origin = goal
    return dict(initial_q=initial.tolist(), outcomes=outcomes)


def numerical_warnings(env):
    import mujoco
    warning = getattr(env.unwrapped.data, 'warning', None)
    if warning is None:
        return np.zeros(3, dtype=int)
    return np.array([warning[int(k)].number for k in
                     (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                      mujoco.mjtWarning.mjWARN_BADQACC)])


def matrices(records, size):
    y = np.full((len(records), size), np.nan)
    valid = np.zeros(y.shape, dtype=bool)
    for i, record in enumerate(records):
        for j, item in enumerate(record['outcomes'][:size]):
            if item is not None:
                y[i, j] = item['achieved']
                valid[i, j] = item['valid']
    return y, valid


def coverage(records, low, high):
    values = [x['achieved'] for x in records if x is not None and x['valid']]
    return max(0., min(high, max(values)) - max(low, min(values))) / (high-low) if values else 0.


def audit_policy(source, inputs, root):
    """Offline audit only: RL reach, then freeze the applied command for 2 s.

    This policy is never used by the simple controller. Three fixed target bins
    and the eight Calibration resets are audited without retries.
    """
    from stable_baselines3 import SAC
    env, _, _, key, lo, hi = _build_env(source['env_id'], inputs, source['split_seed'],
        research_task=True, scissors_control_version=source.get('rl_scissors_control_version', 'v2'),
        object_position_noise=0., max_episode_steps=500)
    results = []
    def advance(action):
        before = numerical_warnings(env)
        obs, reward, done, trunc, info = env.step(action)
        bad = (np.any(numerical_warnings(env) > before)
               or not np.isfinite(_actuated_q(env)).all()
               or not np.isfinite(obs['achieved_goal']).all())
        if bad:
            info = dict(info, task_failure=True, grasp_maintained=False, is_success=False,
                        is_stable_success=False, failure_numerical=True)
        return obs, reward, bool(done or bad), trunc, info
    try:
        policy = SAC.load(inputs/'policy.zip', env=env, device='cpu')
        for pose in source['calibration_poses']:
            for fraction in (.1, .5, .9):
                target = lo + fraction*(hi-lo)
                obs, info = env.reset(seed=source['split_seed']+1000*pose,
                                     options={'initial_pose_index': pose, key: target})
                reached = False
                for _ in range(200):
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, _, done, trunc, info = advance(action)
                    reached = bool(info.get('is_stable_success', False)) and not(done or trunc)
                    if reached or done or trunc:
                        break
                result = dict(pose=pose, target=target, rl_stable_success=reached,
                              rl_success_hold_steps=int(info.get('success_hold_steps', 0)),
                              dt=env.unwrapped.dt, frozen_hold_success=None, frozen_grasp_success=None,
                              continued_rl_hold_success=None, continued_rl_grasp_success=None,
                              continued_rl_reached_success=None)
                if reached:
                    # Applied (not requested) command, including any RL rate limit.
                    _, _, half = _control_bounds(env)
                    u = env.unwrapped
                    command = u.data.ctrl[u._actuator_ids].copy()
                    frozen = np.clip((command-u._neutral_ctrl)/half, -1, 1).astype(np.float32)
                    success, held = [], []
                    count = steps(2., u.dt)
                    for _ in range(count):
                        obs, _, done, trunc, info = advance(frozen)
                        success.append(bool(info.get('is_success', False)))
                        held.append(state(info)['held'] and not state(info)['failed'])
                        if done or trunc:
                            break
                    result.update(frozen_hold_success=len(success)==count and all(success) and not(done or trunc),
                                  frozen_grasp_success=len(held)==count and all(held) and not(done or trunc),
                                  final_angle=float(obs['achieved_goal'][0]), final_state=state(info))
                    # Matched-reset continued-RL control, offline audit only.
                    obs, info = env.reset(seed=source['split_seed']+1000*pose,
                                         options={'initial_pose_index': pose, key: target})
                    reached_again = False
                    for _ in range(200):
                        action, _ = policy.predict(obs, deterministic=True)
                        obs, _, done, trunc, info = advance(action)
                        reached_again = bool(info.get('is_stable_success', False)) and not(done or trunc)
                        if reached_again or done or trunc:
                            break
                    result['continued_rl_reached_success'] = reached_again
                    if reached_again:
                        successes, holds = [], []
                        for _ in range(count):
                            action, _ = policy.predict(obs, deterministic=True)
                            obs, _, done, trunc, info = advance(action)
                            successes.append(bool(info.get('is_success', False)))
                            holds.append(state(info)['held'] and not state(info)['failed'])
                            if done or trunc:
                                break
                        result.update(continued_rl_hold_success=len(successes)==count and all(successes) and not(done or trunc),
                                      continued_rl_grasp_success=len(holds)==count and all(holds) and not(done or trunc))
                results.append(result)
    finally:
        env.close()
    (root/'hold_audit.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-policy', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    def progress(phase):
        from datetime import datetime, timezone
        (output/'progress.json').write_text(json.dumps(dict(phase=phase,
            utc=datetime.now(timezone.utc).isoformat())),encoding='utf-8')
    progress('selecting_fixed_preparation')
    source = json.loads((args.source_root/'strict_reports'/f'{args.task}_seed{args.seed}.json').read_text())
    previous = json.loads((args.source_root/'reports'/f'{args.task}_seed{args.seed}.json').read_text())
    assert not set(source['calibration_poses']) & set(source['test_poses'])
    assert not set(source['fit_poses']) & set(source['test_poses'])
    with np.load(source['database']) as z:
        data = {k:z[k].copy() for k in z.files if k != 'metadata_json'}
    inputs = args.source_root/'inputs'/args.task
    _register_tool_manip_gym_environments()
    env, _, _, key, lo, hi = _build_env(source['env_id'], inputs, args.seed, research_task=True,
        object_position_noise=0., max_episode_steps=10000,
        render_mode='rgb_array' if args.render else None)
    result = dict(protocol='simple_ff_v1', task=args.task, split_seed=args.seed, target_range=[lo, hi],
        source_report=str((args.source_root/'strict_reports'/f'{args.task}_seed{args.seed}.json').resolve()),
        fit_rows=source['fit_rows'], fit_poses=source['fit_poses'], calibration_poses=source['calibration_poses'],
        test_poses=source['test_poses'], online_rl=False, state_restoration=False, test_retries=0,
        preparation_ramp_seconds=5., operation_ramp_seconds=5.,
        preparation_hold_seconds=1., operation_hold_seconds=2., configurations={})
    try:
        offsets, rows = preparation_candidates(data, source['calibration_poses'])
        candidates = []
        for index, offset in enumerate(offsets):
            for duration in (5.,):
                records = [trajectory(env, pose, key, lo+.3*(hi-lo), args.seed+1000*pose,
                                      offset, duration, [])['preparation'] for pose in source['calibration_poses']]
                candidates.append(dict(index=index, duration=duration, offset=offset.tolist(), records=records,
                    valid_count=sum(x['valid'] for x in records)))
        # Same preparation for PCA and PLS; not selected by Test coverage/error.
        selected = max(candidates, key=lambda c:(c['valid_count'], c['duration'], -np.linalg.norm(c['offset'])))
        offset = np.array(selected['offset'])
        result.update(preparation_candidates=candidates, preparation_source_rows=rows, preparation=selected)
        (output/'preparation.json').write_text(json.dumps(result, indent=2, default=_jsonable), encoding='utf-8')
        low, high, _ = _control_bounds(env)
        levels = np.linspace(-2., 2., 17)
        path = np.r_[levels, levels[::-1]]
        for method in ('pca', 'pls'):
            progress('evaluating_'+method)
            axis = np.array(previous['configurations'][method]['operation_axis'])
            scale = amplitude(axis, low, high)
            deltas = [rho*scale*axis for rho in path]
            # Identify the range without letting an unsafe extreme censor all
            # subsequent scores. Each point has the same fixed preparation.
            def independent(pose):
                trials = [trajectory(env, pose, key, lo+.3*(hi-lo), args.seed+1000*pose,
                                     offset, selected['duration'], [rho*scale*axis]) for rho in levels]
                return dict(preparation=trials[0]['preparation'],
                            outcomes=[x['outcomes'][0] for x in trials])
            cal = [independent(pose) for pose in source['calibration_poses']]
            y, valid = matrices(cal, len(levels))
            selection = select_interval(levels, y, valid, [lo, hi])
            sweep_levels = (np.linspace(selection['rho_start'], selection['rho_end'], 13)
                            if selection is not None else levels)
            path = np.r_[sweep_levels, sweep_levels[::-1]]
            deltas = [rho*scale*axis for rho in path]
            tests, images = [], []
            for pose in source['test_poses']:
                frames = []
                do_render = args.render and pose == source['test_poses'][0]
                def capture(step, obs, info):
                    from improved_tds.experiments.evaluate_synergy_sweep_coverage import _caption
                    frames.append(_caption(env.render().copy(), [
                        f'{args.task} / {method.upper()} / continuous forward and reverse / step {step}',
                        f'tool={float(obs["achieved_goal"][0]):.4f} held={bool(info.get("grasp_maintained",False))}',
                        'fixed open-loop preparation; no online RL; no state restoration']))
                trial = trajectory(env, pose, key, lo+.3*(hi-lo), args.seed+1000*pose,
                    offset, selected['duration'], deltas, capture=capture if do_render else None)
                diagnostic = independent(pose)
                trial.update(pose=pose, coverage=coverage(trial['outcomes'], lo, hi), target_trials=[],
                             diagnostic=diagnostic,
                             diagnostic_coverage=coverage(diagnostic['outcomes'], lo, hi))
                if selection is not None:
                    for target in np.linspace(lo, hi, 9):
                        score = float(np.interp(target, selection['map_tool'], selection['map_u']))
                        rho = selection['rho_start']+(score+1)/2*(selection['rho_end']-selection['rho_start'])
                        inv = trajectory(env, pose, key, target, args.seed+1000*pose, offset,
                                         selected['duration'], [rho*scale*axis])
                        item = inv['outcomes'][0]
                        trial['target_trials'].append(dict(target=float(target), score=score, rho=rho,
                            preparation=inv['preparation'], outcome=item,
                            abs_error=abs(item['achieved']-target) if item is not None else None))
                tests.append(trial)
                if frames:
                    from improved_tds.experiments.evaluate_synergy_sweep_coverage import _contact_sheet
                    # Page images keep every captured frame, including early failures.
                    for page, start in enumerate(range(0, len(frames), 40)):
                        name=f'{method}_timeline_{page}.png'
                        _contact_sheet(frames[start:start+40],f'{method.upper()} / every 20 steps / page {page+1}').save(output/name)
                        images.append(name)
            # Identical Fit100 overlays. Physical projection is diagnostic, not causal calibration.
            ids=np.asarray(source['fit_rows'],dtype=int)
            projection=(data['q'][ids]-data['initial_q'][ids]-offset)@axis/scale
            result['configurations'][method]=dict(axis=axis.tolist(), scale=scale, levels=path.tolist(),
                diagnostic_levels=levels.tolist(),
                calibration=cal, selection=selection, test=tests, images=images,
                fit_rho=projection.tolist(), fit_tool=data['tool_state'][ids].tolist(),
                mean_coverage=float(np.mean([x['coverage'] for x in tests])))
            print(args.task, args.seed, method, 'coverage',result['configurations'][method]['mean_coverage'],flush=True)
        (output/'report.json').write_text(json.dumps(result,indent=2,default=_jsonable),encoding='utf-8')
    finally:
        env.close()
    if args.audit_policy:
        progress('offline_retention_audit')
        audit_policy(source, inputs, output)
    progress('completed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
