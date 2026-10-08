"""Matched 5 s FF + 2 s feedback experiments against a frozen paper bundle."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from improved_tds.control.axis_feedback import AxisFeedback, AxisFeedbackConfig, estimate_sensitivity


CONDITIONS = ('ff', 'axis1_fb', 'axis2_fb')
DEFAULT_TASKS = ('scissors_lite_3', 'scissors_lite_4', 'push_lite_3', 'spray_lite_3')
BASELINE_MANIFEST_SHA256 = '0f91d3853eb90080649de62dc2ae77d3d5ac5daf660011ae8c61b3e903e6733d'


def reset_pristine_simulator(env):
    """Undo pose-pool model/neutral edits before the base task's settling reset.

    The pool changes the mount and neutral command AFTER base.reset(). Without
    restoring their construction values, its next settling phase inherits the
    previous trial's pose even if qpos/qvel or MjData have been reset.
    """
    import mujoco
    raw = env.unwrapped
    if not hasattr(raw, '_axis_feedback_reset_reference'):
        raw._axis_feedback_reset_reference = (raw.model.body_pos.copy(),
            raw.model.body_quat.copy(), raw._neutral_ctrl.copy())
    pos, quat, neutral = raw._axis_feedback_reset_reference
    raw.model.body_pos[:] = pos
    raw.model.body_quat[:] = quat
    raw._neutral_ctrl = neutral.copy()
    mujoco.mj_resetData(raw.model, raw.data)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    """Strict JSON: missing/numerically failed observations are null, never NaN."""
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, np.generic):
            return v.item()
        return v
    # Trial traces dominate the full experiment. Compact JSON preserves every
    # value while avoiding many gigabytes of whitespace-only output.
    compact = path.name in ('report.json', 'partial_report.json')
    path.write_text(json.dumps(clean(value), ensure_ascii=False,
                               separators=(',', ':') if compact else None,
                               indent=None if compact else 2, allow_nan=False),
                    encoding='utf-8')


def inverse_score(config, target):
    selection = config['selection']
    domain = config['inverse_target_range']
    if not selection or domain is None or not domain[0] <= target <= domain[1]:
        raise ValueError('target outside frozen inverse domain; no extrapolation')
    if not np.isfinite(selection['map_tool']).all() or not np.all(np.diff(selection['map_tool'])>0):
        raise ValueError('frozen inverse tool knots must be strictly increasing')
    u = float(np.interp(target, selection['map_tool'], selection['map_u']))
    return float(selection['rho_start'] + (u+1)/2 *
                 (selection['rho_end']-selection['rho_start']))


def common_targets(configs, count):
    domains = [c['inverse_target_range'] for c in configs]
    if any(d is None for d in domains):
        raise ValueError('frozen inverse missing')
    low, high = max(d[0] for d in domains), min(d[1] for d in domains)
    if count < 1 or high <= low:
        raise ValueError('no common inverse domain')
    # Interior points: do not evaluate only the calibration endpoints.
    return np.linspace(low, high, count+2)[1:-1].tolist(), [low, high]


def additional_clipping(command, reference, low, high, neutral, half, tolerance=1e-8):
    """Whether a correction worsens either FF joint-limit or action saturation.

    A baseline FF joint may already be clipped. Motion into the feasible range
    is still allowed; only an increased saturation magnitude is rejected.
    """
    command, reference = np.asarray(command), np.asarray(reference)
    command_q, reference_q = np.clip(command, low, high), np.clip(reference, low, high)
    joint_excess = abs(command-command_q) - abs(reference-reference_q)
    command_action, reference_action = (command_q-neutral)/half, (reference_q-neutral)/half
    action_excess = (abs(command_action-np.clip(command_action,-1.,1.)) -
                     abs(reference_action-np.clip(reference_action,-1.,1.)))
    return bool(np.any(joint_excess>tolerance) or np.any(action_excess>tolerance))


class ContactAudit:
    """Hand/external-body contacts only; self contacts and floor are excluded.

    MuJoCo normal forces [N] are simulation diagnostics, NOT measured tendons.
    Unique hand-link/body pairs avoid contact-point tessellation dependence.
    """
    def __init__(self, env):
        import mujoco
        self.raw = env.unwrapped
        m = self.raw.model
        self.hand = {i for i in range(m.nbody)
                     if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or '')
                     .startswith(('rh_', 'robot0:'))}
        if not self.hand:
            raise ValueError('no hand bodies found for contact audit')

    def read(self):
        import mujoco
        m, d = self.raw.model, self.raw.data
        pairs, load = set(), 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if a in self.hand and b not in self.hand and b != 0:
                pair = (a, b)
            elif b in self.hand and a not in self.hand and a != 0:
                pair = (b, a)
            else:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if force[0] > 1e-6:
                pairs.add(pair)
                load += float(force[0])
        return tuple(sorted(pairs)), load


def run_trial(env, key, pose, seed, target, mean_q, axes, rho_ff, condition,
              sensitivity, settings, support, floor, *, probe_offset=None):
    from improved_tds.experiments.collect_posture_database import _actuated_q
    from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
    from improved_tds.experiments.simple_synergy import (
        numerical_warnings, simulation_step_seconds, state, steps,
    )
    # Scissors' legacy reset resets qpos/qvel but retains solver warm-start
    # state/time. Clear MjData BEFORE the authored env reset, not to a saved
    # successful state. All conditions still begin from the same pose pool.
    reset_pristine_simulator(env)
    obs, info = env.reset(seed=seed, options={'initial_pose_index': int(pose), key: float(target)})
    low, high, half = _control_bounds(env)
    raw = env.unwrapped
    neutral = np.asarray(raw._neutral_ctrl, dtype=float)
    initial = _actuated_q(env)
    dt = simulation_step_seconds(env)
    ramp_steps, hold_steps = steps(5., dt), steps(2., dt)
    ff_raw = mean_q + rho_ff * axes[0]
    ff_goal = np.clip(ff_raw, low, high)
    controller = AxisFeedback(sensitivity, settings['tool_span'], settings['feedback'])
    audit = ContactAudit(env)
    axis = axes[0 if condition == 'axis1_fb' else 1]
    trace, held, contacts = [], [], []
    offset = 0.0
    ff_clipped = bool(np.any(abs(ff_raw-ff_goal) > 1e-8) or
                      np.any(abs((ff_goal-neutral)/half)>1.+1e-8))
    ever_clipped = False
    stopped, numerical = False, False
    # Non-overlapping response windows make sign checking less sensitive to a
    # single noisy sample. This check may STOP, never retune, evaluation control.
    block = max(1, steps(.1, dt))
    history = []
    rejected_clips = 0
    baseline_curve = sensitivity.get('zero_probe_curve', [])
    for i in range(ramp_steps + hold_steps):
        candidate = offset
        if i < ramp_steps:
            q = initial + (i+1)/ramp_steps*(ff_goal-initial)
            clipped = False
        else:
            current = state(info, support_criterion=support)
            if not current['held'] or current['failed']:
                controller.stop('support_lost')
            if probe_offset is not None:
                # Diagnostic probes are fixed open-loop offsets, not feedback.
                limit = min(settings['feedback'].max_increment,
                            settings['feedback'].rate_per_second*dt)
                candidate = offset + float(np.clip(probe_offset-offset, -limit, limit))
            elif condition != 'ff':
                candidate = controller.propose(target, float(obs['achieved_goal'][0]), dt)
            candidate_raw = ff_raw + candidate*axis
            clipped = additional_clipping(candidate_raw,ff_raw,low,high,neutral,half)
            if clipped and probe_offset is None and condition != 'ff':
                controller.stop('joint_limit')
                candidate = offset
                rejected_clips += 1
            offset = float(candidate)
            controller.accept(offset)
            requested_q = ff_raw+offset*axis
            q = np.clip(requested_q, low, high)
            clipped = additional_clipping(requested_q,ff_raw,low,high,neutral,half)
            ever_clipped |= clipped
        warnings_before = numerical_warnings(env)
        obs, _, terminated, truncated, info = env.step(
            np.clip((q-neutral)/half, -1., 1.).astype(np.float32))
        if floor and info.get('floor_contact_enabled') is not True:
            raise RuntimeError('Spray floor contact disabled')
        numerical = bool(info.get('numerical_failure', False) or info.get('failure_numerical', False)
                         or np.any(numerical_warnings(env) > warnings_before)
                         or not np.isfinite(_actuated_q(env)).all()
                         or not np.isfinite(obs['achieved_goal']).all())
        current = state(info, support_criterion=support)
        signature, load = audit.read()
        y = float(obs['achieved_goal'][0])
        sample = dict(time=(i+1)*dt, tool=y, rho=rho_ff+(offset if condition=='axis1_fb' else 0.),
                      g=offset if condition=='axis2_fb' else 0., offset=offset,
                      held=bool(current['held'] and not current['failed']),
                      clipped=clipped, contact_links=len({a for a, _ in signature}),
                      normal_load_n=load, applied_q=raw.data.ctrl[raw._actuator_ids].tolist(),
                      command_q=q.tolist(), contact_signature=signature,
                      dropped=bool(info.get('object_dropped',False) or info.get('bottle_dropped',False)),
                      tipped=bool(info.get('object_tipped',False) or info.get('bottle_tipped',False)),
                      score_saturated=controller.saturated, rate_limited=controller.rate_limited,
                      stop_reason=controller.stop_reason)
        trace.append(sample)
        if i >= ramp_steps:
            held.append(sample['held'])
            contacts.append(signature)
            drift = baseline_curve[min(i-ramp_steps,len(baseline_curve)-1)] if baseline_curve else 0.
            history.append((offset, y-drift))
            if probe_offset is None and condition != 'ff' and baseline_curve and len(history) >= 2*block:
                first, second = np.mean(history[-2*block:-block], axis=0), np.mean(history[-block:], axis=0)
                du, dy = second-first
                b = sensitivity.get('slope')
                if b is not None and abs(du) > .005 and abs(dy) > sensitivity['noise'] and dy*du*b < 0:
                    controller.stop('observed_response_sign_changed')
                history = history[-block:]
        stopped = bool(terminated or truncated or numerical)
        if stopped:
            break
    complete = not stopped and len(trace) == ramp_steps+hold_steps
    tail = trace[-steps(.5, dt):] if len(held) >= steps(.5, dt) else []
    finite = not numerical and np.isfinite(y)
    valid = bool(complete and len(held) == hold_steps and all(held))
    # Convergence must persist to the end, rather than just cross the target.
    tail_errors = [abs(s['tool']-target) for s in trace[ramp_steps:]]
    convergence = None
    if complete and valid:
        for j in range(len(tail_errors)):
            if all(e <= .03*settings['tool_span'] for e in tail_errors[j:]):
                convergence = j*dt
                break
    terminal_signature = Counter(contacts[-steps(.5, dt):]).most_common(1)[0][0] if contacts else ()
    # Preserve per-step metrics above, then keep a sparse trajectory for the
    # visual report. Full terminal hold response is retained for drift removal.
    saved_trace = trace[::10]
    if trace and saved_trace[-1] is not trace[-1]:
        saved_trace.append(trace[-1])
    return dict(pose=int(pose), seed=seed, target=target, condition=condition, rho_ff=rho_ff,
                valid=valid, complete=complete, numerical_failure=numerical,
                achieved=y if finite and complete else None,
                abs_error=abs(y-target) if finite and complete else None,
                last_observed_tool=y if finite else None,
                last_observed_abs_error=abs(y-target) if finite else None,
                terminal_mean=float(np.mean([s['tool'] for s in tail])) if tail and finite else None,
                terminal_std=float(np.std([s['tool'] for s in tail])) if tail and finite else None,
                clipped=ever_clipped, stop_reason=controller.stop_reason if condition!='ff' else None,
                ff_clipped=ff_clipped,
                rejected_clip_commands=rejected_clips, sensitivity_enabled=bool(sensitivity['enabled']),
                terminal_contact_signature=terminal_signature,
                saturated_steps=sum(s['score_saturated'] for s in trace),
                dropped=any(s['dropped'] for s in trace), tipped=any(s['tipped'] for s in trace),
                elapsed_seconds=len(trace)*dt, dt=dt, convergence_after_ff_seconds=convergence,
                recorded_trace_stride_steps=10, executed_steps=len(trace),
                hold_tool_curve=[s['tool'] for s in trace[ramp_steps:]],
                score_total_variation=float(np.sum(abs(np.diff([0.]+[s['offset'] for s in trace])))),
                tool_total_variation=float(np.sum(abs(np.diff([s['tool'] for s in trace[ramp_steps:]])))),
                peak_normal_load_n=max(s['normal_load_n'] for s in trace),
                trace=saved_trace)


def summarize_trials(trials, target_range):
    low, high = target_range
    span = high-low
    result = {}
    for condition in CONDITIONS:
        group = [r for r in trials if r['condition'] == condition]
        finite = [r for r in group if r['abs_error'] is not None]
        valid = [r for r in finite if r['valid']]
        ys = [r['achieved'] for r in valid]
        result[condition] = dict(trials=len(group), finite=len(finite), valid=len(valid),
            all_finite_mae_percent=float(np.mean([r['abs_error'] for r in finite]))/span*100 if finite else None,
            valid_mae_percent=float(np.mean([r['abs_error'] for r in valid]))/span*100 if valid else None,
            valid_fraction_percent=100*len(valid)/len(group) if group else None,
            achieved_span_percent=100*max(0., min(high,max(ys))-max(low,min(ys)))/span if ys else 0.,
            complete=sum(r['complete'] for r in group),
            numerical_failures=sum(r.get('numerical_failure',False) for r in group),
            failed_terminal_observations=len(group)-len(finite),
            failure_penalized_mae_percent=100*float(np.mean([
                r['abs_error']/span if r['valid'] and r['abs_error'] is not None else
                max(1.,r['abs_error']/span if r['abs_error'] is not None else 1.) for r in group])) if group else None,
            stopped_feedback=sum(r['stop_reason'] is not None and r.get('sensitivity_enabled',False) for r in group),
            disabled_feedback=sum(not r.get('sensitivity_enabled',False) for r in group) if condition!='ff' else 0,
            enabled_trials=sum(r.get('sensitivity_enabled',False) for r in group),
            saturated_trials=sum(r.get('saturated_steps',0)>0 for r in group),
            ff_clipped_trials=sum(r.get('ff_clipped',False) for r in group),
            dropped_trials=sum(r.get('dropped',False) for r in group),
            tipped_trials=sum(r.get('tipped',False) for r in group),
            mean_score_variation=float(np.mean([r['score_total_variation'] for r in group])) if group else None)
    return result


def paired_comparisons(trials, span):
    """Same pose/target intersection, plus all-attempt failure-penalized delta.

    Negative delta favors axis2. Intersection-only MAE never stands alone.
    A failed trial costs at least 100% of the RL span, explicitly not an MAE.
    """
    keyed = {(r['pose'],r['target'],r['condition']):r for r in trials}
    result = {}
    def metrics(pairs):
        common=[(a,b) for a,b in pairs if a['valid'] and b['valid']
                and a['abs_error'] is not None and b['abs_error'] is not None]
        def cost(r):
            e=r['abs_error']/span if r['abs_error'] is not None else 1.
            return e if r['valid'] else max(1.,e)
        return dict(pairs=len(pairs),jointly_valid=len(common),
            jointly_valid_error_delta_percent=100*float(np.mean([b['abs_error']-a['abs_error'] for a,b in common]))/span if common else None,
            all_attempt_penalized_delta_percent=100*float(np.mean([cost(b)-cost(a) for a,b in pairs])) if pairs else None,
            improved_valid_pairs=sum(b['abs_error']<a['abs_error'] for a,b in common),
            validity_losses=sum(a['valid'] and not b['valid'] for a,b in pairs),
            validity_gains=sum(not a['valid'] and b['valid'] for a,b in pairs))
    for baseline in ('ff','axis1_fb'):
        pairs=[(keyed[(p,t,baseline)],r) for (p,t,c),r in keyed.items()
               if c=='axis2_fb' and (p,t,baseline) in keyed]
        result[baseline]=metrics(pairs)
        groups = {}
        for a,b in pairs:
            enabled2=bool(b.get('sensitivity_enabled',False))
            if baseline=='ff':
                name='axis2_enabled' if enabled2 else 'axis2_disabled'
            else:
                enabled1=bool(a.get('sensitivity_enabled',False))
                name=('both_enabled' if enabled1 and enabled2 else 'neither_enabled'
                      if not enabled1 and not enabled2 else 'axis1_only' if enabled1 else 'axis2_only')
            groups.setdefault(name,[]).append((a,b))
        result[baseline]['by_calibration_eligibility']={name:metrics(group) for name,group in groups.items()}
    return result


def run_task(bundle, output, task, ncal, ntest, ntargets, *, full_rl_grid=False):
    from improved_tds.environments.scissors_smooth import SmoothReachConfig
    from improved_tds.experiments.collect_posture_database import _build_env
    from improved_tds.experiments.reach_tds import load_database
    from improved_tds.experiments.reach_tds_direct import configure_scissors_evaluation_palm_radius
    from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    if sha256(bundle/'manifest.json') != BASELINE_MANIFEST_SHA256:
        raise ValueError('bundle is not the documented frozen paper baseline')
    entry = manifest['tasks'][task]
    report_path = bundle/entry['selected_report']
    dbpath = bundle/task/'database/success_postures.npz'
    for path, digest in ((report_path, entry['selected_report_sha256']),
                         (dbpath, entry['database_sha256'])):
        if sha256(path) != digest:
            raise ValueError(f'frozen input changed: {path}')
    source = json.loads(report_path.read_text(encoding='utf-8'))
    data, meta = load_database(dbpath)
    if not np.allclose(np.mean(data['q'], axis=0), source['mean_q']):
        raise ValueError('source mean does not match frozen DB')
    calposes, testposes = source['calibration_poses'][:ncal], source['test_poses'][:ntest]
    if len(calposes)!=ncal or len(testposes)!=ntest or set(calposes)&set(testposes):
        raise ValueError('insufficient or overlapping calibration/test poses')
    configs = {m: source['configurations'][m+'_1d'] for m in ('pca','pls')}
    _, domain = common_targets(configs.values(), ntargets)
    targets = (np.linspace(*source['target_range'], ntargets).tolist()
               if full_rl_grid else common_targets(configs.values(), ntargets)[0])
    contract = meta['contract']
    version = meta.get('scissors_control_version', 'simple')
    _register_tool_manip_gym_environments()
    smooth = SmoothReachConfig(float(contract.get('action_delta_fraction', .05)),
                               float(contract.get('request_gap_penalty_weight', 1.)))
    env, pool, poolpath, key, lo, hi = _build_env(meta['env_id'], bundle/'pose_pool', source['seed'],
        research_task=True, scissors_control_version=version,
        scissors_smooth_config=smooth if version=='smooth' else None,
        tool_action_slew_rate=float(contract.get('tool_action_slew_rate', 0.)),
        tool_request_gap_penalty_weight=float(contract.get('tool_request_gap_penalty_weight', 1.)),
        object_position_noise=0., max_episode_steps=10000, force_floor_contact=task.startswith('spray'))
    # Capture immediately after construction, before any trial or pool reset.
    reset_pristine_simulator(env)
    try:
        if sha256(poolpath) != entry['pose_pool_sha256'] or not np.allclose([lo,hi], source['target_range']):
            raise ValueError('pose pool or target range differs from paper baseline')
        if task.startswith('scissors'):
            configure_scissors_evaluation_palm_radius(env,
                source['scissors_evaluation_palm_radius_multiplier'], contract.get('palm_sphere_radius'))
        output.mkdir(parents=True, exist_ok=False)
        inputs = output/'inputs'
        inputs.mkdir()
        for src, name in ((report_path,'baseline_report.json'),(dbpath,'success_postures.npz'),
                          (poolpath,'pose_pool.npz')):
            shutil.copy2(src,inputs/name)
        feedback = AxisFeedbackConfig()
        settings = dict(tool_span=hi-lo, feedback=feedback)
        result = dict(protocol=('second_axis_feedback_full_rl_nine_v1' if full_rl_grid
                                else 'second_axis_feedback_v1'),
            task=task, tool_unit='m' if task.startswith('push') else 'rad',
            source_report_sha256=sha256(report_path), database_sha256=sha256(dbpath),
            pose_pool_sha256=sha256(poolpath), baseline_manifest_sha256=sha256(bundle/'manifest.json'),
            calibration_poses=calposes, test_poses=testposes, targets=targets,
            common_inverse_domain=domain, target_range=[lo,hi], feedback_settings=asdict(feedback),
            full_rl_target_grid=full_rl_grid,
            trajectory_seconds=5., feedback_seconds=2., online_rl=False, test_retries=0,
            sensitivity_uses_test_poses=False, unseen_pose_distribution=False,
            source_evaluation_criterion=source['evaluation_grasp_criterion'],
            reset_protocol='restore_construction_mount_and_neutral_then_mj_resetData_then_authored_reset',
            methods={})
        support = 'task' if task.startswith('scissors') else 'no_fall'
        if source['evaluation_support_criterion'] != support:
            raise ValueError('support criterion differs from frozen report')
        floor = task.startswith('spray')
        disabled = dict(enabled=False, reason='diagnostic_open_loop', slope=None, noise=0.,
                        offset_low=-feedback.probe_offset, offset_high=feedback.probe_offset)
        for method, config in configs.items():
            axes = [np.array(config['fit']['operation_axis'])*config['operation_scale'],
                    np.array(config['fit']['grasp_axis'])*config['grasp_scale']]
            method_result = dict(axis2_definition='PC2' if method=='pca' else 'PC1 of residual after PLS1',
                                 authority=dict(
                                     probe_coordinate=feedback.probe_offset,
                                     axis1_joint_l2_rad=float(np.linalg.norm(axes[0])*feedback.probe_offset),
                                     axis2_joint_l2_rad=float(np.linalg.norm(axes[1])*feedback.probe_offset),
                                     axis1_database_score_std=float(np.std((data['q']-np.array(source['mean_q']))@np.array(config['fit']['operation_axis'])/config['operation_scale'])),
                                     axis2_database_score_std=float(np.std((data['q']-np.array(source['mean_q']))@np.array(config['fit']['grasp_axis'])/config['grasp_scale']))),
                                 calibration=[], trials=[])
            result['methods'][method] = method_result
            for target in targets:
                inverse_domain = config['inverse_target_range']
                inverse_clipped = bool(target < inverse_domain[0] or target > inverse_domain[1])
                if inverse_clipped and not full_rl_grid:
                    raise ValueError('target unexpectedly outside inverse domain')
                rho = inverse_score(config,float(np.clip(target,*inverse_domain)))
                sensitivities = {}
                for condition in ('axis1_fb','axis2_fb'):
                    probes = []
                    for pose in calposes:
                        triplet = [run_trial(env,key,pose,source['seed']+1000*pose,target,
                            np.array(source['mean_q']),axes,rho,condition,disabled,settings,support,floor,
                            probe_offset=delta) for delta in (-feedback.probe_offset,0.,feedback.probe_offset)]
                        probes.append(triplet)
                    sensitivity = estimate_sensitivity(probes,feedback.probe_offset,hi-lo,feedback.noise_fraction)
                    zeros=[p[1] for p in probes if p[1]['complete'] and p[1]['valid']]
                    sensitivity['zero_probe_curve_poses'] = len(zeros)
                    if len(zeros)>=2:
                        sensitivity['zero_probe_curve'] = np.mean([
                            p['hold_tool_curve'] for p in zeros],axis=0).tolist()
                    sensitivities[condition] = sensitivity
                    method_result['calibration'].append(dict(target=target,condition=condition,
                        sensitivity=sensitivity,probes=probes,
                        inverse_clipped=inverse_clipped))
                for pose in testposes:
                    for condition in CONDITIONS:
                        trial = run_trial(env,key,pose,source['seed']+1000*pose,
                            target,np.array(source['mean_q']),axes,rho,condition,
                            sensitivities.get(condition,disabled),settings,support,floor)
                        trial['inverse_clipped'] = inverse_clipped
                        method_result['trials'].append(trial)
                dump(output/'progress.json',dict(method=method,last_completed_target=target))
                dump(output/'partial_report.json',result)
            method_result['summary'] = summarize_trials(method_result['trials'],[lo,hi])
            method_result['paired_comparisons'] = paired_comparisons(method_result['trials'],hi-lo)
        dump(output/'report.json',result)
        return result
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path('analysis/paper15_paper_bundle_20260928'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=DEFAULT_TASKS,required=True)
    parser.add_argument('--calibration-poses',type=int,default=8)
    parser.add_argument('--test-poses',type=int,default=8)
    parser.add_argument('--targets',type=int,default=9)
    parser.add_argument('--full-rl-grid',action='store_true',
                        help='Use identical full RL targets, clipping frozen FF inverse at its domain')
    args = parser.parse_args(argv)
    if args.calibration_poses<2 or args.test_poses<1 or args.targets<1:
        parser.error('at least 2 calibration poses, 1 test pose and 1 target required')
    run_task(args.bundle.resolve(),args.output.resolve(),args.task,
             args.calibration_poses,args.test_poses,args.targets,
             full_rl_grid=args.full_rl_grid)


if __name__ == '__main__':
    main()
