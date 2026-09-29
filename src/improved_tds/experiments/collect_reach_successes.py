"""Frozen Best Policy -> equal-size, success-only, angle-balanced posture DB.

No calibration prerequisites. Retry missing bins until the requested quota;
write incomplete data only to .partial.npz, never pad it with failed poses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import mujoco
import numpy as np
import torch
from stable_baselines3 import SAC
from .collect_posture_database import _build_env, _actuated_q, _grasp_metric, _grasp_force
from .train_sb3 import _register_tool_manip_gym_environments
from improved_tds.environments.scissors_smooth import SmoothReachConfig


def successful_state(info, q, angle, low, high):
    tolerance=float(info.get('angle_tolerance',
        info.get('push_tracking_tolerance',info.get('trigger_tracking_tolerance',0.))))
    return bool(info.get('is_stable_success',False) and info.get('is_success',False)
        and info.get('grasp_maintained',False) and not info.get('task_failure',False)
        and not info.get('numerical_failure',False) and np.isfinite(q).all()
        and np.isfinite(angle) and low-tolerance <= angle <= high+tolerance)


def success_tolerance(env):
    current=env
    while current is not None:
        for name in ('angle_tolerance','push_tracking_tolerance','trigger_tracking_tolerance'):
            value=getattr(current,name,None)
            if value is not None:
                return float(value)
        current=getattr(current,'env',None)
    raw=env.unwrapped
    for name in ('angle_threshold','push_tracking_tolerance','trigger_tracking_tolerance'):
        value=getattr(raw,name,None)
        if value is not None:
            return float(value)
    raise ValueError('No task success tolerance found')


def contact_breakdown(env):
    """Viewer-only contact audit; never changes collection eligibility."""
    raw=env.unwrapped
    model=raw.model
    finger_geoms=env.env._digits
    groups={'blade':set(),'moving_handle':set(),'mirror_handle':set()}
    for contact in raw.data.contact:
        a,b=int(contact.geom1),int(contact.geom2)
        for tool,hand in ((a,b),(b,a)):
            if hand not in finger_geoms:continue
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,tool) or ''
            match=re.match(r'scissors_part(\d+)',name)
            if match:
                part=int(match[1])
                group='mirror_handle' if part==1 else 'moving_handle' if part>=29 else 'blade'
                groups[group].add(int(finger_geoms[hand]))
    return {key:len(value) for key,value in groups.items()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-id',required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--pose-pool-dir',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--per-bin',type=int,default=20,help='20 successes in each of 10 angle bins (200 total)')
    p.add_argument('--extend-from',type=Path,help='verified completed smaller DB to top up without overwriting it')
    p.add_argument('--max-episodes',type=int,default=0,help='0 retries until complete; otherwise a bounded diagnostic')
    p.add_argument('--seed',type=int,default=9300)
    p.add_argument('--scissors-control-version',choices=('simple','smooth'),default='simple')
    p.add_argument('--scissors-action-delta-fraction',type=float,default=0.05)
    p.add_argument('--scissors-request-gap-penalty-weight',type=float,default=1.0)
    p.add_argument('--tool-action-slew-rate',type=float,default=0.,
        help='Push/Spray rate in normalized action units per simulated second')
    p.add_argument('--tool-request-gap-penalty-weight',type=float,default=1.)
    p.add_argument('--preview',action='store_true',help='watch the same collection loop without writing or resuming a DB')
    p.add_argument('--no-render',action='store_true',help='headless preview for testing')
    p.add_argument('--speed',type=float,default=1.,help='viewer playback speed relative to simulation time')
    p.add_argument('--print-every',type=int,default=10,help='print a status line every N control steps')
    p.add_argument('--pause-on-success',action='store_true',help='wait for Enter after an accepted posture')
    p.add_argument('--preview-hold-seconds',type=float,default=0.,
        help='preview only: after an accepted posture, freeze the last applied command for this many simulation seconds')
    p.add_argument('--preview-bin',type=int,choices=range(10),help='preview only one target-angle bin')
    p.add_argument('--initial-pose-index',type=int,help='preview a fixed authored initial pose')
    a=p.parse_args(argv)
    if a.per_bin<1 or a.max_episodes<0 or a.speed<=0 or a.print_every<1: p.error('invalid quota, episode budget, speed or print interval')
    if not np.isfinite(a.preview_hold_seconds) or a.preview_hold_seconds<0 or (a.preview_hold_seconds and not a.preview):
        p.error('--preview-hold-seconds requires --preview and a non-negative duration')
    if (not np.isfinite(a.scissors_action_delta_fraction) or not 0<a.scissors_action_delta_fraction<=1
            or not np.isfinite(a.scissors_request_gap_penalty_weight) or a.scissors_request_gap_penalty_weight<0):
        p.error('invalid smooth control configuration')
    if (not np.isfinite(a.tool_action_slew_rate) or a.tool_action_slew_rate<0
            or not np.isfinite(a.tool_request_gap_penalty_weight) or a.tool_request_gap_penalty_weight<0):
        p.error('invalid tool slew configuration')
    if a.preview and a.max_episodes==0: p.error('--preview requires a finite --max-episodes')
    if not a.preview and (a.output is None or a.no_render or a.preview_bin is not None or a.initial_pose_index is not None):
        p.error('normal collection requires --output and does not accept preview-only options')
    if a.preview and a.extend_from is not None:p.error('--extend-from is only for normal collection')
    if not a.preview and a.output.exists(): raise FileExistsError(a.output)
    _register_tool_manip_gym_environments();torch.set_num_threads(1)
    env,pool,pool_path,key,low,high=_build_env(a.env_id,a.pose_pool_dir,a.seed,
        research_task=True,scissors_control_version=a.scissors_control_version,object_position_noise=0.,
        scissors_smooth_config=SmoothReachConfig(a.scissors_action_delta_fraction,
            a.scissors_request_gap_penalty_weight) if a.scissors_control_version=='smooth' else None,
        tool_action_slew_rate=a.tool_action_slew_rate,
        tool_request_gap_penalty_weight=a.tool_request_gap_penalty_weight,
        render_mode='human' if a.preview and not a.no_render else None)
    if a.initial_pose_index is not None and not 0<=a.initial_pose_index<pool.pose_count:
        env.close();p.error('initial pose index outside the saved pool')
    with a.model.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    with pool_path.open('rb') as f: pool_digest=hashlib.file_digest(f,'sha256').hexdigest()
    is_scissors=a.env_id.startswith('Scissors')
    grasp_criterion=('hinge_center_inside_palm_attached_original_sphere' if is_scissors
        else 'two_finger_contacts_without_drop_or_tip')
    contract=dict(env_id=a.env_id,model_sha256=digest,pool_sha256=pool_digest,
        per_bin=a.per_bin,seed=a.seed,target_range=[low,high],scissors_control_version=a.scissors_control_version,
        grasp_criterion=grasp_criterion,success_tolerance=success_tolerance(env))
    if is_scissors:
        contract.update(palm_sphere_radius=float(env.unwrapped.grasp_threshold),
            palm_sphere_local_center=env.unwrapped._tds_palm_sphere_local_center.tolist())
    if is_scissors and a.scissors_control_version=='smooth':
        contract.update(action_delta_fraction=a.scissors_action_delta_fraction,
            request_gap_penalty_weight=a.scissors_request_gap_penalty_weight)
    if not is_scissors:
        contract.update(tool_action_slew_rate=a.tool_action_slew_rate,
            tool_request_gap_penalty_weight=a.tool_request_gap_penalty_weight)
    partial=a.output.with_name(a.output.stem+'.partial.npz') if not a.preview else None
    progress=a.output.with_suffix('.progress.json') if not a.preview else None
    rows=[];counts=np.zeros(10,dtype=int);attempts=np.zeros(10,dtype=int);endpoints=[False,False];episode=0
    rng=np.random.default_rng(a.seed)
    if partial is not None and partial.exists():
        with np.load(partial,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json']))
            if meta['contract']!=contract: raise ValueError('resume requires the same frozen policy, pool and collection settings')
            rows=[{k:z[k][i] for k in z.files if k!='metadata_json'} for i in range(len(z['q']))]
        counts=np.array(meta['counts']);attempts=np.array(meta['attempts']);endpoints=meta['endpoints']
        episode=meta['episodes'];rng.bit_generator.state=meta['rng_state']
    elif a.extend_from is not None:
        with np.load(a.extend_from,allow_pickle=False) as z:
            meta=json.loads(z['metadata_json'].item())
            old_contract=dict(meta.get('contract',{}))
            old_quota=old_contract.pop('per_bin',None)
            new_contract=dict(contract);new_contract.pop('per_bin')
            if (not meta.get('complete') or old_contract!=new_contract or
                    not isinstance(old_quota,int) or not 0<old_quota<a.per_bin or
                    len(z['q'])!=10*old_quota or not z['success'].all()):
                raise ValueError('extension requires a completed, compatible smaller success-only DB')
            rows=[{k:z[k][i] for k in z.files if k!='metadata_json'} for i in range(len(z['q']))]
        counts=np.array(meta['counts']);attempts=np.array(meta['attempts']);endpoints=meta['endpoints']
        if not np.all(counts==old_quota):
            raise ValueError('extension source does not meet its per-bin quota')
        episode=meta['episodes'];rng.bit_generator.state=meta['rng_state']
    if not a.preview:a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        complete=bool(np.all(counts==a.per_bin) and all(endpoints))
        meta=dict(contract=contract,task_version=(f'posture_database_{a.scissors_control_version}_palm_v1'
            if is_scissors else 'posture_database_slew_success_v1'),env_id=a.env_id,model=str(a.model.resolve()),model_sha256=digest,
            pose_pool=str(pool_path.resolve()),pose_count=pool.pose_count,target_range=[low,high],
            scissors_control_version=a.scissors_control_version,success_only=True,strict_success_window=True,
            representative='last',success_hold_steps=5,rows=len(rows),successful_rows=len(rows),
            complete=complete,counts=counts.tolist(),attempts=attempts.tolist(),endpoints=endpoints,
            episodes=episode,rng_state=rng.bit_generator.state,
            target_fractions=sorted(set(float(r['target_fraction']) for r in rows)),
            collection_mode='best_policy_deterministic_and_stochastic_success_only',
            achieved_range=[min(float(r['tool_state']) for r in rows),max(float(r['tool_state']) for r in rows)] if rows else None)
        data={k:np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {'q':np.empty((0,env.action_space.shape[0]))}
        destination=a.output if complete else partial
        tmp=destination.with_name(destination.stem+'.tmp.npz')
        np.savez_compressed(tmp,**data,metadata_json=np.asarray(json.dumps(meta)))
        tmp.replace(destination)
        # The NPZ above is the authoritative, resumable checkpoint. Windows
        # indexers can briefly hold the auxiliary JSON open; that must not
        # abort an otherwise valid success-posture collection.
        temp=progress.with_name(f'{progress.name}.{os.getpid()}.{time.time_ns()}.tmp')
        try:
            temp.write_text(json.dumps(meta,indent=2),encoding='utf-8')
            for attempt in range(10):
                try:
                    temp.replace(progress)
                    break
                except PermissionError:
                    if attempt==9:
                        print(f'warning: progress JSON locked; NPZ checkpoint is current: {progress}',
                              file=sys.stderr,flush=True)
                    else:
                        time.sleep(.2)
        except PermissionError:
            print(f'warning: progress JSON unavailable; NPZ checkpoint is current: {progress}',
                  file=sys.stderr,flush=True)
        return complete
    initial_episode=episode
    try:
        policy=SAC.load(a.model,env=env,device='cpu')
        if a.preview:
            print('VIEWER PREVIEW: no database files will be created or modified; Ctrl+C stops',flush=True)
            print('Contact columns are diagnostics only. Grasp and success use the hinge centre '
                  'inside the palm-attached original green sphere.',flush=True)
        while not (bool(np.all(counts==a.per_bin) and all(endpoints)) if a.preview else save()):
            if a.max_episodes and episode-initial_episode>=a.max_episodes: break
            if a.preview_bin is not None and counts[a.preview_bin]>=a.per_bin and (
                a.preview_bin not in (0,9) or endpoints[0 if a.preview_bin==0 else 1]
            ): break
            missing=np.array([a.preview_bin]) if a.preview_bin is not None else np.flatnonzero(counts<a.per_bin)
            b=int(missing[np.argmin(attempts[missing])])
            fraction=0. if b==0 and not endpoints[0] else 1. if b==9 and not endpoints[1] else (b+rng.uniform(.02,.98))/10
            sampled_pose=int(rng.integers(pool.pose_count))
            pose=sampled_pose if a.initial_pose_index is None else a.initial_pose_index
            target=low+fraction*(high-low)
            episode+=1;attempts[b]+=1;torch.manual_seed(a.seed+episode)
            obs,info=env.reset(seed=a.seed+episode,options={'initial_pose_index':pose,key:target})
            initial=_actuated_q(env);take=False
            if a.preview:
                print(f'\nepisode={episode} pose={pose} requested_bin={b} target={target:.4f} '
                      f'policy={"deterministic" if episode%2==0 else "stochastic"}',flush=True)
            for step in range(1,int(env.spec.max_episode_steps or 200)+1):
                action,_=policy.predict(obs,deterministic=episode%2==0)
                obs,_,done,trunc,info=env.step(action)
                if a.preview:
                    if not a.no_render: time.sleep(env.unwrapped.dt/a.speed)
                    if step==1 or step%a.print_every==0 or info.get('is_stable_success') or done or trunc:
                        angle_now=float(obs['achieved_goal'][0]);contacts=(contact_breakdown(env) if is_scissors else
                            {'blade':0,'moving_handle':0,'mirror_handle':0})
                        print(f'  step={step:3d} tool={angle_now:.4f} error={abs(angle_now-target):.4f} '
                              f'grasp_fingers={int(info.get("grasp_contact_count",0))} '
                              f'blade/handle/mirror={contacts["blade"]}/{contacts["moving_handle"]}/{contacts["mirror_handle"]} '
                              f'applied_delta={float(info.get("applied_action_delta_max",0.)):.3f} '
                              f'limited={float(info.get("action_rate_limited_fraction",0.)):.2f} '
                              f'streak={int(info.get("success_streak",0))} '
                              f'failure={bool(info.get("task_failure",False))}',flush=True)
                if info.get('numerical_failure',False):
                    if a.preview:print('  REJECT: numerical failure',flush=True)
                    break
                if info.get('is_stable_success',False):
                    angle=float(obs['achieved_goal'][0]);q=_actuated_q(env)
                    actual_bin=int(np.clip((angle-low)/(high-low)*10,0,9))
                    duplicate=any(np.sqrt(np.mean((q-r['q'])**2))<1e-6 for r in rows)
                    eligible=successful_state(info,q,angle,low,high)
                    take=eligible and actual_bin==b and not duplicate and counts[b]<a.per_bin
                    if take:
                        rows.append(dict(q=q,command_q=env.unwrapped.data.ctrl[env.unwrapped._actuator_ids].copy(),
                            initial_q=initial,tool_state=angle,target=target,target_fraction=fraction,
                            target_index=b,pose_index=pose,success=True,hold_steps=int(info['success_hold_steps']),
                            grasp_metric=_grasp_metric(info),grasp_force=_grasp_force(info)))
                        counts[b]+=1
                        if fraction==0.:endpoints[0]=True
                        if fraction==1.:endpoints[1]=True
                    if a.preview:
                        reason='ACCEPT' if take else 'REJECT: '+(
                            'invalid success state' if not eligible else
                            f'achieved bin {actual_bin} differs from requested bin {b}' if actual_bin!=b else
                            'duplicate posture' if duplicate else 'bin quota reached')
                        print(f'  {reason}; achieved={angle:.4f}; bin={actual_bin}; '
                              f'q-command RMS={float(np.sqrt(np.mean((q-env.unwrapped.data.ctrl[env.unwrapped._actuator_ids])**2))):.4f}',flush=True)
                        if take and a.pause_on_success and sys.stdin.isatty():
                            input('  Saved posture visible in Viewer. Press Enter to continue... ')
                        if take and a.preview_hold_seconds:
                            frozen=np.asarray(info.get('applied_action',action),dtype=np.float32).copy()
                            hold_steps=int(np.ceil(a.preview_hold_seconds/env.unwrapped.dt))
                            supported=0
                            consecutive=longest=0
                            first_loss=None
                            for hold_step in range(1,hold_steps+1):
                                obs,_,hold_done,hold_trunc,hold_info=env.step(frozen)
                                held=bool(hold_info.get('grasp_maintained',False))
                                supported+=int(held)
                                consecutive=consecutive+1 if held else 0
                                longest=max(longest,consecutive)
                                if not held and first_loss is None:first_loss=hold_step
                                if not a.no_render:time.sleep(env.unwrapped.dt/a.speed)
                                if hold_step==1 or hold_step%a.print_every==0 or hold_done or hold_trunc or hold_step==hold_steps:
                                    print(f'  frozen_command step={hold_step}/{hold_steps} '
                                          f'tool={float(obs["achieved_goal"][0]):.4f} '
                                          f'both_handles={bool(hold_info.get("grasp_maintained",False))} '
                                          f'angle_error={float(hold_info.get("angle_error",np.nan)):.4f}',flush=True)
                                if hold_done or hold_trunc:break
                            print(f'  frozen_command support_fraction={supported/max(hold_step,1):.3f} '
                                  f'longest_consecutive={longest} '
                                  f'first_loss_step={first_loss}',flush=True)
                    break
                if done or trunc:
                    if a.preview:print(f'  REJECT: {"physical failure" if done else "episode limit"}',flush=True)
                    break
            if take or episode%25==0:
                if not a.preview:print(json.dumps(dict(episodes=episode,counts=counts.tolist(),endpoints=endpoints)),flush=True)
        complete=bool(np.all(counts==a.per_bin) and all(endpoints)) if a.preview else save()
        if a.preview:
            print(f'PREVIEW END: attempts={episode-initial_episode}, accepted={len(rows)}, '
                  f'counts={counts.tolist()}, endpoints={endpoints}',flush=True)
    finally:env.close()
    return 0 if a.preview or complete else 2


if __name__=='__main__':raise SystemExit(main())
