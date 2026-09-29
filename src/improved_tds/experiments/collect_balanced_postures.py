"""Bounded, independent-episode collection of 10 x 10 supported postures.

The audit is a pose/target reachability test under the supplied policy, not a
proof of mechanical reachability. Failed trials are retained in the JSON report.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from improved_tds.experiments.collect_posture_database import _build_env, _actuated_q
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.learning.algorithms import algorithm_class


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-id', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--pose-pool-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=9100)
    parser.add_argument('--per-bin', type=int, default=10)
    parser.add_argument('--max-episodes', type=int, default=6400)
    parser.add_argument('--min-posture-rms', type=float, default=0.01)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--scissors-control-version', choices=('v2','v3'), default='v2')
    args = parser.parse_args(argv)
    if args.per_bin < 1 or args.max_episodes < 1 or args.min_posture_rms < 0:
        parser.error('invalid quota, budget or uniqueness threshold')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('choose a new output path; existing results are preserved')
    _register_tool_manip_gym_environments()
    env,pool,pool_path,key,low,high = _build_env(args.env_id,args.pose_pool_dir,args.seed,research_task=True,
        scissors_control_version=args.scissors_control_version)
    policy=algorithm_class('sac').load(args.model,env=env,device='cpu')
    rng=np.random.default_rng(args.seed)
    counts=np.zeros(10,dtype=int)
    attempts=np.zeros(10,dtype=int)
    accepted=[]
    trials=[]
    # Audit all reset poses against endpoints and the full 0.1-spaced grid first.
    grid=[(p,float(f)) for p in range(pool.pose_count) for f in np.linspace(0,1,11)]
    try:
        for episode in range(args.max_episodes):
            if episode < len(grid):
                pose,fraction=grid[episode]
                b=min(9,int(fraction*10))
            else:
                if args.audit_only or np.all(counts>=args.per_bin):
                    break
                candidates=np.flatnonzero(counts<args.per_bin)
                b=int(candidates[np.argmin(attempts[candidates])])
                fraction=(b+rng.random())/10
                pose=int(attempts[b]%pool.pose_count)
            attempts[b]+=1
            target=low+fraction*(high-low)
            obs,info=env.reset(seed=args.seed+episode,options={'initial_pose_index':pose,key:target})
            initial=_actuated_q(env)
            supported_angles=[]
            success=False
            for step in range(int(env.spec.max_episode_steps or 200)):
                action,_=policy.predict(obs,deterministic=True)
                obs,_,terminated,truncated,info=env.step(action)
                angle=float(obs['achieved_goal'][0])
                if info.get('grasp_maintained',False):
                    supported_angles.append(angle)
                if info.get('is_stable_success',False):
                    success=True
                    break
                if terminated or truncated:
                    break
            q=_actuated_q(env)
            actual_bin=min(9,max(0,int((angle-low)/(high-low)*10)))
            duplicate=any(np.sqrt(np.mean((q-row['q'])**2)) < args.min_posture_rms for row in accepted)
            take=bool(success and counts[b]<args.per_bin and actual_bin==b and not duplicate)
            if take:
                accepted.append(dict(q=q,command_q=np.asarray(env.unwrapped.data.ctrl[env.unwrapped._actuator_ids]).copy(),
                    initial_q=initial,tool_state=angle,target=target,grasp_metric=float(info['grasp_maintained']),
                    grasp_force=float(info.get('grasp_contact_count',0)),pose_index=pose,target_index=b,
                    target_fraction=fraction,success=True,hold_steps=int(info['success_hold_steps']),episode_id=episode))
                counts[b]+=1
            trials.append(dict(episode=episode,pose_index=pose,target=target,success=success,accepted=take,
                duplicate=duplicate,final_error=abs(angle-target),tool_speed=info.get('tool_speed'),
                support_bodies=info.get('grasp_contact_count'),displacement=info.get('object_displacement'),
                rotation=info.get('object_rotation_error'),failure=info.get('task_failure',False),
                supported_min=min(supported_angles) if supported_angles else None,
                supported_max=max(supported_angles) if supported_angles else None))
            if (episode+1)%100==0:
                print(f'episodes={episode+1} accepted_per_bin={counts.tolist()}',flush=True)
    finally:
        env.close()
    complete=bool(np.all(counts==args.per_bin))
    with args.model.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    report=dict(task_version='posture_database_v2',env_id=args.env_id,model=str(args.model),model_sha256=digest,
        scissors_control_version=args.scissors_control_version,
        strict_success_window=True,representative='last_actual_state_of_first_stable_window',
        target_fractions=sorted(set(r['target_fraction'] for r in accepted)),
        pose_count=pool.pose_count,rows=len(accepted),successful_rows=len(accepted),
        pose_pool=str(pool_path),criteria=asdict(env.criteria) if hasattr(env,'criteria') else None,
        complete=complete,counts=counts.tolist(),required_per_bin=args.per_bin,min_posture_rms=args.min_posture_rms,
        audit_grid_complete=len(trials)>=len(grid),seed=args.seed,trials=trials)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Partial artifacts have an explicit name and cannot masquerade as a full DB.
    destination=args.output if complete else args.output.with_name(args.output.stem+'.partial.npz')
    if accepted:
        np.savez_compressed(destination,**{k:np.asarray([r[k] for r in accepted]) for k in accepted[0]},
                            metadata_json=np.asarray(json.dumps({k:v for k,v in report.items() if k!='trials'})))
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials','criteria')}))
    return 0 if complete or (args.audit_only and report['audit_grid_complete']) else 2


if __name__=='__main__':
    raise SystemExit(main())
