"""Success-only resumable collection. No episode limit by default; fixed Fit splits stay fixed."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from stable_baselines3 import SAC
from improved_tds.experiments.collect_posture_database import _build_env,_actuated_q,_grasp_force,_grasp_metric
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.experiments.evaluate_strict_synergy_control import balanced_angle_success_indices,_reference_deltas

SEEDS=tuple(range(20260911,20260916))


def inspect_splits(data,metadata):
    results={};deficits=[]
    for seed in SEEDS:
        shuffled=np.random.default_rng(seed).permutation(int(metadata['pose_count']))
        fit,cal=shuffled[:16],shuffled[16:24]
        ids=balanced_angle_success_indices(data,fit,100,seed,metadata['target_range'],allow_partial=True)
        bins=np.minimum(9,(data['target_fraction'][ids]*10+1e-8).astype(int))
        counts=np.bincount(bins,minlength=10)
        for b in np.flatnonzero(counts<10):deficits.append((seed,int(b),fit.tolist()))
        try:_reference_deltas(data,cal);reference_ok=True
        except ValueError:
            reference_ok=False;deficits.append((seed,3,cal.tolist()))
        results[str(seed)]=dict(counts=counts.tolist(),reference_ok=reference_ok,
            complete=bool(np.all(counts==10) and reference_ok))
    return results,deficits


def successful_source_rows(data,metadata):
    low,high=metadata['target_range']
    mask=data['success'].astype(bool)&(data['tool_state']>=low)&(data['tool_state']<=high)
    width=10 if metadata['env_id'].startswith('Scissors') and metadata.get('scissors_control_version')!='simple' else 5
    mask &= data['hold_steps']>=width
    mask &= np.isfinite(data['q']).all(axis=1)
    return {k:v[mask].copy() for k,v in data.items()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-database',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--pose-pool-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--max-new-episodes',type=int,default=0,help='0 continues until all quotas are satisfied')
    a=p.parse_args(argv)
    partial=a.output.with_name(a.output.stem+'.partial.npz')
    progress=a.output.with_suffix('.progress.json')
    with a.model.open('rb') as f:model_hash=hashlib.file_digest(f,'sha256').hexdigest()
    source=partial if partial.exists() else a.source_database
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(source,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json']));data={k:z[k].copy() for k in z.files if k!='metadata_json'}
    if model_hash!=meta['model_sha256']:raise ValueError('Only the same frozen Best Policy may contribute samples')
    if not meta['strict_success_window'] or meta['representative']!='last':raise ValueError('Actual stable-success states required')
    data=successful_source_rows(data,meta)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    old=json.loads(progress.read_text()) if progress.exists() else {}
    episode=int(old.get('new_episodes',0));rng=np.random.default_rng(a.seed)
    if 'rng_state' in old:rng.bit_generator.state=old['rng_state']
    last_added=int(old.get('last_added_episode',episode));initial_episode=episode
    def save(complete,checks):
        metadata={**meta,'model':str(a.model.resolve()),'pose_pool':str(a.pose_pool_dir.resolve()),
            'rows':len(data['q']),'successful_rows':len(data['q']),'success_only':True,
            'target_fractions':np.unique(data['target_fraction']).tolist(),
            'complete':complete,'source_database':str(a.source_database.resolve()),
            'collection_mode':'best_policy_deterministic_and_stochastic_success_only',
            'successful_rows_by_target':{str(float(f)):int(np.sum(np.isclose(data['target_fraction'],f)))
                for f in np.unique(data['target_fraction'])}}
        destination=a.output if complete else partial
        tmp=destination.with_name(destination.stem+'.tmp.npz')
        np.savez_compressed(tmp,**data,metadata_json=np.asarray(json.dumps(metadata)))
        tmp.replace(destination)
        record=dict(state='completed' if complete else 'collecting',utc=datetime.now(timezone.utc).isoformat(),
            success_only=True,rows=len(data['q']),new_episodes=episode,last_added_episode=last_added,
            no_new_success_episodes=episode-last_added,splits=checks,rng_state=rng.bit_generator.state)
        temp=progress.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2),encoding='utf-8');temp.replace(progress)
        if complete:a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    checks,deficits=inspect_splits(data,meta);save(not deficits,checks)
    if not deficits:return 0
    _register_tool_manip_gym_environments()
    env,pool,_,key,low,high=_build_env(meta['env_id'],a.pose_pool_dir,a.seed,research_task=True,
        scissors_control_version=meta.get('scissors_control_version','v2'),object_position_noise=0.)
    policy=SAC.load(a.model,env=env,device='cpu')
    # Failures are counters/diagnostic logs only, never posture rows in the DB.
    try:
        while deficits:
            if a.max_new_episodes and episode-initial_episode>=a.max_new_episodes:break
            _,b,poses=deficits[int(rng.integers(len(deficits)))]
            pose=int(rng.choice(poses));fraction=float((b+rng.uniform(.02,.98))/10)
            target=low+fraction*(high-low);episode+=1
            obs,info=env.reset(seed=a.seed+episode,options={'initial_pose_index':pose,key:target})
            initial=_actuated_q(env);success=False
            # Repeating a deterministic reset verbatim cannot add diversity.
            # Both modes use the same frozen policy; there are no weight updates.
            deterministic=episode%2==0
            for _ in range(int(env.spec.max_episode_steps or 200)):
                action,_=policy.predict(obs,deterministic=deterministic)
                obs,_,done,trunc,info=env.step(action)
                if bool(info.get('is_stable_success',False)):
                    success=True;break
                if done or trunc:break
            angle=float(obs['achieved_goal'][0]);q=_actuated_q(env)
            actual_bin=min(9,int((angle-low)/(high-low)*10))
            duplicate=bool(len(data['q']) and np.any((data['pose_index']==pose)
                &(np.sqrt(np.mean((data['q']-q)**2,axis=1))<1e-6)))
            take=bool(success and low<=angle<=high and actual_bin==b and np.isfinite(q).all() and not duplicate)
            if take:
                row=dict(q=q,command_q=env.unwrapped.data.ctrl[env.unwrapped._actuator_ids].copy(),initial_q=initial,
                    tool_state=angle,target=target,grasp_metric=_grasp_metric(info),grasp_force=_grasp_force(info),
                    pose_index=pose,target_index=int(round(fraction*100)),target_fraction=fraction,
                    success=True,hold_steps=int(info['success_hold_steps']))
                data={k:np.concatenate([v,np.asarray([row[k]],dtype=v.dtype)],axis=0) for k,v in data.items()}
                last_added=episode
                checks,deficits=inspect_splits(data,meta)
            if take or episode%25==0:
                save(not deficits,checks)
                print(f'episodes={episode} accepted_rows={len(data["q"])} remaining_split_bins={len(deficits)}',flush=True)
            with a.output.with_suffix('.attempts.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(dict(episode=episode,pose_index=pose,target=target,success=success,
                    accepted=take,deterministic=deterministic,tool_state=angle))+'\n')
        save(not deficits,checks)
    finally:env.close()
    return 0 if not deficits else 2


if __name__=='__main__':raise SystemExit(main())
