"""Rank reset candidates with full-episode hold and gentle bidirectional probes.

Selection never uses a target success rate. Failed probes remain in the report;
ranking is NOT a proof of controllability and never certifies failed candidates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-dir',type=Path,required=True)
    p.add_argument('--static-audit',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    static=json.loads(a.static_audit.read_text())
    _register_tool_manip_gym_environments()
    report={}
    for v in (3,4,5):
        env,pool,path,*_=_build_env(f'ScissorsVariant{v}-v0',a.pool_dir,0,research_task=True,
            scissors_control_version='v3')
        rows=[]
        try:
            for item in static[str(v)]['poses']:
                if not item['hold_survived']: continue
                pose=item['pose_index']
                probes=[]
                for mode in ('hold','gentle'):
                    obs,info=env.reset(seed=pose,options={'initial_pose_index':pose,'target_angle':.5})
                    names=env.unwrapped._actuator_names
                    mask=np.array([any(s in name for s in ('THJ','FFJ','MFJ','RFJ')) for name in names])
                    supported=0; speeds=[]; angles=[]
                    for step in range(200):
                        amplitude=0. if mode=='hold' else .10*np.sin(2*np.pi*step/199)
                        obs,_,done,trunc,info=env.step((mask*amplitude).astype(np.float32))
                        supported+=int(info['grasp_maintained'])
                        speeds.append(info['tool_speed'])
                        angles.append(float(obs['achieved_goal'][0]))
                        if done or trunc: break
                    probes.append(dict(mode=mode,steps=step+1,survived=not bool(info['task_failure']),
                        supported_fraction=supported/200,slow_fraction=float(np.mean(np.array(speeds)<=.1)),
                        travel=max(angles)-min(angles)))
                rows.append(dict(pose_index=pose,probes=probes))
        finally: env.close()
        def rank(row):
            probes=row['probes']
            return (min(p['steps'] for p in probes),min(p['supported_fraction'] for p in probes),
                    sum(p['slow_fraction'] for p in probes))
        ranked=sorted(rows,key=rank,reverse=True)
        if len(ranked)<32: raise ValueError('need 32 distinct statically qualified candidates')
        selected=[r['pose_index'] for r in ranked[:32]]
        with np.load(path,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        for k in ('poses','mount_dpos','mount_drot'): arrays[k]=arrays[k][selected]
        arrays['accepted']=np.asarray(32)
        arrays['long_horizon_ranked']=np.asarray(True)
        arrays['source_pose_indices']=np.asarray(selected)
        np.savez_compressed(a.output_dir/path.name,**arrays)
        report[str(v)]=dict(selected=ranked[:32],all_probes=rows,
            selected_dual_survivors=sum(all(p['survived'] for p in r['probes']) for r in ranked[:32]))
        (a.output_dir/'long_horizon_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(v,'dual survivors',report[str(v)]['selected_dual_survivors'],'/32',flush=True)


if __name__=='__main__': main()
