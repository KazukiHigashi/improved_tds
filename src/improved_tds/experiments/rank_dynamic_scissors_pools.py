"""Rank statically qualified poses by supported gentle opening/return probes.

The scripted probe is not a mechanical reachability test. Failed probes remain
in the audit; ranking is shared across all five variants without target exclusion.
"""
import json
from pathlib import Path
import argparse
import numpy as np
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-dir',type=Path,required=True)
    p.add_argument('--static-audit',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--amplitude',type=float,default=0.10)
    a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    static=json.loads(a.static_audit.read_text())
    _register_tool_manip_gym_environments()
    report={}
    for v in range(1,6):
        env,pool,path,*_=_build_env(f'ScissorsVariant{v}-v0',a.pool_dir,0,research_task=True)
        rows=[]
        try:
            for item in static[str(v)]['poses']:
                if not item['hold_survived']: continue
                pose=item['pose_index']
                obs,info=env.reset(options={'initial_pose_index':pose,'target_angle':.5})
                u=env.unwrapped
                ranges=u.model.actuator_ctrlrange[u._actuator_ids]
                base=(u.data.ctrl[u._actuator_ids]-u._neutral_ctrl)/((ranges[:,1]-ranges[:,0])/2)
                names=getattr(u,'_actuator_names',getattr(u,'actuator_names',[]))
                mask=np.array([any(s in name for s in ('THJ','FFJ','MFJ','RFJ')) for name in names])
                angles=[];supported=0;failed=False
                for t in range(60):
                    amplitude=a.amplitude*np.sin(np.pi*t/59)**2
                    obs,_,done,trunc,info=env.step(np.clip(base+mask*amplitude,-1,1))
                    supported+=int(info['grasp_maintained'])
                    if info['grasp_maintained']: angles.append(float(obs['achieved_goal'][0]))
                    if done or trunc: failed=True;break
                travel=max(angles)-min(angles) if angles else 0.
                rows.append(dict(pose_index=pose,survived=not failed,supported_fraction=supported/60,
                                 supported_travel=travel))
        finally: env.close()
        ranked=sorted(rows,key=lambda x:(x['survived'],x['supported_fraction']>=.9,x['supported_travel'],x['supported_fraction']),reverse=True)
        if len(ranked)<32: raise ValueError(f'Insufficient static poses for variant {v}')
        selected=[r['pose_index'] for r in ranked[:32]]
        with np.load(path,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        for k in ('poses','mount_dpos','mount_drot'): arrays[k]=arrays[k][selected]
        arrays['accepted']=np.asarray(32)
        arrays['dynamic_ranked']=np.asarray(True)
        np.savez_compressed(a.output_dir/path.name,**arrays)
        report[str(v)]=dict(selected=ranked[:32],all_probes=rows,
            amplitude=a.amplitude,
            selected_survivors=sum(r['survived'] for r in ranked[:32]))
        (a.output_dir/'dynamic_audit.json').write_text(json.dumps(report,indent=2))
        print(v,'selected survivors',report[str(v)]['selected_survivors'],'/32',flush=True)

if __name__=='__main__':main()
