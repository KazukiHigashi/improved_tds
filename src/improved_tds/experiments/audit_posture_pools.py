"""Audit support and static hold of all Scissors poses under shared criteria."""
import argparse
import json
from pathlib import Path
import numpy as np
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qualified-dir',type=Path)
    p.add_argument('--required-count',type=int,default=32)
    args=p.parse_args(argv)
    _register_tool_manip_gym_environments()
    report={}
    for variant in range(1,6):
        env,pool,pool_path,*_=_build_env(f'ScissorsVariant{variant}-v0',args.pool_dir,0,research_task=True)
        rows=[]
        try:
            for pose in range(pool.pose_count):
                obs,initial=env.reset(options={'initial_pose_index':pose,'target_angle':.1})
                u=env.unwrapped
                ranges=u.model.actuator_ctrlrange[u._actuator_ids]
                action=np.clip((u.data.ctrl[u._actuator_ids]-u._neutral_ctrl)/((ranges[:,1]-ranges[:,0])/2),-1,1)
                contacts=[]
                failed=False
                for _ in range(env.success_hold_steps):
                    obs,_,terminated,truncated,info=env.step(action)
                    contacts.append(info['grasp_contact_count'])
                    failed|=terminated
                    if terminated or truncated:
                        break
                rows.append(dict(pose_index=pose,initial_support=initial['grasp_contact_count'],
                    minimum_support=min(contacts),final_displacement=info['object_displacement'],
                    final_rotation=info['object_rotation_error'],hold_survived=not failed,
                    final_tool_speed=info['tool_speed']))
            report[str(variant)]=dict(criteria=env.goal_sampling_report()['criteria'],poses=rows,
                hold_survivors=sum(r['hold_survived'] for r in rows),pose_count=pool.pose_count)
            if args.qualified_dir and sum(r['hold_survived'] for r in rows)>=args.required_count:
                selected=[r['pose_index'] for r in rows if r['hold_survived']][:args.required_count]
                with np.load(pool_path,allow_pickle=False) as artifact:
                    arrays={k:artifact[k].copy() for k in artifact.files}
                for k in ('poses','mount_dpos','mount_drot'):
                    arrays[k]=arrays[k][selected]
                arrays['accepted']=np.asarray(len(selected))
                arrays['posture_criteria_json']=np.asarray(json.dumps(env.goal_sampling_report()['criteria'],sort_keys=True))
                arrays['posture_qualified']=np.asarray(True)
                args.qualified_dir.mkdir(parents=True,exist_ok=True)
                destination=args.qualified_dir/pool_path.name
                if destination.exists():
                    raise FileExistsError(destination)
                np.savez_compressed(destination,**arrays)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({v:{'survivors':r['hold_survivors'],'total':r['pose_count']} for v,r in report.items()}))
    if args.qualified_dir and any(r['hold_survivors']<args.required_count for r in report.values()):
        raise SystemExit('Insufficient supported poses; regenerate candidates. Audit written; no quota relaxation.')


if __name__=='__main__':
    main()
