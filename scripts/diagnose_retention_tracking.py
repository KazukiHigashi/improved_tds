"""Bounded same-policy diagnostic for servo tracking gap and hold failures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import types
import numpy as np
import torch
from stable_baselines3 import SAC
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.environments.scissors_retention_v6 import ScissorsRetentionV6Wrapper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--poses', type=int, default=4)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    _register_tool_manip_gym_environments()
    root = Path('runs/scissors_retention_v6_20260921')
    report = dict(source=str(root), targets=[.1,.5,.9], poses=list(range(a.poses)), results=[])
    for v in (3,4,5):
        model_path = root/f'scissors_lite_{v}/best_success_policy.zip'
        policy = SAC.load(model_path, device='cpu')
        for bound in (None,.15,.3):
            env,*_ = _build_env(f'ScissorsVariant{v}-v0',root/'pose_pool',0,
                research_task=True,scissors_control_version='v6',max_episode_steps=400)
            w = env
            while not isinstance(w,ScissorsRetentionV6Wrapper): w=w.env
            if bound is not None:
                original = w.step
                def guarded(self,action):
                    if self.freeze_remaining<=0 and not self.certified:
                        raw=self.unwrapped
                        limits=raw.model.actuator_ctrlrange[raw._actuator_ids]
                        half=np.diff(limits,axis=1).ravel()/2
                        q=raw.data.actuator_length[raw._actuator_ids]
                        action=np.clip(action,(q-bound-raw._neutral_ctrl)/half,(q+bound-raw._neutral_ctrl)/half)
                    return original(action)
                w.step=types.MethodType(guarded,w)
            c=Counter(); gaps=[]; entry_gaps=[]; first_breaks=Counter(); holds=[]
            try:
                for target in report['targets']:
                    for pose in report['poses']:
                        obs,_=env.reset(seed=pose,options={'initial_pose_index':pose,'target_angle':target})
                        success=entered=previous_frozen=broken=False
                        max_hold=0
                        while True:
                            action,_=policy.predict(obs,deterministic=True)
                            obs,_,done,trunc,info=env.step(action)
                            raw=env.unwrapped
                            gap=float(np.max(abs(raw.data.ctrl[raw._actuator_ids]-raw.data.actuator_length[raw._actuator_ids])))
                            gaps.append(gap)
                            frozen=bool(info.get('retention_frozen'))
                            c['frames']+=1;c['frozen_frames']+=int(frozen)
                            if frozen and not previous_frozen:
                                c['probes']+=1;entry_gaps.append(gap);broken=False;entered=True
                            if frozen and not info.get('is_success') and not broken:
                                for key,fail in [('angle',info.get('angle_error',0)>.03),('speed',info.get('tool_speed',0)>.1),('support',not info.get('grasp_maintained')),('physical',info.get('task_failure'))]:
                                    first_breaks[key]+=int(fail)
                                broken=True
                            previous_frozen=frozen
                            max_hold=max(max_hold,info.get('retention_success_steps',0))
                            success|=bool(info.get('is_stable_success'))
                            if done or trunc:break
                        c['episodes']+=1;c['strict_success']+=int(success and not info.get('numerical_failure'))
                        c['entered_hold_episodes']+=int(entered)
                        c['physical_failure']+=int(info.get('task_failure',False))
                        c['early_failure_le10']+=int(info.get('task_failure',False) and w.episode_step<=10)
                        c['contact_failure']+=int(info.get('failure_contact',False))
                        c['numeric_failure']+=int(info.get('numerical_failure',False))
                        holds.append(max_hold)
            finally:env.close()
            row=dict(variant=v,tracking_bound=bound,counts=dict(c),first_breaks=dict(first_breaks),
                gap_median=float(np.median(gaps)),gap_p95=float(np.quantile(gaps,.95)),
                entry_gap_median=float(np.median(entry_gaps)) if entry_gaps else None,
                max_hold_steps=holds)
            report['results'].append(row)
            a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
