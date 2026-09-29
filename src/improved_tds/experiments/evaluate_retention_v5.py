"""Paired acquisition-window evaluation; keep strict success thresholds fixed."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.experiments.validation import score_policy_on_grid
from improved_tds.environments.scissors_retention_v5 import ScissorsRetentionV5Wrapper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--pool-dir',type=Path,required=True)
    p.add_argument('--variant',type=int,choices=(3,4,5),required=True)
    p.add_argument('--version',choices=('v5','v6','v7'),default='v5')
    a=p.parse_args();_register_tool_manip_gym_environments()
    model_path=a.run_dir/'best_success_policy.zip'
    if not model_path.exists():model_path=a.run_dir/'model.zip'
    results={}
    for grace in (.4,.12):
        env,*_=_build_env(f'ScissorsVariant{a.variant}-v0',a.pool_dir,0,research_task=True,
                          scissors_control_version=a.version,max_episode_steps=400)
        wrapper=env
        while not isinstance(wrapper,ScissorsRetentionV5Wrapper):wrapper=wrapper.env
        wrapper.acquisition_seconds=grace
        try:
            policy=SAC.load(model_path,env=env,device='cpu')
            result=score_policy_on_grid(env,policy,range(env.get_wrapper_attr('pose_count')),
                                       np.linspace(0,1,11),'target_angle')
            results[str(grace)]=result
            print(grace,result['success_rate'],result['final_success_rate'],flush=True)
        finally:env.close()
    report=dict(model=str(model_path),comparison=f'same {a.version} policy; initial acquisition window only',
                final_fixed_hold_seconds=2.,results=results)
    (a.run_dir/'acquisition_reference_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
