"""Evaluate a v3 checkpoint on an explicitly named reset pool."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.experiments.validation import score_policy_on_grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',type=int,required=True,choices=range(1,6))
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--pool-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    _register_tool_manip_gym_environments()
    env,pool,path,key,low,high=_build_env(f'ScissorsVariant{a.variant}-v0',a.pool_dir,0,
        research_task=True,scissors_control_version='v3')
    try:
        model=SAC.load(a.model,env=env,device='cpu')
        result=score_policy_on_grid(env,model,range(pool.pose_count),np.linspace(low,high,11),key)
        result.update(model=str(a.model),pose_pool=str(path),control_version='v3')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('success_rate','final_success_rate','mean_abs_error')}))
    finally: env.close()


if __name__=='__main__': main()
