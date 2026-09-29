"""Bounded deterministic replay audit; does not modify models or reset pools."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--pool-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    _register_tool_manip_gym_environments()
    report = {}
    for variant in (3, 4, 5):
        env, pool, *_ = _build_env(f'ScissorsVariant{variant}-v0', a.pool_dir, 123, research_task=True)
        model = SAC.load(a.run_root / f'scissors_lite_{variant}/best_success_policy.zip', env=env, device='cpu')
        rows = []
        try:
            for mode in ('zero', 'policy'):
                for pose in range(0, pool.pose_count, 4):
                    for target in (0., .5, 1.):
                        obs, info = env.reset(seed=123, options={'initial_pose_index': pose, 'target_angle': target})
                        speeds, errors, deltas, angles = [], [], [], []
                        previous = np.zeros(env.action_space.shape)
                        for step in range(200):
                            action = np.zeros(env.action_space.shape) if mode == 'zero' else model.predict(obs, deterministic=True)[0]
                            obs, _, done, trunc, info = env.step(action)
                            speeds.append(info['tool_speed'])
                            errors.append(info['angle_error'])
                            angles.append(float(obs['achieved_goal'][0]))
                            deltas.append(float(np.mean(np.abs(action - previous))))
                            previous = action
                            if done or trunc:
                                break
                        rows.append(dict(mode=mode, pose=pose, target=target, steps=step+1,
                            failure=bool(info['task_failure']), support=int(info['grasp_contact_count']),
                            displacement=float(info['object_displacement']), rotation=float(info['object_rotation_error']),
                            dropped=bool(info['object_dropped']), speed_median=float(np.median(speeds)),
                            speed_p90=float(np.quantile(speeds,.9)), slow_fraction=float(np.mean(np.array(speeds)<=.1)),
                            near_fraction=float(np.mean(np.array(errors)<=.03)),
                            near_slow_fraction=float(np.mean((np.array(errors)<=.03)&(np.array(speeds)<=.1))),
                            action_delta=float(np.mean(deltas)), angle_min=min(angles), angle_max=max(angles)))
        finally:
            env.close()
        report[str(variant)] = rows
        a.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        for mode in ('zero','policy'):
            subset=[r for r in rows if r['mode']==mode]
            print(variant, mode, {k: round(float(np.mean([r[k] for r in subset])),4)
                for k in ('steps','failure','speed_median','slow_fraction','near_fraction','near_slow_fraction','action_delta','rotation','displacement')}, flush=True)


if __name__ == '__main__':
    main()
