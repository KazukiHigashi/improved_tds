"""Paired OLD-policy rollout, not an estimate of retrained v6 performance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('runs/scissors_retention_v5_20260921'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--poses', type=int, default=4)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    _register_tool_manip_gym_environments()
    report = dict(description=__doc__, source=str(a.source), targets=[.1, .5, .9], poses=a.poses, results=[])
    for variant in (3, 4, 5):
        policy = SAC.load(a.source/f'scissors_lite_{variant}/best_success_policy.zip', device='cpu')
        for version in ('v5', 'v6'):
            for deterministic in (True, False):
                env, *_ = _build_env(f'ScissorsVariant{variant}-v0', a.source/'pose_pool', 0,
                    research_task=True, scissors_control_version=version, max_episode_steps=400)
                c = Counter()
                try:
                    for target in report['targets']:
                        for pose in range(a.poses):
                            torch.manual_seed(1234+pose)
                            obs, _ = env.reset(seed=pose, options={'initial_pose_index':pose, 'target_angle':target})
                            was_frozen = success = False
                            while True:
                                action, _ = policy.predict(obs, deterministic=deterministic)
                                obs, _, done, trunc, info = env.step(action)
                                good = bool(info.get('is_success'))
                                frozen = bool(info.get('retention_frozen'))
                                c['steps'] += 1
                                c['instant_good_frames'] += int(good)
                                c['good_and_delta_le_002'] += int(good and info.get('applied_command_change', 1.) <= .02)
                                c['frozen_frames'] += int(frozen)
                                c['probe_entries'] += int(frozen and not was_frozen)
                                c['probe_aborts'] += int(info.get('retention_aborted', False))
                                was_frozen = frozen
                                success |= bool(info.get('is_stable_success'))
                                if done or trunc: break
                            c['episodes'] += 1
                            c['strict_successes'] += int(success and not info.get('numerical_failure'))
                            c['physical_failures'] += int(info.get('task_failure', False))
                            c['numerical_failures'] += int(info.get('numerical_failure', False))
                finally:
                    env.close()
                row = dict(variant=variant, version=version, deterministic=deterministic, **c)
                report['results'].append(row)
                a.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
                print(json.dumps(row), flush=True)


if __name__ == '__main__': main()
