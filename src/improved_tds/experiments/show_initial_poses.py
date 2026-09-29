"""Step through an initial-pose pool in the viewer, with a numeric audit.

A pose pool that trains badly usually looks wrong long before the policy does.
This walks the pool one pose at a time so the poses can be watched, and reports
for each the things that make a pose unusable:

- does the hand actually touch the tool, and how deeply
- does the pose hold still when nothing is commanded, or does it drift
- does the tool move on its own, meaning the pose is not a stable start

Run without ``--render`` for the table alone, which is the quick way to compare
a pool that trains well against one that does not.

    python -m improved_tds.experiments.show_initial_poses \
        --task scissors --variant 3 \
        --pose-pool-dir ../tool-pose-pipeline/grasps_v3 --render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from improved_tds.environments.pose_pool import (
    InitialPosePoolWrapper,
    pool_filename_for_environment,
)
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.experiments.visualize_tds_control import TASKS, seconds_to_steps


def hand_tool_contacts(unwrapped) -> tuple[int, float]:
    """Contacts between the hand and anything else, and the deepest overlap.

    A contact's ``dist`` is negative when the geoms interpenetrate, so the
    deepest penetration is the most negative distance among those contacts.
    """
    model, data = unwrapped.model, unwrapped.data
    count, deepest = 0, 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
            for geom in (contact.geom1, contact.geom2)
        ]
        on_hand = [name is not None and name.startswith("rh_") for name in bodies]
        if sum(on_hand) != 1:  # hand-vs-hand and world-vs-tool are not of interest
            continue
        count += 1
        deepest = min(deepest, float(contact.dist))
    return count, -deepest


def tool_value(observation, info) -> float:
    if isinstance(observation, dict) and "achieved_goal" in observation:
        return float(np.asarray(observation["achieved_goal"]).reshape(-1)[0])
    for key in ("scissors_angle", "trigger_angle", "push_depth"):
        if key in info:
            return float(np.asarray(info[key]).reshape(-1)[0])
    return float("nan")


def inspect(env, pose: int, *, settle_steps: int, pause: float) -> dict[str, Any]:
    """Hold the pose with a zero command and see whether anything moves."""
    unwrapped = env.unwrapped
    observation, info = env.reset(options={"initial_pose_index": int(pose)})
    start_q = unwrapped.data.qpos.copy()
    start_tool = tool_value(observation, info)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(settle_steps):
        observation, _, terminated, truncated, info = env.step(zero)
        if pause:
            time.sleep(pause)
        if terminated or truncated:
            break
    contacts, penetration = hand_tool_contacts(unwrapped)
    drift = float(np.max(np.abs(unwrapped.data.qpos - start_q)))
    return {
        "pose": int(pose),
        "contacts": contacts,
        "penetration_mm": 1000.0 * penetration,
        "tool_start": start_tool,
        "tool_end": tool_value(observation, info),
        "joint_drift": drift,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--variants", default="1", help="space or comma separated")
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--render", action="store_true", help="open the viewer")
    parser.add_argument(
        "--seconds", type=float, default=2.0, help="how long each pose is held"
    )
    parser.add_argument("--speed", type=float, default=1.0, help="playback rate when rendering")
    parser.add_argument("--poses", default="", help="only these pose indices (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _register_tool_manip_gym_environments()
    variants = [int(v) for v in args.variants.replace(",", " ").split()]
    wanted = [int(v) for v in args.poses.replace(",", " ").split()] if args.poses else None

    for variant in variants:
        template, kwargs, _ = TASKS[args.task]
        env_id = template.format(v=variant)
        options = dict(kwargs)
        if args.render:
            options["render_mode"] = "human"
        base = gym.make(env_id, **options)
        pool_path = args.pose_pool_dir / pool_filename_for_environment(base)
        env = InitialPosePoolWrapper(base, pool_path, seed=args.seed)
        settle = seconds_to_steps(env, args.seconds)
        pause = (args.seconds / settle) / args.speed if args.render and args.speed > 0 else 0.0
        poses = wanted if wanted is not None else list(range(env.pose_count))

        print(f"\n=== {env_id}  ({pool_path.name}, {env.pose_count} poses) ===", flush=True)
        print(
            f"{'pose':>5s} {'contacts':>9s} {'penetration':>12s} "
            f"{'tool start':>11s} {'tool end':>9s} {'drift':>8s}",
            flush=True,
        )
        rows = []
        try:
            for pose in poses:
                row = inspect(env, pose, settle_steps=settle, pause=pause)
                rows.append(row)
                print(
                    f"{row['pose']:5d} {row['contacts']:9d} {row['penetration_mm']:10.2f}mm "
                    f"{row['tool_start']:11.4f} {row['tool_end']:9.4f} {row['joint_drift']:8.4f}",
                    flush=True,
                )
        finally:
            env.close()

        if rows:
            contacts = np.array([r["contacts"] for r in rows])
            penetration = np.array([r["penetration_mm"] for r in rows])
            moved = np.array([abs(r["tool_end"] - r["tool_start"]) for r in rows])
            drift = np.array([r["joint_drift"] for r in rows])
            print(
                f"  touching the tool: {int((contacts > 0).sum())}/{len(rows)} poses"
                f" | median contacts {np.median(contacts):.0f}"
                f" | penetration median {np.median(penetration):.2f}mm max {penetration.max():.2f}mm",
                flush=True,
            )
            print(
                f"  tool moved on its own: median {np.median(moved):.4f} max {moved.max():.4f}"
                f" | joint drift median {np.median(drift):.4f} max {drift.max():.4f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
