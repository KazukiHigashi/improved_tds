"""Keep only the initial poses that hold the tool still under a zero command.

A pose the tool slips out of is not a starting state a policy can learn from:
the episode begins with the tool angle moving on its own, so the policy cannot
tell its own effect from the pose's. Measured over the scissors pools, this
separates the variants that trained from the ones that did not --

    scissors 1, 2 (success 1.00): tool drifts 0.0001 rad, 32/32 poses under 0.01
    scissors 3, 4 (success 0.01, 0.05): drifts 0.05-0.08 rad, 9/32 under 0.01

-- so the pools are filtered on it before training rather than after.

    python -m improved_tds.experiments.filter_still_poses \
        --task scissors --variants "3 4 5" \
        --source ../tool-pose-pipeline/artifacts_v3 \
        --out ../tool-pose-pipeline/grasps_v4 --keep 32
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np

from improved_tds.environments.pose_pool import (
    InitialPosePoolWrapper,
    pool_filename_for_environment,
)
from improved_tds.experiments.show_initial_poses import inspect
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.experiments.visualize_tds_control import TASKS, seconds_to_steps


def audit(env, settle_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Per pose: how far the tool drifts, and how many contacts hold it.

    A pose touching nothing is no more usable than one the tool slips out of --
    the hand has to be on the tool for the task to start at all.
    """
    rows = [
        inspect(env, pose, settle_steps=settle_steps, pause=0.0)
        for pose in range(env.pose_count)
    ]
    drift = np.array([abs(row["tool_end"] - row["tool_start"]) for row in rows])
    contacts = np.array([row["contacts"] for row in rows])
    penetration = np.array([row["penetration_mm"] for row in rows])
    return drift, contacts, penetration


def filter_pool(
    task: str,
    variant: int,
    source: Path,
    destination: Path,
    *,
    keep: int,
    threshold: float,
    min_contacts: int,
    max_penetration_mm: float,
    seconds: float,
    seed: int,
    env_kwargs: dict | None = None,
) -> dict[str, float | int]:
    template, kwargs, _ = TASKS[task]
    base = gym.make(template.format(v=variant), **{**kwargs, **(env_kwargs or {})})
    name = pool_filename_for_environment(base)
    env = InitialPosePoolWrapper(base, source / name, seed=seed)
    try:
        drift, contacts, penetration = audit(env, seconds_to_steps(env, seconds))
        poses = np.asarray(env._poses, dtype=np.float64)
        mount_dpos = np.asarray(env._mount_dpos, dtype=np.float64)
        mount_drot = np.asarray(env._mount_drot, dtype=np.float64)
        actuator_names = list(env._pool_actuator_names)
        pool_task, pool_hand, pool_variant = env.task, env.hand, env.variant
    finally:
        env.close()

    # a pose the hand is buried in is not a grasp, it is a modelling failure --
    # one scissors pose sat 9 mm inside the handle and destabilised the solver
    eligible = np.flatnonzero(
        (drift < threshold) & (contacts >= min_contacts) & (penetration < max_penetration_mm)
    )
    # stillest first, so a short pool is still the best of what there was
    order = eligible[np.argsort(drift[eligible])][:keep]
    if order.size == 0:
        raise RuntimeError(
            f"{name}: no pose holds the tool within {threshold}, touches it at least "
            f"{min_contacts} times, and stays under {max_penetration_mm} mm of overlap"
        )

    destination.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination / name,
        task=pool_task,
        hand=pool_hand,
        variant=pool_variant,
        actuator_names=np.asarray(actuator_names),
        poses=poses[order],
        mount_dpos=mount_dpos[order],
        mount_drot=mount_drot[order],
    )
    return {
        "candidates": int(drift.size),
        "eligible": int(eligible.size),
        "kept": int(order.size),
        "drift_median": float(np.median(drift[order])),
        "drift_max": float(drift[order].max()),
        "contacts_min": int(contacts[order].min()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--variants", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--keep", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="largest tool movement allowed during the hold, in the tool's own unit",
    )
    parser.add_argument(
        "--min-contacts", type=int, default=1, help="contacts with the tool the pose must have"
    )
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-penetration-mm", type=float, default=2.0, help="deepest overlap allowed"
    )
    parser.add_argument(
        "--object-dx",
        type=float,
        default=0.0,
        help="shift the tool along x before judging the poses, so the pool is "
        "selected for the placement it will be trained at",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--copy-others",
        default="",
        help="pool files to copy through unchanged, e.g. variants that already pass",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _register_tool_manip_gym_environments()
    print(
        f"{'pool':22s} {'cands':>6s} {'ok':>5s} {'kept':>5s} "
        f"{'drift med':>10s} {'max':>9s} {'min contacts':>13s}"
    )
    for variant in [int(v) for v in args.variants.replace(",", " ").split()]:
        report = filter_pool(
            args.task,
            variant,
            args.source,
            args.out,
            keep=args.keep,
            threshold=args.threshold,
            min_contacts=args.min_contacts,
            max_penetration_mm=args.max_penetration_mm,
            seconds=args.seconds,
            env_kwargs=(
                {
                    "initial_object_qpos": (1.07 + args.object_dx, 0.892, 0.402, 1.0, 0.0, 0.0, 0.0)
                }
                if args.object_dx
                else None
            ),
            seed=args.seed,
        )
        print(
            f"{args.task}_lite_{variant:<12d} {report['candidates']:6d} {report['eligible']:5d} "
            f"{report['kept']:5d} {report['drift_median']:10.5f} {report['drift_max']:9.5f} "
            f"{report['contacts_min']:13d}",
            flush=True,
        )
    for name in args.copy_others.replace(",", " ").split():
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source.parent / name, args.out / Path(name).name)
        print(f"copied {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
