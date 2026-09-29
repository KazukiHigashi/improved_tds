"""Drive the tool along one synergy axis and see what the tool DoF does.

This is the scissors procedure of the TDS paper (its Eq. 6-8) carried over to a
tool that has to be *held*. There, the hand starts already threaded through the
scissor handles, so a posture swept along the first principal component is a
valid posture throughout and the tool cannot be dropped. A spray bottle is not
attached to the hand: the same sweep can open the grasp, and then the tool angle
stops being a function of the hand posture at all.

Two changes follow from that.

The trajectory is in two phases. The hand first closes from its authored initial
posture onto the reference held posture for that pose -- the mean posture the
policy held the tool in -- and only then moves along the synergy axis. The
authored poses deliberately start clear of the bottle, so without this the sweep
would begin from a posture that is not holding anything.

The reference posture is per initial pose. The authored poses differ from one
another by much more than the command moves the hand, so a single global mean
posture -- the paper's ``x̄`` -- is not a posture that grasps anything here.

    x(rho) = x_pose_mean + w * rho

Everything orthogonal to ``w`` therefore stays at the grasped reference, so the
grasp is preserved by construction unless ``w`` itself opens the hand. Whether
it does is measured, not assumed, and an axis orthogonalised against the
grasp-force direction is evaluated alongside the raw one.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from improved_tds.data.trajectory import TrajectoryDataset
from improved_tds.environments.pose_pool import (
    InitialPosePoolWrapper,
    SprayGoalWrapper,
    pool_filename_for_environment,
)
from improved_tds.evaluation.reports import write_report
from improved_tds.experiments.analyze_synergy import _jsonable, actuated_qpos_indices
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.emergence import extract_manipulation_samples


def _axis(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The first PLS direction of ``target`` on ``design``, unit length.

    PLS fixes the direction only up to sign, so it is oriented to make a larger
    score mean a larger target: a control input whose sign flips between tools
    would be a trap for whoever commands it.
    """
    weights = PLSRegression(1, scale=False).fit(design, target).x_weights_[:, 0]
    axis = weights / np.linalg.norm(weights)
    return axis if float(np.corrcoef(design @ axis, target)[0, 1]) >= 0.0 else -axis


def control_axes(samples) -> dict[str, np.ndarray]:
    """The axis that moves the tool, and a version that avoids the grasp axis.

    The grasp-force axis is the direction along which the held posture changes
    how hard the hand squeezes. Removing that component from the operation axis
    gives a command that, to first order, changes the tool state without
    changing grip force -- the property a real hand needs to keep hold of the
    tool while working it.
    """
    operation = _axis(samples.q, samples.tool_state)
    grasp = _axis(samples.q, samples.grasp_force)
    orthogonalised = operation - float(operation @ grasp) * grasp
    norm = float(np.linalg.norm(orthogonalised))
    return {
        "pls": operation,
        "pls_grasp_orthogonal": orthogonalised / norm if norm > 1e-9 else operation,
        "_grasp_axis": grasp,
    }


def _horizon(*phases: int) -> int:
    """Episode length a sweep needs, with a little slack."""
    return int(sum(phases)) + 10


def seconds_to_steps(env, seconds: float) -> int:
    """Convert a duration to env steps using the simulated step length."""
    unwrapped = env.unwrapped
    substeps = getattr(unwrapped, "frame_skip", None) or getattr(unwrapped, "n_substeps", 1)
    return max(1, int(round(seconds / (float(unwrapped.model.opt.timestep) * int(substeps)))))


def reference_postures(
    dataset: TrajectoryDataset,
    indices: np.ndarray,
    pose_env,
    mode: str,
    axis: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """The posture each sweep starts from, taken one of four ways.

    ``measured`` is what the analysis records: the joint angles the hand was
    actually in while holding the tool. Commanding those back is not the same
    command the policy sent, though -- contact pushes a position-servo joint off
    its target, so the measured angle is always short of what was asked for, and
    replaying it under-drives the hand. ``commanded`` reconstructs the control
    target instead, and is the like-for-like replay.

    ``measured_clipped`` isolates how much of any difference is just the
    commandable range, and ``commanded_min_tool`` anchors at the episode that
    held the tool closest to shut, so the whole sweep runs in one direction
    rather than starting from the middle.
    """
    unwrapped = pose_env.unwrapped
    halfspan = np.asarray(unwrapped._ctrl_halfspan, dtype=np.float64)
    low = np.asarray(unwrapped._ctrl_low, dtype=np.float64)
    high = np.asarray(unwrapped._ctrl_high, dtype=np.float64)
    authored = np.clip(np.asarray(pose_env._poses, dtype=np.float64), low, high)

    per_pose: dict[int, list[tuple[float, np.ndarray]]] = {}
    for trajectory in dataset.trajectories:
        if not trajectory.metadata.success:
            continue
        started = [
            index for index, step in enumerate(trajectory.steps) if step.is_stable_success
        ]
        if not started or started[0] >= len(trajectory.steps) - 1:
            continue
        window = trajectory.steps[started[0] :]
        pose = int(trajectory.metadata.simulator_parameters["initial_pose_index"])
        tool = float(np.mean([float(s.tool_state.reshape(-1)[0]) for s in window]))
        if mode == "commanded" or mode == "commanded_min_tool":
            actions = np.array([s.action for s in window], dtype=np.float64)
            posture = (authored[pose] + actions.mean(axis=0) * halfspan)
        else:
            posture = np.array([s.q[indices] for s in window], dtype=np.float64).mean(axis=0)
            if mode == "measured_clipped":
                posture = np.clip(posture, low, high)
        per_pose.setdefault(pose, []).append((tool, posture))

    references: dict[int, np.ndarray] = {}
    for pose, entries in per_pose.items():
        if mode == "commanded_min_tool":
            references[pose] = min(entries, key=lambda item: item[0])[1]
        elif mode == "max_headroom":
            # How far rho can travel depends only on the part of the reference
            # perpendicular to the axis: shifting the reference along the axis
            # slides the feasible interval without widening it. So the choice is
            # made among postures the policy actually held the tool in -- each is
            # known to grasp -- by which leaves the most travel.
            if axis is None:
                raise ValueError("max_headroom needs the axis it will travel along")
            def travel(posture: np.ndarray) -> float:
                lower, upper = kinematic_rho_limits(posture, axis, low, high)
                return upper - lower
            references[pose] = max((posture for _, posture in entries), key=travel)
        else:
            references[pose] = np.mean([posture for _, posture in entries], axis=0)
    return references


def kinematic_rho_limits(
    reference: np.ndarray, axis: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[float, float]:
    """How far ``rho`` can go before the first actuator hits its travel limit.

    Beyond this the commanded posture is clipped, so the hand stops moving along
    the axis and further ``rho`` buys nothing: it is the hard ceiling on the
    range this synergy can command, independent of what the policy happened to
    visit.
    """
    # The reference is a *measured* posture, and contact can hold a joint past
    # what its actuator can command; taking the limits from the raw reference
    # then yields an inverted range. The sweep clips its commands anyway, so the
    # limits are measured from the posture that is actually commandable.
    commandable = np.clip(reference, low, high)
    upper = np.inf
    lower = -np.inf
    for value, direction, bottom, top in zip(commandable, axis, low, high):
        if abs(direction) < 1e-12:
            continue
        to_top = (top - value) / direction
        to_bottom = (bottom - value) / direction
        upper = min(upper, max(to_top, to_bottom))
        lower = max(lower, min(to_top, to_bottom))
    return float(lower), float(upper)


def sweep(
    goal_env,
    pose_index: int,
    reference: np.ndarray,
    axis: np.ndarray,
    rho: float,
    *,
    grasp_steps: int,
    ramp_steps: int,
    hold_steps: int,
) -> dict[str, Any]:
    """Close onto the tool, move along ``axis`` by ``rho``, and hold."""
    unwrapped = goal_env.unwrapped
    _, info = goal_env.reset(
        seed=1000 + pose_index, options={"initial_pose_index": int(pose_index)}
    )
    start = np.asarray(unwrapped._neutral_ctrl, dtype=np.float64).copy()
    halfspan = np.asarray(unwrapped._ctrl_halfspan, dtype=np.float64)
    target = reference + axis * rho

    # close onto the tool, move along the axis, then hold there. Only the last
    # two segments are scored: the grasp is not established during the first.
    segments = (
        ("grasp", start, reference, grasp_steps, False),
        ("operate", reference, target, ramp_steps, True),
        ("hold", target, target, hold_steps, True),
    )

    held_min = 5
    lost = False
    final = info
    done = False
    for _, origin, destination, steps, scored in segments:
        for step in range(steps):
            fraction = (step + 1) / steps
            posture = origin + (destination - origin) * fraction
            action = np.clip((posture - start) / halfspan, -1.0, 1.0)
            _, _, terminated, truncated, final = goal_env.step(action.astype(np.float32))
            done = bool(terminated or truncated)
            if scored:
                held_min = min(held_min, int(final["grasp_contact_count"]))
            lost = lost or bool(final["bottle_dropped"] or final["bottle_tipped"])
            if done:
                break
        if done:
            break

    return {
        "rho": float(rho),
        "tool_state": float(final["trigger_angle"]),
        "grasp_contact_count": int(final["grasp_contact_count"]),
        "min_grasp_contacts_after_grasp": int(held_min),
        "grasp_normal_force": float(final["grasp_normal_force_total"]),
        "tool_normal_force": float(final["trigger_normal_force_total"]),
        "lost": bool(lost),
    }


def evaluate_variant(
    dataset_path: Path,
    pose_pool_dir: Path,
    *,
    poses: int,
    rho_points: int,
    rho_scale: float | str,
    grasp_steps: int,
    ramp_steps: int,
    hold_steps: int,
    axis_names: tuple[str, ...] = ("pls",),
    reference_mode: str = "measured",
) -> dict[str, Any]:
    dataset = TrajectoryDataset.load(dataset_path)
    parameters = dataset.trajectories[0].metadata.simulator_parameters
    env_id = str(parameters["env_id"])
    indices, actuator_names = actuated_qpos_indices(env_id)
    samples = extract_manipulation_samples(dataset, indices)
    axes = control_axes(samples)
    grasp_axis = axes.pop("_grasp_axis")
    axes = {name: axes[name] for name in axis_names}

    # each commanded posture is applied for several seconds, as the TDS paper
    # does, so the horizon must exceed the 200 steps the policy trained on
    horizon = _horizon(grasp_steps, ramp_steps, hold_steps)
    base_env = gym.make(env_id, bottle_xy_noise=0.0, max_episode_steps=horizon)
    if getattr(base_env.unwrapped, "max_episode_steps", None):
        base_env.unwrapped.max_episode_steps = horizon
    pool_env = InitialPosePoolWrapper(
        base_env, pose_pool_dir / pool_filename_for_environment(base_env), seed=0
    )
    goal_env = SprayGoalWrapper(pool_env, seed=0)

    references = reference_postures(
        dataset,
        np.asarray(indices, dtype=int),
        pool_env,
        reference_mode,
        axis=axes.get("pls"),
    )
    available = sorted(set(samples.pose_index.tolist()))
    chosen = available[:: max(1, len(available) // poses)][:poses]
    results: dict[str, Any] = {
        "dataset": dataset_path.name,
        "env_id": env_id,
        "actuator_names": actuator_names,
        "poses": chosen,
        "operation_grasp_axis_angle_deg": float(
            np.degrees(np.arccos(min(1.0, abs(float(axes["pls"] @ grasp_axis)))))
        ),
        "rho_scale": rho_scale,
        "reference_mode": reference_mode,
        "axes": {name: axis for name, axis in axes.items()},
        "sweeps": {},
    }
    try:
        for axis_name, axis in axes.items():
            per_pose: dict[str, Any] = {}
            for pose in chosen:
                rows = samples.pose_index == pose
                reference = references[int(pose)]
                # scores stay relative to the reference actually being commanded,
                # so a shifted reference shifts the sweep with it
                scores = (samples.raw_q[rows] - reference) @ axis
                # The paper takes the range straight from the scores the policy
                # visited (its Eq. 9). Widening it asks whether the tool range
                # was limited by the axis or only by how far the data went.
                low_limit, high_limit = kinematic_rho_limits(
                    reference,
                    axis,
                    np.asarray(goal_env.unwrapped._ctrl_low, dtype=np.float64),
                    np.asarray(goal_env.unwrapped._ctrl_high, dtype=np.float64),
                )
                if rho_scale == "kinematic":
                    # the conservative reading: as far as the axis goes before
                    # any one actuator saturates and the commanded direction
                    # stops being the axis
                    start_rho, stop_rho = low_limit, high_limit
                else:
                    # deliberately not clamped to the limits above -- individual
                    # joints saturating is exactly what this is measuring, and
                    # the per-joint clipping in `sweep` still moves the rest
                    start_rho = float(scores.min()) * float(rho_scale)
                    stop_rho = float(scores.max()) * float(rho_scale)
                grid = np.linspace(start_rho, stop_rho, rho_points)
                per_pose[str(pose)] = [
                    sweep(
                        goal_env,
                        pose,
                        reference,
                        axis,
                        float(rho),
                        grasp_steps=grasp_steps,
                        ramp_steps=ramp_steps,
                        hold_steps=hold_steps,
                    )
                    for rho in grid
                ]
                for point in per_pose[str(pose)]:
                    point["rho_data_range"] = [float(scores.min()), float(scores.max())]
                    point["rho_kinematic_range"] = [low_limit, high_limit]
                observed = np.array([point["tool_state"] for point in per_pose[str(pose)]])
                kept = np.array(
                    [not point["lost"] for point in per_pose[str(pose)]], dtype=bool
                )
                print(
                    f"  {axis_name} pose {pose}: rho {start_rho:+.3f}..{stop_rho:+.3f} "
                    f"(data {scores.min():+.3f}..{scores.max():+.3f}, "
                    f"limit {low_limit:+.3f}..{high_limit:+.3f}) "
                    f"tool {observed.min():.3f}-{observed.max():.3f} rad, "
                    f"span {np.ptp(observed[kept]) if kept.any() else 0.0:.3f}, "
                    f"held {int(kept.sum())}/{len(kept)}",
                    flush=True,
                )
            results["sweeps"][axis_name] = per_pose
    finally:
        goal_env.close()
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--poses", type=int, default=4)
    parser.add_argument("--rho-points", type=int, default=12)
    parser.add_argument(
        "--rho-scale",
        default="1",
        help="multiplier on the observed score range, or 'kinematic' to run out "
        "to the actuator travel limits along the axis",
    )
    parser.add_argument(
        "--reference",
        default="measured",
        choices=(
            "measured",
            "measured_clipped",
            "commanded",
            "commanded_min_tool",
            "max_headroom",
        ),
        help="how the posture each sweep starts from is taken",
    )
    parser.add_argument(
        "--axes",
        default="pls pls_grasp_orthogonal",
        help="space-separated axis names (pls, pls_grasp_orthogonal)",
    )
    # the paper applies each posture for 5 s; ramp + hold is that budget
    parser.add_argument("--grasp-seconds", type=float, default=5.0)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--step-seconds", type=float, default=0.02, help="env step length")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _register_tool_manip_gym_environments()
    datasets = sorted(
        args.rollout_dir.glob(args.pattern),
        key=lambda path: [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)
        ],
    )
    report: dict[str, Any] = {"rollout_dir": str(args.rollout_dir), "variants": {}}
    for path in datasets:
        print(f"=== {path.name} ===", flush=True)
        report["variants"][path.stem] = evaluate_variant(
            path,
            args.pose_pool_dir,
            poses=args.poses,
            rho_points=args.rho_points,
            rho_scale=(
                "kinematic" if args.rho_scale == "kinematic" else float(args.rho_scale)
            ),
            axis_names=tuple(args.axes.split()),
            reference_mode=args.reference,
            grasp_steps=max(1, round(args.grasp_seconds / args.step_seconds)),
            ramp_steps=max(1, round(args.ramp_seconds / args.step_seconds)),
            hold_steps=max(1, round(args.hold_seconds / args.step_seconds)),
        )
    write_report(args.output, _jsonable(report))
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
