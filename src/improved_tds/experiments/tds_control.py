"""Control the tool through one synergy axis, feedforward and with force feedback.

The TDS paper controls scissors open-loop: sweep the synergy score, record the
tool angle, fit ``f``, and thereafter command ``f^-1(c_target)`` and trust it.
That works when the tool is threaded onto the hand. A spray bottle is held, so
two things can go wrong that no amount of feedforward fixes: the object is not
exactly where it was when ``f`` was identified, and the grip that holds it is
itself a function of the same joints that work the tool.

So two controllers are compared here.

``feedforward``  commands x_ref + w * f^-1(c_target) and nothing else. This is
                 the paper's scheme.

``force_feedback`` adds a second, orthogonal command. Grip force is measured and
                 regulated back to the value seen during identification by
                 moving along the grasp-force axis, with the axis first made
                 orthogonal to the operation axis so that holding on does not
                 also move the tool.

The comparison that matters is under disturbance: identification happens with
the object in one place, and evaluation with it moved.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from improved_tds.data.trajectory import TrajectoryDataset
from improved_tds.environments.pose_pool import (
    InitialPosePoolWrapper,
    SprayGoalWrapper,
    pool_filename_for_environment,
)
from improved_tds.evaluation.reports import write_report
from improved_tds.experiments.analyze_synergy import _jsonable, actuated_qpos_indices
from improved_tds.experiments.evaluate_tds_control import (
    control_axes,
    kinematic_rho_limits,
    reference_postures,
)
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.emergence import extract_manipulation_samples


class ToolMap:
    """The identified relation between synergy score and tool state.

    The paper fits a ramp. A monotone interpolation over the identification
    points is used instead: it makes no assumption about the shape, and the
    inverse is exact on the identified range, so any tracking error that shows
    up is the controller's rather than the fit's.
    """

    def __init__(self, rho: np.ndarray, tool: np.ndarray, grasp_force: np.ndarray):
        order = np.argsort(rho)
        self.rho = np.asarray(rho, dtype=np.float64)[order]
        self.tool = np.asarray(tool, dtype=np.float64)[order]
        # the grip this command scheme actually produces, which is far below what
        # the policy produced: the policy squeezed with saturated commands, and
        # regulating towards a force this scheme cannot reach only winds the
        # correction out to its limit
        self.grasp_force = np.asarray(grasp_force, dtype=np.float64)[order]
        # enforce monotonicity so the inverse is single valued
        self.monotone_tool = np.maximum.accumulate(self.tool)

    @property
    def reachable(self) -> tuple[float, float]:
        return float(self.monotone_tool.min()), float(self.monotone_tool.max())

    def forward(self, rho: float) -> float:
        return float(np.interp(rho, self.rho, self.monotone_tool))

    def inverse(self, tool_state: float) -> float:
        low, high = self.reachable
        return float(np.interp(np.clip(tool_state, low, high), self.monotone_tool, self.rho))


def identify(env, reference, axis, limits, *, points: int, hold: dict[str, int]) -> ToolMap:
    """Sweep the axis once and record where the tool ends up."""
    from improved_tds.experiments.evaluate_tds_control import sweep

    grid = np.linspace(limits[0], limits[1], points)
    observed = [
        sweep(env, hold["pose"], reference, axis, float(value), **hold["steps"])
        for value in grid
    ]
    return ToolMap(
        grid,
        np.array([point["tool_state"] for point in observed]),
        np.array([point["grasp_normal_force"] for point in observed]),
    )


def run_target(
    goal_env,
    pose_index: int,
    reference: np.ndarray,
    axis: np.ndarray,
    grasp_axis: np.ndarray,
    tool_map: ToolMap,
    target: float,
    *,
    trial: int,
    controller: str,
    force_setpoint: float,
    force_gain: float,
    grasp_steps: int,
    ramp_steps: int,
    hold_steps: int,
    sigma_limit: float,
) -> dict[str, Any]:
    """Command one target tool state and report what the tool actually did."""
    unwrapped = goal_env.unwrapped
    # the seed decides where the object lands, so it varies per target and is
    # shared between controllers: both meet the same disturbance
    _, _ = goal_env.reset(
        seed=1000 + 97 * pose_index + trial,
        options={"initial_pose_index": int(pose_index)},
    )
    start = np.asarray(unwrapped._neutral_ctrl, dtype=np.float64).copy()
    halfspan = np.asarray(unwrapped._ctrl_halfspan, dtype=np.float64)
    rho = tool_map.inverse(target)
    sigma = 0.0
    forces: list[float] = []
    sigmas: list[float] = []
    lost = False
    held_min = 5
    info: dict[str, Any] = {}

    def send(posture: np.ndarray):
        action = np.clip((posture - start) / halfspan, -1.0, 1.0)
        return goal_env.step(action.astype(np.float32))

    # close onto the tool, then move to the commanded score and hold there
    for step in range(grasp_steps):
        _, _, terminated, truncated, info = send(
            start + (reference - start) * (step + 1) / grasp_steps
        )
        if terminated or truncated:
            break
    for phase, steps in (("ramp", ramp_steps), ("hold", hold_steps)):
        for step in range(steps):
            fraction = (step + 1) / steps if phase == "ramp" else 1.0
            if controller == "force_feedback" and phase == "hold":
                # hold the grip where identification found it, without letting
                # that correction move the tool: the axis is orthogonal to w
                error = force_setpoint - float(info.get("grasp_normal_force_total", 0.0))
                sigma = float(np.clip(sigma + force_gain * error, -sigma_limit, sigma_limit))
            posture = reference + axis * (rho * fraction) + grasp_axis * sigma
            _, _, terminated, truncated, info = send(posture)
            held_min = min(held_min, int(info["grasp_contact_count"]))
            forces.append(float(info["grasp_normal_force_total"]))
            sigmas.append(sigma)
            lost = lost or bool(info["bottle_dropped"] or info["bottle_tipped"])
            if terminated or truncated:
                break
        if lost:
            break

    achieved = float(info.get("trigger_angle", np.nan))
    return {
        "target": float(target),
        "rho": rho,
        "achieved": achieved,
        "abs_error": abs(achieved - float(target)),
        "sigma": sigma,
        "grasp_force_mean": float(np.mean(forces)) if forces else float("nan"),
        "grasp_force_final": forces[-1] if forces else float("nan"),
        "sigma_saturated": bool(sigmas and abs(sigmas[-1]) >= 0.999 * sigma_limit),
        "min_grasp_contacts": int(held_min),
        "lost": bool(lost),
    }


def evaluate_variant(
    dataset_path: Path,
    pose_pool_dir: Path,
    *,
    poses: int,
    identify_points: int,
    targets: int,
    margin: float,
    force_gain: float,
    sigma_limit: float,
    disturbance: float,
    grasp_steps: int,
    ramp_steps: int,
    hold_steps: int,
) -> dict[str, Any]:
    dataset = TrajectoryDataset.load(dataset_path)
    env_id = str(dataset.trajectories[0].metadata.simulator_parameters["env_id"])
    indices, _ = actuated_qpos_indices(env_id)
    samples = extract_manipulation_samples(dataset, indices)
    axes = control_axes(samples)
    grasp_axis = axes["_grasp_axis"]
    axis = axes["pls"]
    # regulating grip must not move the tool, so the correction travels along the
    # part of the grasp axis that the operation axis does not already cover
    orthogonal = grasp_axis - float(grasp_axis @ axis) * axis
    norm = float(np.linalg.norm(orthogonal))
    grasp_correction = orthogonal / norm if norm > 1e-9 else np.zeros_like(axis)

    horizon = grasp_steps + ramp_steps + hold_steps + 10

    def build(noise: float):
        base = gym.make(env_id, bottle_xy_noise=noise, max_episode_steps=horizon)
        if getattr(base.unwrapped, "max_episode_steps", None):
            base.unwrapped.max_episode_steps = horizon
        pool = InitialPosePoolWrapper(
            base, pose_pool_dir / pool_filename_for_environment(base), seed=0
        )
        return SprayGoalWrapper(pool, seed=0), pool

    identify_env, pool_env = build(0.0)
    references = reference_postures(
        dataset, np.asarray(indices, dtype=int), pool_env, "max_headroom", axis=axis
    )
    low = np.asarray(identify_env.unwrapped._ctrl_low, dtype=np.float64)
    high = np.asarray(identify_env.unwrapped._ctrl_high, dtype=np.float64)
    steps = {"grasp_steps": grasp_steps, "ramp_steps": ramp_steps, "hold_steps": hold_steps}

    available = sorted(set(samples.pose_index.tolist()))
    chosen = available[:: max(1, len(available) // poses)][:poses]

    results: dict[str, Any] = {
        "dataset": dataset_path.name,
        "env_id": env_id,
        "poses": chosen,
        "disturbance": disturbance,
        "force_gain": force_gain,
        "runs": {},
    }
    try:
        maps: dict[int, ToolMap] = {}
        setpoints: dict[int, float] = {}
        for pose in chosen:
            reference = references[int(pose)]
            limits = kinematic_rho_limits(reference, axis, low, high)
            maps[pose] = identify(
                identify_env,
                reference,
                axis,
                limits,
                points=identify_points,
                hold={"pose": pose, "steps": steps},
            )
            setpoints[pose] = float(np.median(maps[pose].grasp_force))
            low_c, high_c = maps[pose].reachable
            print(
                f"  pose {pose}: identified over tool {low_c:.3f}..{high_c:.3f} rad, "
                f"grip setpoint {setpoints[pose]:.1f} N",
                flush=True,
            )
    finally:
        identify_env.close()

    # evaluation runs in a world where the object is no longer exactly where it
    # was identified, which is the only place feedback can earn its keep
    test_env, _ = build(disturbance)
    try:
        for controller in ("feedforward", "force_feedback"):
            per_pose: dict[str, Any] = {}
            for pose in chosen:
                tool_map = maps[pose]
                low_c, high_c = tool_map.reachable
                grid = np.linspace(low_c + margin, high_c - margin, targets)
                per_pose[str(pose)] = [
                    run_target(
                        test_env,
                        pose,
                        references[int(pose)],
                        axis,
                        grasp_correction,
                        tool_map,
                        float(target),
                        trial=trial,
                        controller=controller,
                        force_setpoint=setpoints[pose],
                        force_gain=force_gain,
                        sigma_limit=sigma_limit,
                        **steps,
                    )
                    for trial, target in enumerate(grid)
                ]
                errors = np.array([point["abs_error"] for point in per_pose[str(pose)]])
                kept = np.array([not point["lost"] for point in per_pose[str(pose)]])
                print(
                    f"  {controller} pose {pose}: mean |e| {np.degrees(errors.mean()):.2f} deg, "
                    f"max {np.degrees(errors.max()):.2f} deg, held {int(kept.sum())}/{len(kept)}",
                    flush=True,
                )
            results["runs"][controller] = per_pose
    finally:
        test_env.close()
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--identify-points", type=int, default=12)
    parser.add_argument("--targets", type=int, default=11)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--force-gain", type=float, default=0.002)
    parser.add_argument("--sigma-limit", type=float, default=0.3)
    parser.add_argument(
        "--disturbance",
        type=float,
        default=0.01,
        help="object placement noise during evaluation; identification is always "
        "done with the object fixed",
    )
    parser.add_argument("--grasp-seconds", type=float, default=5.0)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--step-seconds", type=float, default=0.02)
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
            identify_points=args.identify_points,
            targets=args.targets,
            margin=args.margin,
            force_gain=args.force_gain,
            sigma_limit=args.sigma_limit,
            disturbance=args.disturbance,
            grasp_steps=max(1, round(args.grasp_seconds / args.step_seconds)),
            ramp_steps=max(1, round(args.ramp_seconds / args.step_seconds)),
            hold_steps=max(1, round(args.hold_seconds / args.step_seconds)),
        )
    write_report(args.output, _jsonable(report))
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
