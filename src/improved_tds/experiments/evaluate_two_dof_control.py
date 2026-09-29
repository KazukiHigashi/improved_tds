"""Evaluate PCA/PLS operation plus residual-variance grasp synergy control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from improved_tds.evaluation.reports import write_report
from improved_tds.experiments.collect_posture_database import (
    _actuated_q,
    _build_env,
    _grasp_force,
    _grasp_metric,
)
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.two_axis import balanced_cell_indices, fit_two_axis_synergy


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _kinematic_limits(
    reference: np.ndarray,
    axis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[float, float]:
    reference = np.clip(reference, low, high)
    lower, upper = -np.inf, np.inf
    for value, direction, bottom, top in zip(reference, axis, low, high):
        if abs(direction) <= 1e-12:
            continue
        endpoints = ((bottom - value) / direction, (top - value) / direction)
        lower = max(lower, min(endpoints))
        upper = min(upper, max(endpoints))
    return float(lower), float(upper)


def _task_state(env, observation: dict, info: dict) -> dict[str, Any]:
    name = type(env.unwrapped).__name__
    achieved = float(np.asarray(observation["achieved_goal"]).reshape(-1)[0])
    if name == "ShadowHandSprayBottleEnv":
        failed = bool(info.get("bottle_dropped", False) or info.get("bottle_tipped", False))
        tolerance = float(env.unwrapped.trigger_tracking_tolerance)
    elif name == "PushBottleEnv":
        failed = bool(info.get("object_dropped", False) or info.get("object_tipped", False))
        tolerance = float(env.unwrapped.push_tracking_tolerance)
    else:
        failed = not bool(info.get("is_in_grasp_space", False))
        tolerance = float(env.unwrapped.angle_threshold)
    return {
        "achieved": achieved,
        "grasp_metric": _grasp_metric(info),
        "grasp_force": _grasp_force(info),
        "held": bool(info.get("grasp_maintained", False)) and not failed,
        "failed": failed,
        "tolerance": tolerance,
    }


def _control_bounds(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected-actuator bounds for every supported hand environment."""
    unwrapped = env.unwrapped
    ranges = np.asarray(
        unwrapped.model.actuator_ctrlrange[unwrapped._actuator_ids],
        dtype=np.float64,
    )
    low = np.asarray(getattr(unwrapped, "_ctrl_low", ranges[:, 0]), dtype=np.float64)
    high = np.asarray(getattr(unwrapped, "_ctrl_high", ranges[:, 1]), dtype=np.float64)
    halfspan = np.asarray(
        getattr(unwrapped, "_ctrl_halfspan", (high - low) / 2.0),
        dtype=np.float64,
    )
    return low, high, halfspan


def _execute(
    env,
    pool_pose: int,
    option_key: str,
    target: float,
    grasp_posture: np.ndarray,
    operation_posture: np.ndarray,
    *,
    seed: int,
    grasp_steps: int,
    ramp_steps: int,
    hold_steps: int,
    frame_callback=None,
) -> dict[str, Any]:
    observation, info = env.reset(
        seed=seed,
        options={"initial_pose_index": int(pool_pose), option_key: float(target)},
    )
    unwrapped = env.unwrapped
    start = np.asarray(unwrapped._neutral_ctrl, dtype=np.float64).copy()
    _, _, halfspan = _control_bounds(env)
    phase_states: list[dict[str, Any]] = []
    done = False
    total_steps = 0

    def run(origin, destination, steps: int, score: bool) -> None:
        nonlocal observation, info, done, total_steps
        for step in range(steps):
            fraction = (step + 1) / steps
            posture = origin + fraction * (destination - origin)
            action = np.clip((posture - start) / halfspan, -1.0, 1.0)
            observation, _, terminated, truncated, info = env.step(action.astype(np.float32))
            total_steps += 1
            if frame_callback is not None and (total_steps % 20 == 0 or terminated or truncated):
                frame_callback(total_steps,observation,info)
            if score:
                phase_states.append(_task_state(env, observation, info))
            done = bool(terminated or truncated)
            if done:
                break

    run(start, grasp_posture, grasp_steps, False)
    if not done:
        run(grasp_posture, operation_posture, ramp_steps, True)
    if not done:
        run(operation_posture, operation_posture, hold_steps, True)
    final = _task_state(env, observation, info)
    held = bool(phase_states) and all(state["held"] for state in phase_states)
    return {
        **final,
        "executed_steps": total_steps,
        "held": held,
        "grasp_metric_mean": (
            float(np.mean([state["grasp_metric"] for state in phase_states]))
            if phase_states
            else 0.0
        ),
        "grasp_force_mean": (
            float(np.mean([state["grasp_force"] for state in phase_states]))
            if phase_states
            else 0.0
        ),
        "saturated_action_fraction": float(
            np.mean(
                np.abs(
                    np.clip((operation_posture - start) / halfspan, -1.0, 1.0)
                )
                >= 0.999
            )
        ),
    }


def _choose_grasp_coordinate(
    env,
    pose: int,
    option_key: str,
    nominal_target: float,
    reference: np.ndarray,
    grasp_axis: np.ndarray,
    operation_axis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    points: int,
    seed: int,
    steps: dict[str, int],
) -> tuple[float, list[dict[str, Any]]]:
    lower, upper = _kinematic_limits(reference, grasp_axis, low, high)
    candidates: list[dict[str, Any]] = []
    for index, sigma in enumerate(np.linspace(lower, upper, points)):
        posture = np.clip(reference + grasp_axis * sigma, low, high)
        result = _execute(
            env,
            pose,
            option_key,
            nominal_target,
            posture,
            posture,
            seed=seed + index,
            **steps,
        )
        result["sigma"] = float(sigma)
        op_lower, op_upper = _kinematic_limits(
            posture, operation_axis, low, high
        )
        result["operation_headroom"] = float(op_upper - op_lower)
        candidates.append(result)
    best = max(
        candidates,
        key=lambda item: (
            int(item["held"]),
            item["operation_headroom"],
            item["grasp_metric_mean"],
            item["grasp_force_mean"],
            -int(item["failed"]),
            -abs(item["sigma"]),
        ),
    )
    return float(best["sigma"]), candidates


def _identify_operation_map(
    env,
    pose: int,
    option_key: str,
    nominal_target: float,
    grasp_posture: np.ndarray,
    operation_axis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    points: int,
    seed: int,
    steps: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    lower, upper = _kinematic_limits(grasp_posture, operation_axis, low, high)
    records: list[dict[str, Any]] = []
    for index, rho in enumerate(np.linspace(lower, upper, points)):
        operation_posture = np.clip(grasp_posture + operation_axis * rho, low, high)
        result = _execute(
            env,
            pose,
            option_key,
            nominal_target,
            grasp_posture,
            operation_posture,
            seed=seed + index,
            **steps,
        )
        result["rho"] = float(rho)
        records.append(result)
    usable = [item for item in records if item["held"] and not item["failed"]]
    if len(usable) < 3:
        return np.array([]), np.array([]), records
    usable.sort(key=lambda item: item["rho"])
    rho = np.asarray([item["rho"] for item in usable], dtype=np.float64)
    tool = np.maximum.accumulate(
        np.asarray([item["achieved"] for item in usable], dtype=np.float64)
    )
    distinct = np.r_[True, np.diff(tool) > 1e-9]
    return rho[distinct], tool[distinct], records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--methods", default="pca pls")
    parser.add_argument("--split-seed", type=int, default=20260909)
    parser.add_argument("--fit-poses", type=int, default=16)
    parser.add_argument("--calibration-poses", type=int, default=8)
    parser.add_argument("--grasp-points", type=int, default=9)
    parser.add_argument("--reference-points", type=int, default=5)
    parser.add_argument("--rho-points", type=int, default=13)
    parser.add_argument("--target-points", type=int, default=9)
    parser.add_argument("--grasp-steps", type=int, default=100)
    parser.add_argument("--ramp-steps", type=int, default=50)
    parser.add_argument("--hold-steps", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with np.load(args.database, allow_pickle=False) as artifact:
        data = {name: np.asarray(artifact[name]) for name in artifact.files}
    metadata = json.loads(str(data.pop("metadata_json")))
    poses = np.unique(data["pose_index"]).astype(int)
    if args.fit_poses + args.calibration_poses >= len(poses):
        raise SystemExit("fit and calibration poses must leave at least one test pose")
    shuffled = np.random.default_rng(args.split_seed).permutation(poses)
    fit_poses = np.sort(shuffled[: args.fit_poses])
    calibration_poses = np.sort(
        shuffled[args.fit_poses : args.fit_poses + args.calibration_poses]
    )
    test_poses = np.sort(shuffled[args.fit_poses + args.calibration_poses :])
    fit_rows = np.isin(data["pose_index"], fit_poses) & data["success"].astype(bool)
    selected = balanced_cell_indices(
        data["pose_index"], data["target_index"], fit_rows, per_cell=1
    )
    if selected.size < 3:
        raise SystemExit("too few balanced successful rows to fit two axes")

    methods = tuple(args.methods.split())
    axes = {
        method: fit_two_axis_synergy(
            data["q"][selected],
            data["tool_state"][selected],
            data["pose_index"][selected],
            method=method,
            grasp_metric=data["grasp_force"][selected],
        )
        for method in methods
    }
    _register_tool_manip_gym_environments()
    horizon = args.grasp_steps + args.ramp_steps + args.hold_steps + 10
    env, _, _, option_key, target_low, target_high = _build_env(
        metadata["env_id"],
        args.pose_pool_dir,
        args.split_seed,
        max_episode_steps=horizon,
        object_position_noise=0.0,
    )
    low, high, _ = _control_bounds(env)
    nominal_target = target_low + 0.3 * (target_high - target_low)
    targets = target_low + np.linspace(0.1, 0.5, args.target_points) * (
        target_high - target_low
    )
    steps = {
        "grasp_steps": args.grasp_steps,
        "ramp_steps": args.ramp_steps,
        "hold_steps": args.hold_steps,
    }
    report: dict[str, Any] = {
        "database": str(args.database),
        "env_id": metadata["env_id"],
        "split": {
            "seed": args.split_seed,
            "fit_poses": fit_poses,
            "calibration_poses": calibration_poses,
            "test_poses": test_poses,
        },
        "balanced_fit_rows": int(selected.size),
        "axes": {method: axes[method].__dict__ for method in methods},
        "methods": {},
    }
    try:
        for method in methods:
            synergy = axes[method]
            method_report: dict[str, Any] = {"poses": {}}
            for pose in test_poses:
                pose_rows = (data["pose_index"] == pose) & data["success"].astype(bool)
                if not np.any(pose_rows):
                    method_report["poses"][str(pose)] = {
                        "available": False,
                        "reason": "no successful reference posture",
                    }
                    continue
                # Contact loads hold measured joints short of their servo
                # targets. Replaying measured q under-drives the grasp, so use
                # the command posture that produced the successful hold.
                measured_reference = np.mean(data["q"][pose_rows], axis=0)
                command_reference = np.mean(
                    data.get("command_q", data["q"])[pose_rows], axis=0
                )
                # PCA and PLS receive identical reset seeds so every reported
                # comparison is paired by pose and physical initial condition.
                base_seed = args.split_seed + 100 * int(pose)
                grasp_scan: list[dict[str, Any]] = []
                reference_candidates: list[np.ndarray] = []
                for reference_index, preload in enumerate(
                    np.linspace(0.0, 1.0, args.reference_points)
                ):
                    candidate_reference = measured_reference + preload * (
                        command_reference - measured_reference
                    )
                    _, candidate_scan = _choose_grasp_coordinate(
                        env,
                        int(pose),
                        option_key,
                        nominal_target,
                        candidate_reference,
                        synergy.grasp_axis,
                        synergy.operation_axis,
                        low,
                        high,
                        points=args.grasp_points,
                        seed=base_seed + 100 * reference_index,
                        steps=steps,
                    )
                    for item in candidate_scan:
                        item["preload"] = float(preload)
                        item["reference_index"] = reference_index
                    grasp_scan.extend(candidate_scan)
                    reference_candidates.append(candidate_reference)
                best_grasp = max(
                    grasp_scan,
                    key=lambda item: (
                        int(item["held"]),
                        item["operation_headroom"],
                        item["grasp_metric_mean"],
                        item["grasp_force_mean"],
                        -abs(item["sigma"]),
                    ),
                )
                sigma = float(best_grasp["sigma"])
                reference = reference_candidates[int(best_grasp["reference_index"])]
                same_reference = [
                    item
                    for item in grasp_scan
                    if item["reference_index"] == best_grasp["reference_index"]
                ]
                zero_grasp = min(same_reference, key=lambda item: abs(item["sigma"]))
                grasp_posture = np.clip(reference + synergy.grasp_axis * sigma, low, high)
                rho, tool, identification = _identify_operation_map(
                    env,
                    int(pose),
                    option_key,
                    nominal_target,
                    grasp_posture,
                    synergy.operation_axis,
                    low,
                    high,
                    points=args.rho_points,
                    seed=base_seed + 1000,
                    steps=steps,
                )
                pose_report: dict[str, Any] = {
                    "available": bool(rho.size >= 2),
                    "sigma": sigma,
                    "reference_preload": best_grasp["preload"],
                    "grasp_metric_improvement": float(
                        best_grasp["grasp_metric_mean"]
                        - zero_grasp["grasp_metric_mean"]
                    ),
                    "grasp_force_improvement": float(
                        best_grasp["grasp_force_mean"]
                        - zero_grasp["grasp_force_mean"]
                    ),
                    "grasp_acquired_by_axis": bool(
                        best_grasp["held"] and not zero_grasp["held"]
                    ),
                    "grasp_scan": grasp_scan,
                    "identification": identification,
                    "reachable_tool_range": (
                        [float(tool[0]), float(tool[-1])] if tool.size else None
                    ),
                    "targets": [],
                }
                if rho.size >= 2:
                    for target_index, target in enumerate(targets):
                        reachable = bool(tool[0] <= target <= tool[-1])
                        command_rho = float(np.interp(target, tool, rho))
                        operation_posture = np.clip(
                            grasp_posture + synergy.operation_axis * command_rho,
                            low,
                            high,
                        )
                        outcome = _execute(
                            env,
                            int(pose),
                            option_key,
                            float(target),
                            grasp_posture,
                            operation_posture,
                            seed=base_seed + 2000 + target_index,
                            **steps,
                        )
                        outcome.update(
                            {
                                "target": float(target),
                                "rho": command_rho,
                                "reachable": reachable,
                                "abs_error": abs(outcome["achieved"] - target),
                                "success": bool(
                                    reachable
                                    and outcome["held"]
                                    and not outcome["failed"]
                                    and abs(outcome["achieved"] - target)
                                    <= outcome["tolerance"]
                                ),
                            }
                        )
                        pose_report["targets"].append(outcome)
                method_report["poses"][str(pose)] = pose_report
            report["methods"][method] = method_report
    finally:
        env.close()
    write_report(args.output, _jsonable(report))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
