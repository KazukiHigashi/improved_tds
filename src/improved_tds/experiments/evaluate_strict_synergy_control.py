"""Leakage-free one-/two-DoF synergy control with calibrated grasp ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from improved_tds.experiments.collect_posture_database import _actuated_q, _build_env
from improved_tds.experiments.evaluate_two_dof_control import (
    _control_bounds,
    _execute,
)
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.two_axis import (
    clipped_posture_from_normalised,
    fit_orthogonal_grasp_axis,
    fit_two_axis_synergy,
)


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _equal_success_indices(
    data: dict[str, np.ndarray], fit_poses: np.ndarray, count: int, seed: int
) -> np.ndarray:
    valid = np.flatnonzero(
        np.isin(data["pose_index"], fit_poses) & data["success"].astype(bool)
    )
    if valid.size < count:
        raise ValueError(f"only {valid.size} successful fit rows; {count} required")
    rng = np.random.default_rng(seed)
    groups: dict[int, list[int]] = {}
    for target in np.unique(data["target_index"][valid]).astype(int):
        rows = valid[data["target_index"][valid] == target].copy()
        rng.shuffle(rows)
        groups[target] = rows.tolist()
    selected: list[int] = []
    targets = sorted(groups)
    while len(selected) < count:
        progressed = False
        for target in targets:
            if groups[target] and len(selected) < count:
                selected.append(groups[target].pop())
                progressed = True
        if not progressed:
            break
    return np.asarray(selected, dtype=int)


def balanced_angle_success_indices(data, fit_poses, count, seed, target_range, bins=10, *, allow_partial=False):
    """Exact bin quotas from actual successful states; never duplicate/pad rows."""
    if count % bins or bins < 1:
        raise ValueError('Fit count must be divisible by the number of angle bins')
    low,high=target_range
    fractions=(np.asarray(data['tool_state'])-low)/(high-low)
    target_bins=np.minimum(bins-1,(np.asarray(data['target_fraction'])*bins+1e-8).astype(int))
    actual_bins=np.minimum(bins-1,np.floor(fractions*bins).astype(int))
    valid=(np.isin(data['pose_index'],fit_poses)&data['success'].astype(bool)
           &(fractions>=0)&(fractions<=1)&(actual_bins==target_bins))
    rng=np.random.default_rng(seed)
    groups={b:rng.permutation(np.flatnonzero(valid&(target_bins==b))).tolist() for b in range(bins)}
    selected=[]; counts=np.zeros(bins,int)
    # Scarce bins get priority during global posture duplicate removal.
    for b in sorted(groups,key=lambda b:len(groups[b])):
        for index in groups[b]:
            q=data['q'][index]
            if any(np.sqrt(np.mean((q-data['q'][j])**2))<.01 for j in selected):
                continue
            selected.append(index);counts[b]+=1
            if counts[b]==count//bins:break
    if not allow_partial and np.any(counts!=count//bins):
        raise ValueError(f'Insufficient unique successful Fit samples per angle bin: {counts.tolist()} (need {count//bins} each)')
    return np.asarray(selected,dtype=int)


def _reference_deltas(
    data: dict[str, np.ndarray], calibration_poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    measured_deltas = []
    command_deltas = []
    contributing = []
    for pose in calibration_poses:
        rows = np.flatnonzero(
            (data["pose_index"] == pose) & data["success"].astype(bool)
        )
        if not rows.size:
            continue
        nearest = rows[
            np.argmin(np.abs(np.asarray(data["target_fraction"][rows]) - 0.3))
        ]
        measured_deltas.append(data["q"][nearest] - data["initial_q"][nearest])
        command_deltas.append(
            data.get("command_q", data["q"])[nearest] - data["initial_q"][nearest]
        )
        contributing.append(int(pose))
    if len(measured_deltas) < 3:
        raise ValueError("fewer than three calibration poses have successful references")
    return (
        np.median(np.asarray(measured_deltas), axis=0),
        np.median(np.asarray(command_deltas), axis=0),
        contributing,
    )


def _reference_for_pose(
    env,
    pose: int,
    option_key: str,
    target: float,
    delta: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    seed: int,
) -> np.ndarray:
    env.reset(seed=seed, options={"initial_pose_index": int(pose), option_key: target})
    return np.clip(_actuated_q(env) + delta, low, high)


def _posture_on_axis(
    reference: np.ndarray,
    axis: np.ndarray,
    value: float,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, float]:
    return clipped_posture_from_normalised(reference, axis, low, high, value)


def _axis_sweep_span(
    reference: np.ndarray,
    axis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> float:
    negative, _ = _posture_on_axis(reference, axis, -1.0, low, high)
    positive, _ = _posture_on_axis(reference, axis, 1.0, low, high)
    return float(np.linalg.norm(positive - negative))


def _choose_grasp_value(
    env,
    poses: np.ndarray,
    option_key: str,
    nominal_target: float,
    reference_delta: np.ndarray,
    grasp_axis: np.ndarray,
    operation_axis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    candidates = np.linspace(-1.0, 1.0, 9)
    records: list[dict[str, Any]] = []
    for candidate_index, value in enumerate(candidates):
        outcomes = []
        for pose in poses:
            pose_seed = seed + 1000 * int(pose) + candidate_index
            reference = _reference_for_pose(
                env,
                int(pose),
                option_key,
                nominal_target,
                reference_delta,
                low,
                high,
                pose_seed,
            )
            grasp, coordinate = _posture_on_axis(
                reference, grasp_axis, float(value), low, high
            )
            outcome = _execute(
                env,
                int(pose),
                option_key,
                nominal_target,
                grasp,
                grasp,
                seed=pose_seed,
                grasp_steps=100,
                ramp_steps=50,
                hold_steps=50,
            )
            outcomes.append(
                {
                    "pose": int(pose),
                    "coordinate": coordinate,
                    "headroom": _axis_sweep_span(
                        grasp, operation_axis, low, high
                    ),
                    **outcome,
                }
            )
        record = {
            "normalised_grasp": float(value),
            "held_rate": float(np.mean([item["held"] for item in outcomes])),
            "failure_rate": float(np.mean([item["failed"] for item in outcomes])),
            "headroom": float(np.mean([item["headroom"] for item in outcomes])),
            "grasp_metric": float(np.mean([item["grasp_metric_mean"] for item in outcomes])),
            "grasp_force": float(np.mean([item["grasp_force_mean"] for item in outcomes])),
            "outcomes": outcomes,
        }
        records.append(record)
    best = max(
        records,
        key=lambda item: (
            item["held_rate"],
            -item["failure_rate"],
            item["headroom"],
            item["grasp_metric"],
            item["grasp_force"],
            -abs(item["normalised_grasp"]),
        ),
    )
    return float(best["normalised_grasp"]), records


def _choose_reference_preload(
    env,
    poses: np.ndarray,
    option_key: str,
    nominal_target: float,
    measured_delta: np.ndarray,
    command_delta: np.ndarray,
    operation_axis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    records = []
    for preload_index, preload in enumerate(np.linspace(0.0, 1.0, 5)):
        delta = measured_delta + preload * (command_delta - measured_delta)
        outcomes = []
        for pose in poses:
            pose_seed = seed + 1000 * int(pose) + preload_index
            reference = _reference_for_pose(
                env,
                int(pose),
                option_key,
                nominal_target,
                delta,
                low,
                high,
                pose_seed,
            )
            outcome = _execute(
                env,
                int(pose),
                option_key,
                nominal_target,
                reference,
                reference,
                seed=pose_seed,
                grasp_steps=100,
                ramp_steps=50,
                hold_steps=50,
            )
            outcomes.append(
                {
                    "pose": int(pose),
                    "headroom": _axis_sweep_span(
                        reference, operation_axis, low, high
                    ),
                    **outcome,
                }
            )
        records.append(
            {
                "preload": float(preload),
                "held_rate": float(np.mean([item["held"] for item in outcomes])),
                "failure_rate": float(np.mean([item["failed"] for item in outcomes])),
                "headroom": float(np.mean([item["headroom"] for item in outcomes])),
                "grasp_metric": float(np.mean([item["grasp_metric_mean"] for item in outcomes])),
                "outcomes": outcomes,
            }
        )
    best = max(
        records,
        key=lambda item: (
            item["held_rate"],
            -item["failure_rate"],
            item["headroom"],
            item["grasp_metric"],
            -item["preload"],
        ),
    )
    return float(best["preload"]), records


def _operation_calibration(
    env,
    poses: np.ndarray,
    option_key: str,
    nominal_target: float,
    reference_delta: np.ndarray,
    operation_axis: np.ndarray,
    grasp_axis: np.ndarray | None,
    grasp_value: float,
    low: np.ndarray,
    high: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    levels = np.linspace(-1.0, 1.0, 13)
    records: list[dict[str, Any]] = []
    means = []
    for level_index, value in enumerate(levels):
        outcomes = []
        for pose in poses:
            pose_seed = seed + 1000 * int(pose) + level_index
            reference = _reference_for_pose(
                env,
                int(pose),
                option_key,
                nominal_target,
                reference_delta,
                low,
                high,
                pose_seed,
            )
            if grasp_axis is None:
                grasp, grasp_coordinate = reference, 0.0
            else:
                grasp, grasp_coordinate = _posture_on_axis(
                    reference, grasp_axis, grasp_value, low, high
                )
            operation, rho = _posture_on_axis(
                grasp, operation_axis, float(value), low, high
            )
            outcome = _execute(
                env,
                int(pose),
                option_key,
                nominal_target,
                grasp,
                operation,
                seed=pose_seed,
                grasp_steps=100,
                ramp_steps=50,
                hold_steps=50,
            )
            outcomes.append(
                {
                    "pose": int(pose),
                    "grasp_coordinate": grasp_coordinate,
                    "operation_coordinate": rho,
                    **outcome,
                }
            )
        usable = [item for item in outcomes if item["held"] and not item["failed"]]
        record = {
            "level_index": level_index,
            "normalised_operation": float(value),
            "usable_poses": len(usable),
            "outcomes": outcomes,
        }
        records.append(record)
        if usable:
            means.append((float(value), float(np.mean([item["achieved"] for item in usable]))))
    if len(means) < 2:
        return np.array([]), np.array([]), records
    coordinate = np.asarray([item[0] for item in means])
    tool = np.maximum.accumulate(np.asarray([item[1] for item in means]))
    distinct = np.r_[True, np.diff(tool) > 1e-9]
    return coordinate[distinct], tool[distinct], records


def _evaluate_targets(
    env,
    poses: np.ndarray,
    targets: np.ndarray,
    option_key: str,
    nominal_target: float,
    reference_delta: np.ndarray,
    operation_axis: np.ndarray,
    grasp_axis: np.ndarray | None,
    grasp_value: float,
    map_coordinate: np.ndarray,
    map_tool: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    outcomes = []
    for pose in poses:
        for target_index, target in enumerate(targets):
            trial_seed = seed + 1000 * int(pose) + target_index
            reference = _reference_for_pose(
                env,
                int(pose),
                option_key,
                nominal_target,
                reference_delta,
                low,
                high,
                trial_seed,
            )
            if grasp_axis is None:
                grasp, grasp_coordinate = reference, 0.0
            else:
                grasp, grasp_coordinate = _posture_on_axis(
                    reference, grasp_axis, grasp_value, low, high
                )
            reachable = bool(
                map_tool.size >= 2 and map_tool[0] <= target <= map_tool[-1]
            )
            operation_value = (
                float(np.interp(target, map_tool, map_coordinate))
                if map_tool.size
                else 0.0
            )
            operation, rho = _posture_on_axis(
                grasp, operation_axis, operation_value, low, high
            )
            result = _execute(
                env,
                int(pose),
                option_key,
                float(target),
                grasp,
                operation,
                seed=trial_seed,
                grasp_steps=100,
                ramp_steps=50,
                hold_steps=50,
            )
            error = abs(float(result["achieved"]) - float(target))
            result.update(
                {
                    "pose": int(pose),
                    "target_index": target_index,
                    "target": float(target),
                    "normalised_grasp": grasp_value,
                    "grasp_coordinate": grasp_coordinate,
                    "normalised_operation": operation_value,
                    "operation_coordinate": rho,
                    "reachable": reachable,
                    "abs_error": error,
                    "success": bool(
                        reachable
                        and result["held"]
                        and not result["failed"]
                        and error <= result["tolerance"]
                    ),
                }
            )
            outcomes.append(result)
    return outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--common-fit-rows", type=int, default=100)
    parser.add_argument('--balanced-angle-bins',type=int,default=0)
    parser.add_argument(
        "--require-all-fit-targets",
        action="store_true",
        help="fail if any target bin has no successful row in the Fit poses",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with np.load(args.database, allow_pickle=False) as artifact:
        data = {name: np.asarray(artifact[name]) for name in artifact.files}
    metadata = json.loads(str(data.pop("metadata_json")))
    if not metadata.get("strict_success_window", False):
        raise SystemExit("strict evaluator requires a strict-success-window database")
    poses = np.arange(int(metadata['pose_count'])) if args.balanced_angle_bins else np.unique(data["pose_index"]).astype(int)
    if args.balanced_angle_bins and len(poses)!=32:
        raise ValueError('This protocol requires all 32 authored reset poses')
    shuffled = np.random.default_rng(args.split_seed).permutation(poses)
    fit_poses = np.sort(shuffled[:16])
    calibration_poses = np.sort(shuffled[16:24])
    test_poses = np.sort(shuffled[24:])
    if args.require_all_fit_targets:
        fit_success = np.isin(data["pose_index"], fit_poses) & data["success"].astype(bool)
        all_targets = set(np.unique(data["target_index"]).astype(int).tolist())
        covered_targets = set(
            np.unique(data["target_index"][fit_success]).astype(int).tolist()
        )
        missing_targets = sorted(all_targets - covered_targets)
        if missing_targets:
            fractions = np.asarray(data["target_fraction"])
            missing_fractions = [
                float(fractions[np.flatnonzero(data["target_index"] == index)[0]])
                for index in missing_targets
            ]
            raise ValueError(
                "Fit poses have no successful row for target fractions "
                f"{missing_fractions}"
            )
    selected = (balanced_angle_success_indices(data,fit_poses,args.common_fit_rows,args.split_seed,
        metadata['target_range'],args.balanced_angle_bins) if args.balanced_angle_bins else
        _equal_success_indices(data, fit_poses, args.common_fit_rows, args.split_seed))
    axes = {
        method: fit_two_axis_synergy(
            data["q"][selected],
            data["tool_state"][selected],
            data["pose_index"][selected],
            method=method,
            grasp_metric=data["grasp_force"][selected],
        )
        for method in ("pca", "pls")
    }
    measured_delta, command_delta, reference_poses = _reference_deltas(
        data, calibration_poses
    )
    _register_tool_manip_gym_environments()
    env, _, _, option_key, target_low, target_high = _build_env(
        metadata["env_id"],
        args.pose_pool_dir,
        args.split_seed,
        max_episode_steps=210,
        object_position_noise=0.0,
        research_task=metadata.get("task_version") == "posture_database_v2",
    )
    low, high, _ = _control_bounds(env)
    nominal_target = target_low + 0.3 * (target_high - target_low)
    # Evaluate the complete RL goal range.  Calibration still uses a nominal
    # interior target only to establish a repeatable reference posture.
    targets = target_low + np.linspace(0.0, 1.0, 11) * (target_high - target_low)
    is_scissors = metadata["env_id"].startswith("Scissors")
    report: dict[str, Any] = {
        "schema_version": 1,
        "research_task": metadata.get('task_version')=='posture_database_v2',
        "rl_scissors_control_version": metadata.get('scissors_control_version','v2'),
        "ff_control_version": "direct_absolute_command",
        "balanced_angle_bins": args.balanced_angle_bins,
        "grasp_signal_definition": metadata.get('grasp_signal_definition','legacy grasp metric'),
        "database": str(args.database.resolve()),
        "env_id": metadata["env_id"],
        "split_seed": args.split_seed,
        "fit_poses": fit_poses,
        "calibration_poses": calibration_poses,
        "test_poses": test_poses,
        "target_values": targets,
        "fit_rows": selected,
        "common_fit_rows": int(selected.size),
        "database_target_fractions": np.asarray(metadata["target_fractions"]),
        "database_success_by_target": {
            str(float(fraction)): int(
                np.sum(
                    data["success"].astype(bool)
                    & np.isclose(data["target_fraction"], float(fraction))
                )
            )
            for fraction in metadata["target_fractions"]
        },
        "fit_target_fractions": np.asarray(data["target_fraction"][selected]),
        "fit_success_by_target": {
            str(float(fraction)): int(
                np.sum(np.isclose(data["target_fraction"][selected], float(fraction)))
            )
            for fraction in metadata["target_fractions"]
        },
        "measured_reference_delta": measured_delta,
        "command_reference_delta": command_delta,
        "reference_calibration_poses": reference_poses,
        "configurations": {},
    }
    try:
        for operation_method, synergy in axes.items():
            reference_preload, reference_calibration = _choose_reference_preload(
                env,
                calibration_poses,
                option_key,
                nominal_target,
                measured_delta,
                command_delta,
                synergy.operation_axis,
                low,
                high,
                args.split_seed + 50_000,
            )
            reference_delta = measured_delta + reference_preload * (
                command_delta - measured_delta
            )
            if is_scissors:
                grasp_methods = ("one_dof",)
            else:
                grasp_methods = ("none", "residual_pca", "force_pls")
            for grasp_method in grasp_methods:
                key = f"{operation_method}__{grasp_method}"
                if grasp_method in {"one_dof", "none"}:
                    grasp_axis = None
                    grasp_value = 0.0
                    grasp_calibration = []
                else:
                    try:
                        grasp_axis = fit_orthogonal_grasp_axis(
                            data["q"][selected], data["pose_index"][selected],
                            synergy.operation_axis, data["grasp_force"][selected],
                            method=grasp_method)
                    except ValueError as exc:
                        if grasp_method != 'force_pls':
                            raise
                        report['configurations'][key] = dict(available=False,
                            reason=str(exc),operation_method=operation_method,grasp_method=grasp_method)
                        continue
                    grasp_value, grasp_calibration = _choose_grasp_value(
                        env,
                        calibration_poses,
                        option_key,
                        nominal_target,
                        reference_delta,
                        grasp_axis,
                        synergy.operation_axis,
                        low,
                        high,
                        args.split_seed + 100_000,
                    )
                map_coordinate, map_tool, operation_calibration = _operation_calibration(
                    env,
                    calibration_poses,
                    option_key,
                    nominal_target,
                    reference_delta,
                    synergy.operation_axis,
                    grasp_axis,
                    grasp_value,
                    low,
                    high,
                    args.split_seed + 200_000,
                )
                outcomes = _evaluate_targets(
                    env,
                    test_poses,
                    targets,
                    option_key,
                    nominal_target,
                    reference_delta,
                    synergy.operation_axis,
                    grasp_axis,
                    grasp_value,
                    map_coordinate,
                    map_tool,
                    low,
                    high,
                    args.split_seed + 300_000,
                )
                report["configurations"][key] = {
                    "operation_method": operation_method,
                    "grasp_method": grasp_method,
                    "operation_axis": synergy.operation_axis,
                    "grasp_axis": grasp_axis,
                    "operation_tool_correlation": synergy.operation_tool_correlation,
                    "reference_preload": reference_preload,
                    "reference_calibration": reference_calibration,
                    "reference_delta": reference_delta,
                    "normalised_grasp": grasp_value,
                    "grasp_calibration": grasp_calibration,
                    "operation_calibration": operation_calibration,
                    "map_coordinate": map_coordinate,
                    "map_tool": map_tool,
                    "outcomes": outcomes,
                }
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=_jsonable), encoding="utf-8"
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
