"""Evaluate held tool-range coverage from fixed PCA/PLS synergy sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from improved_tds.experiments.collect_posture_database import _actuated_q, _build_env
from improved_tds.experiments.evaluate_strict_synergy_control import (
    _posture_on_axis,
    _reference_for_pose,
)
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds, _execute
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def interval_coverage(values: np.ndarray, low: float, high: float) -> float:
    """Fraction of [low, high] covered by the continuous observed value span."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2 or high <= low:
        return 0.0
    overlap = max(0.0, min(high, float(values.max())) - max(low, float(values.min())))
    return float(overlap / (high - low))


def _caption(frame: np.ndarray, lines: list[str]) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    banner = 12 + 15 * len(lines)
    result = Image.new("RGB", (image.width, image.height + banner), "white")
    result.paste(image, (0, banner))
    draw = ImageDraw.Draw(result)
    for index, line in enumerate(lines):
        draw.text((8, 5 + 15 * index), line, fill="black")
    return result


def _contact_sheet(items: list[Image.Image], title: str) -> Image.Image:
    columns = 4
    width = max(image.width for image in items)
    height = max(image.height for image in items)
    rows = (len(items) + columns - 1) // columns
    result = Image.new("RGB", (columns * width, 30 + rows * height), "white")
    ImageDraw.Draw(result).text((8, 8), title, fill="black")
    for index, image in enumerate(items):
        result.paste(image, ((index % columns) * width, 30 + (index // columns) * height))
    return result


def _configuration_names(report: dict[str, Any]) -> list[str]:
    if "scissors" in report["env_id"].lower():
        names = ["pca__one_dof", "pls__one_dof"]
    else:
        names = ["pca__force_pls", "pls__force_pls"]
    missing = [name for name in names if name not in report["configurations"]]
    if missing:
        raise ValueError(f"missing configurations: {missing}")
    return names


def _representative_render_pose(
    env,
    poses: list[int],
    option_key: str,
    nominal_target: float,
    reference_delta: np.ndarray,
    operation_axis: np.ndarray,
    grasp_axis: np.ndarray | None,
    grasp_value: float,
    low: np.ndarray,
    high: np.ndarray,
    seed: int,
) -> int:
    spans = []
    for pose in poses:
        reference = _reference_for_pose(
            env, pose, option_key, nominal_target, reference_delta,
            low, high, seed + 1000 * pose,
        )
        grasp = reference
        if grasp_axis is not None:
            grasp, _ = _posture_on_axis(reference, grasp_axis, grasp_value, low, high)
        negative, _ = _posture_on_axis(grasp, operation_axis, -1.0, low, high)
        positive, _ = _posture_on_axis(grasp, operation_axis, 1.0, low, high)
        spans.append((float(np.linalg.norm(positive - negative)), int(pose)))
    nonzero = sorted(item for item in spans if item[0] > 1e-8)
    candidates = nonzero if nonzero else sorted(spans)
    return candidates[(len(candidates) - 1) // 2][1]


def _target_errors(
    levels: np.ndarray,
    achieved: np.ndarray,
    map_coordinate: np.ndarray,
    map_tool: np.ndarray,
    low: float,
    high: float,
) -> list[dict[str, float | bool]]:
    targets = np.linspace(low, high, 9)
    records = []
    for target in targets:
        score = float(np.interp(target, map_tool, map_coordinate)) if map_tool.size else 0.0
        prediction = float(np.interp(score, levels, achieved))
        records.append(
            {
                "target": float(target),
                "score": score,
                "achieved_from_sweep": prediction,
                "abs_error": abs(prediction - float(target)),
                "calibration_reachable": bool(
                    map_tool.size >= 2 and map_tool[0] <= target <= map_tool[-1]
                ),
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-report", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--sweep-points", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sweep_points < 3:
        raise SystemExit("sweep-points must be at least three")
    report = json.loads(args.strict_report.read_text(encoding="utf-8"))
    _register_tool_manip_gym_environments()
    env, pool, _, option_key, target_low, target_high = _build_env(
        report["env_id"],
        args.pose_pool_dir,
        int(report["split_seed"]),
        object_position_noise=0.0,
        render_mode="rgb_array" if args.render_dir else None,
    )
    low, high, _ = _control_bounds(env)
    nominal_target = target_low + 0.3 * (target_high - target_low)
    levels = np.linspace(-1.0, 1.0, args.sweep_points)
    configurations: dict[str, Any] = {}
    try:
        for config_name in _configuration_names(report):
            source = report["configurations"][config_name]
            operation_axis = np.asarray(source["operation_axis"], dtype=np.float64)
            grasp_axis = (
                np.asarray(source["grasp_axis"], dtype=np.float64)
                if source["grasp_axis"] is not None
                else None
            )
            reference_delta = np.asarray(source["reference_delta"], dtype=np.float64)
            grasp_value = float(source["normalised_grasp"])
            map_coordinate = np.asarray(source["map_coordinate"], dtype=np.float64)
            map_tool = np.asarray(source["map_tool"], dtype=np.float64)
            poses = []
            render_items: list[Image.Image] = []
            render_pose = _representative_render_pose(
                env,
                [int(pose) for pose in report["test_poses"]],
                option_key,
                nominal_target,
                reference_delta,
                operation_axis,
                grasp_axis,
                grasp_value,
                low,
                high,
                int(report["split_seed"]),
            )
            for pose in report["test_poses"]:
                outcomes = []
                for level_index, level in enumerate(levels):
                    trial_seed = int(report["split_seed"]) + 1000 * int(pose) + level_index
                    reference = _reference_for_pose(
                        env, int(pose), option_key, nominal_target, reference_delta,
                        low, high, trial_seed,
                    )
                    if grasp_axis is None:
                        grasp = reference
                    else:
                        grasp, _ = _posture_on_axis(
                            reference, grasp_axis, grasp_value, low, high
                        )
                    operation, coordinate = _posture_on_axis(
                        grasp, operation_axis, float(level), low, high
                    )
                    outcome = _execute(
                        env, int(pose), option_key, nominal_target, grasp, operation,
                        seed=trial_seed, grasp_steps=100, ramp_steps=50, hold_steps=50,
                    )
                    measured_q = _actuated_q(env)
                    outcome.update(
                        {
                            "normalised_score": float(level),
                            "operation_coordinate": coordinate,
                            "command_delta_from_grasp": float(
                                np.linalg.norm(operation - grasp)
                            ),
                            "measured_delta_from_grasp": float(
                                np.linalg.norm(measured_q - grasp)
                            ),
                        }
                    )
                    outcomes.append(outcome)
                    if args.render_dir and int(pose) == render_pose:
                        rendered = _caption(
                            env.render().copy(),
                            [
                                f"{config_name} | test pose {pose}",
                                f"score={level:+.2f} | coord={coordinate:+.3f} | tool={outcome['achieved']:.4f}",
                                f"command delta={outcome['command_delta_from_grasp']:.3f} | measured delta={outcome['measured_delta_from_grasp']:.3f} | held={outcome['held']}",
                            ],
                        )
                        frame_path = (
                            args.render_dir / config_name / f"score_{level_index:02d}.png"
                        )
                        frame_path.parent.mkdir(parents=True, exist_ok=True)
                        rendered.save(frame_path)
                        render_items.append(rendered)
                achieved = np.asarray([item["achieved"] for item in outcomes])
                held_achieved = np.asarray(
                    [item["achieved"] for item in outcomes if item["held"] and not item["failed"]]
                )
                errors = _target_errors(
                    levels, achieved, map_coordinate, map_tool, target_low, target_high
                )
                poses.append(
                    {
                        "pose": int(pose),
                        "outcomes": outcomes,
                        "coverage": interval_coverage(achieved, target_low, target_high),
                        "held_coverage": interval_coverage(
                            held_achieved, target_low, target_high
                        ),
                        "target_errors": errors,
                        "target_mae": float(np.mean([item["abs_error"] for item in errors])),
                        "target_rmse": float(
                            np.sqrt(np.mean([item["abs_error"] ** 2 for item in errors]))
                        ),
                    }
                )
            if args.render_dir:
                image_path = args.render_dir / f"{config_name}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                _contact_sheet(
                    render_items,
                    f"{report['env_id']} | {config_name} | normalised synergy sweep",
                ).save(image_path)
            configurations[config_name] = {
                "operation_method": source["operation_method"],
                "grasp_method": source["grasp_method"],
                "target_range": [target_low, target_high],
                "render_pose": render_pose,
                "sweep_levels": levels.tolist(),
                "calibration_map_coordinate": map_coordinate.tolist(),
                "calibration_map_tool": map_tool.tolist(),
                "poses": poses,
            }
    finally:
        env.close()
        pool.close()
    output = {
        "schema_version": 1,
        "source_report": str(args.strict_report.resolve()),
        "env_id": report["env_id"],
        "split_seed": report["split_seed"],
        "test_poses": report["test_poses"],
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
