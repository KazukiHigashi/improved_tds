"""Render representative two-axis control and synergy-score sweep images."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.evaluate_two_dof_control import (
    _control_bounds,
    _execute,
    _kinematic_limits,
)
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


CAPTURE_STEPS = tuple(range(0, 201, 20))


def _safe_score(value: float) -> str:
    return f"{value:+.5f}".replace("+", "p").replace("-", "m").replace(".", "d")


def _sheet(items: list[tuple[str, Image.Image]], columns: int, title: str) -> Image.Image:
    if not items:
        canvas = Image.new("RGB", (640, 120), "white")
        ImageDraw.Draw(canvas).text((16, 16), f"{title}: unavailable", fill="black")
        return canvas
    width = max(image.width for _, image in items)
    height = max(image.height for _, image in items)
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, 34 + rows * (height + 26)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), title, fill="black")
    for index, (label, image) in enumerate(items):
        x = (index % columns) * width
        y = 34 + (index // columns) * (height + 26)
        canvas.paste(image, (x, y))
        draw.text((x + 7, y + height + 6), label, fill="black")
    return canvas


def _caption(image: Image.Image, lines: tuple[str, ...]) -> Image.Image:
    banner_height = 18 + 14 * len(lines)
    captioned = Image.new("RGB", (image.width, image.height + banner_height), "white")
    captioned.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(captioned)
    for index, line in enumerate(lines):
        draw.text((8, 7 + 14 * index), line, fill="black")
    return captioned


def _save_frame(
    frame: np.ndarray,
    path: Path,
    *subtitle: str,
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    if subtitle:
        image = _caption(image, tuple(subtitle))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image


def _run(
    env,
    pose: int,
    option_key: str,
    target: float,
    grasp_posture: np.ndarray,
    operation_posture: np.ndarray,
    *,
    seed: int,
    capture_every: int | None,
) -> tuple[list[tuple[int, np.ndarray]], dict[str, Any]]:
    observation, info = env.reset(
        seed=seed,
        options={"initial_pose_index": pose, option_key: target},
    )
    unwrapped = env.unwrapped
    start = np.asarray(unwrapped._neutral_ctrl, dtype=np.float64).copy()
    _, _, halfspan = _control_bounds(env)
    frames: list[tuple[int, np.ndarray]] = [(0, env.render().copy())]
    global_step = 0
    done = False

    def phase(origin: np.ndarray, destination: np.ndarray, steps: int) -> None:
        nonlocal observation, info, global_step, done
        for step in range(steps):
            fraction = (step + 1) / steps
            posture = origin + fraction * (destination - origin)
            action = np.clip((posture - start) / halfspan, -1.0, 1.0)
            observation, _, terminated, truncated, info = env.step(action.astype(np.float32))
            global_step += 1
            if capture_every and global_step % capture_every == 0:
                frames.append((global_step, env.render().copy()))
            done = bool(terminated or truncated)
            if done:
                break

    phase(start, grasp_posture, 100)
    if not done:
        phase(grasp_posture, operation_posture, 50)
    if not done:
        phase(operation_posture, operation_posture, 50)
    if capture_every is None:
        frames = [(global_step, env.render().copy())]
    return frames, {
        "steps": global_step,
        "terminated_early": global_step < 200,
        "achieved": float(np.asarray(observation["achieved_goal"]).reshape(-1)[0]),
        "held": bool(info.get("grasp_maintained", False)),
    }


def _reference(data: dict[str, np.ndarray], pose: int, preload: float) -> np.ndarray:
    rows = (data["pose_index"] == pose) & data["success"].astype(bool)
    if not np.any(rows):
        raise ValueError(f"pose {pose} has no successful reference rows")
    measured = np.mean(data["q"][rows], axis=0)
    commanded = np.mean(data.get("command_q", data["q"])[rows], axis=0)
    return measured + preload * (commanded - measured)


def _choose_pose(report: dict[str, Any], method: str) -> tuple[int, dict[str, Any]]:
    poses = report["methods"][method]["poses"]
    usable = [
        (int(key), value)
        for key, value in poses.items()
        if value.get("available") and value.get("targets")
    ]
    if usable:
        return sorted(usable)[0]
    diagnostic = [
        (int(key), value)
        for key, value in poses.items()
        if "sigma" in value and value.get("grasp_scan")
    ]
    if not diagnostic:
        raise ValueError(f"{method} has no renderable pose")
    return sorted(diagnostic)[0]


def _group_mean(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for level in sorted({int(item[group_key]) for item in records}):
        rows = [item for item in records if int(item[group_key]) == level]
        grouped.append(
            {
                "level": level,
                "score": float(np.mean([item["score"] for item in rows])),
                "tool_state": float(np.mean([item["tool_state"] for item in rows])),
                "samples": len(rows),
            }
        )
    return grouped


def _collection_curve(
    data: dict[str, np.ndarray],
    report: dict[str, Any],
    operation_axis: np.ndarray,
) -> list[dict[str, Any]]:
    fit_poses = np.asarray(report["split"]["fit_poses"], dtype=int)
    valid = np.isin(data["pose_index"], fit_poses) & data["success"].astype(bool)
    records: list[dict[str, Any]] = []
    for pose in fit_poses:
        rows = np.flatnonzero(valid & (data["pose_index"] == pose))
        if not rows.size:
            continue
        centred_q = data["q"][rows] - np.mean(data["q"][rows], axis=0)
        scores = centred_q @ operation_axis
        for row, score in zip(rows, scores, strict=True):
            records.append(
                {
                    "target_index": int(data["target_index"][row]),
                    "score": float(score),
                    "tool_state": float(data["tool_state"][row]),
                }
            )
    return _group_mean(records, "target_index")


def _database_points(
    data: dict[str, np.ndarray],
    report: dict[str, Any],
    operation_axis: np.ndarray,
) -> list[dict[str, Any]]:
    split_role = {
        int(pose): role
        for role, key in (
            ("fit", "fit_poses"),
            ("calibration", "calibration_poses"),
            ("test", "test_poses"),
        )
        for pose in report["split"][key]
    }
    points: list[dict[str, Any]] = []
    for pose in np.unique(data["pose_index"]).astype(int):
        pose_rows = np.flatnonzero(data["pose_index"] == pose)
        successful = pose_rows[data["success"][pose_rows].astype(bool)]
        centre_rows = successful if successful.size else pose_rows
        centre = np.mean(data["q"][centre_rows], axis=0)
        scores = (data["q"][pose_rows] - centre) @ operation_axis
        for row, score in zip(pose_rows, scores, strict=True):
            points.append(
                {
                    "pose": int(pose),
                    "target_index": int(data["target_index"][row]),
                    "score": float(score),
                    "tool_state": float(data["tool_state"][row]),
                    "success": bool(data["success"][row]),
                    "split": split_role[pose],
                }
            )
    return points


def _two_dof_curve(
    report: dict[str, Any], method: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pose_report in report["methods"][method]["poses"].values():
        for index, item in enumerate(pose_report.get("identification", [])):
            records.append(
                {
                    "sweep_index": index,
                    "score": float(item["rho"]),
                    "tool_state": float(item["achieved"]),
                }
            )
    return _group_mean(records, "sweep_index")


def _one_dof_scissors_curve(
    env,
    option_key: str,
    nominal_target: float,
    data: dict[str, np.ndarray],
    report: dict[str, Any],
    method: str,
    operation_axis: np.ndarray,
) -> list[dict[str, Any]]:
    low, high, _ = _control_bounds(env)
    records: list[dict[str, Any]] = []
    for pose in report["split"]["test_poses"]:
        pose_report = report["methods"][method]["poses"].get(str(pose), {})
        if "reference_preload" not in pose_report:
            continue
        preload = float(pose_report["reference_preload"])
        reference = _reference(data, int(pose), preload)
        lower, upper = _kinematic_limits(reference, operation_axis, low, high)
        base_seed = int(report["split"]["seed"]) + 100 * int(pose) + 5000
        for index, rho in enumerate(np.linspace(lower, upper, 13)):
            operation = np.clip(reference + operation_axis * rho, low, high)
            outcome = _execute(
                env,
                int(pose),
                option_key,
                nominal_target,
                reference,
                operation,
                seed=base_seed + index,
                grasp_steps=100,
                ramp_steps=50,
                hold_steps=50,
            )
            records.append(
                {
                    "sweep_index": index,
                    "score": float(rho),
                    "tool_state": float(outcome["achieved"]),
                }
            )
    return _group_mean(records, "sweep_index")


def _plot_operation_score_comparison(
    env,
    option_key: str,
    target_low: float,
    target_high: float,
    data: dict[str, np.ndarray],
    report: dict[str, Any],
    task: str,
    output: Path,
) -> dict[str, Any]:
    is_scissors = task.startswith("scissors")
    nominal_target = target_low + 0.3 * (target_high - target_low)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160, sharey=True)
    curve_data: dict[str, Any] = {
        "task": task,
        "score_definition": "joint posture displacement projected onto the unit operation axis",
        "aggregation": "mean at each target or sweep level across poses",
        "methods": {},
    }
    for axis_plot, method in zip(axes, ("pca", "pls"), strict=True):
        operation_axis = np.asarray(report["axes"][method]["operation_axis"], dtype=np.float64)
        database = _database_points(data, report, operation_axis)
        collection = _collection_curve(data, report, operation_axis)
        if is_scissors:
            experiment = _one_dof_scissors_curve(
                env,
                option_key,
                nominal_target,
                data,
                report,
                method,
                operation_axis,
            )
            experiment_label = "1-DoF operation sweep"
            dof = 1
        else:
            experiment = _two_dof_curve(report, method)
            experiment_label = "2-DoF operation sweep"
            dof = 2
        database_success = [item for item in database if item["success"]]
        database_failure = [item for item in database if not item["success"]]
        axis_plot.scatter(
            [item["score"] for item in database_success],
            [item["tool_state"] for item in database_success],
            s=10,
            alpha=0.18,
            color="#1f77b4",
            linewidths=0,
            label="DB success (all poses)",
        )
        axis_plot.scatter(
            [item["score"] for item in database_failure],
            [item["tool_state"] for item in database_failure],
            s=11,
            alpha=0.22,
            color="#7f7f7f",
            marker="x",
            linewidths=0.7,
            label="DB failure (all poses)",
        )
        axis_plot.plot(
            [item["score"] for item in collection],
            [item["tool_state"] for item in collection],
            "o-",
            linewidth=1.8,
            markersize=4,
            label="Successful DB mean (fit)",
        )
        axis_plot.plot(
            [item["score"] for item in experiment],
            [item["tool_state"] for item in experiment],
            "s-",
            linewidth=1.8,
            markersize=4,
            label=experiment_label,
        )
        axis_plot.set(
            title=f"{method.upper()} | {dof}-DoF control",
            xlabel="Operation synergy score",
        )
        axis_plot.grid(alpha=0.25)
        axis_plot.legend(frameon=False, fontsize=8)
        curve_data["methods"][method] = {
            "control_dof": dof,
            "database": database,
            "recollection": collection,
            "operation_experiment": experiment,
        }
    axes[0].set_ylabel("Tool DoF")
    figure.suptitle(
        f"{task}: operation synergy score vs. tool DoF\n"
        "Points are all database rows; lines are pose-averaged target/sweep levels",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output / "operation_score_vs_tool_dof.png", bbox_inches="tight")
    figure.savefig(output / "operation_score_vs_tool_dof.pdf", bbox_inches="tight")
    plt.close(figure)
    (output / "operation_score_vs_tool_dof.json").write_text(
        json.dumps(curve_data, indent=2), encoding="utf-8"
    )
    return curve_data


def _render_method(
    env,
    option_key: str,
    target_low: float,
    target_high: float,
    data: dict[str, np.ndarray],
    report: dict[str, Any],
    method: str,
    output: Path,
) -> dict[str, Any]:
    pose, pose_report = _choose_pose(report, method)
    axis = report["axes"][method]
    grasp_axis = np.asarray(axis["grasp_axis"], dtype=np.float64)
    operation_axis = np.asarray(axis["operation_axis"], dtype=np.float64)
    preload = float(pose_report["reference_preload"])
    sigma = float(pose_report["sigma"])
    reference = _reference(data, pose, preload)
    low, high, _ = _control_bounds(env)
    grasp_posture = np.clip(reference + grasp_axis * sigma, low, high)
    nominal_target = target_low + 0.3 * (target_high - target_low)
    base_seed = int(report["split"]["seed"]) + 100 * pose
    method_dir = output / method
    method_dir.mkdir(parents=True, exist_ok=True)
    task = output.name

    control = None
    targets = list(pose_report.get("targets", []))
    if targets:
        target_index = len(targets) // 2
        target_record = targets[target_index]
        rho = float(target_record["rho"])
        target = float(target_record["target"])
        operation_posture = np.clip(grasp_posture + operation_axis * rho, low, high)
        frames, outcome = _run(
            env,
            pose,
            option_key,
            target,
            grasp_posture,
            operation_posture,
            seed=base_seed + 2000 + target_index,
            capture_every=20,
        )
        saved: list[tuple[str, Image.Image]] = []
        seen = {step: frame for step, frame in frames}
        last = frames[0][1]
        for step in CAPTURE_STEPS:
            if step in seen:
                last = seen[step]
            label = f"step {step}"
            phase = (
                "Reset"
                if step == 0
                else "Grasp ramp"
                if step <= 100
                else "Operation ramp"
                if step <= 150
                else "Hold"
            )
            saved.append(
                (
                    label,
                    _save_frame(
                        last,
                        method_dir / "control" / f"step_{step:03d}.png",
                        f"{task} | {method.upper()} | Target control",
                        f"step={step:03d} | phase={phase}",
                        f"pose={pose} | target={target:.5f}",
                    ),
                )
            )
        _sheet(saved, 4, f"{method.upper()} target control | pose {pose}").save(
            method_dir / "control_contact_sheet.png"
        )
        control = {
            "pose": pose,
            "target_index": target_index,
            "target": target,
            "rho": rho,
            "reported_success": bool(target_record["success"]),
            "reported_reachable": bool(target_record["reachable"]),
            **outcome,
        }

    grasp_records = [
        item
        for item in pose_report.get("grasp_scan", [])
        if int(item["reference_index"]) == int(round(preload * 4))
    ]
    grasp_items: list[tuple[str, Image.Image]] = []
    for index, record in enumerate(grasp_records):
        sweep_sigma = float(record["sigma"])
        posture = np.clip(reference + grasp_axis * sweep_sigma, low, high)
        frames, _ = _run(
            env,
            pose,
            option_key,
            nominal_target,
            posture,
            posture,
            seed=base_seed + 100 * int(record["reference_index"]) + index,
            capture_every=None,
        )
        label = f"sigma={sweep_sigma:+.3f}"
        image = _save_frame(
            frames[-1][1],
            method_dir / "grasp_sweep" / f"{index:02d}_sigma_{_safe_score(sweep_sigma)}.png",
            f"{task} | {method.upper()} | Synergy score sweep",
            "Residual / candidate grasp axis (operation fixed)",
            f"pose={pose} | sigma={sweep_sigma:+.5f}",
        )
        grasp_items.append((label, image))
    _sheet(grasp_items, 3, f"{method.upper()} residual/grasp-axis sweep | pose {pose}").save(
        method_dir / "grasp_sweep_contact_sheet.png"
    )

    operation_items: list[tuple[str, Image.Image]] = []
    for index, record in enumerate(pose_report.get("identification", [])):
        rho = float(record["rho"])
        posture = np.clip(grasp_posture + operation_axis * rho, low, high)
        frames, _ = _run(
            env,
            pose,
            option_key,
            nominal_target,
            grasp_posture,
            posture,
            seed=base_seed + 1000 + index,
            capture_every=None,
        )
        label = f"rho={rho:+.3f}"
        image = _save_frame(
            frames[-1][1],
            method_dir / "operation_sweep" / f"{index:02d}_rho_{_safe_score(rho)}.png",
            f"{task} | {method.upper()} | Synergy score sweep",
            "Operation axis (grasp score fixed)",
            f"pose={pose} | rho={rho:+.5f}",
        )
        operation_items.append((label, image))
    _sheet(operation_items, 4, f"{method.upper()} operation-axis sweep | pose {pose}").save(
        method_dir / "operation_sweep_contact_sheet.png"
    )
    return {
        "method": method,
        "pose": pose,
        "preload": preload,
        "sigma": sigma,
        "control": control,
        "grasp_sweep_points": len(grasp_items),
        "operation_sweep_points": len(operation_items),
    }


def _write_index(output: Path, task: str, methods: list[dict[str, Any]]) -> None:
    blocks = []
    for item in methods:
        method = item["method"]
        control = (
            f'<img src="{method}/control_contact_sheet.png">'
            if item["control"] is not None
            else "<p>Target control unavailable: no identified operation map.</p>"
        )
        blocks.append(
            f"<h2>{html.escape(method.upper())}</h2>"
            "<h3>Target control: reset, grasp ramp, operation ramp, and hold at 20-step intervals</h3>"
            f"{control}"
            "<h3>Residual / candidate grasp-axis score sweep (operation score fixed)</h3>"
            f'<img src="{method}/grasp_sweep_contact_sheet.png">'
            "<h3>Operation-axis score sweep (grasp score fixed)</h3>"
            f'<img src="{method}/operation_sweep_contact_sheet.png">'
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>Two-DoF visuals</title>"
        "<style>body{font-family:sans-serif;margin:24px}img{display:block;max-width:100%;"
        "margin:12px 0;border:1px solid #bbb}h2{margin-top:36px}</style>"
        f"<h1>{html.escape(task)}: PCA/PLS control and score sweeps</h1>"
        "<h2>Operation synergy score vs. Tool DoF</h2>"
        "<p>Each PCA/PLS panel shows all database rows (success/failure), the successful fit-data mean, and the 1- or 2-DoF operation experiment.</p>"
        '<img src="operation_score_vs_tool_dof.png">'
        + "".join(blocks)
    )
    (output / "index.html").write_text(document, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plots-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with np.load(args.database, allow_pickle=False) as artifact:
        data = {name: np.asarray(artifact[name]) for name in artifact.files}
    metadata = json.loads(str(data.pop("metadata_json")))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    _register_tool_manip_gym_environments()
    env, _, _, option_key, target_low, target_high = _build_env(
        metadata["env_id"],
        args.pose_pool_dir,
        int(report["split"]["seed"]),
        max_episode_steps=210,
        object_position_noise=0.0,
        render_mode="rgb_array",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    methods = []
    try:
        if args.plots_only:
            manifest_path = args.output / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"plots-only requires {manifest_path}")
            methods = json.loads(manifest_path.read_text(encoding="utf-8"))["methods"]
        else:
            for method in ("pca", "pls"):
                methods.append(
                    _render_method(
                        env,
                        option_key,
                        target_low,
                        target_high,
                        data,
                        report,
                        method,
                        args.output,
                    )
                )
        _plot_operation_score_comparison(
            env,
            option_key,
            target_low,
            target_high,
            data,
            report,
            args.database.stem,
            args.output,
        )
    finally:
        env.close()
    manifest = {"task": args.database.stem, "methods": methods}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_index(args.output, args.database.stem, methods)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
