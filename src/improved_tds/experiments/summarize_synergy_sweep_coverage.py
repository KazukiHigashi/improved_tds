"""Summarize synergy-sweep range coverage and build a visual HTML report."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from improved_tds.experiments.collect_posture_database import _build_env
from improved_tds.experiments.evaluate_strict_synergy_control import _posture_on_axis
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def _ci(values: list[float]) -> dict[str, object]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    mean = float(data.mean())
    if data.size < 2 or np.std(data, ddof=1) <= 1e-15:
        interval = [mean, mean]
    else:
        margin = float(stats.t.ppf(0.975, data.size - 1) * data.std(ddof=1) / np.sqrt(data.size))
        interval = [mean - margin, mean + margin]
    return {"mean": mean, "ci95": interval}


def _metrics(configuration: dict) -> dict[str, float]:
    poses = configuration["poses"]
    span = float(configuration["target_range"][1] - configuration["target_range"][0])
    covered_errors = []
    for pose in poses:
        held_values = [
            outcome["achieved"]
            for outcome in pose["outcomes"]
            if outcome["held"] and not outcome["failed"]
        ]
        if len(held_values) < 2:
            continue
        held_low, held_high = min(held_values), max(held_values)
        covered_errors.extend(
            error["abs_error"]
            for error in pose["target_errors"]
            if error["calibration_reachable"]
            and held_low <= error["target"] <= held_high
        )
    return {
        "coverage": float(np.mean([pose["coverage"] for pose in poses])),
        "held_coverage": float(np.mean([pose["held_coverage"] for pose in poses])),
        "target_mae": float(np.mean([pose["target_mae"] for pose in poses])),
        "target_rmse": float(np.mean([pose["target_rmse"] for pose in poses])),
        "normalised_target_mae": float(np.mean([pose["target_mae"] for pose in poses]) / span),
        "covered_target_mae": (
            float(np.mean(covered_errors)) if covered_errors else float("nan")
        ),
        "covered_normalised_target_mae": (
            float(np.mean(covered_errors) / span) if covered_errors else float("nan")
        ),
        "covered_target_samples": float(len(covered_errors)),
        "held_rate": float(
            np.mean([outcome["held"] for pose in poses for outcome in pose["outcomes"]])
        ),
    }


def _task_name(path: Path) -> str:
    return path.name.split("_seed", 1)[0]


def _fit_dataset_points(report: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    strict_path = Path(report["source_report"])
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    database_path = Path(strict["database"])
    with np.load(database_path) as archive:
        data = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    _register_tool_manip_gym_environments()
    env, pool, _, _, _, _ = _build_env(
        report["env_id"],
        Path(metadata["pose_pool"]).parent,
        int(report["split_seed"]),
        object_position_noise=0.0,
    )
    low, high, _ = _control_bounds(env)
    rows = np.asarray(strict["fit_rows"], dtype=int)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        for name, configuration in report["configurations"].items():
            source = strict["configurations"][name]
            axis = np.asarray(source["operation_axis"], dtype=np.float64)
            grasp_axis = (
                np.asarray(source["grasp_axis"], dtype=np.float64)
                if source["grasp_axis"] is not None
                else None
            )
            reference_delta = np.asarray(source["reference_delta"], dtype=np.float64)
            scores = []
            candidate_scores = np.linspace(-1.0, 1.0, 401)
            for row in rows:
                reference = np.clip(data["initial_q"][row] + reference_delta, low, high)
                if grasp_axis is None:
                    grasp = reference
                else:
                    grasp, _ = _posture_on_axis(
                        reference,
                        grasp_axis,
                        float(source["normalised_grasp"]),
                        low,
                        high,
                    )
                path = np.asarray(
                    [
                        _posture_on_axis(grasp, axis, float(score), low, high)[0]
                        for score in candidate_scores
                    ]
                )
                closest = int(np.argmin(np.linalg.norm(path - data["q"][row], axis=1)))
                scores.append(float(candidate_scores[closest]))
            result[configuration["operation_method"]] = (
                np.asarray(scores),
                np.asarray(data["tool_state"][rows], dtype=np.float64),
            )
    finally:
        env.close()
        pool.close()
    return result


def _plot_task(
    report: dict,
    task: str,
    output: Path,
    dataset_points: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.7), dpi=160)
    colors = {"pca": "#4c78a8", "pls": "#e45756"}
    for name, configuration in report["configurations"].items():
        method = configuration["operation_method"]
        levels = np.asarray(configuration["sweep_levels"], dtype=np.float64)
        curves = np.asarray(
            [[item["achieved"] for item in pose["outcomes"]] for pose in configuration["poses"]]
        )
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        color = colors[method]
        fit_score, fit_tool = dataset_points[method]
        axis.scatter(
            fit_score,
            fit_tool,
            marker="x",
            s=18,
            linewidths=0.8,
            alpha=0.28,
            color=color,
            label=f"{method.upper()} fit samples (n={fit_score.size})",
        )
        axis.plot(levels, mean, marker="o", color=color, label=f"{method.upper()} test mean")
        axis.fill_between(levels, mean - std, mean + std, color=color, alpha=0.14)
        calibration_x = np.asarray(configuration["calibration_map_coordinate"])
        calibration_y = np.asarray(configuration["calibration_map_tool"])
        if calibration_x.size:
            axis.plot(
                calibration_x, calibration_y, linestyle="--", color=color, alpha=0.8,
                label=f"{method.upper()} calibration map",
            )
    target_range = next(iter(report["configurations"].values()))["target_range"]
    axis.axhspan(target_range[0], target_range[1], color="#54a24b", alpha=0.07, label="RL target range")
    axis.set(
        xlabel="Normalised operation-synergy score",
        ylabel="Tool angle / displacement",
        title=f"{task}: held-out synergy sweep",
        xlim=(-1.03, 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--visual-seed", type=int, default=20260911)
    parser.add_argument("--baseline-summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = sorted(args.report_dir.glob("*_seed*.json"))
    if not paths:
        raise SystemExit("no sweep reports found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    paired: dict[str, list[dict[str, float]]] = defaultdict(list)
    visual: dict[str, dict] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        task = _task_name(path)
        family = task.split("_", 1)[0]
        reports.append((task, report))
        if int(report["split_seed"]) == args.visual_seed:
            visual[task] = report
        report_metrics = {
            configuration["operation_method"]: _metrics(configuration)
            for configuration in report["configurations"].values()
        }
        for name, configuration in report["configurations"].items():
            grouped[(family, name)].append(report_metrics[configuration["operation_method"]])
        if {"pca", "pls"}.issubset(report_metrics):
            paired[family].append(
                {
                    "held_coverage": report_metrics["pls"]["held_coverage"]
                    - report_metrics["pca"]["held_coverage"],
                    "normalised_target_mae": report_metrics["pls"]["normalised_target_mae"]
                    - report_metrics["pca"]["normalised_target_mae"],
                }
            )
    aggregate = {}
    for (family, name), rows in sorted(grouped.items()):
        aggregate[f"{family}/{name}"] = {
            "replicates": len(rows),
            **{
                metric: _ci([row[metric] for row in rows])
                for metric in (
                    "coverage", "held_coverage", "target_mae", "target_rmse",
                    "normalised_target_mae", "covered_target_mae",
                    "covered_normalised_target_mae", "covered_target_samples", "held_rate",
                )
            },
        }
    comparisons = {
        family: {
            "definition": "PLS minus PCA on the same task and split seed",
            "replicates": len(rows),
            "held_coverage_difference": _ci([row["held_coverage"] for row in rows]),
            "normalised_target_mae_difference": _ci(
                [row["normalised_target_mae"] for row in rows]
            ),
            "paired_t_p_coverage": float(
                stats.ttest_1samp([row["held_coverage"] for row in rows], 0.0).pvalue
            ),
            "paired_t_p_normalised_mae": float(
                stats.ttest_1samp(
                    [row["normalised_target_mae"] for row in rows], 0.0
                ).pvalue
            ),
        }
        for family, rows in sorted(paired.items())
    }
    baseline_comparison = {}
    if args.baseline_summary:
        baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        for key, value in aggregate.items():
            if key not in baseline["aggregate"]:
                continue
            before = baseline["aggregate"][key]
            baseline_comparison[key] = {
                "baseline_held_coverage": before["held_coverage"]["mean"],
                "updated_held_coverage": value["held_coverage"]["mean"],
                "held_coverage_difference": value["held_coverage"]["mean"]
                - before["held_coverage"]["mean"],
                "baseline_normalised_target_mae": before["normalised_target_mae"]["mean"],
                "updated_normalised_target_mae": value["normalised_target_mae"]["mean"],
                "normalised_target_mae_difference": value["normalised_target_mae"]["mean"]
                - before["normalised_target_mae"]["mean"],
            }
    summary = {
        "report_count": len(reports),
        "visual_seed": args.visual_seed,
        "metric_definition": {
            "held_coverage": "intersection of the RL target range and the min-max tool span at sweep points held throughout, divided by target-range width",
            "target_mae": "nine full-range targets inverted through the calibration map and evaluated by interpolation of the held-out sweep response",
            "covered_target_mae": "target MAE restricted to targets inside both the held test-sweep span and calibration-map range",
        },
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "baseline_comparison": baseline_comparison,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sections = []
    for task, report in sorted(visual.items()):
        plot = args.output_dir / "plots" / f"{task}.png"
        _plot_task(report, task, plot, _fit_dataset_points(report))
        rows = []
        images = []
        for name, configuration in report["configurations"].items():
            metrics = _metrics(configuration)
            rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td><td>{100*metrics['held_coverage']:.1f}%</td>"
                f"<td>{metrics['target_mae']:.5f}</td>"
                f"<td>{100*metrics['normalised_target_mae']:.1f}%</td>"
                f"<td>{metrics['covered_target_mae']:.5f}</td>"
                f"<td>{100*metrics['held_rate']:.1f}%</td></tr>"
            )
            image = args.image_dir / task / f"{name}.png"
            if image.is_file():
                relative = Path("..") / "images" / task / image.name
                images.append(
                    f"<h4>{html.escape(name)} sweep posture</h4>"
                    f"<a href='{relative.as_posix()}'><img src='{relative.as_posix()}'></a>"
                )
        sections.append(
            f"<section><h2>{html.escape(task)}</h2>"
            f"<img src='plots/{task}.png'>"
            "<table><tr><th>Configuration</th><th>Held range coverage</th>"
            "<th>Full-range target MAE</th><th>Normalised MAE</th>"
            "<th>Covered-range MAE</th><th>Sweep held rate</th></tr>"
            + "".join(rows) + "</table>" + "".join(images) + "</section>"
        )
    aggregate_rows = []
    for key, value in aggregate.items():
        coverage = value["held_coverage"]
        aggregate_rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td><td>{100*coverage['mean']:.1f}%</td>"
            f"<td>[{100*coverage['ci95'][0]:.1f}, {100*coverage['ci95'][1]:.1f}]%</td>"
            f"<td>{value['target_mae']['mean']:.5f}</td>"
            f"<td>{100*value['normalised_target_mae']['mean']:.1f}%</td>"
            f"<td>{value['covered_target_mae']['mean']:.5f}</td>"
            "</tr>"
        )
    comparison_rows = []
    for family, value in comparisons.items():
        coverage = value["held_coverage_difference"]
        error = value["normalised_target_mae_difference"]
        comparison_rows.append(
            "<tr>"
            f"<td>{html.escape(family)}</td>"
            f"<td>{100*coverage['mean']:+.1f} pp "
            f"[{100*coverage['ci95'][0]:+.1f}, {100*coverage['ci95'][1]:+.1f}]</td>"
            f"<td>{value['paired_t_p_coverage']:.4g}</td>"
            f"<td>{100*error['mean']:+.1f} pp "
            f"[{100*error['ci95'][0]:+.1f}, {100*error['ci95'][1]:+.1f}]</td>"
            f"<td>{value['paired_t_p_normalised_mae']:.4g}</td></tr>"
        )
    baseline_rows = []
    for key, value in baseline_comparison.items():
        baseline_rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{100*value['baseline_held_coverage']:.1f}%</td>"
            f"<td>{100*value['updated_held_coverage']:.1f}%</td>"
            f"<td>{100*value['held_coverage_difference']:+.1f} pp</td>"
            f"<td>{100*value['baseline_normalised_target_mae']:.1f}%</td>"
            f"<td>{100*value['updated_normalised_target_mae']:.1f}%</td>"
            "</tr>"
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>Synergy sweep coverage</title>"
        "<style>body{font-family:sans-serif;margin:28px;max-width:1500px}"
        "table{border-collapse:collapse;margin:12px 0 24px}td,th{border:1px solid #bbb;padding:6px 9px}"
        "img{max-width:100%;height:auto}section{border-top:2px solid #ddd;margin-top:32px}</style>"
        "<h1>Feed-forward synergy sweep coverage</h1>"
        "<p>Primary metric: held sweep coverage of the complete RL target range. "
        "Target error is reported continuously and is not thresholded by the RL success tolerance. "
        "Plots and posture sheets use the designated held-out visual split. Cross markers show exactly "
        "the 100 success-only fit samples used to estimate each PCA/PLS operation axis; their scores are "
        "mapped to the same pose-specific normalised control coordinate as the sweep.</p>"
        "<h2>Five-task by five-split aggregate</h2>"
        "<table><tr><th>Family/configuration</th><th>Held coverage</th><th>95% CI</th>"
        "<th>Full-range target MAE</th><th>Normalised MAE</th><th>Covered-range MAE</th></tr>"
        + "".join(aggregate_rows)
        + "</table><h2>Correction effect versus the previous exact-line limiter</h2>"
        "<table><tr><th>Family/configuration</th><th>Old coverage</th><th>Updated coverage</th>"
        "<th>Coverage change</th><th>Old normalised MAE</th><th>Updated normalised MAE</th></tr>"
        + "".join(baseline_rows)
        + "</table><h2>Paired PLS - PCA comparisons</h2>"
        "<table><tr><th>Family</th><th>Held coverage difference</th><th>p</th>"
        "<th>Normalised MAE difference</th><th>p</th></tr>"
        + "".join(comparison_rows) + "</table>" + "".join(sections)
        + "<p><a href='summary.json'>Machine-readable summary</a></p>"
    )
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    print(f"wrote {args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
