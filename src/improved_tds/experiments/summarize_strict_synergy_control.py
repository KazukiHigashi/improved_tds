"""Aggregate strict synergy-control reports with seed-level confidence intervals."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _family(task: str) -> str:
    return task.split("_", 1)[0]


def _ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if array.size < 2 or np.std(array, ddof=1) <= 1e-15:
        return mean, mean, mean
    margin = float(
        stats.t.ppf(0.975, array.size - 1)
        * np.std(array, ddof=1)
        / np.sqrt(array.size)
    )
    return mean, mean - margin, mean + margin


def _metrics(outcomes: list[dict]) -> dict[str, float]:
    reachable = [item for item in outcomes if item["reachable"]]
    return {
        "success_rate": float(np.mean([item["success"] for item in outcomes])),
        "reachable_rate": float(np.mean([item["reachable"] for item in outcomes])),
        "held_rate": float(np.mean([item["held"] for item in outcomes])),
        "mean_abs_error": float(np.mean([item["abs_error"] for item in outcomes])),
        "reachable_success_rate": (
            float(np.mean([item["success"] for item in reachable]))
            if reachable
            else 0.0
        ),
    }


def _paired(report: dict, left: str, right: str) -> dict[str, float]:
    left_outcomes = report["configurations"][left]["outcomes"]
    right_outcomes = report["configurations"][right]["outcomes"]
    keys = lambda item: (int(item["pose"]), int(item["target_index"]))
    left_map = {keys(item): item for item in left_outcomes}
    right_map = {keys(item): item for item in right_outcomes}
    common = sorted(set(left_map) & set(right_map))
    left_only = sum(left_map[key]["success"] and not right_map[key]["success"] for key in common)
    right_only = sum(right_map[key]["success"] and not left_map[key]["success"] for key in common)
    discordant = left_only + right_only
    pvalue = (
        float(stats.binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "cells": len(common),
        "left_only": left_only,
        "right_only": right_only,
        "mcnemar_exact_p": pvalue,
        "success_rate_difference": float(
            np.mean([right_map[key]["success"] for key in common])
            - np.mean([left_map[key]["success"] for key in common])
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = []
    for path in sorted(args.report_dir.glob("*_seed*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["task"] = path.name.split("_seed", 1)[0]
        reports.append(report)
    if not reports:
        raise SystemExit("no strict reports found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replicate_rows = []
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for report in reports:
        family = _family(report["task"])
        for config, value in report["configurations"].items():
            metrics = _metrics(value["outcomes"])
            row = {
                "task": report["task"],
                "family": family,
                "seed": report["split_seed"],
                "configuration": config,
                **metrics,
            }
            replicate_rows.append(row)
            grouped[(family, config)].append(metrics)
    metric_names = (
        "success_rate",
        "reachable_rate",
        "held_rate",
        "mean_abs_error",
        "reachable_success_rate",
    )
    aggregate = {}
    for (family, config), rows in sorted(grouped.items()):
        key = f"{family}/{config}"
        aggregate[key] = {"replicates": len(rows)}
        for metric in metric_names:
            mean, lower, upper = _ci([row[metric] for row in rows])
            aggregate[key][metric] = {
                "mean": mean,
                "ci95": [lower, upper],
            }

    comparisons: dict[str, dict] = {}
    comparison_specs = []
    for grasp in ("none", "residual_pca", "force_pls"):
        comparison_specs.append((f"operation:{grasp}", f"pca__{grasp}", f"pls__{grasp}"))
    comparison_specs.append(("operation:one_dof", "pca__one_dof", "pls__one_dof"))
    for operation in ("pca", "pls"):
        comparison_specs.extend(
            [
                (f"grasp:{operation}:residual_vs_none", f"{operation}__none", f"{operation}__residual_pca"),
                (f"grasp:{operation}:force_pls_vs_none", f"{operation}__none", f"{operation}__force_pls"),
                (f"grasp:{operation}:force_vs_residual", f"{operation}__residual_pca", f"{operation}__force_pls"),
            ]
        )
    for name, left, right in comparison_specs:
        rows = []
        for report in reports:
            if left in report["configurations"] and right in report["configurations"]:
                rows.append(_paired(report, left, right))
        if not rows:
            continue
        differences = [row["success_rate_difference"] for row in rows]
        mean, lower, upper = _ci(differences)
        left_only = sum(row["left_only"] for row in rows)
        right_only = sum(row["right_only"] for row in rows)
        discordant = left_only + right_only
        comparisons[name] = {
            "left": left,
            "right": right,
            "replicates": len(rows),
            "cells": sum(row["cells"] for row in rows),
            "mean_success_difference_right_minus_left": mean,
            "difference_ci95": [lower, upper],
            "left_only": left_only,
            "right_only": right_only,
            "mcnemar_exact_p": (
                float(stats.binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            ),
        }

    summary = {
        "report_count": len(reports),
        "seeds": sorted({int(report["split_seed"]) for report in reports}),
        "tasks": sorted({report["task"] for report in reports}),
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "replicate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=replicate_rows[0].keys())
        writer.writeheader()
        writer.writerows(replicate_rows)

    labels, means, lower_error, upper_error = [], [], [], []
    for key, value in aggregate.items():
        labels.append(key.replace("/", "\n"))
        metric = value["success_rate"]
        means.append(metric["mean"])
        lower_error.append(metric["mean"] - metric["ci95"][0])
        upper_error.append(metric["ci95"][1] - metric["mean"])
    figure, axis = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5.5), dpi=160)
    positions = np.arange(len(labels))
    axis.bar(positions, means, color="#4c78a8")
    axis.errorbar(
        positions,
        means,
        yerr=np.asarray([lower_error, upper_error]),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axis.set(
        ylabel="Success rate",
        ylim=(0.0, 1.0),
        xticks=positions,
        xticklabels=labels,
        title="Strict held-out synergy control (mean and 95% CI across task/seed replicates)",
    )
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "success_rates.png", bbox_inches="tight")
    figure.savefig(args.output_dir / "success_rates.pdf", bbox_inches="tight")
    plt.close(figure)

    table_rows = []
    for key, value in aggregate.items():
        success = value["success_rate"]
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{100*success['mean']:.1f}%</td>"
            f"<td>[{100*success['ci95'][0]:.1f}, {100*success['ci95'][1]:.1f}]%</td>"
            f"<td>{100*value['held_rate']['mean']:.1f}%</td>"
            f"<td>{100*value['reachable_rate']['mean']:.1f}%</td>"
            "</tr>"
        )
    comparison_rows = []
    for name, value in comparisons.items():
        difference = value["mean_success_difference_right_minus_left"]
        interval = value["difference_ci95"]
        comparison_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(value['left'])}</td>"
            f"<td>{html.escape(value['right'])}</td>"
            f"<td>{100*difference:+.1f} pp</td>"
            f"<td>[{100*interval[0]:+.1f}, {100*interval[1]:+.1f}] pp</td>"
            f"<td>{value['mcnemar_exact_p']:.4g}</td>"
            "</tr>"
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>Strict synergy experiment</title>"
        "<style>body{font-family:sans-serif;margin:28px}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:6px 9px}img{max-width:100%}</style>"
        "<h1>Strict one-/two-DoF synergy experiment</h1>"
        "<p>Scissors: one DoF. Push/Spray: operation plus none, residual-PCA, or force-PLS grasp ablation.</p>"
        "<p>Each family/configuration has 25 task/split-seed replicates. Confidence intervals describe "
        "task and held-out split sensitivity for a shared trained policy; they are not independent "
        "training-seed confidence intervals.</p>"
        "<img src='success_rates.png'>"
        "<table><tr><th>Family/configuration</th><th>Success</th><th>95% CI</th>"
        "<th>Held</th><th>Reachable</th></tr>"
        + "".join(table_rows)
        + "</table><h2>Paired comparisons</h2>"
        "<p>Difference is right minus left. The interval is computed across paired report replicates; "
        "the exact McNemar test uses paired pose-target outcomes.</p>"
        "<table><tr><th>Comparison</th><th>Left</th><th>Right</th><th>Difference</th>"
        "<th>95% CI</th><th>McNemar p</th></tr>"
        + "".join(comparison_rows)
        + "</table><p>Full results: <a href='summary.json'>summary.json</a>. "
        "Replicate data: <a href='replicate_metrics.csv'>replicate_metrics.csv</a>.</p>"
    )
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
