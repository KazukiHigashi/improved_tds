"""Aggregate paired PCA/PLS two-DoF control reports across tool variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from improved_tds.evaluation.reports import write_report


def _method_rows(report: dict, method: str) -> dict[tuple[int, int], dict]:
    rows: dict[tuple[int, int], dict] = {}
    for pose_text, pose in report["methods"][method]["poses"].items():
        for target_index, outcome in enumerate(pose.get("targets", [])):
            rows[(int(pose_text), target_index)] = outcome
    return rows


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"attempts": 0, "success_rate": 0.0, "held_rate": 0.0}
    return {
        "attempts": len(rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "held_rate": float(np.mean([bool(row["held"]) for row in rows])),
        "reachable_rate": float(np.mean([bool(row["reachable"]) for row in rows])),
        "mean_abs_error": float(np.mean([float(row["abs_error"]) for row in rows])),
        "median_abs_error": float(np.median([float(row["abs_error"]) for row in rows])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default="*_control.json")
    args = parser.parse_args(argv)
    import json

    reports = sorted(args.report_dir.glob(args.pattern))
    if not reports:
        raise SystemExit(f"no reports matched {args.pattern} in {args.report_dir}")
    variants: dict[str, dict] = {}
    paired_differences: list[float] = []
    paired_success = {"pca_only": 0, "pls_only": 0, "both": 0, "neither": 0}
    all_rows = {"pca": [], "pls": []}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        pca = _method_rows(report, "pca")
        pls = _method_rows(report, "pls")
        shared = sorted(set(pca) & set(pls))
        for cell in shared:
            paired_differences.append(float(pls[cell]["abs_error"]) - float(pca[cell]["abs_error"]))
            key = (
                "both"
                if pca[cell]["success"] and pls[cell]["success"]
                else "pca_only"
                if pca[cell]["success"]
                else "pls_only"
                if pls[cell]["success"]
                else "neither"
            )
            paired_success[key] += 1
        pca_rows, pls_rows = list(pca.values()), list(pls.values())
        all_rows["pca"].extend(pca_rows)
        all_rows["pls"].extend(pls_rows)
        variants[path.stem.removesuffix("_control")] = {
            "env_id": report["env_id"],
            "balanced_fit_rows": report["balanced_fit_rows"],
            "pca": _summary(pca_rows),
            "pls": _summary(pls_rows),
            "paired_cells": len(shared),
        }
    differences = np.asarray(paired_differences, dtype=np.float64)
    payload = {
        "report_dir": str(args.report_dir),
        "variant_count": len(variants),
        "variants": variants,
        "overall": {method: _summary(rows) for method, rows in all_rows.items()},
        "paired": {
            "cells": int(differences.size),
            "mean_pls_minus_pca_abs_error": (
                float(np.mean(differences)) if differences.size else None
            ),
            "median_pls_minus_pca_abs_error": (
                float(np.median(differences)) if differences.size else None
            ),
            "success_table": paired_success,
        },
    }
    write_report(args.output, payload)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
