from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from improved_tds.evaluation.reports import write_report
from improved_tds.experiments.calibration_metrics import calibrated_subset
from improved_tds.experiments.common import load_estimator, load_samples
from improved_tds.synergy.calibration import ToolStateCalibrator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a monotone tool-state calibration")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=("linear", "piecewise_linear", "isotonic", "pchip"), default="isotonic")
    parser.add_argument("--shots", type=int, default=0, help="0 uses all synchronized samples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-steps-only", action="store_true")
    parser.add_argument("--stable-success-only", action="store_true")
    args = parser.parse_args(argv)
    estimator = load_estimator(args.model)
    q, state = load_samples(
        args.dataset,
        success_steps_only=args.success_steps_only,
        stable_success_only=args.stable_success_only,
        expected_dataset_role=(
            "formal_balanced_calibration" if args.stable_success_only else None
        ),
    )
    rho = estimator.encode(q)[:, 0]
    if args.shots > 0:
        count = min(args.shots, q.shape[0])
        order = np.argsort(rho)
        indices = order[np.linspace(0, order.size - 1, count).round().astype(int)]
    else:
        indices = np.arange(q.shape[0])
    calibrator = ToolStateCalibrator(args.method).fit(rho[indices], state.reshape(-1)[indices])
    calibrator.save(args.output)
    target_in_domain, predicted, coverage = calibrated_subset(calibrator, rho, state)
    metrics = {
        "method": args.method,
        "shots": int(indices.size),
        "invertible_domain_coverage": coverage,
        "rmse": float(np.sqrt(np.mean((predicted - target_in_domain) ** 2))),
        "safe_ranges": calibrator.safe_ranges(),
    }
    write_report(Path(args.output).with_suffix(".metrics.json"), metrics)
    print(metrics)


if __name__ == "__main__":
    main()
