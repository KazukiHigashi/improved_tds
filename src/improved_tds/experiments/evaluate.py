from __future__ import annotations

import argparse


from improved_tds.evaluation.tracking import compute_tracking_metrics
from improved_tds.experiments.calibration_metrics import calibrated_subset
from improved_tds.experiments.common import load_estimator, load_samples
from improved_tds.synergy.calibration import ToolStateCalibrator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline TDS/calibration evaluation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--dt", type=float, default=0.04)
    args = parser.parse_args(argv)
    q, state = load_samples(args.dataset)
    estimator = load_estimator(args.model)
    calibrator = ToolStateCalibrator.load(args.calibration)
    target_in_domain, predicted, coverage = calibrated_subset(calibrator, estimator.encode(q)[:, 0], state)
    metrics = compute_tracking_metrics(target_in_domain, predicted, dt=args.dt)
    print({**metrics.as_dict(), "invertible_domain_coverage": coverage})


if __name__ == "__main__":
    main()

