from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from improved_tds.evaluation.reports import write_report
from improved_tds.experiments.common import load_samples
from improved_tds.synergy.pca import PCATDS
from improved_tds.synergy.random import RandomTDS
from improved_tds.synergy.supervised import SupervisedLinearTDS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a one-dimensional TDS estimator")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=("pca", "pls", "covariance", "random"), default="pca")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args(argv)
    q, state = load_samples(args.dataset)
    estimator = (PCATDS() if args.method == "pca" else RandomTDS(args.seed) if args.method == "random" else SupervisedLinearTDS(method=args.method))
    estimator.fit(q, state)
    estimator.save(args.output)
    rho = estimator.encode(q)[:, 0]
    reconstructed = estimator.decode(rho)
    metrics = {
        "method": args.method,
        "samples": int(q.shape[0]),
        "joints": int(q.shape[1]),
        "reconstruction_rmse": float(np.sqrt(np.mean((reconstructed - q) ** 2))),
        "tool_state_correlation": float(np.corrcoef(rho, state.reshape(-1))[0, 1]),
    }
    metrics_path = args.metrics or str(Path(args.output).with_suffix(".metrics.json"))
    write_report(metrics_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()

