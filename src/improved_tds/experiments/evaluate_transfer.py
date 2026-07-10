from __future__ import annotations

import argparse
import json
from pathlib import Path

from improved_tds.evaluation.reports import write_report
from improved_tds.evaluation.transfer import leave_one_instance_out
from improved_tds.experiments.common import load_samples
from improved_tds.synergy.pca import PCATDS
from improved_tds.synergy.supervised import SupervisedLinearTDS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Leave-one-instance-out family TDS evaluation")
    parser.add_argument(
        "--manifest",
        required=True,
        help='JSON object mapping instance IDs to dataset paths, e.g. {"s1":"s1.npz"}',
    )
    parser.add_argument("--method", choices=("pca", "pls", "covariance"), default="pls")
    parser.add_argument("--shots", default="3,6,12")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    samples = {
        str(instance): load_samples(
            path if Path(path).is_absolute() else manifest_path.parent / path
        )
        for instance, path in manifest.items()
    }
    factory = (
        (lambda: PCATDS())
        if args.method == "pca"
        else (lambda: SupervisedLinearTDS(method=args.method))
    )
    shots = tuple(int(value) for value in args.shots.split(",") if value)
    results = leave_one_instance_out(samples, factory, calibration_shots=shots)
    write_report(args.output, {"method": args.method, "instances": results})
    print(results)


if __name__ == "__main__":
    main()

