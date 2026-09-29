"""Validate the structural invariants of the strict synergy experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate(root: Path) -> None:
    databases = sorted((root / "databases").glob("*.npz"))
    reports = sorted((root / "reports").glob("*.json"))
    assert len(databases) == 15, f"expected 15 databases, found {len(databases)}"
    assert len(reports) == 75, f"expected 75 reports, found {len(reports)}"

    for path in databases:
        with np.load(path) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            assert metadata["strict_success_window"] is True
            assert metadata["success_window_steps"] == 5
            expected_rows = metadata["pose_count"] * len(metadata["target_fractions"]) * metadata["episodes_per_cell"]
            assert len(data["q"]) == metadata["rows"] == expected_rows
            assert min(metadata["target_fractions"]) == 0.0
            assert max(metadata["target_fractions"]) == 1.0
            successful_holds = data["hold_steps"][data["success"].astype(bool)]
            assert set(successful_holds.tolist()) == {5}

    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        fit = set(report["fit_poses"])
        calibration = set(report["calibration_poses"])
        test = set(report["test_poses"])
        assert (len(fit), len(calibration), len(test)) == (16, 8, 8)
        assert len(fit | calibration | test) == 32
        assert fit.isdisjoint(calibration)
        assert fit.isdisjoint(test)
        assert calibration.isdisjoint(test)
        assert len(report["fit_rows"]) == report["common_fit_rows"] == 100
        assert min(report["database_target_fractions"]) == 0.0
        assert max(report["database_target_fractions"]) == 1.0

        expected_configurations = 2 if "scissors" in path.name else 6
        assert len(report["configurations"]) == expected_configurations
        for configuration in report["configurations"].values():
            outcomes = configuration["outcomes"]
            expected_outcomes = len(test) * len(report["target_values"])
            assert len(outcomes) == expected_outcomes
            assert {outcome["pose"] for outcome in outcomes} == test
            assert len(
                {(outcome["pose"], outcome["target_index"]) for outcome in outcomes}
            ) == expected_outcomes
            for stage in (
                "reference_calibration",
                "operation_calibration",
                "grasp_calibration",
            ):
                assert all(
                    outcome["pose"] in calibration
                    for candidate in configuration[stage]
                    for outcome in candidate["outcomes"]
                )

    print(
        f"validated databases={len(databases)} reports={len(reports)}; "
        "all invariants passed"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    validate(args.root)


if __name__ == "__main__":
    main()
