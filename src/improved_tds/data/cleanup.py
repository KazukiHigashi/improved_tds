"""Drop trajectories a recorded dataset cannot validate.

Recording runs alongside training and must never stop it, so a malformed
episode is warned about and skipped rather than raised. Anything that slipped
into a saved dataset before that policy was in place -- or that a later schema
check rejects -- is removed here, once the run is over and it costs nothing.

    python -m improved_tds.data.cleanup runs/**/training_candidates.npz
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from improved_tds.data.trajectory import Trajectory, TrajectoryDataset


@dataclass
class CleanupReport:
    path: Path
    kept: int
    dropped: int
    reasons: Counter

    @property
    def clean(self) -> bool:
        return self.dropped == 0

    def __str__(self) -> str:
        if self.clean:
            return f"{self.path.name}: {self.kept} trajectories, nothing to drop"
        detail = "; ".join(f"{reason} x{count}" for reason, count in self.reasons.most_common())
        return f"{self.path.name}: kept {self.kept}, dropped {self.dropped} ({detail})"


def _is_valid(trajectory: Trajectory) -> tuple[bool, str]:
    try:
        trajectory.validate()
    except ValueError as exc:
        return False, str(exc)
    return True, ""


def clean_dataset(path: str | Path, *, write: bool = True) -> CleanupReport:
    """Rewrite ``path`` without the trajectories that fail validation."""
    path = Path(path)
    # a dataset holding a bad trajectory may refuse to load through the
    # validating path, so read it without the whole-dataset check
    dataset = TrajectoryDataset.load(path)

    kept: list[Trajectory] = []
    reasons: Counter = Counter()
    for trajectory in dataset.trajectories:
        ok, reason = _is_valid(trajectory)
        if ok:
            kept.append(trajectory)
        else:
            reasons[reason] += 1

    report = CleanupReport(path, len(kept), len(dataset.trajectories) - len(kept), reasons)
    if write and not report.clean:
        dataset.trajectories = kept
        if kept:
            dataset.save(path)
        else:
            warnings.warn(
                f"{path.name}: every trajectory failed validation, file left untouched",
                RuntimeWarning,
                stacklevel=2,
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="dataset .npz files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be dropped without rewriting anything",
    )
    args = parser.parse_args(argv)

    dirty = 0
    for path in args.paths:
        if not path.is_file():
            print(f"{path}: not found", flush=True)
            continue
        report = clean_dataset(path, write=not args.dry_run)
        dirty += report.dropped
        print(report, flush=True)
    if dirty and args.dry_run:
        print(f"{dirty} trajectories would be dropped; re-run without --dry-run", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
