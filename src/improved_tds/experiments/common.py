from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from improved_tds.data.legacy import LegacySynergyConverter
from improved_tds.data.trajectory import TrajectoryDataset
from improved_tds.synergy.family import FamilyTDS
from improved_tds.synergy.pca import PCATDS
from improved_tds.synergy.random import RandomTDS
from improved_tds.synergy.supervised import SupervisedLinearTDS


def load_samples(
    path: str | Path,
    *,
    success_steps_only: bool = False,
    stable_success_only: bool = False,
    expected_dataset_role: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if success_steps_only and stable_success_only:
        raise ValueError("success_steps_only and stable_success_only are mutually exclusive")
    source = Path(path)
    if source.suffix == ".npy":
        if stable_success_only:
            raise ValueError("legacy .npy datasets do not contain stable-success markers")
        return LegacySynergyConverter.load_arrays(source)
    dataset = TrajectoryDataset.load(source)
    if expected_dataset_role is not None:
        roles = {trajectory.metadata.dataset_role for trajectory in dataset.trajectories}
        if roles != {expected_dataset_role}:
            raise ValueError(
                f"expected dataset role {expected_dataset_role!r}, found {sorted(roles)}"
            )
    if stable_success_only:
        return dataset.stable_success_samples()
    return dataset.samples(success_steps_only=success_steps_only)


def load_estimator(path: str | Path) -> Any:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        estimator_type = str(data["estimator_type"].item())
    loaders = {
        "pca_tds": PCATDS.load,
        "random_tds": RandomTDS.load,
        "supervised_linear_tds": SupervisedLinearTDS.load,
        "family_tds": FamilyTDS.load,
    }
    if estimator_type not in loaders:
        raise ValueError(f"unknown estimator type {estimator_type!r}")
    return loaders[estimator_type](source)
