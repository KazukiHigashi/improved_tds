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


def load_samples(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    if source.suffix == ".npy":
        return LegacySynergyConverter.load_arrays(source)
    return TrajectoryDataset.load(source).samples()


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

