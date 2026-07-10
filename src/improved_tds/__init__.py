"""Closed-loop Tool DoF Synergy (TDS) research toolkit."""

from improved_tds.data.trajectory import (
    SCHEMA_VERSION,
    Trajectory,
    TrajectoryDataset,
    TrajectoryMetadata,
    TrajectoryStep,
)

__all__ = [
    "SCHEMA_VERSION",
    "Trajectory",
    "TrajectoryDataset",
    "TrajectoryMetadata",
    "TrajectoryStep",
    "register_environments",
]

__version__ = "0.1.0"


def register_environments() -> None:
    """Register MuJoCo environments without making simulation a core dependency."""

    from improved_tds.environments.registration import register_environments as _register

    _register()


try:
    register_environments()
except ImportError:
    # Numerical estimation/control remains importable without Gymnasium or MuJoCo.
    pass

