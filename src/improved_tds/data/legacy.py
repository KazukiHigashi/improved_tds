from __future__ import annotations

from pathlib import Path

import numpy as np

from improved_tds.data.trajectory import (
    Trajectory,
    TrajectoryDataset,
    TrajectoryMetadata,
    TrajectoryStep,
)


class LegacySynergyConverter:
    """Strict converter for legacy object arrays `[successful_postures, tool_angles]`."""

    @staticmethod
    def load_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        raw = np.load(Path(path), allow_pickle=True)
        if raw.dtype != object or raw.ndim not in {1, 2} or raw.shape[0] != 2:
            raise ValueError("legacy dataset must have two object-array rows: postures and tool states")
        q = np.asarray(list(raw[0]), dtype=np.float64)
        c = np.asarray(list(raw[1]), dtype=np.float64).reshape(-1)
        if q.ndim != 2:
            raise ValueError(f"legacy posture array must be two-dimensional, got {q.shape}")
        if q.shape[0] != c.shape[0] or q.shape[0] == 0:
            raise ValueError("legacy posture and tool-state lengths do not match")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(c)):
            raise ValueError("legacy dataset contains NaN or infinity")
        return q, c

    @classmethod
    def convert(
        cls,
        path: str | Path,
        *,
        tool_family: str = "scissors",
        tool_instance_id: str = "legacy",
        seed: int = 0,
    ) -> TrajectoryDataset:
        q, c = cls.load_arrays(path)
        trajectories = []
        for index, (posture, tool_state) in enumerate(zip(q, c)):
            metadata = TrajectoryMetadata(
                tool_family=tool_family,
                tool_instance_id=f"{tool_instance_id}:{index}",
                task_name="legacy_success_posture",
                success=True,
                seed=seed,
                simulator_parameters={"source": str(path)},
                controller_name="legacy_unknown",
            )
            step = TrajectoryStep(
                timestamp=float(index),
                q=posture,
                qd=np.zeros_like(posture),
                action=np.zeros_like(posture),
                tool_state=np.asarray([tool_state]),
                phase="legacy_terminal",
                reward=0.0,
                terminated=True,
            )
            trajectories.append(Trajectory(metadata, [step]))
        return TrajectoryDataset.from_iterable(trajectories)

