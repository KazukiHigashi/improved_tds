from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "1.0"
_VECTOR_FIELDS = (
    "q",
    "qd",
    "action",
    "tool_state",
    "tool_state_rate",
    "joint_torque",
    "motor_current",
    "contact_flags",
    "contact_forces",
)
_OPTIONAL_FIELDS = {
    "tool_state_rate",
    "joint_torque",
    "motor_current",
    "contact_flags",
    "contact_forces",
}


def _vector(value: Any, name: str, *, dtype: np.dtype = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array.copy()


@dataclass
class TrajectoryStep:
    """One synchronized interaction sample.

    Joint positions are radians, tool angles are radians, translational tool states
    are metres, force is newtons, and torque is N m. The metadata/config records
    which physical unit applies to a particular tool state.
    """

    timestamp: float
    q: np.ndarray
    qd: np.ndarray
    action: np.ndarray
    tool_state: np.ndarray
    tool_state_rate: np.ndarray | None = None
    joint_torque: np.ndarray | None = None
    motor_current: np.ndarray | None = None
    contact_flags: np.ndarray | None = None
    contact_forces: np.ndarray | None = None
    phase: str = "unknown"
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        self.reward = float(self.reward)
        if not np.isfinite(self.timestamp) or not np.isfinite(self.reward):
            raise ValueError("timestamp and reward must be finite")
        self.q = _vector(self.q, "q")
        self.qd = _vector(self.qd, "qd")
        self.action = _vector(self.action, "action")
        self.tool_state = _vector(self.tool_state, "tool_state")
        for name in _OPTIONAL_FIELDS:
            value = getattr(self, name)
            if value is not None:
                dtype = np.bool_ if name == "contact_flags" else np.float64
                setattr(self, name, _vector(value, name, dtype=dtype))
        self.phase = str(self.phase)


@dataclass
class TrajectoryMetadata:
    tool_family: str
    tool_instance_id: str
    task_name: str
    success: bool
    seed: int
    simulator_parameters: dict[str, Any] = field(default_factory=dict)
    controller_name: str = "unknown"
    policy_checkpoint: str | None = None
    tool_state_unit: str = "rad"

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Reject configuration values that cannot be persisted deterministically.
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class Trajectory:
    metadata: TrajectoryMetadata
    steps: list[TrajectoryStep]

    def validate(self) -> None:
        if not self.steps:
            raise ValueError("trajectory contains no steps")
        timestamps = np.asarray([step.timestamp for step in self.steps], dtype=np.float64)
        if np.any(np.diff(timestamps) < 0.0):
            raise ValueError("timestamps must be monotone non-decreasing")
        expected: dict[str, tuple[int, ...]] = {}
        for index, step in enumerate(self.steps):
            for name in _VECTOR_FIELDS:
                value = getattr(step, name)
                if value is None:
                    continue
                shape = value.shape
                if name in expected and shape != expected[name]:
                    raise ValueError(
                        f"step {index} field {name} has shape {shape}; expected {expected[name]}"
                    )
                expected.setdefault(name, shape)
        if self.metadata.success and not any(step.terminated or step.truncated for step in self.steps):
            # Successful trajectories from online collection should include an episode boundary.
            # Legacy conversion is allowed to use its explicit legacy task marker.
            if self.metadata.task_name != "legacy_success_posture":
                raise ValueError("successful trajectory has no terminated/truncated boundary")


@dataclass
class TrajectoryDataset:
    trajectories: list[Trajectory] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def append(self, trajectory: Trajectory, *, successful_only: bool = True) -> bool:
        trajectory.validate()
        if successful_only and not trajectory.metadata.success:
            return False
        self.trajectories.append(trajectory)
        return True

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported trajectory schema {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if not self.trajectories:
            raise ValueError("dataset contains no trajectories")
        global_shapes: dict[str, tuple[int, ...]] = {}
        for episode, trajectory in enumerate(self.trajectories):
            trajectory.validate()
            for step in trajectory.steps:
                for name in _VECTOR_FIELDS:
                    value = getattr(step, name)
                    if value is None:
                        continue
                    if name in global_shapes and value.shape != global_shapes[name]:
                        raise ValueError(
                            f"trajectory {episode} field {name} shape {value.shape} "
                            f"does not match dataset shape {global_shapes[name]}"
                        )
                    global_shapes.setdefault(name, value.shape)

    @property
    def successful(self) -> list[Trajectory]:
        return [trajectory for trajectory in self.trajectories if trajectory.metadata.success]

    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Return synchronized `(q, tool_state)` arrays from successful trajectories."""

        selected = self.successful
        if not selected:
            raise ValueError("dataset contains no successful trajectories")
        q = np.vstack([step.q for trajectory in selected for step in trajectory.steps])
        tool_state = np.vstack(
            [step.tool_state for trajectory in selected for step in trajectory.steps]
        )
        return q.astype(np.float64), tool_state.astype(np.float64)

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lengths = np.asarray([len(item.steps) for item in self.trajectories], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        steps = [step for trajectory in self.trajectories for step in trajectory.steps]
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(self.schema_version),
            "episode_offsets": offsets,
            "metadata_json": np.asarray(
                [json.dumps(item.metadata.to_json_dict(), sort_keys=True) for item in self.trajectories]
            ),
            "timestamp": np.asarray([step.timestamp for step in steps], dtype=np.float64),
            "phase": np.asarray([step.phase for step in steps]),
            "reward": np.asarray([step.reward for step in steps], dtype=np.float64),
            "terminated": np.asarray([step.terminated for step in steps], dtype=np.bool_),
            "truncated": np.asarray([step.truncated for step in steps], dtype=np.bool_),
        }
        for name in _VECTOR_FIELDS:
            arrays[name] = self._stack_field(steps, name)
        np.savez_compressed(path, **arrays)

    @staticmethod
    def _stack_field(steps: list[TrajectoryStep], name: str) -> np.ndarray:
        values = [getattr(step, name) for step in steps]
        shape = next((value.shape for value in values if value is not None), None)
        if shape is None:
            return np.empty((len(steps), 0), dtype=np.float64)
        dtype = np.bool_ if name == "contact_flags" else np.float64
        fill = False if dtype == np.bool_ else np.nan
        output = np.full((len(steps), *shape), fill, dtype=dtype)
        for index, value in enumerate(values):
            if value is not None:
                output[index] = value
        return output

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryDataset":
        with np.load(Path(path), allow_pickle=False) as data:
            version = str(np.asarray(data["schema_version"]).item())
            if version != SCHEMA_VERSION:
                raise ValueError(f"unsupported trajectory schema {version!r}")
            offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
            if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0:
                raise ValueError("invalid episode_offsets")
            n_steps = int(offsets[-1])
            required = {
                "timestamp",
                "phase",
                "reward",
                "terminated",
                "truncated",
                *_VECTOR_FIELDS,
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"dataset is missing fields: {missing}")
            for name in required:
                if np.asarray(data[name]).shape[0] != n_steps:
                    raise ValueError(f"field {name} length does not match episode_offsets")
            metadata_raw = np.asarray(data["metadata_json"])
            if metadata_raw.shape != (offsets.size - 1,):
                raise ValueError("metadata count does not match trajectory count")
            trajectories: list[Trajectory] = []
            for episode, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
                metadata = TrajectoryMetadata(**json.loads(str(metadata_raw[episode])))
                steps = [cls._step_from_arrays(data, index) for index in range(int(start), int(stop))]
                trajectories.append(Trajectory(metadata=metadata, steps=steps))
        dataset = cls(trajectories=trajectories, schema_version=version)
        dataset.validate()
        return dataset

    @staticmethod
    def _step_from_arrays(data: Any, index: int) -> TrajectoryStep:
        values: dict[str, Any] = {}
        for name in _VECTOR_FIELDS:
            array = np.asarray(data[name])
            row = array[index]
            if name in _OPTIONAL_FIELDS and (row.size == 0 or np.any(~np.isfinite(row))):
                values[name] = None
            else:
                values[name] = row
        return TrajectoryStep(
            timestamp=float(data["timestamp"][index]),
            phase=str(data["phase"][index]),
            reward=float(data["reward"][index]),
            terminated=bool(data["terminated"][index]),
            truncated=bool(data["truncated"][index]),
            **values,
        )

    @classmethod
    def from_iterable(cls, trajectories: Iterable[Trajectory]) -> "TrajectoryDataset":
        dataset = cls(list(trajectories))
        dataset.validate()
        return dataset

