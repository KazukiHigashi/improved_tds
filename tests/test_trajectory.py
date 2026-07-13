from __future__ import annotations

import numpy as np
import pytest

from improved_tds.data.legacy import LegacySynergyConverter
from improved_tds.data.trajectory import (
    Trajectory,
    TrajectoryDataset,
    TrajectoryMetadata,
    TrajectoryStep,
)
from improved_tds.experiments.common import load_samples


def make_dataset() -> TrajectoryDataset:
    metadata = TrajectoryMetadata(
        tool_family="scissors",
        tool_instance_id="s1",
        task_name="tracking",
        success=True,
        seed=3,
        simulator_parameters={"friction": 0.1},
        controller_name="test",
        stable_success_steps=2,
    )
    steps = [
        TrajectoryStep(
            timestamp=0.01 * index,
            q=np.array([index, index + 1.0]),
            qd=np.array([0.1, 0.2]),
            action=np.array([0.0, 0.1]),
            tool_state=np.array([0.2 * index]),
            desired_tool_state=np.array([0.4]),
            tool_state_rate=np.array([0.2]),
            joint_torque=np.array([0.3, 0.4]),
            contact_flags=np.array([index > 0]),
            phase="open",
            reward=float(index),
            terminated=index == 2,
            step_index=index,
            is_success=index >= 1,
            is_stable_success=index == 2,
            success_streak=max(index, 0),
        )
        for index in range(3)
    ]
    return TrajectoryDataset([Trajectory(metadata, steps)])


def test_trajectory_npz_round_trip(tmp_path) -> None:
    dataset = make_dataset()
    path = tmp_path / "trajectory.npz"
    dataset.save(path)
    loaded = TrajectoryDataset.load(path)
    q, state = loaded.samples()
    assert q.shape == (3, 2)
    assert state.shape == (3, 1)
    assert loaded.trajectories[0].metadata.tool_instance_id == "s1"
    np.testing.assert_allclose(q, np.array([[0, 1], [1, 2], [2, 3]]))
    success_q, _, desired = loaded.goal_conditioned_samples(success_steps_only=True)
    assert success_q.shape == (2, 2)
    np.testing.assert_allclose(desired, 0.4)
    stable_q, stable_state = loaded.stable_success_samples()
    np.testing.assert_allclose(stable_q, np.array([[1, 2], [2, 3]]))
    np.testing.assert_allclose(stable_state[:, 0], np.array([0.2, 0.4]))
    assert not list(tmp_path.glob(".*.tmp.npz"))


def test_schema_10_loads_without_goal_but_goal_analysis_fails(tmp_path) -> None:
    source = tmp_path / "source.npz"
    legacy = tmp_path / "legacy.npz"
    make_dataset().save(source)
    with np.load(source, allow_pickle=False) as data:
        omitted = {
            "desired_tool_state",
            "step_index",
            "is_success",
            "is_stable_success",
            "success_streak",
        }
        payload = {
            name: np.asarray(data[name])
            for name in data.files
            if name not in omitted
        }
    payload["schema_version"] = np.asarray("1.0")
    np.savez_compressed(legacy, **payload)
    loaded = TrajectoryDataset.load(legacy)
    assert loaded.schema_version == "1.0"
    assert loaded.trajectories[0].steps[0].desired_tool_state is None
    with pytest.raises(ValueError, match="desired_tool_state"):
        loaded.goal_conditioned_samples()


def test_trajectory_rejects_shape_and_nan() -> None:
    dataset = make_dataset()
    dataset.trajectories[0].steps[1].q = np.ones(3)
    with pytest.raises(ValueError, match="shape"):
        dataset.validate()
    with pytest.raises(ValueError, match="NaN"):
        TrajectoryStep(
            timestamp=0.0,
            q=np.array([np.nan]),
            qd=np.zeros(1),
            action=np.zeros(1),
            tool_state=np.zeros(1),
        )


def test_stable_success_samples_reject_inconsistent_streak() -> None:
    dataset = make_dataset()
    dataset.trajectories[0].steps[-1].success_streak = 1
    with pytest.raises(ValueError, match="success streak"):
        dataset.stable_success_samples()


def test_load_samples_enforces_formal_split_role(tmp_path) -> None:
    path = tmp_path / "training_candidate.npz"
    make_dataset().save(path)
    with pytest.raises(ValueError, match="expected dataset role"):
        load_samples(
            path,
            stable_success_only=True,
            expected_dataset_role="formal_balanced_fit",
        )


def test_legacy_converter_is_strict_and_preserves_samples(tmp_path) -> None:
    q = np.arange(12, dtype=np.float64).reshape(4, 3)
    state = np.linspace(0.0, 1.0, 4)
    legacy = np.empty(2, dtype=object)
    legacy[0] = list(q)
    legacy[1] = list(state)
    path = tmp_path / "synergy_dataset.npy"
    np.save(path, legacy, allow_pickle=True)
    converted = LegacySynergyConverter.convert(path)
    converted_q, converted_state = converted.samples()
    np.testing.assert_array_equal(converted_q, q)
    np.testing.assert_array_equal(converted_state[:, 0], state)
