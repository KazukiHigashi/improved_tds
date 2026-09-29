from __future__ import annotations

import numpy as np
import pytest

from improved_tds.experiments.evaluate_synergy_sweep_coverage import interval_coverage
from improved_tds.data.trajectory import (
    Trajectory,
    TrajectoryDataset,
    TrajectoryMetadata,
    TrajectoryStep,
)
from improved_tds.synergy.emergence import (
    analyse,
    extract_manipulation_samples,
    principal_angles,
)
from improved_tds.synergy.two_axis import (
    balanced_cell_indices,
    clipped_posture_from_normalised,
    coordinate_from_normalised,
    fit_orthogonal_grasp_axis,
    fit_two_axis_synergy,
)


def test_interval_coverage_is_clipped_to_the_target_range() -> None:
    assert interval_coverage(np.array([-0.2, 0.25, 0.8]), 0.0, 1.0) == pytest.approx(0.8)
    assert interval_coverage(np.array([0.3, 0.6]), 0.0, 1.0) == pytest.approx(0.3)
    assert interval_coverage(np.array([1.1, 1.2]), 0.0, 1.0) == 0.0


def _episode(
    *,
    pose: int,
    command: float,
    n_actuated: int = 4,
    hold: int = 6,
    grasp_digits: int = 2,
    success: bool = True,
    drop_grasp_at: int | None = None,
) -> Trajectory:
    """An episode that reaches ``command`` and then holds there.

    The held posture is a fixed pose offset plus the command along one joint
    direction, so the analysis has a synergy to find and a known answer.
    """
    direction = np.array([1.0, 0.5, 0.0, -0.25])[:n_actuated]
    offset = np.full(n_actuated, 0.1 * pose)
    steps: list[TrajectoryStep] = []
    for index in range(hold + 2):
        settled = index >= 2
        q = offset + (command * direction if settled else np.zeros(n_actuated))
        flags = np.zeros(10, dtype=bool)
        held = 1 if (drop_grasp_at is not None and index == drop_grasp_at) else grasp_digits
        flags[:held] = True
        flags[5 + 1] = True  # the index digit works the driven part
        forces = np.zeros(10)
        forces[:held] = 3.0 + command
        forces[5 + 1] = 1.0 + command
        steps.append(
            TrajectoryStep(
                timestamp=0.02 * index,
                step_index=index,
                q=q,
                qd=np.zeros(n_actuated),
                action=np.zeros(n_actuated),
                tool_state=np.array([command if settled else 0.0]),
                contact_flags=flags,
                contact_forces=forces,
                is_success=settled,
                is_stable_success=settled,
                terminated=index == hold + 1,
            )
        )
    metadata = TrajectoryMetadata(
        tool_family="spray",
        tool_instance_id="test",
        task_name="test",
        success=success,
        seed=0,
        simulator_parameters={"initial_pose_index": pose},
    )
    return Trajectory(metadata, steps)


def _dataset(**kwargs) -> TrajectoryDataset:
    dataset = TrajectoryDataset()
    dataset.trajectories = [
        _episode(pose=pose, command=command, **kwargs)
        for pose in range(3)
        for command in (0.1, 0.2, 0.3, 0.4)
    ]
    return dataset


def test_held_posture_is_averaged_and_pose_offset_removed() -> None:
    samples = extract_manipulation_samples(_dataset(), [0, 1, 2, 3])
    assert len(samples) == 12
    # each pose contributes four commands whose centred tool states sum to zero
    for pose in range(3):
        rows = samples.pose_index == pose
        assert np.isclose(samples.tool_state[rows].sum(), 0.0)
        assert np.allclose(samples.q[rows].sum(axis=0), 0.0, atol=1e-9)
    # the pose offset is gone: the same command from different poses coincides
    first = samples.q[samples.pose_index == 0]
    second = samples.q[samples.pose_index == 1]
    assert np.allclose(first, second, atol=1e-9)


def test_episodes_that_let_go_while_holding_are_dropped() -> None:
    with pytest.raises(ValueError, match="grasp lost while holding x12"):
        extract_manipulation_samples(_dataset(drop_grasp_at=4), [0, 1, 2, 3])


def test_unsuccessful_episodes_do_not_reach_the_analysis() -> None:
    with pytest.raises(ValueError, match="not successful x12"):
        extract_manipulation_samples(_dataset(success=False), [0, 1, 2, 3])


def test_a_dropped_episode_does_not_take_the_good_ones_with_it() -> None:
    dataset = _dataset()
    dataset.trajectories.append(_episode(pose=0, command=0.25, drop_grasp_at=4))
    samples = extract_manipulation_samples(dataset, [0, 1, 2, 3])
    assert len(samples) == 12
    assert samples.rejected_episodes == {"grasp lost while holding": 1}


def test_one_synergy_is_found_when_the_data_holds_exactly_one() -> None:
    samples = extract_manipulation_samples(_dataset(), [0, 1, 2, 3])
    result = analyse(
        samples,
        actuator_names=("a", "b", "c", "d"),
        digit_names=("thumb", "index", "middle", "ring", "little"),
    )
    assert result.k == 1
    assert result.criterion_met
    assert result.tool_r2_by_k[0] == pytest.approx(1.0, abs=1e-9)
    assert abs(result.operation_tool_corr) == pytest.approx(1.0, abs=1e-6)
    # the operation axis recovers the direction the postures were built along
    expected = np.array([1.0, 0.5, 0.0, -0.25])
    expected = expected / np.linalg.norm(expected)
    assert abs(float(result.operation_axis @ expected)) == pytest.approx(1.0, abs=1e-6)
    # with k = 1 there is no room for a grasp axis, and that is reported as such
    assert result.grasp_axes.shape == (0, 4)
    assert not result.grasp_axis_supported


def test_digit_roles_come_from_contact_not_from_the_fit() -> None:
    samples = extract_manipulation_samples(_dataset(), [0, 1, 2, 3])
    roles = analyse(
        samples, digit_names=("thumb", "index", "middle", "ring", "little")
    ).digit_roles
    assert roles["holding_digits"] == ["index", "thumb"]
    assert roles["driving_digits"] == ["index"]
    assert roles["shared_digits"] == ["index"]
    assert not roles["roles_are_separate"]


def test_k_is_none_when_the_criterion_is_never_met() -> None:
    samples = extract_manipulation_samples(_dataset(), [0, 1, 2, 3])
    # a threshold no fit can reach, so the criterion must be reported unmet
    result = analyse(samples, r2_threshold=1.0)
    if result.tool_r2_by_k.max() < 1.0:
        assert result.k is None
        assert not result.criterion_met


def test_identical_subspaces_have_zero_principal_angle() -> None:
    basis = np.eye(4)[:2]
    assert np.allclose(principal_angles(basis, basis), 0.0, atol=1e-6)
    rotated = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]]) / np.sqrt(2)
    assert np.allclose(principal_angles(basis, rotated), 0.0, atol=1e-6)
    assert np.max(principal_angles(basis, np.eye(4)[2:])) == pytest.approx(90.0, abs=1e-6)


def test_pls_operation_and_residual_grasp_axes_recover_known_directions() -> None:
    rng = np.random.default_rng(4)
    operation = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    grasp = np.array([0.0, 0.0, 1.0, -1.0]) / np.sqrt(2.0)
    poses = np.repeat(np.arange(4), 20)
    tool = rng.normal(size=len(poses))
    grip = 3.0 * rng.normal(size=len(poses))
    offsets = rng.normal(scale=2.0, size=(4, 4))[poses]
    q = offsets + np.outer(tool, operation) + np.outer(grip, grasp)
    result = fit_two_axis_synergy(
        q, tool, poses, method="pls", grasp_metric=grip
    )
    assert float(result.operation_axis @ operation) > 0.99
    assert float(result.grasp_axis @ grasp) > 0.99
    assert abs(float(result.operation_axis @ result.grasp_axis)) < 1e-9
    assert result.grasp_metric_correlation > 0.99


def test_pca_uses_first_two_components_as_operation_and_grasp() -> None:
    operation = np.array([1.0, 0.0, 0.0])
    grasp = np.array([0.0, 1.0, 0.0])
    poses = np.repeat(np.arange(3), 10)
    score = np.tile(np.linspace(-2.0, 2.0, 10), 3)
    grip = np.tile(np.array([-0.5, 0.5] * 5), 3)
    q = np.outer(score, operation) + np.outer(grip, grasp)
    result = fit_two_axis_synergy(q, score, poses, method="pca", grasp_metric=grip)
    assert float(result.operation_axis @ operation) > 0.99
    assert float(result.grasp_axis @ grasp) > 0.99


def test_global_centering_preserves_between_initial_pose_success_variation() -> None:
    poses=np.repeat(np.arange(4),10)
    score=np.tile(np.linspace(-1.,1.,10),4)
    q=np.column_stack((score,2.*poses+.1*score,0.1*np.arange(40)))
    global_fit=fit_two_axis_synergy(q,score,poses,method='pca',centering='global')
    local_fit=fit_two_axis_synergy(q,score,poses,method='pca',centering='within_pose')
    assert abs(global_fit.operation_axis[1])>abs(global_fit.operation_axis[0])
    assert abs(local_fit.operation_axis[0])>abs(local_fit.operation_axis[1])


def test_balanced_cell_indices_never_fills_missing_cells_by_duplication() -> None:
    poses = np.array([0, 0, 0, 0, 1, 1])
    targets = np.array([0, 0, 1, 1, 0, 1])
    valid = np.array([True, True, True, False, True, False])
    selected = balanced_cell_indices(poses, targets, valid, per_cell=1)
    np.testing.assert_array_equal(selected, np.array([0, 2, 4]))


def test_force_pls_grasp_axis_is_supervised_and_orthogonal() -> None:
    rng = np.random.default_rng(91)
    poses = np.repeat(np.arange(5), 30)
    operation = np.array([1.0, 0.0, 0.0, 0.0])
    grasp = np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2.0)
    tool = rng.normal(size=poses.size)
    force = rng.normal(size=poses.size)
    nuisance = 3.0 * rng.normal(size=poses.size)
    q = (
        np.outer(tool, operation)
        + np.outer(force, grasp)
        + np.outer(nuisance, np.array([0.0, 0.0, 0.0, 1.0]))
    )
    axis = fit_orthogonal_grasp_axis(
        q, poses, operation, force, method="force_pls"
    )
    assert float(axis @ grasp) > 0.95
    assert abs(float(axis @ operation)) < 1e-9


def test_normalised_coordinate_reaches_axis_limits() -> None:
    reference = np.array([0.0, 0.5])
    axis = np.array([1.0, 0.0])
    low = np.array([-2.0, -1.0])
    high = np.array([1.0, 1.0])
    assert coordinate_from_normalised(reference, axis, low, high, -1.0) == -2.0
    assert coordinate_from_normalised(reference, axis, low, high, 1.0) == 1.0
    assert coordinate_from_normalised(reference, axis, low, high, -0.5) == -1.0


def test_clipped_synergy_does_not_collapse_at_opposing_joint_limits() -> None:
    reference = np.array([1.0, -1.0])
    axis = np.array([1.0, 1.0])
    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])
    negative, negative_coordinate = clipped_posture_from_normalised(
        reference, axis, low, high, -1.0
    )
    positive, positive_coordinate = clipped_posture_from_normalised(
        reference, axis, low, high, 1.0
    )
    assert negative_coordinate < 0.0 < positive_coordinate
    assert np.linalg.norm(positive - negative) > 0.0
    assert np.all(negative >= low) and np.all(negative <= high)
    assert np.all(positive >= low) and np.all(positive <= high)
