"""Does a synergy matching the tool DoF emerge in a learned policy?

The analysis this implements is specified in ``docs/synergy_analysis_plan.md``.

What is measured is the *quasi-static* relation: the hand posture that holds the
tool at a commanded state. Instantaneous samples do not carry it -- the trigger
rings after it is pressed, swinging ~0.12 rad even in episodes commanded to
0.04, so step-by-step the joint angles and the tool angle are near uncorrelated.
Averaging over the settled hold window gives one posture per episode, and the
episodes span the commanded tool states.

Postures are centred within each authored initial pose, not globally: the
authored poses differ from one another by far more than any command changes the
hand, so an uncentred fit would return the pose catalogue as its leading
component.

The subspace a PCA returns is not tied to any particular basis -- rotating
within it leaves the fit unchanged -- so "the first component is the tool DoF"
is not a claim the data can support. What is well defined is the subspace, and
within it the single direction most aligned with tool motion. That direction is
the operation axis; its orthogonal complement inside the subspace holds the
grasp axes. Whether those complement directions really do hold the tool is then
an empirical question, answered against measured contact force rather than
assumed.

This module stays free of MuJoCo and Gymnasium: the caller supplies the mapping
from recorded ``qpos`` to actuated joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.linalg import subspace_angles

from improved_tds.data.trajectory import Trajectory, TrajectoryDataset

#: contact_flags/contact_forces are recorded as two blocks of one flag per digit
DIGITS_PER_CHANNEL = 5


@dataclass
class ManipulationSamples:
    """One held posture per episode, centred within each authored pose.

    Rows are episodes, not steps: each is the average over the window where the
    policy held the tool at the commanded state.
    """

    q: np.ndarray                    # (n_episodes, n_actuated) pose-centred
    tool_state: np.ndarray           # (n_episodes,) pose-centred
    grasp_force: np.ndarray          # (n_episodes,) pose-centred, N
    tool_force: np.ndarray           # (n_episodes,) pose-centred, N
    raw_q: np.ndarray                # (n_episodes, n_actuated) as held, rad
    raw_tool_state: np.ndarray       # (n_episodes,) as achieved, rad
    raw_grasp_force: np.ndarray      # (n_episodes,) as measured, N
    grasp_contacts: np.ndarray       # (n_episodes, 5) fraction of the window
    tool_contacts: np.ndarray        # (n_episodes, 5) fraction of the window
    pose_index: np.ndarray           # (n_episodes,) authored initial pose
    hold_steps: np.ndarray           # (n_episodes,) length of the averaged window
    rejected_episodes: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.q.shape[0])


@dataclass
class EmergenceResult:
    """What the held postures say about the tool-DoF synergy."""

    n_episodes: int
    n_actuated: int
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    tool_r2_by_k: np.ndarray          # R^2 of tool state from the first k scores
    k: int | None                     # None when the criterion is never met
    k_criterion: float
    components: np.ndarray            # (k, n_actuated) PCA basis of the subspace
    operation_axis: np.ndarray        # (n_actuated,) in joint space
    grasp_axes: np.ndarray            # (k-1, n_actuated) in joint space
    operation_tool_corr: float
    operation_grasp_force_corr: float
    grasp_tool_corr: np.ndarray       # (k-1,)
    grasp_force_corr: np.ndarray      # (k-1,)
    pls_r2_by_k: np.ndarray
    digit_roles: dict[str, Any]
    actuator_names: tuple[str, ...] = ()

    @property
    def criterion_met(self) -> bool:
        return self.k is not None

    @property
    def grasp_axis_supported(self) -> bool:
        """Do the complement directions behave like grasp axes?

        They must move the tool less than the operation axis does, and relate to
        contact force more strongly than the operation axis does. Failing this
        is a reportable result, not an error.
        """
        if self.grasp_axes.size == 0:
            return False
        return bool(
            np.max(np.abs(self.grasp_tool_corr)) < abs(self.operation_tool_corr)
            and np.max(np.abs(self.grasp_force_corr)) > abs(self.operation_grasp_force_corr)
        )


def _hold_window(trajectory: Trajectory) -> slice | None:
    """From the first stable success to the end of the episode."""
    for index, step in enumerate(trajectory.steps):
        if step.is_stable_success:
            if index >= len(trajectory.steps) - 1:
                return None
            return slice(index, len(trajectory.steps))
    return None


def extract_manipulation_samples(
    dataset: TrajectoryDataset,
    actuated_qpos_indices: Sequence[int],
    *,
    required_grasp_contacts: int = 2,
) -> ManipulationSamples:
    """One held posture per successful, grasp-holding episode."""
    indices = np.asarray(actuated_qpos_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("actuated_qpos_indices must be a non-empty 1-D index array")

    rows: dict[str, list[Any]] = {
        name: []
        for name in (
            "q",
            "tool",
            "grasp_force",
            "tool_force",
            "grasp_contacts",
            "tool_contacts",
            "pose",
            "hold_steps",
        )
    }
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for trajectory in dataset.trajectories:
        if not trajectory.metadata.success:
            reject("not successful")
            continue
        if any(step.contact_flags is None for step in trajectory.steps):
            reject("no contact record")
            continue
        window = _hold_window(trajectory)
        if window is None:
            reject("no settled hold window")
            continue
        steps = trajectory.steps[window]

        flags = np.array([step.contact_flags for step in steps], dtype=bool)
        forces = np.array([step.contact_forces for step in steps], dtype=np.float64)
        if int(flags[:, :DIGITS_PER_CHANNEL].sum(axis=1).min()) < required_grasp_contacts:
            # the tool was let go while nominally being held at the target
            reject("grasp lost while holding")
            continue

        rows["q"].append(np.array([step.q[indices] for step in steps]).mean(axis=0))
        rows["tool"].append(
            float(np.mean([float(step.tool_state.reshape(-1)[0]) for step in steps]))
        )
        rows["grasp_force"].append(float(forces[:, :DIGITS_PER_CHANNEL].sum(axis=1).mean()))
        rows["tool_force"].append(float(forces[:, DIGITS_PER_CHANNEL:].sum(axis=1).mean()))
        rows["grasp_contacts"].append(flags[:, :DIGITS_PER_CHANNEL].mean(axis=0))
        rows["tool_contacts"].append(flags[:, DIGITS_PER_CHANNEL:].mean(axis=0))
        rows["pose"].append(
            int(trajectory.metadata.simulator_parameters.get("initial_pose_index", -1))
        )
        rows["hold_steps"].append(len(steps))

    if not rows["q"]:
        # why nothing survived is the useful part of this failure
        detail = ", ".join(f"{reason} x{count}" for reason, count in rejected.items())
        raise ValueError(f"no episode contributed a held posture ({detail or 'empty dataset'})")

    q = np.asarray(rows["q"], dtype=np.float64)
    tool = np.asarray(rows["tool"], dtype=np.float64)
    grasp_force = np.asarray(rows["grasp_force"], dtype=np.float64)
    tool_force = np.asarray(rows["tool_force"], dtype=np.float64)
    pose = np.asarray(rows["pose"], dtype=int)

    q_centred, tool_centred, grasp_centred, tool_force_centred = (
        _centre_within_pose(values, pose) for values in (q, tool, grasp_force, tool_force)
    )
    return ManipulationSamples(
        q=q_centred,
        tool_state=tool_centred,
        grasp_force=grasp_centred,
        tool_force=tool_force_centred,
        raw_q=q,
        raw_tool_state=tool,
        raw_grasp_force=grasp_force,
        grasp_contacts=np.asarray(rows["grasp_contacts"], dtype=np.float64),
        tool_contacts=np.asarray(rows["tool_contacts"], dtype=np.float64),
        pose_index=pose,
        hold_steps=np.asarray(rows["hold_steps"], dtype=int),
        rejected_episodes=rejected,
    )


def _centre_within_pose(values: np.ndarray, pose_index: np.ndarray) -> np.ndarray:
    """Remove each authored pose's own offset, keeping variation within it."""
    centred = np.array(values, dtype=np.float64, copy=True)
    for pose in np.unique(pose_index):
        rows = pose_index == pose
        centred[rows] -= centred[rows].mean(axis=0)
    return centred


def _pca(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Component directions (rows) and explained-variance ratios."""
    _, singular_values, vh = np.linalg.svd(samples, full_matrices=False)
    variance = singular_values**2
    total = float(variance.sum())
    ratios = variance / total if total > 0.0 else np.zeros_like(variance)
    return vh, ratios


def _r2(target: np.ndarray, design: np.ndarray) -> float:
    """R^2 of a least-squares fit of ``target`` on ``design`` (both centred)."""
    total = float(np.sum(target**2))
    if total <= 0.0:
        return 0.0
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return float(1.0 - np.sum(residual**2) / total)


def _orthogonal_complement(axis: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the complement of ``axis`` in its own space."""
    dimension = axis.shape[0]
    if dimension < 2:
        return np.zeros((0, dimension))
    basis = np.eye(dimension)
    basis[:, 0] = axis
    q, _ = np.linalg.qr(basis)
    complement = q[:, 1:].T
    # QR pins the first column only up to sign; re-orthogonalise against the axis
    complement -= np.outer(complement @ axis, axis)
    norms = np.linalg.norm(complement, axis=1, keepdims=True)
    return complement / np.where(norms > 0.0, norms, 1.0)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _pls_r2_by_k(q: np.ndarray, tool_state: np.ndarray, max_k: int) -> np.ndarray:
    """R^2 from PLS, which targets the tool state instead of maximising variance."""
    try:
        from sklearn.cross_decomposition import PLSRegression
    except ImportError:  # pragma: no cover - sklearn is a declared dependency
        return np.full(max_k, np.nan)
    scores = np.empty(max_k)
    for k in range(1, max_k + 1):
        model = PLSRegression(n_components=k, scale=False).fit(q, tool_state)
        scores[k - 1] = _r2(tool_state, model.x_scores_)
    return scores


def _digit_roles(samples: ManipulationSamples, digit_names: Sequence[str]) -> dict[str, Any]:
    """Which digits hold the tool and which work it, straight from contact."""
    holding = samples.grasp_contacts.mean(axis=0)
    driving = samples.tool_contacts.mean(axis=0)
    names = list(digit_names) or [str(i) for i in range(samples.grasp_contacts.shape[1])]
    holding_set = {name for name, value in zip(names, holding) if value >= 0.5}
    driving_set = {name for name, value in zip(names, driving) if value >= 0.5}
    return {
        "holding_fraction": {name: float(value) for name, value in zip(names, holding)},
        "driving_fraction": {name: float(value) for name, value in zip(names, driving)},
        "holding_digits": sorted(holding_set),
        "driving_digits": sorted(driving_set),
        "shared_digits": sorted(holding_set & driving_set),
        "roles_are_separate": bool(driving_set and not (holding_set & driving_set)),
    }


def analyse(
    samples: ManipulationSamples,
    *,
    r2_threshold: float = 0.9,
    actuator_names: Sequence[str] = (),
    digit_names: Sequence[str] = (),
) -> EmergenceResult:
    """Fit the synergy subspace and split it into an operation and grasp axes."""
    if not 0.0 < r2_threshold <= 1.0:
        raise ValueError("r2_threshold must lie in (0, 1]")
    q = samples.q
    tool = samples.tool_state
    n_actuated = q.shape[1]

    directions, ratios = _pca(q)
    max_k = min(n_actuated, directions.shape[0])
    scores_full = q @ directions.T

    tool_r2 = np.array([_r2(tool, scores_full[:, : k + 1]) for k in range(max_k)])
    reached = np.flatnonzero(tool_r2 >= r2_threshold)
    # When the criterion is never met, the whole space is analysed and `k` is
    # reported as None: that the tool state is not explainable to the stated
    # standard is the result, and hiding it behind a fallback k would misreport it.
    k = int(reached[0]) + 1 if reached.size else max_k
    criterion_k = k if reached.size else None

    components = directions[:k]
    scores = scores_full[:, :k]

    weights, *_ = np.linalg.lstsq(scores, tool, rcond=None)
    norm = float(np.linalg.norm(weights))
    if norm == 0.0:
        raise ValueError("the tool state does not depend on the synergy scores")
    operation_direction = weights / norm
    complement = _orthogonal_complement(operation_direction)

    operation_scores = scores @ operation_direction
    grasp_scores = scores @ complement.T if complement.size else np.zeros((len(tool), 0))

    return EmergenceResult(
        n_episodes=len(samples),
        n_actuated=n_actuated,
        explained_variance_ratio=ratios,
        cumulative_variance_ratio=np.cumsum(ratios),
        tool_r2_by_k=tool_r2,
        k=criterion_k,
        k_criterion=float(r2_threshold),
        components=components,
        operation_axis=operation_direction @ components,
        grasp_axes=complement @ components if complement.size else np.zeros((0, n_actuated)),
        operation_tool_corr=_correlation(operation_scores, tool),
        operation_grasp_force_corr=_correlation(operation_scores, samples.grasp_force),
        grasp_tool_corr=np.array(
            [_correlation(grasp_scores[:, j], tool) for j in range(grasp_scores.shape[1])]
        ),
        grasp_force_corr=np.array(
            [
                _correlation(grasp_scores[:, j], samples.grasp_force)
                for j in range(grasp_scores.shape[1])
            ]
        ),
        pls_r2_by_k=_pls_r2_by_k(q, tool, max_k),
        digit_roles=_digit_roles(samples, digit_names),
        actuator_names=tuple(str(name) for name in actuator_names),
    )


def pls_subspace(samples: ManipulationSamples, k: int) -> np.ndarray:
    """A k-dimensional subspace fitted to explain the tool state, not variance.

    PCA answers "where does the hand move most", which need not be "where does
    the hand move the tool". Where the two disagree -- and they do, sharply --
    this is the subspace the tool-DoF claim should be tested in.
    """
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=k, scale=False).fit(samples.q, samples.tool_state)
    weights = np.asarray(model.x_weights_, dtype=np.float64).T
    basis, _ = np.linalg.qr(weights.T)
    return basis.T


def subspace_by_pose(
    samples: ManipulationSamples, k: int, *, method: str = "pca"
) -> dict[int, np.ndarray]:
    """Fit the k-dimensional subspace separately for each authored pose.

    A synergy tied to the tool should come out the same whatever pose the hand
    started from; a synergy that is really an artefact of pose diversity will not.
    """
    subspaces: dict[int, np.ndarray] = {}
    for pose in sorted(set(samples.pose_index.tolist())):
        rows = samples.pose_index == pose
        if int(rows.sum()) <= k + 1:
            continue
        if method == "pls":
            subset = ManipulationSamples(
                q=samples.q[rows],
                tool_state=samples.tool_state[rows],
                grasp_force=samples.grasp_force[rows],
                tool_force=samples.tool_force[rows],
                raw_q=samples.raw_q[rows],
                raw_tool_state=samples.raw_tool_state[rows],
                raw_grasp_force=samples.raw_grasp_force[rows],
                grasp_contacts=samples.grasp_contacts[rows],
                tool_contacts=samples.tool_contacts[rows],
                pose_index=samples.pose_index[rows],
                hold_steps=samples.hold_steps[rows],
            )
            subspaces[int(pose)] = pls_subspace(subset, k)
        else:
            directions, _ = _pca(samples.q[rows])
            subspaces[int(pose)] = directions[:k]
    return subspaces


def principal_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Angles in degrees between two subspaces given as row-vector bases."""
    return np.degrees(subspace_angles(np.asarray(left).T, np.asarray(right).T))
