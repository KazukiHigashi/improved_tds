"""Run the tool-DoF synergy analysis over a set of collected rollout datasets.

    python -m improved_tds.experiments.analyze_synergy \
        --rollout-dir analysis/synergy_rollouts_v5 \
        --pose-pool-dir ../tool-pose-pipeline/grasps_v3 \
        --output analysis/synergy_rollouts_v5/analysis.json

The analysis itself lives in :mod:`improved_tds.synergy.emergence`, which is
free of MuJoCo; this module supplies the one thing that needs the model -- which
entries of the recorded ``qpos`` the actuators drive -- and reports the result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from improved_tds.data.trajectory import TrajectoryDataset
from improved_tds.evaluation.reports import write_report
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.synergy.emergence import (
    analyse,
    extract_manipulation_samples,
    pls_subspace,
    principal_angles,
    subspace_by_pose,
)


def actuated_qpos_indices(env_id: str) -> tuple[list[int], list[str]]:
    """Where each actuator's joint sits in ``qpos``, and the actuator names.

    Recorded ``q`` is the whole ``qpos`` -- hand joints, the bottle's free joint
    and the tool DoF together. Only the actuated hand joints belong in the
    analysis, and the coupled distal joints must not be counted twice: they are
    copies of the joint driving them, not independent freedoms, so taking one
    index per actuator is what gives the hand's true 13.
    """
    env = gym.make(env_id)
    try:
        unwrapped = env.unwrapped
        model = unwrapped.model
        names = getattr(unwrapped, "actuator_names", None) or unwrapped._actuator_names
        indices = []
        for actuator_id in np.asarray(unwrapped._actuator_ids, dtype=int):
            if int(model.actuator_trntype[actuator_id]) != mujoco.mjtTrn.mjTRN_JOINT:
                raise TypeError(f"actuator {actuator_id} in {env_id} is not joint-driven")
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            indices.append(int(model.jnt_qposadr[joint_id]))
        return indices, [str(name) for name in names]
    finally:
        env.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _top_loadings(axis: np.ndarray, names: list[str], count: int = 5) -> list[dict[str, float]]:
    order = np.argsort(-np.abs(axis))[:count]
    return [{"actuator": names[i], "loading": float(axis[i])} for i in order]


def analyse_variant(
    dataset_path: Path,
    *,
    r2_threshold: float,
    required_grasp_contacts: int,
    common_k: int = 2,
) -> dict[str, Any]:
    dataset = TrajectoryDataset.load(dataset_path)
    parameters = dataset.trajectories[0].metadata.simulator_parameters
    env_id = str(parameters.get("env_id", ""))
    if not env_id:
        raise ValueError(f"{dataset_path.name} does not record which environment it came from")
    indices, actuator_names = actuated_qpos_indices(env_id)
    digit_names = parameters.get("contact_digits") or ()

    samples = extract_manipulation_samples(
        dataset, indices, required_grasp_contacts=required_grasp_contacts
    )
    result = analyse(
        samples,
        r2_threshold=r2_threshold,
        actuator_names=actuator_names,
        digit_names=digit_names,
    )
    analysed_k = result.components.shape[0]

    # A synergy tied to the tool should be the same subspace whatever pose the
    # hand started from. Comparing per-pose fits against the pooled one turns
    # that into a number rather than an assumption.
    per_pose = subspace_by_pose(samples, analysed_k)
    pose_angles = {
        pose: float(np.max(principal_angles(result.components, subspace)))
        for pose, subspace in per_pose.items()
    }
    # PCA maximises variance, which is not the same thing as moving the tool;
    # the same question is therefore asked again of a subspace fitted to the
    # tool state, at a comparable dimension across variants.
    pls_bases = {k: pls_subspace(samples, k) for k in (common_k, 1)}
    pls_pose_angles = {
        pose: float(np.max(principal_angles(pls_bases[common_k], subspace)))
        for pose, subspace in subspace_by_pose(samples, common_k, method="pls").items()
    }

    payload: dict[str, Any] = {
        "dataset": dataset_path.name,
        "env_id": env_id,
        "policy": str(parameters.get("policy_sha256", ""))[:12],
        "episodes_analysed": result.n_episodes,
        "episodes_rejected": samples.rejected_episodes,
        "hold_steps_median": float(np.median(samples.hold_steps)),
        "achieved_tool_state_range": [
            float(samples.raw_tool_state.min()),
            float(samples.raw_tool_state.max()),
        ],
        "grasp_force_median_n": float(np.median(samples.raw_grasp_force)),
        "n_actuated": result.n_actuated,
        "explained_variance_ratio": result.explained_variance_ratio,
        "cumulative_variance_ratio": result.cumulative_variance_ratio,
        "tool_r2_by_k": result.tool_r2_by_k,
        "pls_r2_by_k": result.pls_r2_by_k,
        "k": result.k,
        "k_criterion_r2": result.k_criterion,
        "k_criterion_met": result.criterion_met,
        "analysed_k": analysed_k,
        "variance_explained_by_k": float(result.cumulative_variance_ratio[analysed_k - 1]),
        "tool_r2_at_analysed_k": float(result.tool_r2_by_k[analysed_k - 1]),
        "operation_axis": result.operation_axis,
        "operation_axis_top_loadings": _top_loadings(result.operation_axis, actuator_names),
        "grasp_axes": result.grasp_axes,
        "grasp_axes_top_loadings": [
            _top_loadings(axis, actuator_names) for axis in result.grasp_axes
        ],
        "operation_tool_corr": result.operation_tool_corr,
        "operation_grasp_force_corr": result.operation_grasp_force_corr,
        "grasp_axis_tool_corr": result.grasp_tool_corr,
        "grasp_axis_force_corr": result.grasp_force_corr,
        "grasp_axis_supported": result.grasp_axis_supported,
        "digit_roles": result.digit_roles,
        "pose_subspace_max_angle_deg": pose_angles,
        "pose_subspace_angle_median_deg": (
            float(np.median(list(pose_angles.values()))) if pose_angles else float("nan")
        ),
        "common_k": common_k,
        "pls_pose_subspace_angle_median_deg": (
            float(np.median(list(pls_pose_angles.values()))) if pls_pose_angles else float("nan")
        ),
        "actuator_names": list(actuator_names),
    }
    payload["_components"] = result.components
    payload["_pls_common"] = pls_bases[common_k]
    payload["_pls_first"] = pls_bases[1]
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--r2-threshold", type=float, default=0.9)
    parser.add_argument("--required-grasp-contacts", type=int, default=2)
    parser.add_argument(
        "--common-k",
        type=int,
        default=2,
        help="dimension at which subspaces are compared across variants and poses; "
        "the per-variant k differs, and subspaces of different size are not comparable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _register_tool_manip_gym_environments()

    datasets = sorted(
        args.rollout_dir.glob(args.pattern),
        key=lambda path: [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)
        ],
    )
    if not datasets:
        raise SystemExit(f"no datasets matched {args.pattern} in {args.rollout_dir}")

    variants: dict[str, Any] = {}
    components: dict[str, np.ndarray] = {}
    pls_common: dict[str, np.ndarray] = {}
    pls_first: dict[str, np.ndarray] = {}
    for path in datasets:
        print(f"=== {path.name} ===", flush=True)
        payload = analyse_variant(
            path,
            r2_threshold=args.r2_threshold,
            required_grasp_contacts=args.required_grasp_contacts,
            common_k=args.common_k,
        )
        components[path.stem] = payload.pop("_components")
        pls_common[path.stem] = payload.pop("_pls_common")
        pls_first[path.stem] = payload.pop("_pls_first")
        variants[path.stem] = payload
        k_text = (
            f"k = {payload['k']}"
            if payload["k_criterion_met"]
            else f"k undefined (R2 never reached {payload['k_criterion_r2']}); "
            f"analysed the full {payload['analysed_k']}"
        )
        print(
            f"  {payload['episodes_analysed']} episodes\n"
            f"  {k_text} (tool R2 {payload['tool_r2_at_analysed_k']:.3f}, "
            f"variance {100 * payload['variance_explained_by_k']:.1f}%)\n"
            f"  tool R2 at k=1 {payload['tool_r2_by_k'][0]:.3f}, "
            f"k=2 {payload['tool_r2_by_k'][1]:.3f}, k=3 {payload['tool_r2_by_k'][2]:.3f}\n"
            f"  holding {payload['digit_roles']['holding_digits']} / "
            f"driving {payload['digit_roles']['driving_digits']} "
            f"(separate: {payload['digit_roles']['roles_are_separate']})\n"
            f"  operation axis: tool r={payload['operation_tool_corr']:+.3f}, "
            f"force r={payload['operation_grasp_force_corr']:+.3f}\n"
            f"  grasp axes:     tool r={np.round(payload['grasp_axis_tool_corr'], 3)}, "
            f"force r={np.round(payload['grasp_axis_force_corr'], 3)}\n"
            f"  grasp axis supported: {payload['grasp_axis_supported']}",
            flush=True,
        )

    # Do the variants share a synergy subspace? Each variant's own k differs, and
    # subspaces of different size are not comparable, so the comparison is made
    # at a common dimension -- and separately for the tool-fitted (PLS) subspace,
    # since that is where a tool-tied synergy would live if anywhere.
    def _cross(bases: dict[str, np.ndarray]) -> dict[str, float]:
        angles: dict[str, float] = {}
        names = list(bases)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if bases[left].shape != bases[right].shape:
                    continue
                angles[f"{left} vs {right}"] = float(
                    np.max(principal_angles(bases[left], bases[right]))
                )
        return angles

    cross_own_k = _cross(components)
    cross_pls = _cross(pls_common)
    cross_pls_first = _cross(pls_first)

    report = {
        "rollout_dir": str(args.rollout_dir),
        "k_criterion_r2": args.r2_threshold,
        "required_grasp_contacts": args.required_grasp_contacts,
        "common_k": args.common_k,
        "variants": variants,
        "cross_variant_max_principal_angle_deg": cross_own_k,
        "cross_variant_pls_angle_deg": cross_pls,
        "cross_variant_pls_first_axis_angle_deg": cross_pls_first,
    }
    write_report(args.output, _jsonable(report))
    print(f"\nwrote {args.output}", flush=True)
    for label, angles in (
        (f"PCA subspaces at each variant's own k", cross_own_k),
        (f"PLS subspaces at k={args.common_k}", cross_pls),
        ("PLS first axis", cross_pls_first),
    ):
        if not angles:
            continue
        values = np.array(list(angles.values()))
        print(
            f"{label}: median {np.median(values):.1f} deg, "
            f"range {values.min():.1f}-{values.max():.1f} ({len(values)} pairs)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
