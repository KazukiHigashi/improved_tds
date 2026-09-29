"""Compare commanded and measured hand motion at synergy-sweep endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from improved_tds.experiments.collect_posture_database import _actuated_q, _build_env
from improved_tds.experiments.evaluate_strict_synergy_control import (
    _posture_on_axis,
    _reference_for_pose,
)
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds, _execute
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    sweep = json.loads(args.report.read_text(encoding="utf-8"))
    strict = json.loads(Path(sweep["source_report"]).read_text(encoding="utf-8"))
    with np.load(strict["database"]) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    _register_tool_manip_gym_environments()
    env, pool, _, option_key, target_low, target_high = _build_env(
        sweep["env_id"], Path(metadata["pose_pool"]).parent,
        int(sweep["split_seed"]), object_position_noise=0.0,
    )
    low, high, _ = _control_bounds(env)
    pose = int(sweep["test_poses"][0])
    nominal_target = target_low + 0.3 * (target_high - target_low)
    output = {"env_id": sweep["env_id"], "pose": pose, "configurations": {}}
    try:
        for name in sweep["configurations"]:
            source = strict["configurations"][name]
            operation_axis = np.asarray(source["operation_axis"], dtype=np.float64)
            grasp_axis = (
                np.asarray(source["grasp_axis"], dtype=np.float64)
                if source["grasp_axis"] is not None else None
            )
            postures, measured, tools, coordinates = [], [], [], []
            for level in (-1.0, 1.0):
                seed = int(sweep["split_seed"]) + 1000 * pose
                reference = _reference_for_pose(
                    env, pose, option_key, nominal_target,
                    np.asarray(source["reference_delta"]), low, high, seed,
                )
                grasp = reference
                if grasp_axis is not None:
                    grasp, _ = _posture_on_axis(
                        reference, grasp_axis, float(source["normalised_grasp"]), low, high
                    )
                operation, coordinate = _posture_on_axis(
                    grasp, operation_axis, level, low, high
                )
                outcome = _execute(
                    env, pose, option_key, nominal_target, grasp, operation,
                    seed=seed, grasp_steps=100, ramp_steps=50, hold_steps=50,
                )
                postures.append(operation)
                measured.append(_actuated_q(env))
                tools.append(outcome["achieved"])
                coordinates.append(coordinate)
            at_bound = int(
                np.count_nonzero(
                    np.isclose(reference, low, atol=1e-8)
                    | np.isclose(reference, high, atol=1e-8)
                )
            )
            output["configurations"][name] = {
                "coordinate_endpoints": coordinates,
                "commanded_posture_l2_difference": float(np.linalg.norm(postures[1] - postures[0])),
                "measured_posture_l2_difference": float(np.linalg.norm(measured[1] - measured[0])),
                "tool_difference": float(tools[1] - tools[0]),
                "reference_joints_at_bound": at_bound,
                "joint_count": int(reference.size),
            }
    finally:
        env.close()
        pool.close()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
