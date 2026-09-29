"""Collect an equal-size pose-target posture database from one best policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from improved_tds.environments.pose_pool import (
    InitialPosePoolWrapper,
    PushGoalWrapper,
    ScissorsGoalWrapper,
    SprayGoalWrapper,
    pool_filename_for_environment,
)
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.learning.algorithms import algorithm_class
from improved_tds.environments.scissors_research import ScissorsPostureWrapper
from improved_tds.environments.scissors_control import ScissorsControlWrapper
from improved_tds.environments.scissors_retention import ScissorsRetentionWrapper
from improved_tds.environments.scissors_retention_v5 import ScissorsRetentionV5Wrapper
from improved_tds.environments.scissors_retention_v6 import ScissorsRetentionV6Wrapper
from improved_tds.environments.scissors_retention_v7 import ScissorsRetentionV7Wrapper
from improved_tds.environments.scissors_reach import ScissorsReachWrapper
from improved_tds.environments.scissors_smooth import ScissorsSmoothReachWrapper, SmoothReachConfig
from improved_tds.environments.action_slew import ToolActionSlewWrapper
from improved_tds.environments.scissors_research import PostureCriteria
from improved_tds.environments.control_refinement import ControlRefinement


DEFAULT_TARGET_FRACTIONS = tuple(np.linspace(0.0, 1.0, 21).tolist())


def _fractions(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.replace(",", " ").split())
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("target fractions must lie in [0, 1]")
    return values


def _build_env(
    env_id: str,
    pool_dir: Path,
    seed: int,
    *,
    max_episode_steps: int | None = None,
    object_position_noise: float | None = None,
    render_mode: str | None = None,
    research_task: bool = False,
    scissors_control_version: str = 'v2',
    scissors_smooth_config: SmoothReachConfig | None = None,
    tool_action_slew_rate: float = 0.0,
    tool_request_gap_penalty_weight: float = 1.0,
    force_floor_contact: bool = False,
):
    if scissors_control_version not in ('simple','smooth','v2', 'v3', 'v4', 'v5', 'v6', 'v7'):
        raise ValueError('unknown Scissors control version')
    if scissors_control_version in ('simple','smooth','v3','v4','v5','v6','v7') and not research_task:
        raise ValueError('versioned Scissors control requires research_task=True')
    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = int(max_episode_steps)
    if object_position_noise is not None:
        if env_id.startswith("ShadowHandLiteSpray"):
            env_kwargs["bottle_xy_noise"] = float(object_position_noise)
        elif env_id.startswith("ShadowHandLitePush"):
            env_kwargs["object_xy_noise"] = float(object_position_noise)
        elif env_id.startswith("Scissors"):
            env_kwargs["object_position_noise"] = float(object_position_noise)
    if env_id.startswith("Scissors"):
        env_kwargs.update({"hold_initial_steps": 0, "reward_type": "dense"})
    if force_floor_contact:
        if not env_id.startswith("ShadowHandLiteSpray"):
            raise ValueError('force_floor_contact is only supported for Spray')
        env_kwargs.update(floor_contact_disable_fraction=None,
                          floor_contact_disable_curriculum=None)
    base = gym.make(env_id, **env_kwargs)
    if max_episode_steps is not None and hasattr(base.unwrapped, 'max_episode_steps'):
        # Gymnasium's TimeLimit override does not change a task's own internal
        # 200-step truncation. Both limits must cover the 5 s ramp + 2 s hold.
        base.unwrapped.max_episode_steps = int(max_episode_steps)
    pool_path = pool_dir / pool_filename_for_environment(base)
    if not pool_path.is_file():
        base.close()
        raise FileNotFoundError(f"initial-pose pool not found: {pool_path}")
    pool = InitialPosePoolWrapper(base, pool_path, seed=seed)
    task_name = type(pool.unwrapped).__name__
    common = {"success_hold_steps": 5}
    if task_name == "ShadowHandSprayBottleEnv":
        env = SprayGoalWrapper(pool, required_grasp_contacts=2, seed=seed, **common)
        option_key = "target_trigger_angle"
        low, high = 0.0, float(env.unwrapped.trigger_limit)
    elif task_name == "PushBottleEnv":
        env = PushGoalWrapper(pool, required_grasp_contacts=2, seed=seed, **common)
        option_key = "target_push_depth"
        low, high = 0.0, float(env.unwrapped.push_limit)
    elif task_name == "ScissorsManipulateEnv":
        wrapper_class = {'v2':ScissorsPostureWrapper,'v3':ScissorsControlWrapper,
                         'v4':ScissorsRetentionWrapper,'v5':ScissorsRetentionV5Wrapper,
                         'v6':ScissorsRetentionV6Wrapper,'v7':ScissorsRetentionV7Wrapper,
                         'simple':ScissorsReachWrapper,
                         'smooth':ScissorsSmoothReachWrapper}[scissors_control_version]
        extra = {'criteria':PostureCriteria(hold_seconds=2.),'retention_training_steps':0} if scissors_control_version in ('v4','v5','v6','v7') else {}
        if scissors_control_version == 'smooth' and scissors_smooth_config is not None:
            extra['smooth_config'] = scissors_smooth_config
        env = wrapper_class(pool, seed=seed, **extra) if research_task else ScissorsGoalWrapper(pool, **common)
        option_key = "target_angle"
        low, high = (float(value) for value in env.unwrapped.target_angle_range)
    else:
        pool.close()
        raise TypeError(f"unsupported task environment: {task_name}")
    if task_name in ('ShadowHandSprayBottleEnv', 'PushBottleEnv') and tool_action_slew_rate:
        env = ToolActionSlewWrapper(env, rate_per_second=tool_action_slew_rate,
            request_gap_penalty_weight=tool_request_gap_penalty_weight)
    if research_task:
        env = ControlRefinement(env)
    return env, pool, pool_path, option_key, low, high


def _actuated_q(env) -> np.ndarray:
    unwrapped = env.unwrapped
    indices = []
    for actuator_id in np.asarray(unwrapped._actuator_ids, dtype=int):
        joint_id = int(unwrapped.model.actuator_trnid[actuator_id, 0])
        indices.append(int(unwrapped.model.jnt_qposadr[joint_id]))
    return np.asarray(unwrapped.data.qpos[indices], dtype=np.float64).copy()


def _grasp_metric(info: dict) -> float:
    if "grasp_contact_fraction" in info:
        return float(info["grasp_contact_fraction"])
    if "grasp_contact_count" in info:
        return float(info["grasp_contact_count"])
    return float(info.get("is_in_grasp_space", info.get("grasp_maintained", 0.0)))


def _grasp_force(info: dict) -> float:
    return float(info.get("grasp_normal_force_total", _grasp_metric(info)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default="sac")
    parser.add_argument("--target-fractions", type=_fractions, default=DEFAULT_TARGET_FRACTIONS)
    parser.add_argument("--episodes-per-cell", type=int, default=1)
    parser.add_argument("--expected-pose-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--object-position-noise", type=float, default=0.0)
    parser.add_argument(
        "--strict-success-window",
        action="store_true",
        help="average only the first fixed stable-success window and stop the rollout",
    )
    parser.add_argument("--success-window-steps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument('--research-task', action='store_true')
    parser.add_argument('--scissors-control-version', choices=('simple','smooth','v2','v3','v4','v5','v6','v7'), default='v2')
    parser.add_argument('--representative', choices=('mean','last'), default='mean')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.episodes_per_cell < 1
        or args.expected_pose_count < 1
        or args.success_window_steps < 1
    ):
        raise SystemExit("episode, pose, and success-window counts must be positive")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output} (pass --overwrite)")
    _register_tool_manip_gym_environments()
    env, pool, pool_path, option_key, target_low, target_high = _build_env(
        args.env_id,
        args.pose_pool_dir,
        args.seed,
        research_task=args.research_task,
        scissors_control_version=args.scissors_control_version,
        object_position_noise=args.object_position_noise,
    )
    if pool.pose_count != args.expected_pose_count:
        env.close()
        raise SystemExit(
            f"{args.env_id} has {pool.pose_count} poses; expected {args.expected_pose_count}"
        )
    model_path = args.model if args.model.suffix == ".zip" else args.model.with_suffix(".zip")
    if not model_path.is_file():
        env.close()
        raise SystemExit(f"model not found: {model_path}")
    model = algorithm_class(args.algorithm).load(model_path, env=env, device="cpu")
    with model_path.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    rows: dict[str, list] = {
        key: []
        for key in (
            "q",
            "command_q",
            "initial_q",
            "tool_state",
            "target",
            "grasp_metric",
            "grasp_force",
            "pose_index",
            "target_index",
            "target_fraction",
            "success",
            "hold_steps",
        )
    }
    episode = 0
    try:
        for pose in range(pool.pose_count):
            for target_index, fraction in enumerate(args.target_fractions):
                target = target_low + fraction * (target_high - target_low)
                for _ in range(args.episodes_per_cell):
                    observation, info = env.reset(
                        seed=args.seed + episode,
                        options={"initial_pose_index": pose, option_key: target},
                    )
                    initial_q = _actuated_q(env)
                    unwrapped = env.unwrapped
                    control_ranges = np.asarray(
                        unwrapped.model.actuator_ctrlrange[unwrapped._actuator_ids],
                        dtype=np.float64,
                    )
                    halfspan = (control_ranges[:, 1] - control_ranges[:, 0]) / 2.0
                    neutral = np.asarray(unwrapped._neutral_ctrl, dtype=np.float64).copy()
                    q_steps: list[np.ndarray] = []
                    command_steps: list[np.ndarray] = []
                    tool_steps: list[float] = []
                    grasp_steps: list[float] = []
                    force_steps: list[float] = []
                    task_success_steps: list[bool] = []
                    stable_start: int | None = None
                    done = False
                    while not done:
                        action, _ = model.predict(observation, deterministic=True)
                        observation, _, terminated, truncated, info = env.step(action)
                        q_steps.append(_actuated_q(env))
                        # Record the command actually applied, including v3's
                        # rate limiter, rather than reconstructing the request.
                        command_steps.append(np.asarray(
                            unwrapped.data.ctrl[unwrapped._actuator_ids],dtype=np.float64).copy())
                        tool_steps.append(
                            float(np.asarray(observation["achieved_goal"]).reshape(-1)[0])
                        )
                        grasp_steps.append(_grasp_metric(info))
                        force_steps.append(_grasp_force(info))
                        task_success_steps.append(bool(info.get("is_success", False)))
                        if stable_start is None and bool(info.get("is_stable_success", False)):
                            stable_start = len(q_steps) - 1
                        if (
                            args.strict_success_window
                            and stable_start is not None
                            and len(q_steps) >= args.success_window_steps
                            and all(task_success_steps[-args.success_window_steps :])
                            and all(
                                value > 0.0
                                for value in grasp_steps[-args.success_window_steps :]
                            )
                        ):
                            break
                        done = bool(terminated or truncated)
                    if args.strict_success_window and stable_start is not None:
                        start = max(0, len(q_steps) - args.success_window_steps)
                        valid_hold = (
                            len(q_steps) - start == args.success_window_steps
                            and all(task_success_steps[start:])
                            and all(value > 0.0 for value in grasp_steps[start:])
                        )
                    else:
                        start = (
                            stable_start
                            if stable_start is not None
                            else max(0, len(q_steps) - 5)
                        )
                        valid_hold = stable_start is not None and all(
                            value > 0.0 for value in grasp_steps[start:]
                        )
                    representative_start = len(q_steps)-1 if args.representative == 'last' else start
                    rows["q"].append(np.mean(q_steps[representative_start:], axis=0))
                    rows["command_q"].append(
                        np.mean(command_steps[representative_start:], axis=0)
                    )
                    rows["initial_q"].append(initial_q)
                    rows["tool_state"].append(float(np.mean(tool_steps[representative_start:])))
                    rows["target"].append(float(target))
                    rows["grasp_metric"].append(float(np.mean(grasp_steps[representative_start:])))
                    rows["grasp_force"].append(float(np.mean(force_steps[representative_start:])))
                    rows["pose_index"].append(pose)
                    rows["target_index"].append(target_index)
                    rows["target_fraction"].append(float(fraction))
                    rows["success"].append(bool(valid_hold))
                    rows["hold_steps"].append(len(q_steps) - start)
                    episode += 1
            print(f"pose {pose + 1}/{pool.pose_count} collected", flush=True)
    finally:
        env.close()

    expected_rows = pool.pose_count * len(args.target_fractions) * args.episodes_per_cell
    if len(rows["q"]) != expected_rows:
        raise RuntimeError(f"collected {len(rows['q'])} rows; expected {expected_rows}")
    metadata = {
        "task_version": "posture_database_v2" if args.research_task else "legacy",
        "scissors_control_version": args.scissors_control_version,
        "representative": args.representative,
        "grasp_signal_definition": "normal force if provided; otherwise grasp contact fraction/count (not force in newtons)",
        "target_range": [target_low,target_high],
        "schema_version": 1,
        "env_id": args.env_id,
        "model": str(model_path.resolve()),
        "model_sha256": model_hash,
        "pose_pool": str(pool_path.resolve()),
        "pose_count": pool.pose_count,
        "target_fractions": list(args.target_fractions),
        "episodes_per_cell": args.episodes_per_cell,
        "rows": expected_rows,
        "successful_rows": int(np.sum(rows["success"])),
        "successful_rows_by_target": {
            str(float(fraction)): int(
                np.sum(
                    np.asarray(rows["success"], dtype=bool)
                    & np.isclose(
                        np.asarray(rows["target_fraction"], dtype=float),
                        float(fraction),
                    )
                )
            )
            for fraction in args.target_fractions
        },
        "seed": args.seed,
        "object_position_noise": args.object_position_noise,
        "strict_success_window": bool(args.strict_success_window),
        "success_window_steps": int(args.success_window_steps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{key: np.asarray(value) for key, value in rows.items()},
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"wrote {expected_rows} rows ({metadata['successful_rows']} successful) "
        f"to {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
