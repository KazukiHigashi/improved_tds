"""Watch TDS-based tool control, one representative example per task.

Shows the same tool being worked twice: first by the trained policy, then by
commanding a single synergy coordinate. Seeing them back to back is the point --
the policy uses all 13 actuators freely, the synergy command uses one number.

    python -m improved_tds.experiments.visualize_tds_control \
        --task spray --variant 1 \
        --model runs/.../spray_lite_1/checkpoints/sac_her_900000_steps.zip \
        --pose-pool-dir ../tool-pose-pipeline/grasps_v3

The synergy is identified on the spot from a short set of policy rollouts, so
this needs nothing but a trained model and the pose pool. Every task is driven
through the same absolute posture command,

    action = (posture - neutral) / halfspan

which is what all three environments implement, and the tool state is read from
``achieved_goal``, which all three expose.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from improved_tds.environments.pose_pool import (
    InitialPosePoolWrapper,
    PushGoalWrapper,
    SprayGoalWrapper,
    pool_filename_for_environment,
)
from improved_tds.experiments.analyze_synergy import actuated_qpos_indices
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from improved_tds.learning.algorithms import algorithm_class

TASKS = {
    # env id template, extra kwargs, and the goal wrapper the task trains with.
    # Scissors is already goal conditioned and needs no wrapper.
    "spray": ("ShadowHandLiteSpray{v}-v0", {"bottle_xy_noise": 0.0}, SprayGoalWrapper),
    "push": ("ShadowHandLitePush{v}-v0", {"object_xy_noise": 0.0}, PushGoalWrapper),
    "scissors": ("ScissorsVariant{v}-v0", {"hold_initial_steps": 0}, None),
}


def step_duration(env) -> float:
    """Seconds of simulated time one env step advances."""
    unwrapped = env.unwrapped
    substeps = getattr(unwrapped, "frame_skip", None) or getattr(unwrapped, "n_substeps", 1)
    return float(unwrapped.model.opt.timestep) * int(substeps)


def seconds_to_steps(env, seconds: float) -> int:
    return max(1, int(round(seconds / step_duration(env))))


def build_env(
    task: str,
    variant: int,
    pose_pool_dir: Path,
    *,
    render: bool,
    seed: int,
    horizon_steps: int | None = None,
):
    template, kwargs, wrapper = TASKS[task]
    env_id = template.format(v=variant)
    options = dict(kwargs)
    if render:
        options["render_mode"] = "human"
    if horizon_steps is not None:
        options["max_episode_steps"] = horizon_steps
    base = gym.make(env_id, **options)
    if horizon_steps is not None and getattr(base.unwrapped, "max_episode_steps", None):
        # gym.make consumes max_episode_steps for its TimeLimit; the tool-manip-gym
        # envs truncate on a counter of their own, which has to be raised too or a
        # five-second phase is cut off at the trained horizon
        base.unwrapped.max_episode_steps = horizon_steps
    pool = InitialPosePoolWrapper(
        base, pose_pool_dir / pool_filename_for_environment(base), seed=seed
    )
    return (wrapper(pool, seed=seed) if wrapper is not None else pool), env_id


def pool_wrapper(env) -> InitialPosePoolWrapper:
    """Find the pose pool in the wrapper stack.

    The tasks stack differently -- scissors has no goal wrapper -- so the pool
    cannot be reached by a fixed number of `.env` hops.
    """
    current = env
    while current is not None:
        if isinstance(current, InitialPosePoolWrapper):
            return current
        current = getattr(current, "env", None)
    raise TypeError("no initial-pose pool in the wrapper stack")


def command_frame(env) -> tuple[np.ndarray, np.ndarray]:
    """The centre and half-range every task's action is expressed against."""
    unwrapped = env.unwrapped
    if getattr(unwrapped, "relative_action", False) or getattr(
        unwrapped, "relative_control", False
    ):
        raise RuntimeError("this script commands absolute postures; the env is relative")
    ctrlrange = unwrapped.model.actuator_ctrlrange[unwrapped._actuator_ids]
    halfspan = 0.5 * (ctrlrange[:, 1] - ctrlrange[:, 0])
    return np.asarray(unwrapped._neutral_ctrl, dtype=np.float64).copy(), halfspan


def tool_state(observation) -> float:
    return float(np.asarray(observation["achieved_goal"]).reshape(-1)[0])


def succeeded(info: dict[str, Any]) -> bool:
    """Scissors reports only `is_success`; the goal wrappers add a held version."""
    value = info.get("is_stable_success", info.get("is_success", 0.0))
    return bool(float(np.asarray(value).reshape(-1)[0]))


def held_posture(env, policy, pose: int, indices, *, max_steps: int) -> tuple[np.ndarray, float] | None:
    """Run one policy episode and average the posture over the settled window."""
    observation, _ = env.reset(options={"initial_pose_index": int(pose)})
    postures, tools, success = [], [], []
    for _ in range(max_steps):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        postures.append(env.unwrapped.data.qpos[indices].copy())
        tools.append(tool_state(observation))
        success.append(succeeded(info))
        if terminated or truncated:
            break
    if not postures:
        return None
    flags = np.asarray(success)
    window = np.flatnonzero(flags)
    # settled on the target if it ever succeeded; otherwise fall back to the tail,
    # which is still the part of the episode where the hand has stopped moving
    rows = slice(int(window[0]), len(postures)) if window.size else slice(
        int(0.75 * len(postures)), len(postures)
    )
    return np.asarray(postures)[rows].mean(axis=0), float(np.mean(np.asarray(tools)[rows]))


def identify_synergy(env, policy, poses, indices, *, episodes: int, max_steps: int):
    """Fit the one direction that moves the tool, from short policy rollouts.

    With a single response variable the first PLS direction is closed form: the
    covariance between each joint and the tool state, normalised. Postures are
    centred within each initial pose first, because the authored poses differ
    from one another by far more than the tool command moves the hand.
    """
    rows: dict[int, list[tuple[np.ndarray, float]]] = {}
    for pose in poses:
        for _ in range(episodes):
            sample = held_posture(env, policy, pose, indices, max_steps=max_steps)
            if sample is not None:
                rows.setdefault(pose, []).append(sample)
    if not rows:
        raise RuntimeError("no episode produced a settled posture")
    # centring within a pose annihilates a pose that contributed one episode, so
    # a single episode per pose leaves nothing to fit
    if max(len(entries) for entries in rows.values()) < 2:
        raise RuntimeError(
            "identification needs at least 2 episodes per pose "
            "(--identify-episodes); with one, centring leaves no variation"
        )

    postures, tools, labels = [], [], []
    for pose, entries in rows.items():
        for posture, tool in entries:
            postures.append(posture)
            tools.append(tool)
            labels.append(pose)
    postures = np.asarray(postures)
    tools = np.asarray(tools)
    labels = np.asarray(labels)

    centred_q = postures.copy()
    centred_tool = tools.copy()
    for pose in np.unique(labels):
        mask = labels == pose
        centred_q[mask] -= centred_q[mask].mean(axis=0)
        centred_tool[mask] -= centred_tool[mask].mean()

    axis = centred_q.T @ centred_tool
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise RuntimeError("the tool state does not co-vary with any joint")
    axis = axis / norm
    if float(np.corrcoef(centred_q @ axis, centred_tool)[0, 1]) < 0.0:
        axis = -axis  # a larger score must mean a larger tool state
    references = {
        pose: np.asarray([posture for posture, _ in entries]).mean(axis=0)
        for pose, entries in rows.items()
    }
    scores = centred_q @ axis
    return axis, references, float(scores.min()), float(scores.max()), postures, tools


def play_policy(env, policy, pose: int, *, max_steps: int, pause: float) -> None:
    observation, _ = env.reset(options={"initial_pose_index": int(pose)})
    print(f"  policy: target {float(np.asarray(observation['desired_goal']).reshape(-1)[0]):+.3f}")
    for _ in range(max_steps):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        time.sleep(pause)
        if terminated or truncated:
            break
    print(f"  policy: tool ended at {tool_state(observation):+.3f}")


def play_synergy(
    env,
    pose: int,
    reference: np.ndarray,
    axis: np.ndarray,
    span: tuple[float, float],
    *,
    grasp_steps: int,
    ramp_steps: int,
    hold_steps: int,
    pause: float,
) -> None:
    """Close onto the tool, then sweep the single synergy coordinate.

    Each commanded posture is applied over several seconds of simulated time, as
    the TDS paper does (its Eq. 9 applies every posture for 5 s), so the hand and
    the tool settle before the tool state is read.
    """
    observation, _ = env.reset(options={"initial_pose_index": int(pose)})
    start, halfspan = command_frame(env)

    def send(posture: np.ndarray):
        action = np.clip((posture - start) / halfspan, -1.0, 1.0).astype(np.float32)
        return env.step(action)

    for step in range(grasp_steps):
        observation, *_ = send(start + (reference - start) * (step + 1) / grasp_steps)
        time.sleep(pause)
    print(f"  synergy: engaged, tool at {tool_state(observation):+.3f}")

    low, high = span
    current = 0.0
    for label, target in (("open", high), ("close", low), ("mid", 0.5 * (low + high))):
        for step in range(ramp_steps):
            value = current + (target - current) * (step + 1) / ramp_steps
            observation, *_ = send(reference + axis * value)
            time.sleep(pause)
        for _ in range(hold_steps):
            observation, *_ = send(reference + axis * target)
            time.sleep(pause)
        current = target
        print(f"  synergy: rho {target:+.3f} ({label}) -> tool {tool_state(observation):+.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--variant", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pose-pool-dir", required=True, type=Path)
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default="sac")
    parser.add_argument(
        "--pose", type=int, default=None, help="initial pose to show (default: the first identified)"
    )
    parser.add_argument("--identify-poses", type=int, default=4)
    parser.add_argument("--identify-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--grasp-seconds", type=float, default=5.0)
    parser.add_argument(
        "--ramp-seconds", type=float, default=2.0, help="time to travel to a commanded rho"
    )
    parser.add_argument(
        "--hold-seconds", type=float, default=3.0, help="time held there before reading the tool"
    )
    parser.add_argument("--rho-scale", type=float, default=2.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback rate; 1.0 shows the motion at the speed it is simulated",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("both", "policy", "synergy"), default="both")
    parser.add_argument("--repeat", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _register_tool_manip_gym_environments()

    model_path = args.model
    if not model_path.is_file() and model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    if not model_path.is_file():
        raise SystemExit(f"model not found: {args.model}")

    # identification runs headless: opening the viewer for it would take minutes
    fit_env, env_id = build_env(
        args.task, args.variant, args.pose_pool_dir, render=False, seed=args.seed
    )
    indices, names = actuated_qpos_indices(env_id)
    indices = np.asarray(indices, dtype=int)
    policy = algorithm_class(args.algorithm).load(model_path, env=fit_env, device="cpu")

    pose_count = pool_wrapper(fit_env).pose_count
    if args.mode == "policy":
        # nothing to identify: watching the policy needs no synergy
        fit_env.close()
        view_env, _ = build_env(
            args.task, args.variant, args.pose_pool_dir, render=True, seed=args.seed
        )
        policy.set_env(view_env)
        pause = max(0.0, step_duration(view_env) / args.speed) if args.speed > 0 else 0.0
        try:
            for pose in ([args.pose] if args.pose is not None else list(range(0, pose_count, max(1, pose_count // args.identify_poses)))[: args.identify_poses]):
                print(f"--- pose {pose} ---", flush=True)
                play_policy(view_env, policy, pose, max_steps=args.max_steps, pause=pause)
        finally:
            view_env.close()
        return 0
    step = max(1, pose_count // args.identify_poses)
    poses = list(range(0, pose_count, step))[: args.identify_poses]
    print(f"=== {args.task} variant {args.variant} ({env_id}) ===", flush=True)
    print(f"identifying the synergy from poses {poses} ...", flush=True)
    try:
        axis, references, low, high, postures, tools = identify_synergy(
            fit_env,
            policy,
            poses,
            indices,
            episodes=args.identify_episodes,
            max_steps=args.max_steps,
        )
    finally:
        fit_env.close()

    order = np.argsort(-np.abs(axis))[:4]
    joints = ", ".join(f"{names[i].replace('rh_A_', '')}{axis[i]:+.2f}" for i in order)
    print(f"  tool state seen: {tools.min():+.3f} .. {tools.max():+.3f}")
    print(f"  synergy: {joints}")
    print(f"  rho range: {low:+.3f} .. {high:+.3f} (x{args.rho_scale:g})", flush=True)

    pose = args.pose if args.pose is not None else poses[0]
    if pose not in references:
        raise SystemExit(f"pose {pose} was not identified; choose one of {sorted(references)}")

    # the viewer run has to outlast three five-second postures plus the grasp,
    # which is well beyond the horizon the policy was trained on
    probe, _ = build_env(args.task, args.variant, args.pose_pool_dir, render=False, seed=args.seed)
    dt = step_duration(probe)
    grasp_steps = seconds_to_steps(probe, args.grasp_seconds)
    ramp_steps = seconds_to_steps(probe, args.ramp_seconds)
    hold_steps = seconds_to_steps(probe, args.hold_seconds)
    probe.close()
    horizon = grasp_steps + 3 * (ramp_steps + hold_steps) + 10
    pause = max(0.0, dt / args.speed) if args.speed > 0 else 0.0
    print(
        f"  timing: dt {dt:.3f}s -> grasp {args.grasp_seconds:g}s ({grasp_steps} steps), "
        f"move {args.ramp_seconds:g}s, hold {args.hold_seconds:g}s",
        flush=True,
    )

    view_env, _ = build_env(
        args.task,
        args.variant,
        args.pose_pool_dir,
        render=True,
        seed=args.seed,
        horizon_steps=horizon,
    )
    policy.set_env(view_env)
    try:
        for round_index in range(args.repeat):
            if args.repeat > 1:
                print(f"--- round {round_index + 1}/{args.repeat} ---", flush=True)
            if args.mode in ("both", "policy"):
                play_policy(view_env, policy, pose, max_steps=args.max_steps, pause=pause)
            if args.mode in ("both", "synergy"):
                play_synergy(
                    view_env,
                    pose,
                    references[pose],
                    axis,
                    (low * args.rho_scale, high * args.rho_scale),
                    grasp_steps=grasp_steps,
                    ramp_steps=ramp_steps,
                    hold_steps=hold_steps,
                    pause=pause,
                )
    finally:
        view_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
