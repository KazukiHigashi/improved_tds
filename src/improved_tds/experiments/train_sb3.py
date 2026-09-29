from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import numpy as np

from improved_tds.evaluation.reports import write_report
from improved_tds.learning.collector import SuccessfulTrajectoryCollector
from improved_tds.learning.algorithms import algorithm_class
from improved_tds.experiments.validation import ValidationEarlyStoppingCallback
from improved_tds.environments.scissors_research import ScissorsPostureWrapper, PostureCriteria
from improved_tds.environments.scissors_control import ScissorsControlWrapper
from improved_tds.environments.scissors_retention import ScissorsRetentionWrapper
from improved_tds.environments.scissors_retention_v5 import ScissorsRetentionV5Wrapper
from improved_tds.environments.scissors_retention_v6 import ScissorsRetentionV6Wrapper
from improved_tds.environments.scissors_retention_v7 import ScissorsRetentionV7Wrapper
from improved_tds.environments.scissors_reach import ScissorsReachWrapper
from improved_tds.environments.scissors_smooth import ScissorsSmoothReachWrapper, SmoothReachConfig
from improved_tds.environments.action_slew import ToolActionSlewWrapper
from improved_tds.learning.retention_policy import RetentionSACPolicy
from improved_tds.learning.safe_replay import NumericalSafeReplayBuffer
from improved_tds.learning.tool_features import ScissorsControlFeatures
from improved_tds.environments.control_refinement import ControlRefinement
from improved_tds.learning.tool_features import ToolFeaturesExtractor
from improved_tds.environments.pose_pool import (
    InitialPosePoolWrapper,
    PushGoalWrapper,
    ScissorsGoalWrapper,
    SprayGoalWrapper,
    pool_filename_for_environment,
)


def _register_tool_manip_gym_environments() -> None:
    """Import the sibling tool-manip-gym packages when running from this repository."""
    try:
        import gym_scissors  # noqa: F401
        import gym_slider  # noqa: F401
        import gym_spray  # noqa: F401
        import gym_push  # noqa: F401
        return
    except ModuleNotFoundError:
        workspace = Path(__file__).resolve().parents[4]
        package_roots = (
            workspace / "tool-manip-gym" / "gym-scissors",
            workspace / "tool-manip-gym" / "gym-slider",
            workspace / "tool-manip-gym" / "gym-spray",
            workspace / "tool-manip-gym" / "gym-push",
        )
        for root in reversed(package_roots):
            if root.is_dir():
                sys.path.insert(0, str(root))
    try:
        import gym_scissors  # noqa: F401
        import gym_slider  # noqa: F401
        import gym_spray  # noqa: F401
        import gym_push  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "tool-manip-gym is required for randomized grasp-pool training. "
            "Install its gym-scissors, gym-slider, gym-spray, and gym-push packages."
        ) from error


def _tool_manip_env_id(env_id: str) -> str:
    """Resolve legacy scissors aliases to unambiguous tool-manip-gym IDs."""
    match = re.fullmatch(r"(?:Exp|Improved)Scissor([1-5])-v0", env_id)
    if match:
        return f"ScissorsVariant{match.group(1)}-v0"
    return env_id


def _fractions(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.replace(",", " ").split())
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("fractions must lie in [0, 1]")
    return values


class TrainingDiagnosticsCallback(BaseCallback):
    def __init__(self, n_envs: int):
        super().__init__(verbose=0)
        self._episode_success = np.zeros(n_envs, dtype=bool)
        self.successful_trajectory_count = 0
        self.action_components = 0
        self.saturated_action_components = 0
        self.retention_probe_attempts = np.zeros(n_envs, dtype=np.int64)
        self.retention_probe_completions = np.zeros(n_envs, dtype=np.int64)
        self.retention_probe_successes = np.zeros(n_envs, dtype=np.int64)
        self.retention_stages = np.zeros(n_envs, dtype=np.int64)
        self.retention_observed = False

    @property
    def action_saturation_fraction(self) -> float:
        if self.action_components == 0:
            return 0.0
        return self.saturated_action_components / self.action_components

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", []), dtype=np.float64)
        if actions.size:
            self.action_components += int(actions.size)
            self.saturated_action_components += int(np.sum(np.abs(actions) > 0.95))
        infos = self.locals.get("infos", [])
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        for index, info in enumerate(infos):
            if 'retention_probe_attempts' in info:
                self.retention_observed = True
                self.retention_probe_attempts[index] = info['retention_probe_attempts']
                self.retention_probe_completions[index] = info['retention_probe_completions']
                self.retention_probe_successes[index] = info['retention_probe_successes']
                self.retention_stages[index] = info['retention_stage']
            self._episode_success[index] |= bool(info.get("is_stable_success", False))
            if index < len(dones) and dones[index]:
                self.successful_trajectory_count += int(self._episode_success[index])
                self._episode_success[index] = False
        self.logger.record(
            "rollout/stable_success_trajectories", self.successful_trajectory_count
        )
        self.logger.record(
            "rollout/action_saturation_fraction",
            self.action_saturation_fraction,
        )
        if self.retention_observed:
            self.logger.record('retention/probe_attempts', int(self.retention_probe_attempts.sum()))
            self.logger.record('retention/probe_completions', int(self.retention_probe_completions.sum()))
            self.logger.record('retention/probe_successes', int(self.retention_probe_successes.sum()))
            self.logger.record('retention/stage_min', int(self.retention_stages.min()))
            self.logger.record('retention/stage_max', int(self.retention_stages.max()))
        return True


class TargetRangeCurriculumCallback(BaseCallback):
    """Expand the sampled goal range in equal-duration stages."""

    def __init__(self, max_fractions: tuple[float, ...], total_timesteps: int):
        super().__init__(verbose=1)
        self.max_fractions = max_fractions
        self.boundaries = tuple(
            int(total_timesteps * index / len(max_fractions))
            for index in range(len(max_fractions))
        )
        self.stage = -1

    def _on_step(self) -> bool:
        stage = max(
            index
            for index, boundary in enumerate(self.boundaries)
            if self.num_timesteps >= boundary
        )
        if stage != self.stage:
            self.stage = stage
            maximum = self.max_fractions[stage]
            self.training_env.env_method("set_target_max_fraction", maximum)
            self.logger.record("curriculum/target_max_fraction", maximum)
            if self.verbose:
                print(
                    f"target curriculum stage {stage + 1}/{len(self.max_fractions)}: "
                    f"fraction [0, {maximum}]",
                    flush=True,
                )
        return True


def main(argv: list[str] | None = None) -> None:
    """SB3 SAC/TD3/DDPG+HER学習と同期した候補姿勢DB収集を行う。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="ScissorsVariant1-v0")
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default="sac")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=2)
    parser.add_argument("--output", default="runs/sac_her")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--validation-freq", type=int, default=100_000)
    parser.add_argument(
        "--validation-target-fractions",
        type=_fractions,
        default=tuple(np.linspace(0.0, 1.0, 11)),
    )
    parser.add_argument("--validation-success-threshold", type=float, default=0.95)
    parser.add_argument("--validation-consecutive", type=int, default=2)
    parser.add_argument("--best-policy-output", default=None)
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--her-ratio", type=float, default=0.25)
    parser.add_argument(
        "--her-sampled-goals",
        type=int,
        default=4,
        help="HER future goals per transition; zero disables HER",
    )
    parser.add_argument(
        "--sac-ent-coef",
        default="auto",
        help="SAC entropy coefficient: 'auto' or a fixed non-negative float",
    )
    parser.add_argument(
        "--pose-pool-dir",
        default=str(Path(__file__).resolve().parents[4] / "tool-pose-pipeline" / "grasps"),
        help="required directory containing tool-pose-pipeline pose .npz artifacts",
    )
    parser.add_argument("--target-min-fraction", type=float, default=0.0)
    parser.add_argument("--target-max-fraction", type=float, default=1.0)
    parser.add_argument(
        "--target-curriculum-max-fractions",
        type=_fractions,
        default=None,
        help="equal-duration upper bounds, e.g. '0.3 0.5 0.7 1.0'",
    )
    parser.add_argument(
        "--require-drivable-pool",
        action="store_true",
        help="reject pose pools generated without drive validation",
    )
    parser.add_argument("--required-grasp-contacts", type=int, default=2)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--spray-tracking-reward-weight", type=float, default=2.0)
    parser.add_argument("--spray-progress-reward-weight", type=float, default=1.0)
    parser.add_argument("--spray-grasp-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--spray-index-proximity-reward-weight", type=float, default=0.25
    )
    parser.add_argument("--spray-success-bonus", type=float, default=5.0)
    parser.add_argument("--spray-failure-penalty", type=float, default=25.0)
    parser.add_argument(
        "--spray-grasp-gate-floor",
        type=float,
        default=0.25,
        help="share of the angle reward available without a grasp",
    )
    parser.add_argument("--push-tracking-reward-weight", type=float, default=2.0)
    parser.add_argument("--push-progress-reward-weight", type=float, default=1.0)
    parser.add_argument("--push-grasp-reward-weight", type=float, default=1.0)
    parser.add_argument("--push-thumb-contact-reward-weight", type=float, default=0.25)
    parser.add_argument("--push-success-bonus", type=float, default=5.0)
    parser.add_argument("--push-failure-penalty", type=float, default=25.0)
    parser.add_argument("--push-grasp-gate-floor", type=float, default=0.25)
    parser.add_argument(
        "--scissors-object-dx",
        type=float,
        default=0.0,
        help="shift the scissors along x so the handles sit deeper on the fingers; "
        "the variants that train well hold 5 contacts, the ones that do not hold 2",
    )
    parser.add_argument(
        "--scissors-angle-threshold",
        type=float,
        default=0.03,
        help="angle error counted as success, in radians",
    )
    parser.add_argument(
        "--scissors-reward-type", choices=("standard", "dense"), default="standard"
    )
    # One shared Scissors reward configuration is used for variants 1--5.
    parser.add_argument('--scissors-control-version', choices=('simple','smooth','v2','v3','v4','v5','v6','v7'), default='v2')
    parser.add_argument('--scissors-action-delta-fraction', type=float, default=0.05,
        help='smooth mode: maximum change of normalized actuator setpoint per step; Push/Spray relative mode uses 0.05')
    parser.add_argument('--scissors-request-gap-penalty-weight', type=float, default=1.0)
    parser.add_argument('--tool-action-slew-rate', type=float, default=0.0,
        help='Push/Spray only: normalized setpoint change per simulated second; 0 preserves legacy dynamics')
    parser.add_argument('--tool-request-gap-penalty-weight', type=float, default=1.0)
    parser.add_argument("--scissors-coarse-tracking-weight", type=float, default=1.0)
    parser.add_argument("--scissors-fine-tracking-weight", type=float, default=1.0)
    parser.add_argument("--scissors-coarse-tracking-sigma", type=float, default=0.35)
    parser.add_argument("--scissors-progress-weight", type=float, default=1.0)
    parser.add_argument("--scissors-grasp-weight", type=float, default=1.0)
    parser.add_argument("--scissors-contact-auxiliary-weight", type=float, default=0.0,
                        help="optional contact reward only; never changes palm grasp or success")
    parser.add_argument("--scissors-interaction-weight", type=float, default=0.25)
    parser.add_argument("--scissors-success-bonus", type=float, default=5.0)
    parser.add_argument("--scissors-failure-penalty", type=float, default=25.0)
    parser.add_argument("--scissors-grasp-gate-floor", type=float, default=0.25)
    parser.add_argument(
        "--scissors-ctrl-delta-penalty",
        type=float,
        default=0.0,
        help="charge for how far the command moves between steps; a policy that "
        "reaches the target but keeps jittering never settles on it",
    )
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--dataset-save-frequency", type=int, default=100)
    parser.add_argument("--control-delta-penalty-weight", type=float, default=0.1)
    parser.add_argument("--control-magnitude-penalty-weight", type=float, default=0.002)
    args = parser.parse_args(argv)
    if args.scissors_control_version in ('simple','smooth') and args.her_sampled_goals != 0:
        parser.error('the simple/smooth campaign uses standard SAC replay; use --her-sampled-goals 0')
    if args.scissors_control_version in ('v4','v5','v6','v7') and args.her_sampled_goals != 0:
        parser.error('fixed-command phases depend on the original goal; use --her-sampled-goals 0')
    if args.scissors_control_version == 'v7' and (args.algorithm != 'sac' or args.sac_ent_coef == 'auto'):
        parser.error('v7 phase-aware learning requires SAC with a fixed entropy coefficient')
    if not 0 <= args.her_ratio < 1:
        parser.error("her-ratio must lie in [0,1)")
    try:
        sac_ent_coef: str | float = (
            "auto" if args.sac_ent_coef == "auto" else float(args.sac_ent_coef)
        )
    except ValueError:
        parser.error("--sac-ent-coef must be 'auto' or a non-negative float")
    if isinstance(sac_ent_coef, float) and (
        not np.isfinite(sac_ent_coef) or sac_ent_coef < 0.0
    ):
        parser.error("--sac-ent-coef must be finite and non-negative")
    positive_values = {
        "total_timesteps": args.total_timesteps,
        "checkpoint_freq": args.checkpoint_freq,
        "validation_freq": args.validation_freq,
        "validation_consecutive": args.validation_consecutive,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "success_hold_steps": args.success_hold_steps,
        "dataset_save_frequency": args.dataset_save_frequency,
        "n_envs": args.n_envs,
        "gradient_steps": args.gradient_steps,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    invalid_target_range = not (
        0.0 <= args.target_min_fraction < args.target_max_fraction <= 1.0
    )
    curriculum = args.target_curriculum_max_fractions
    invalid_curriculum = bool(
        curriculum
        and (
            any(right <= left for left, right in zip(curriculum, curriculum[1:]))
            or curriculum[0] <= args.target_min_fraction
            or curriculum[-1] != args.target_max_fraction
        )
    )
    push_reward_values = {
        "push_tracking_reward_weight": args.push_tracking_reward_weight,
        "push_progress_reward_weight": args.push_progress_reward_weight,
        "push_grasp_reward_weight": args.push_grasp_reward_weight,
        "push_thumb_contact_reward_weight": args.push_thumb_contact_reward_weight,
        "push_success_bonus": args.push_success_bonus,
        "push_failure_penalty": args.push_failure_penalty,
        "push_grasp_gate_floor": args.push_grasp_gate_floor,
        "scissors_ctrl_delta_penalty": args.scissors_ctrl_delta_penalty,
        "scissors_angle_threshold": args.scissors_angle_threshold,
        "scissors_object_dx": args.scissors_object_dx,
        "control_delta_penalty_weight": args.control_delta_penalty_weight,
        "control_magnitude_penalty_weight": args.control_magnitude_penalty_weight,
    }
    invalid_push_rewards = [
        name
        for name, value in push_reward_values.items()
        if not np.isfinite(value) or value < 0.0
    ]
    push_reward_values["scissors_reward_type"] = args.scissors_reward_type
    spray_reward_values = {
        "spray_tracking_reward_weight": args.spray_tracking_reward_weight,
        "spray_progress_reward_weight": args.spray_progress_reward_weight,
        "spray_grasp_reward_weight": args.spray_grasp_reward_weight,
        "spray_index_proximity_reward_weight": (
            args.spray_index_proximity_reward_weight
        ),
        "spray_success_bonus": args.spray_success_bonus,
        "spray_failure_penalty": args.spray_failure_penalty,
        "spray_grasp_gate_floor": args.spray_grasp_gate_floor,
    }
    invalid_spray_rewards = [
        name
        for name, value in spray_reward_values.items()
        if not np.isfinite(value) or value < 0.0
    ]
    scissors_reward_values = {
        "coarse_tracking_reward_weight": args.scissors_coarse_tracking_weight,
        "fine_tracking_reward_weight": args.scissors_fine_tracking_weight,
        "coarse_tracking_sigma": args.scissors_coarse_tracking_sigma,
        "progress_reward_weight": args.scissors_progress_weight,
        "grasp_reward_weight": args.scissors_grasp_weight,
        "contact_auxiliary_reward_weight": args.scissors_contact_auxiliary_weight,
        "interaction_reward_weight": args.scissors_interaction_weight,
        "success_bonus": args.scissors_success_bonus,
        "failure_penalty": args.scissors_failure_penalty,
        "grasp_gate_floor": args.scissors_grasp_gate_floor,
        "control_delta_penalty_weight": args.control_delta_penalty_weight,
        "control_magnitude_penalty_weight": args.control_magnitude_penalty_weight,
    }
    invalid_scissors_rewards = [
        name
        for name, value in scissors_reward_values.items()
        if not np.isfinite(value) or value < 0.0
    ]
    if (
        invalid
        or args.her_sampled_goals < 0
        or args.action_noise < 0.0
        or args.learning_starts < 0
        or args.required_grasp_contacts < 1
        or not 0.0 < args.validation_success_threshold <= 1.0
        or invalid_target_range
        or invalid_curriculum
        or invalid_push_rewards
        or args.push_failure_penalty <= 0.0
        or invalid_spray_rewards
        or args.spray_failure_penalty <= 0.0
        or invalid_scissors_rewards
        or not 0 < args.scissors_action_delta_fraction <= 1
        or not np.isfinite(args.scissors_action_delta_fraction)
        or not np.isfinite(args.scissors_request_gap_penalty_weight)
        or args.scissors_request_gap_penalty_weight < 0
        or not np.isfinite(args.tool_action_slew_rate)
        or args.tool_action_slew_rate < 0
        or not np.isfinite(args.tool_request_gap_penalty_weight)
        or args.tool_request_gap_penalty_weight < 0
        or args.scissors_coarse_tracking_sigma <= 0.0
        or args.scissors_failure_penalty <= 0.0
        or not 0.0 <= args.scissors_grasp_gate_floor <= 1.0
    ):
        parser.error(
            "invalid training configuration: "
            f"non-positive={invalid}, target_range="
            f"({args.target_min_fraction}, {args.target_max_fraction}), "
            f"required_grasp_contacts={args.required_grasp_contacts}, "
            f"invalid_push_rewards={invalid_push_rewards}, "
            f"invalid_spray_rewards={invalid_spray_rewards}"
            f", invalid_scissors_rewards={invalid_scissors_rewards}, "
            f"invalid_curriculum={invalid_curriculum}"
        )

    # Training always uses the tool-manip-gym task implementations.  Importing
    # them here registers ScissorsVariant*, SliderVariant*, and spray IDs.
    _register_tool_manip_gym_environments()
    train_env_id = _tool_manip_env_id(args.env_id)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = (
        Path(args.dataset)
        if args.dataset is not None
        else destination.with_name(f"{destination.name}_training_candidates.npz")
    )
    checkpoints_dir = (
        Path(args.checkpoints_dir)
        if args.checkpoints_dir is not None
        else destination.with_name(f"{destination.name}_checkpoints")
    )
    env_kwargs = {}
    if train_env_id.startswith("Scissors"):
        # A pool is injected after reset; the task's canonical-pose hold would
        # otherwise overwrite that sampled grasp on the first step.
        env_kwargs = {
            "hold_initial_steps": 0,
            "ctrl_delta_penalty_weight": args.scissors_ctrl_delta_penalty,
            "angle_threshold": args.scissors_angle_threshold,
            "reward_type": args.scissors_reward_type,
        }
        if args.scissors_object_dx:
            default_qpos = (1.07, 0.892, 0.402, 1.0, 0.0, 0.0, 0.0)
            env_kwargs["initial_object_qpos"] = (
                default_qpos[0] + args.scissors_object_dx,
            ) + default_qpos[1:]
    if args.scissors_control_version in ('v4','v5','v6','v7') and train_env_id.startswith('Scissors'):
        env_kwargs['max_episode_steps'] = 400
    base_env = gym.make(train_env_id, **env_kwargs)
    pool_path = Path(args.pose_pool_dir) / pool_filename_for_environment(base_env)
    if not pool_path.is_file():
        base_env.close()
        parser.error(f"initial-pose pool not found: {pool_path}")
    if args.require_drivable_pool:
        with np.load(pool_path, allow_pickle=False) as pool_artifact:
            drive_checked = bool(
                "require_drive" in pool_artifact.files
                and np.asarray(pool_artifact["require_drive"]).reshape(-1)[0]
            )
        if not drive_checked:
            base_env.close()
            parser.error(
                f"pose pool was not drive-validated: {pool_path}; regenerate it "
                "with toolpose.generate --require-drive"
            )

    training_target_max = curriculum[0] if curriculum else args.target_max_fraction

    def wrap_task_env(raw_env, seed: int, *, validation=False):
        wrapped = InitialPosePoolWrapper(raw_env, pool_path, seed=seed)
        if type(wrapped.unwrapped).__name__ == "ShadowHandSprayBottleEnv":
            wrapped = SprayGoalWrapper(
                wrapped,
                target_min_fraction=args.target_min_fraction,
                target_max_fraction=training_target_max,
                required_grasp_contacts=args.required_grasp_contacts,
                success_hold_steps=args.success_hold_steps,
                tracking_reward_weight=args.spray_tracking_reward_weight,
                trigger_progress_reward_weight=args.spray_progress_reward_weight,
                grasp_reward_weight=args.spray_grasp_reward_weight,
                index_proximity_reward_weight=(
                    args.spray_index_proximity_reward_weight
                ),
                success_bonus=args.spray_success_bonus,
                failure_penalty=args.spray_failure_penalty,
                grasp_gate_floor=args.spray_grasp_gate_floor,
                control_delta_penalty_weight=args.control_delta_penalty_weight,
                control_magnitude_penalty_weight=(
                    args.control_magnitude_penalty_weight
                ),
                seed=seed,
            )
        elif type(wrapped.unwrapped).__name__ == "PushBottleEnv":
            wrapped = PushGoalWrapper(
                wrapped,
                target_min_fraction=args.target_min_fraction,
                target_max_fraction=training_target_max,
                required_grasp_contacts=args.required_grasp_contacts,
                success_hold_steps=args.success_hold_steps,
                tracking_reward_weight=args.push_tracking_reward_weight,
                push_progress_reward_weight=args.push_progress_reward_weight,
                grasp_reward_weight=args.push_grasp_reward_weight,
                thumb_contact_reward_weight=args.push_thumb_contact_reward_weight,
                success_bonus=args.push_success_bonus,
                failure_penalty=args.push_failure_penalty,
                grasp_gate_floor=args.push_grasp_gate_floor,
                control_delta_penalty_weight=args.control_delta_penalty_weight,
                control_magnitude_penalty_weight=(
                    args.control_magnitude_penalty_weight
                ),
                seed=seed,
            )
        elif type(wrapped.unwrapped).__name__ == "ScissorsManipulateEnv":
            wrapper_class = {'v2':ScissorsPostureWrapper, 'v3':ScissorsControlWrapper,
                             'v4':ScissorsRetentionWrapper,'v5':ScissorsRetentionV5Wrapper,
                             'v6':ScissorsRetentionV6Wrapper,'v7':ScissorsRetentionV7Wrapper,
                             'simple':ScissorsReachWrapper,
                             'smooth':ScissorsSmoothReachWrapper}[args.scissors_control_version]
            wrapped = wrapper_class(
                wrapped,
                **({'required_grasp_contacts':args.required_grasp_contacts,'angle_tolerance':args.scissors_angle_threshold,
                    'contact_auxiliary_reward_weight':args.scissors_contact_auxiliary_weight} if args.scissors_control_version in ('simple','smooth')
                   else {'criteria':PostureCriteria(angle_tolerance=args.scissors_angle_threshold,
                    hold_seconds=2. if args.scissors_control_version in ('v4','v5','v6','v7') else .4)}),
                **({'smooth_config': SmoothReachConfig(args.scissors_action_delta_fraction,
                    args.scissors_request_gap_penalty_weight)} if args.scissors_control_version == 'smooth' else {}),
                **({'retention_training_steps': 0 if validation else max(1,args.total_timesteps//args.n_envs)}
                   if args.scissors_control_version in ('v4','v5','v6','v7') else {}),
                target_min_fraction=args.target_min_fraction,
                target_max_fraction=training_target_max,
                success_hold_steps=args.success_hold_steps,
                coarse_tracking_reward_weight=args.scissors_coarse_tracking_weight,
                fine_tracking_reward_weight=args.scissors_fine_tracking_weight,
                coarse_tracking_sigma=args.scissors_coarse_tracking_sigma,
                progress_reward_weight=args.scissors_progress_weight,
                grasp_reward_weight=args.scissors_grasp_weight,
                interaction_reward_weight=args.scissors_interaction_weight,
                success_bonus=args.scissors_success_bonus,
                failure_penalty=args.scissors_failure_penalty,
                grasp_gate_floor=args.scissors_grasp_gate_floor,
                control_delta_penalty_weight=args.control_delta_penalty_weight,
                control_magnitude_penalty_weight=(
                    args.control_magnitude_penalty_weight
                ),
                seed=seed,
            )
        if type(wrapped.unwrapped).__name__ in ('ShadowHandSprayBottleEnv', 'PushBottleEnv') and args.tool_action_slew_rate:
            wrapped = ToolActionSlewWrapper(wrapped,
                rate_per_second=args.tool_action_slew_rate,
                request_gap_penalty_weight=args.tool_request_gap_penalty_weight)
        return ControlRefinement(wrapped)

    action_dimension = int(base_env.action_space.shape[0])
    raw = base_env.unwrapped
    goal_span = (float(np.diff(raw.target_angle_range)[0]) if hasattr(raw,'target_angle_range')
                 else float(getattr(raw,'push_limit',getattr(raw,'trigger_limit',1.))))
    base_env.close()
    simulator_parameters = {
            "scissors_control_version": args.scissors_control_version,
            "algorithm": args.algorithm,
            "n_envs": args.n_envs,
            "gradient_steps": args.gradient_steps,
            "learning_rate": args.learning_rate,
            "action_noise": args.action_noise if args.algorithm != "sac" else None,
            "learning_starts": args.learning_starts,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "replay_buffer": {
                "enabled": args.her_sampled_goals > 0,
                "n_sampled_goal": args.her_sampled_goals,
                "goal_selection_strategy": "future",
                "copy_info_dict": True,
            },
            "ent_coef": sac_ent_coef if args.algorithm == "sac" else None,
            "success_hold_steps": (args.success_hold_steps if args.scissors_control_version in ('simple','smooth') else
                50 if args.scissors_control_version in ('v4','v5','v6','v7') else 10)
                if "angle_threshold" in env_kwargs else args.success_hold_steps,
            "hold_initial_steps": 0,
            "pose_pool_dir": str(args.pose_pool_dir),
            "target_min_fraction": args.target_min_fraction,
            "target_max_fraction": args.target_max_fraction,
            "required_grasp_contacts": args.required_grasp_contacts,
            "scissors_contact_auxiliary_reward_weight": args.scissors_contact_auxiliary_weight,
            "scissors_grasp_criterion": ('hinge_center_inside_palm_attached_original_sphere'
                if args.scissors_control_version in ('simple','smooth') else None),
            "scissors_action_delta_fraction": args.scissors_action_delta_fraction if args.scissors_control_version == 'smooth' else None,
            "scissors_request_gap_penalty_weight": args.scissors_request_gap_penalty_weight if args.scissors_control_version == 'smooth' else None,
            "tool_action_slew_rate_per_second": args.tool_action_slew_rate,
            "tool_request_gap_penalty_weight": args.tool_request_gap_penalty_weight if args.tool_action_slew_rate else None,
            "push_reward": push_reward_values,
            "spray_reward": spray_reward_values,
            "scissors_reward": scissors_reward_values,
            "unified_control_delta_penalty_weight": (
                args.control_delta_penalty_weight
            ),
            "unified_control_magnitude_penalty_weight": (
                args.control_magnitude_penalty_weight
            ),
        }

    def worker_dataset_path(rank: int) -> Path:
        if args.n_envs == 1:
            return dataset_path
        return dataset_path.with_name(
            f"{dataset_path.stem}_env{rank}{dataset_path.suffix}"
        )

    def make_training_env(rank: int):
        def initialize():
            # Windows uses spawn, so registrations must also occur in children.
            _register_tool_manip_gym_environments()
            raw = gym.make(train_env_id, **env_kwargs)
            task = wrap_task_env(raw, args.seed + rank)
            return SuccessfulTrajectoryCollector(
                task,
                output_path=worker_dataset_path(rank),
                controller_name=f"{args.algorithm}_her",
                policy_checkpoint=None,
                simulator_parameters={**simulator_parameters, "worker_rank": rank},
                save_failures=args.save_failures,
                dataset_role="training_candidate",
                save_every_trajectories=args.dataset_save_frequency,
            )

        return initialize

    env = SubprocVecEnv(
        [make_training_env(rank) for rank in range(args.n_envs)],
        start_method="spawn",
    )
    action_noise = None
    if args.algorithm in {"td3", "ddpg"}:
        action_noise = NormalActionNoise(
            np.zeros(action_dimension),
            args.action_noise * np.ones(action_dimension),
        )
    algorithm_kwargs = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "seed": args.seed,
        "device": args.device,
        "verbose": 1,
        "gradient_steps": args.gradient_steps,
    }
    if args.her_sampled_goals > 0:
        algorithm_kwargs.update(
            {
                "replay_buffer_class": HerReplayBuffer,
                "replay_buffer_kwargs": {
                    "n_sampled_goal": args.her_sampled_goals,
                    "goal_selection_strategy": "future",
                    "copy_info_dict": True,
                },
            }
        )
    if args.algorithm == "sac":
        algorithm_kwargs.update(
            {
                "ent_coef": sac_ent_coef,
                "policy_kwargs": {"net_arch": [256, 256]},
            }
        )
    else:
        algorithm_kwargs.update(
            {
                "action_noise": action_noise,
                "policy_kwargs": {"net_arch": [256, 256, 256]},
            }
        )
    if args.scissors_control_version in ('simple','smooth','v5','v6','v7') or args.tool_action_slew_rate:
        algorithm_kwargs['replay_buffer_class'] = NumericalSafeReplayBuffer
    feature_class = ScissorsControlFeatures if train_env_id.startswith('Scissors') and args.scissors_control_version in ('v3','v4','v5','v6','v7') else ToolFeaturesExtractor
    algorithm_kwargs['policy_kwargs'].update(features_extractor_class=feature_class,
        features_extractor_kwargs={'goal_span':goal_span})
    model = algorithm_class(args.algorithm)(
        RetentionSACPolicy if args.scissors_control_version == 'v7' else "MultiInputPolicy",
        env,
        **algorithm_kwargs,
    )
    checkpoint = CheckpointCallback(
        # Callback calls count VecEnv steps; num_timesteps counts all workers.
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(checkpoints_dir),
        name_prefix=f"{args.algorithm}_her",
    )
    if args.her_sampled_goals > 0:
        model.replay_buffer.her_ratio = args.her_ratio
    diagnostics = TrainingDiagnosticsCallback(args.n_envs)
    validation_env = wrap_task_env(gym.make(train_env_id, **env_kwargs), args.seed + 10_000, validation=True)
    task_name = type(validation_env.unwrapped).__name__
    if task_name == "ShadowHandSprayBottleEnv":
        option_key = "target_trigger_angle"
        limit = float(validation_env.unwrapped.trigger_limit)
        validation_targets = tuple(
            fraction * limit for fraction in args.validation_target_fractions
        )
    elif task_name == "PushBottleEnv":
        option_key = "target_push_depth"
        limit = float(validation_env.unwrapped.push_limit)
        validation_targets = tuple(
            fraction * limit for fraction in args.validation_target_fractions
        )
    elif task_name == "ScissorsManipulateEnv":
        option_key = "target_angle"
        low, high = np.asarray(validation_env.unwrapped.target_angle_range)
        validation_targets = tuple(
            float(low + fraction * (high - low))
            for fraction in args.validation_target_fractions
        )
    elif task_name == "SliderManipulateEnv":
        option_key = "target_position"
        low, high = np.asarray(validation_env.unwrapped.target_position_range)
        validation_targets = tuple(
            float(low + fraction * (high - low))
            for fraction in args.validation_target_fractions
        )
    else:
        validation_env.close()
        raise TypeError(f"validation is not implemented for {task_name}")
    best_policy_path = (
        Path(args.best_policy_output)
        if args.best_policy_output is not None
        else destination.with_name("best_policy")
    )
    validation_report_path = (
        Path(args.validation_report)
        if args.validation_report is not None
        else destination.with_name("validation_history.json")
    )
    validation = ValidationEarlyStoppingCallback(
        validation_env=validation_env,
        poses=range(validation_env.get_wrapper_attr("pose_count")),
        targets=validation_targets,
        option_key=option_key,
        eval_freq=args.validation_freq,
        success_threshold=args.validation_success_threshold,
        consecutive_evaluations=args.validation_consecutive,
        best_policy_path=best_policy_path,
        report_path=validation_report_path,
    )
    callbacks: list[BaseCallback] = [checkpoint, diagnostics]
    if curriculum:
        callbacks.append(TargetRangeCurriculumCallback(curriculum, args.total_timesteps))
    callbacks.append(validation)
    model.learn(args.total_timesteps, callback=CallbackList(callbacks))
    model.save(destination)
    goal_sampling = (
        env.env_method("goal_sampling_report")
        if task_name == "ScissorsManipulateEnv" else None
    )
    summary = {
        "goal_sampling": goal_sampling,
        "observation_version": f'scissors_reach_{args.scissors_control_version}_palm_v1' if task_name == 'ScissorsManipulateEnv' and args.scissors_control_version in ('simple','smooth') else f"scissors_control_{args.scissors_control_version}" if task_name == 'ScissorsManipulateEnv' and args.scissors_control_version in ('v3','v4','v5','v6','v7') else "tool_control_v2",
        "dropped_numerical_vector_transitions": getattr(model.replay_buffer,'dropped_vector_transitions',0),
        "scissors_control_version": args.scissors_control_version,
        "her_ratio": args.her_ratio if args.her_sampled_goals > 0 else 0,
        "goal_feature_span": goal_span,
        "algorithm": args.algorithm,
        "env_id": train_env_id,
        "requested_env_id": args.env_id,
        "seed": args.seed,
        "total_timesteps": model.num_timesteps,
        "requested_total_timesteps": args.total_timesteps,
        "stopped_early": validation.stopped_early,
        "validation": validation._report(),
        "successful_candidate_trajectories": diagnostics.successful_trajectory_count,
        # episodes the recorder had to skip; clear them from the saved
        # dataset afterwards with improved_tds.data.cleanup
        "rejected_trajectories": int(sum(env.get_attr("rejected_trajectories"))),
        "rejection_reasons": {},
        "action_saturation_fraction": diagnostics.action_saturation_fraction,
        "datasets": [str(worker_dataset_path(rank)) for rank in range(args.n_envs)],
        "n_envs": args.n_envs,
        "gradient_steps": args.gradient_steps,
        "model": str(destination.with_suffix(".zip")),
        "checkpoints_dir": str(checkpoints_dir),
        "learning_starts": args.learning_starts,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "action_noise": args.action_noise if args.algorithm != "sac" else None,
        "ent_coef": sac_ent_coef if args.algorithm == "sac" else None,
        "her_sampled_goals": args.her_sampled_goals,
        "pose_pool": str(pool_path),
        "target_min_fraction": args.target_min_fraction,
        "target_max_fraction": args.target_max_fraction,
        "target_curriculum_max_fractions": list(curriculum) if curriculum else None,
        "require_drivable_pool": args.require_drivable_pool,
        "required_grasp_contacts": args.required_grasp_contacts,
        "push_reward": push_reward_values,
        "spray_reward": spray_reward_values,
        "scissors_reward": scissors_reward_values,
    }
    write_report(destination.with_name(f"{destination.name}_training_summary.json"), summary)
    # The complete per-episode Validation history is in the JSON report. Avoid
    # duplicating tens of MB in one unbounded console line at training end.
    print({k: summary[k] for k in ('env_id','total_timesteps','stopped_early',
        'successful_candidate_trajectories','rejected_trajectories','model')}, flush=True)
    env.close()


if __name__ == "__main__":
    main()
