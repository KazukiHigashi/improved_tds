"""Randomized initial-pose pools for tool-manip-gym training environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


# A finger's middle (J2) and distal (J1) joints are coupled: only J2 is
# commanded and J1 mirrors it. J1 therefore has an actuator of its own but is
# absent from the action space, so injecting a pose has to set it explicitly --
# otherwise the episode starts with a finger shape the pose never described.
_COUPLED_DISTAL_ACTUATORS = {
    "rh_A_FFJ2": "rh_A_FFJ1",
    "rh_A_MFJ2": "rh_A_MFJ1",
    "rh_A_RFJ2": "rh_A_RFJ1",
    "rh_A_LFJ2": "rh_A_LFJ1",
}


def _environment_case(env: gym.Env) -> tuple[str, str, str]:
    unwrapped = env.unwrapped
    class_name = type(unwrapped).__name__
    if class_name == "ScissorsManipulateEnv":
        return "scissors", str(unwrapped.hand), str(unwrapped.scissors_variant)
    if class_name == "SliderManipulateEnv":
        return "slider", str(unwrapped.hand), str(unwrapped.slider_variant)
    if class_name == "ShadowHandSprayBottleEnv":
        return "spray", str(unwrapped.hand), str(unwrapped.spray_type)
    if class_name == "PushBottleEnv":
        return "push", str(unwrapped.hand), str(unwrapped.push_variant)
    raise ValueError(
        "initial-pose pools require a tool-manip-gym scissors, slider, spray, "
        "or push environment"
    )


class InitialPosePoolWrapper(gym.Wrapper):
    """Sample and apply one validated grasp pose after every environment reset.

    Pools are intentionally checked against the environment's selected actuators
    by name, rather than merely by array length.  This prevents a pose generated
    for a different Shadow Hand model from being applied in the wrong order.
    """

    def __init__(self, env: gym.Env, pool_path: str | Path, seed: int | None = None):
        super().__init__(env)
        self.pool_path = Path(pool_path)
        with np.load(self.pool_path, allow_pickle=False) as artifact:
            self.task = str(artifact["task"])
            self.hand = str(artifact["hand"])
            self.variant = str(artifact["variant"])
            self._pool_actuator_names = tuple(
                str(name) for name in artifact["actuator_names"]
            )
            self._poses = np.asarray(artifact["poses"], dtype=np.float64)
            self._mount_dpos = np.asarray(
                artifact["mount_dpos"]
                if "mount_dpos" in artifact.files
                else np.zeros((len(self._poses), 3)),
                dtype=np.float64,
            )
            self._mount_drot = np.asarray(
                artifact["mount_drot"]
                if "mount_drot" in artifact.files
                else np.zeros((len(self._poses), 3)),
                dtype=np.float64,
            )
        if self._poses.ndim != 2 or not len(self._poses):
            raise ValueError(f"pose pool is empty or malformed: {self.pool_path}")

        unwrapped = env.unwrapped
        expected_case = _environment_case(env)
        actual_case = (self.task, self.hand, self.variant)
        if actual_case != expected_case:
            raise ValueError(
                f"pool {self.pool_path.name} belongs to {actual_case}, "
                f"but the environment is {expected_case}"
            )
        required = ("model", "data", "_actuator_ids", "_neutral_ctrl")
        missing = [name for name in required if not hasattr(unwrapped, name)]
        actuator_names = getattr(unwrapped, "_actuator_names", None)
        if actuator_names is None:
            actuator_names = getattr(unwrapped, "actuator_names", None)
        if missing or actuator_names is None:
            raise TypeError(
                "initial-pose pools require a tool-manip-gym Shadow Hand environment; "
                f"missing {missing + ([] if actuator_names is not None else ['actuator_names'])}"
            )
        self._actuator_ids = np.asarray(unwrapped._actuator_ids, dtype=np.int32)
        environment_names = tuple(str(name) for name in actuator_names)
        if environment_names != self._pool_actuator_names:
            raise ValueError(
                f"pool {self.pool_path.name} actuators do not match the environment: "
                f"pool={self._pool_actuator_names}, environment={environment_names}"
            )
        self._rng = np.random.default_rng(seed)
        self._hand_root_id = mujoco.mj_name2id(
            unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "rh_forearm"
        )
        self._default_mount_pos = (
            unwrapped.model.body_pos[self._hand_root_id].copy()
            if self._hand_root_id >= 0
            else None
        )
        self._default_mount_quat = (
            unwrapped.model.body_quat[self._hand_root_id].copy()
            if self._hand_root_id >= 0
            else None
        )

    @property
    def pose_count(self) -> int:
        return len(self._poses)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # An analysis run needs to visit every authored pose the same number of
        # times, so it can ask for one by index instead of drawing at random.
        reset_options = dict(options or {})
        requested = reset_options.pop("initial_pose_index", None)
        _, info = self.env.reset(seed=seed, options=reset_options or None)
        if requested is None:
            index = int(self._rng.integers(len(self._poses)))
        else:
            index = int(requested)
            if not 0 <= index < len(self._poses):
                raise ValueError(
                    f"initial_pose_index {index} is outside the pool of "
                    f"{len(self._poses)} poses in {self.pool_path.name}"
                )
        return self._inject(index, info)

    def _inject(self, index: int, info: dict[str, Any]):
        unwrapped = self.env.unwrapped
        model, data = unwrapped.model, unwrapped.data
        self._set_mount(model, self._mount_dpos[index], self._mount_drot[index])
        pose = np.clip(
            self._poses[index],
            model.actuator_ctrlrange[self._actuator_ids, 0],
            model.actuator_ctrlrange[self._actuator_ids, 1],
        )
        for value, actuator_id in zip(pose, self._actuator_ids, strict=True):
            transmission = int(model.actuator_trntype[actuator_id])
            target = int(model.actuator_trnid[actuator_id, 0])
            if transmission == mujoco.mjtTrn.mjTRN_JOINT:
                data.qpos[int(model.jnt_qposadr[target])] = value
        data.qvel[:] = 0.0
        unwrapped._neutral_ctrl = pose.astype(unwrapped._neutral_ctrl.dtype)
        data.ctrl[self._actuator_ids] = pose
        self._apply_coupled_distal(model, data)
        mujoco.mj_forward(model, data)
        observation = unwrapped._get_obs()
        try:
            info = unwrapped._get_info(observation)
        except TypeError:
            info = unwrapped._get_info()
        info = dict(info)
        info["initial_pose_pool"] = self.pool_path.name
        info["initial_pose_index"] = index
        return observation, info

    def _apply_coupled_distal(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Give each coupled J1 the angle and command of the J2 that drives it."""
        for driver, follower in _COUPLED_DISTAL_ACTUATORS.items():
            driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, driver)
            follower_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, follower)
            if driver_id < 0 or follower_id < 0:
                continue
            data.ctrl[follower_id] = data.ctrl[driver_id]
            driver_joint = int(model.actuator_trnid[driver_id, 0])
            follower_joint = int(model.actuator_trnid[follower_id, 0])
            data.qpos[int(model.jnt_qposadr[follower_joint])] = data.qpos[
                int(model.jnt_qposadr[driver_joint])
            ]

    def _set_mount(self, model: mujoco.MjModel, dpos: np.ndarray, drot: np.ndarray) -> None:
        if self._hand_root_id < 0:
            if np.any(dpos) or np.any(drot):
                raise ValueError(f"pool {self.pool_path.name} requires a rh_forearm body")
            return
        model.body_pos[self._hand_root_id] = self._default_mount_pos + dpos
        angle = float(np.linalg.norm(drot))
        delta_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if angle > 1e-9:
            mujoco.mju_axisAngle2Quat(delta_quat, drot / angle, angle)
        mount_quat = np.zeros(4)
        mujoco.mju_mulQuat(mount_quat, self._default_mount_quat, delta_quat)
        model.body_quat[self._hand_root_id] = mount_quat


def pool_filename_for_environment(env: gym.Env) -> str:
    """Return the pool artifact name for a supported tool-manip-gym environment."""
    task, hand, variant = _environment_case(env)
    return f"{task}_{hand}_{variant}.npz"


class SprayGoalWrapper(gym.Wrapper):
    """Goal-conditioned spray control with grasp and stability shaping.

    The reward follows the same common-term design as :class:`PushGoalWrapper`:
    broad/fine tracking, transition progress, grasp, interaction, success and
    control costs. A drop or tip overrides every shaping term. It remains a pure function of
    ``achieved_goal``, ``desired_goal``, and copied ``info`` so HER can
    recompute rewards for relabelled goals.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        target_min_fraction: float = 0.0,
        target_max_fraction: float = 1.0,
        required_grasp_contacts: int = 2,
        success_hold_steps: int = 5,
        tracking_reward_weight: float = 2.0,
        trigger_progress_reward_weight: float = 1.0,
        grasp_reward_weight: float = 1.0,
        index_proximity_reward_weight: float = 0.25,
        success_bonus: float = 5.0,
        failure_penalty: float = 25.0,
        grasp_gate_floor: float = 0.25,
        control_delta_penalty_weight: float = 0.1,
        control_magnitude_penalty_weight: float = 0.002,
        seed: int | None = None,
    ):
        super().__init__(env)
        if not 0.0 <= grasp_gate_floor <= 1.0:
            raise ValueError("grasp_gate_floor must lie in [0, 1]")
        if not 0.0 <= target_min_fraction < target_max_fraction <= 1.0:
            raise ValueError("target fractions must satisfy 0 <= min < max <= 1")
        if required_grasp_contacts < 1:
            raise ValueError("required_grasp_contacts must be positive")
        if success_hold_steps < 1:
            raise ValueError("success_hold_steps must be positive")
        reward_parameters = {
            "tracking_reward_weight": tracking_reward_weight,
            "trigger_progress_reward_weight": trigger_progress_reward_weight,
            "grasp_reward_weight": grasp_reward_weight,
            "index_proximity_reward_weight": index_proximity_reward_weight,
            "success_bonus": success_bonus,
            "failure_penalty": failure_penalty,
            "grasp_gate_floor": grasp_gate_floor,
            "control_delta_penalty_weight": control_delta_penalty_weight,
            "control_magnitude_penalty_weight": control_magnitude_penalty_weight,
        }
        invalid_reward_parameters = [
            name
            for name, value in reward_parameters.items()
            if not np.isfinite(value) or value < 0.0
        ]
        if invalid_reward_parameters:
            raise ValueError(
                "spray reward parameters must be finite and non-negative: "
                f"{invalid_reward_parameters}"
            )
        if failure_penalty <= 0.0:
            raise ValueError("failure_penalty must be positive")
        unwrapped = env.unwrapped
        if type(unwrapped).__name__ != "ShadowHandSprayBottleEnv":
            raise TypeError(
                "SprayGoalWrapper requires a gym-spray ShadowHandSprayBottleEnv"
            )
        self.target_min_fraction = float(target_min_fraction)
        self.target_max_fraction = float(target_max_fraction)
        self.required_grasp_contacts = int(required_grasp_contacts)
        self.success_hold_steps = int(success_hold_steps)
        self.grasp_gate_floor = float(grasp_gate_floor)
        self.tracking_reward_weight = float(tracking_reward_weight)
        self.trigger_progress_reward_weight = float(trigger_progress_reward_weight)
        self.grasp_reward_weight = float(grasp_reward_weight)
        self.index_proximity_reward_weight = float(index_proximity_reward_weight)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.control_delta_penalty_weight = float(control_delta_penalty_weight)
        self.control_magnitude_penalty_weight = float(control_magnitude_penalty_weight)
        self._rng = np.random.default_rng(seed)
        self._success_streak = 0
        self._previous_action: np.ndarray | None = None
        self._initial_achieved = 0.0
        self._previous_achieved = 0.0
        goal_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "achieved_goal": goal_space,
                "desired_goal": goal_space,
            }
        )

    @staticmethod
    def _goals(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([info["trigger_angle"]], dtype=np.float32),
            np.asarray([info["target_trigger_angle"]], dtype=np.float32),
        )

    def _observation(
        self, observation: np.ndarray, info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        achieved, desired = self._goals(info)
        observation = np.asarray(observation, dtype=np.float32).copy()
        observation[-3] = 0.0  # Goal is supplied only through desired_goal (HER).
        return {
            "observation": np.asarray(observation, dtype=np.float32),
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        reset_options = dict(options or {})
        if "target_trigger_angle" not in reset_options:
            limit = float(self.env.unwrapped.trigger_limit)
            reset_options["target_trigger_angle"] = float(
                self._rng.uniform(
                    self.target_min_fraction * limit,
                    self.target_max_fraction * limit,
                )
            )
        observation, info = self.env.reset(seed=seed, options=reset_options)
        self._success_streak = 0
        self._previous_action = None
        self._initial_achieved = float(info["trigger_angle"])
        self._previous_achieved = self._initial_achieved
        info = self._augment_info(info)
        return self._observation(observation, info), info

    def step(self, action: np.ndarray):
        observation, _, terminated, truncated, info = self.env.step(action)
        info = self._add_control_info(info, action)
        info = self._augment_info(info)
        achieved, desired = self._goals(info)
        reward_terms = self._reward_terms(achieved, desired, info)
        for name, values in reward_terms.items():
            info[f"reward_{name}"] = float(np.asarray(values).reshape(-1)[0])
        reward = info["reward_total"]
        return (
            self._observation(observation, info),
            reward,
            terminated,
            truncated,
            info,
        )

    def _add_control_info(self, info: dict[str, Any], action: np.ndarray):
        result = dict(info)
        current = np.asarray(action, dtype=np.float32)
        previous = current if self._previous_action is None else self._previous_action
        result["reward_control_delta_cost"] = float(np.mean(np.square(current - previous)) / 4.0)
        result["reward_control_magnitude_cost"] = float(np.mean(np.square(current)))
        result["previous_achieved_goal"] = self._previous_achieved
        self._previous_achieved = float(result["trigger_angle"])
        self._previous_action = current.copy()
        return result

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        result = dict(info)
        result.setdefault(
            "trigger_tracking_tolerance",
            float(self.env.unwrapped.trigger_tracking_tolerance),
        )
        grasped = (
            int(result.get("grasp_contact_count", 0))
            >= self.required_grasp_contacts
        )
        fallen = bool(result.get("bottle_dropped", False)) or bool(
            result.get("bottle_tipped", False)
        )
        success = (
            bool(result.get("trigger_target_reached", False))
            and grasped
            and not fallen
        )
        self._success_streak = self._success_streak + 1 if success else 0
        result["grasp_maintained"] = float(grasped and not fallen)
        result["is_success"] = float(success)
        result["success_streak"] = self._success_streak
        result["is_stable_success"] = float(
            self._success_streak >= self.success_hold_steps
        )
        result["success_hold_steps"] = self.success_hold_steps
        result["initial_achieved_goal"] = self._initial_achieved
        return result

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._reward_terms(achieved_goal, desired_goal, info)["total"]

    def _reward_terms(self, achieved_goal, desired_goal, info):
        achieved = np.asarray(achieved_goal, dtype=np.float32).reshape(-1)
        desired = np.asarray(desired_goal, dtype=np.float32).reshape(-1)
        infos = [info] * len(achieved) if isinstance(info, dict) else list(info)
        limits = np.asarray(
            [
                max(
                    float(item.get("trigger_limit", self.env.unwrapped.trigger_limit)),
                    np.finfo(np.float32).eps,
                )
                for item in infos
            ],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [
                float(
                    item.get(
                        "trigger_tracking_tolerance",
                        self.env.unwrapped.trigger_tracking_tolerance,
                    )
                )
                for item in infos
            ],
            dtype=np.float32,
        )
        contacts = np.asarray(
            [int(item.get("grasp_contact_count", 0)) for item in infos],
            dtype=np.float32,
        )
        proximity = np.asarray(
            [
                max(float(item.get("index_to_trigger_distance", 1.0)), 0.0)
                for item in infos
            ],
            dtype=np.float32,
        )
        previous_achieved = np.asarray(
            [
                float(item.get("previous_achieved_goal", achieved[index]))
                for index, item in enumerate(infos)
            ],
            dtype=np.float32,
        )
        control_delta_cost = np.asarray(
            [float(item.get("reward_control_delta_cost", 0.0)) for item in infos],
            dtype=np.float32,
        )
        control_magnitude_cost = np.asarray(
            [float(item.get("reward_control_magnitude_cost", 0.0)) for item in infos],
            dtype=np.float32,
        )
        fallen = np.asarray(
            [
                bool(item.get("bottle_dropped", False))
                or bool(item.get("bottle_tipped", False))
                for item in infos
            ]
        )
        error = np.abs(achieved - desired)
        fine_tracking_score = np.exp(
            -0.5 * np.square(error / np.maximum(tolerance, 1e-6))
        )
        coarse_sigma = np.maximum(0.35 * limits, tolerance)
        coarse_tracking_score = np.exp(-0.5 * np.square(error / coarse_sigma))
        previous_error = np.abs(previous_achieved - desired)
        progress_score = np.clip((previous_error - error) / limits, -1.0, 1.0)
        grasp_score = np.clip(
            contacts / float(self.required_grasp_contacts), 0.0, 1.0
        )
        proximity_score = np.exp(-proximity / 0.03)
        grasped = contacts >= self.required_grasp_contacts
        success = (error <= tolerance) & grasped & ~fallen

        # Angle reward is only partly available without a grasp, so hitting the
        # target by knocking the bottle is worth less than working the trigger
        # while holding it. The floor keeps the gradient alive before contact.
        grasp_gate = self.grasp_gate_floor + (1.0 - self.grasp_gate_floor) * grasp_score
        coarse_tracking = (
            0.5 * self.tracking_reward_weight * coarse_tracking_score * grasp_gate
        )
        fine_tracking = (
            0.5 * self.tracking_reward_weight * fine_tracking_score * grasp_gate
        )
        tracking = coarse_tracking + fine_tracking
        trigger_progress = (
            self.trigger_progress_reward_weight * progress_score * grasp_gate
        )
        grasp = self.grasp_reward_weight * grasp_score
        index_proximity = self.index_proximity_reward_weight * proximity_score
        success_reward = self.success_bonus * success.astype(np.float32)
        control_delta = -getattr(self, "control_delta_penalty_weight", 0.0) * control_delta_cost
        control_magnitude = (
            -getattr(self, "control_magnitude_penalty_weight", 0.0)
            * control_magnitude_cost
        )
        failure = -self.failure_penalty * fallen.astype(np.float32)
        total = (
            tracking
            + trigger_progress
            + grasp
            + index_proximity
            + success_reward
            + control_delta
            + control_magnitude
        )
        total = np.where(fallen, failure, total)
        return {
            "tracking": np.asarray(tracking, dtype=np.float32),
            "coarse_tracking": np.asarray(coarse_tracking, dtype=np.float32),
            "fine_tracking": np.asarray(fine_tracking, dtype=np.float32),
            "progress": np.asarray(trigger_progress, dtype=np.float32),
            "trigger_progress": np.asarray(trigger_progress, dtype=np.float32),
            "grasp": np.asarray(grasp, dtype=np.float32),
            "interaction": np.asarray(index_proximity, dtype=np.float32),
            "index_proximity": np.asarray(index_proximity, dtype=np.float32),
            "success_bonus": np.asarray(success_reward, dtype=np.float32),
            "control_delta_penalty": np.asarray(control_delta, dtype=np.float32),
            "control_magnitude_penalty": np.asarray(control_magnitude, dtype=np.float32),
            "failure_penalty": np.asarray(failure, dtype=np.float32),
            "total": np.asarray(total, dtype=np.float32),
        }

class PushGoalWrapper(gym.Wrapper):
    """Goal-conditioned push control with grasp and stability shaping.

    Its common terms and default scales match Spray and Scissors. A drop or tip
    overrides the shaping terms. The reward remains a
    pure function of ``achieved_goal``, ``desired_goal``, and copied ``info`` so
    that HER can recompute it correctly for relabelled goals.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        target_min_fraction: float = 0.0,
        target_max_fraction: float = 1.0,
        required_grasp_contacts: int = 2,
        success_hold_steps: int = 5,
        tracking_reward_weight: float = 2.0,
        push_progress_reward_weight: float = 1.0,
        grasp_reward_weight: float = 1.0,
        thumb_contact_reward_weight: float = 0.25,
        success_bonus: float = 5.0,
        failure_penalty: float = 25.0,
        grasp_gate_floor: float = 0.25,
        control_delta_penalty_weight: float = 0.1,
        control_magnitude_penalty_weight: float = 0.002,
        seed: int | None = None,
    ):
        super().__init__(env)
        if not 0.0 <= grasp_gate_floor <= 1.0:
            raise ValueError("grasp_gate_floor must lie in [0, 1]")
        if not 0.0 <= target_min_fraction < target_max_fraction <= 1.0:
            raise ValueError("target fractions must satisfy 0 <= min < max <= 1")
        if required_grasp_contacts < 1:
            raise ValueError("required_grasp_contacts must be positive")
        if success_hold_steps < 1:
            raise ValueError("success_hold_steps must be positive")
        reward_parameters = {
            "tracking_reward_weight": tracking_reward_weight,
            "push_progress_reward_weight": push_progress_reward_weight,
            "grasp_reward_weight": grasp_reward_weight,
            "thumb_contact_reward_weight": thumb_contact_reward_weight,
            "success_bonus": success_bonus,
            "failure_penalty": failure_penalty,
            "grasp_gate_floor": grasp_gate_floor,
            "control_delta_penalty_weight": control_delta_penalty_weight,
            "control_magnitude_penalty_weight": control_magnitude_penalty_weight,
        }
        invalid_reward_parameters = [
            name
            for name, value in reward_parameters.items()
            if not np.isfinite(value) or value < 0.0
        ]
        if invalid_reward_parameters:
            raise ValueError(
                "push reward parameters must be finite and non-negative: "
                f"{invalid_reward_parameters}"
            )
        if failure_penalty <= 0.0:
            raise ValueError("failure_penalty must be positive")
        unwrapped = env.unwrapped
        if type(unwrapped).__name__ != "PushBottleEnv":
            raise TypeError("PushGoalWrapper requires a gym-push PushBottleEnv")
        self.target_min_fraction = float(target_min_fraction)
        self.target_max_fraction = float(target_max_fraction)
        self.required_grasp_contacts = int(required_grasp_contacts)
        self.success_hold_steps = int(success_hold_steps)
        self.tracking_reward_weight = float(tracking_reward_weight)
        self.push_progress_reward_weight = float(push_progress_reward_weight)
        self.grasp_reward_weight = float(grasp_reward_weight)
        self.thumb_contact_reward_weight = float(thumb_contact_reward_weight)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.grasp_gate_floor = float(grasp_gate_floor)
        self.control_delta_penalty_weight = float(control_delta_penalty_weight)
        self.control_magnitude_penalty_weight = float(control_magnitude_penalty_weight)
        self._rng = np.random.default_rng(seed)
        self._success_streak = 0
        self._previous_action: np.ndarray | None = None
        self._initial_achieved = 0.0
        self._previous_achieved = 0.0
        goal_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "achieved_goal": goal_space,
                "desired_goal": goal_space,
            }
        )

    @staticmethod
    def _goals(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        achieved = np.asarray([info["push_depth"]], dtype=np.float32)
        desired = np.asarray([info["target_push_depth"]], dtype=np.float32)
        return achieved, desired

    def _observation(
        self, observation: np.ndarray, info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        achieved, desired = self._goals(info)
        observation = np.asarray(observation, dtype=np.float32).copy()
        observation[-3] = 0.0  # Keep shape, remove conflicting embedded HER goal.
        return {
            "observation": np.asarray(observation, dtype=np.float32),
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        reset_options = dict(options or {})
        if "target_push_depth" not in reset_options:
            limit = float(self.env.unwrapped.push_limit)
            reset_options["target_push_depth"] = float(
                self._rng.uniform(
                    self.target_min_fraction * limit,
                    self.target_max_fraction * limit,
                )
            )
        observation, info = self.env.reset(seed=seed, options=reset_options)
        self._success_streak = 0
        self._previous_action = None
        self._initial_achieved = float(info["push_depth"])
        self._previous_achieved = self._initial_achieved
        info = self._augment_info(info)
        return self._observation(observation, info), info

    def step(self, action: np.ndarray):
        observation, _, terminated, truncated, info = self.env.step(action)
        info = self._add_control_info(info, action)
        info = self._augment_info(info)
        achieved, desired = self._goals(info)
        reward_terms = self._reward_terms(achieved, desired, info)
        for name, values in reward_terms.items():
            info[f"reward_{name}"] = float(np.asarray(values).reshape(-1)[0])
        reward = info["reward_total"]
        return (
            self._observation(observation, info),
            reward,
            terminated,
            truncated,
            info,
        )

    def _add_control_info(self, info: dict[str, Any], action: np.ndarray):
        result = dict(info)
        current = np.asarray(action, dtype=np.float32)
        previous = current if self._previous_action is None else self._previous_action
        result["reward_control_delta_cost"] = float(np.mean(np.square(current - previous)) / 4.0)
        result["reward_control_magnitude_cost"] = float(np.mean(np.square(current)))
        result["previous_achieved_goal"] = self._previous_achieved
        self._previous_achieved = float(result["push_depth"])
        self._previous_action = current.copy()
        return result

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        result = dict(info)
        result.setdefault(
            "push_tracking_tolerance",
            float(self.env.unwrapped.push_tracking_tolerance),
        )
        grasped = int(result.get("grasp_contact_count", 0)) >= self.required_grasp_contacts
        fallen = bool(result.get("object_dropped", False)) or bool(
            result.get("object_tipped", False)
        )
        success = bool(result.get("push_target_reached", False)) and grasped and not fallen
        self._success_streak = self._success_streak + 1 if success else 0
        result["grasp_maintained"] = float(grasped and not fallen)
        result["is_success"] = float(success)
        result["success_streak"] = self._success_streak
        result["is_stable_success"] = float(
            self._success_streak >= self.success_hold_steps
        )
        result["success_hold_steps"] = self.success_hold_steps
        result["initial_achieved_goal"] = self._initial_achieved
        return result

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._reward_terms(achieved_goal, desired_goal, info)["total"]

    def _reward_terms(self, achieved_goal, desired_goal, info):
        achieved = np.asarray(achieved_goal, dtype=np.float32).reshape(-1)
        desired = np.asarray(desired_goal, dtype=np.float32).reshape(-1)
        infos = [info] * len(achieved) if isinstance(info, dict) else list(info)
        push_limits = np.asarray(
            [
                max(
                    float(item.get("push_limit", self.env.unwrapped.push_limit)),
                    np.finfo(np.float32).eps,
                )
                for item in infos
            ],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [float(item.get("push_tracking_tolerance", 0.0)) for item in infos],
            dtype=np.float32,
        )
        grasp_contacts = np.asarray(
            [int(item.get("grasp_contact_count", 0)) for item in infos],
            dtype=np.float32,
        )
        grasped = grasp_contacts >= self.required_grasp_contacts
        fallen = np.asarray(
            [
                bool(item.get("object_dropped", False))
                or bool(item.get("object_tipped", False))
                for item in infos
            ]
        )
        thumb_contact = np.asarray(
            [bool(item.get("thumb_on_plunger", False)) for item in infos],
            dtype=np.float32,
        )
        previous_achieved = np.asarray(
            [
                float(item.get("previous_achieved_goal", achieved[index]))
                for index, item in enumerate(infos)
            ],
            dtype=np.float32,
        )
        control_delta_cost = np.asarray(
            [float(item.get("reward_control_delta_cost", 0.0)) for item in infos],
            dtype=np.float32,
        )
        control_magnitude_cost = np.asarray(
            [float(item.get("reward_control_magnitude_cost", 0.0)) for item in infos],
            dtype=np.float32,
        )
        error = np.abs(achieved - desired)
        safe_tolerance = np.maximum(tolerance, np.finfo(np.float32).eps)
        fine_tracking_score = np.exp(-0.5 * np.square(error / safe_tolerance))
        coarse_sigma = np.maximum(0.35 * push_limits, safe_tolerance)
        coarse_tracking_score = np.exp(-0.5 * np.square(error / coarse_sigma))
        previous_error = np.abs(previous_achieved - desired)
        push_progress_score = np.clip(
            (previous_error - error) / push_limits, -1.0, 1.0
        )
        grasp_score = np.clip(
            grasp_contacts / float(self.required_grasp_contacts), 0.0, 1.0
        )
        success = (error <= tolerance) & grasped & ~fallen

        grasp_gate = self.grasp_gate_floor + (1.0 - self.grasp_gate_floor) * grasp_score
        coarse_tracking = (
            0.5 * self.tracking_reward_weight * coarse_tracking_score * grasp_gate
        )
        fine_tracking = (
            0.5 * self.tracking_reward_weight * fine_tracking_score * grasp_gate
        )
        tracking = coarse_tracking + fine_tracking
        push_progress = (
            self.push_progress_reward_weight * push_progress_score * grasp_gate
        )
        grasp = self.grasp_reward_weight * grasp_score
        thumb = self.thumb_contact_reward_weight * thumb_contact
        success_reward = self.success_bonus * success.astype(np.float32)
        control_delta = -getattr(self, "control_delta_penalty_weight", 0.0) * control_delta_cost
        control_magnitude = (
            -getattr(self, "control_magnitude_penalty_weight", 0.0)
            * control_magnitude_cost
        )
        failure = -self.failure_penalty * fallen.astype(np.float32)
        total = (
            tracking
            + push_progress
            + grasp
            + thumb
            + success_reward
            + control_delta
            + control_magnitude
        )
        total = np.where(fallen, failure, total)
        return {
            "tracking": np.asarray(tracking, dtype=np.float32),
            "coarse_tracking": np.asarray(coarse_tracking, dtype=np.float32),
            "fine_tracking": np.asarray(fine_tracking, dtype=np.float32),
            "progress": np.asarray(push_progress, dtype=np.float32),
            "push_progress": np.asarray(push_progress, dtype=np.float32),
            "grasp": np.asarray(grasp, dtype=np.float32),
            "interaction": np.asarray(thumb, dtype=np.float32),
            "thumb_contact": np.asarray(thumb, dtype=np.float32),
            "success_bonus": np.asarray(success_reward, dtype=np.float32),
            "control_delta_penalty": np.asarray(control_delta, dtype=np.float32),
            "control_magnitude_penalty": np.asarray(control_magnitude, dtype=np.float32),
            "failure_penalty": np.asarray(failure, dtype=np.float32),
            "total": np.asarray(total, dtype=np.float32),
        }


class ScissorsGoalWrapper(gym.Wrapper):
    """Goal-conditioned scissors reward shared by every scissors variant.

    Scissors needs a task-specific reward because its one physical DoF spans a
    much wider range than Push or Spray.  The formula and all default weights
    are nevertheless identical for variants 1--5.  A broad tracking kernel
    supplies a learning signal far from the target, a narrow kernel supplies
    stopping precision, and progress is paid only when the error decreases on
    the current transition (rather than forever after partial progress).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        target_min_fraction: float = 0.0,
        target_max_fraction: float = 1.0,
        success_hold_steps: int = 5,
        coarse_tracking_reward_weight: float = 1.0,
        fine_tracking_reward_weight: float = 1.0,
        coarse_tracking_sigma: float = 0.35,
        progress_reward_weight: float = 1.0,
        grasp_reward_weight: float = 1.0,
        interaction_reward_weight: float = 0.25,
        success_bonus: float = 5.0,
        failure_penalty: float = 25.0,
        grasp_gate_floor: float = 0.25,
        control_delta_penalty_weight: float = 0.1,
        control_magnitude_penalty_weight: float = 0.002,
        seed: int | None = None,
    ):
        super().__init__(env)
        if type(env.unwrapped).__name__ != "ScissorsManipulateEnv":
            raise TypeError("ScissorsGoalWrapper requires ScissorsManipulateEnv")
        values = (
            coarse_tracking_reward_weight,
            fine_tracking_reward_weight,
            coarse_tracking_sigma,
            progress_reward_weight,
            grasp_reward_weight,
            interaction_reward_weight,
            success_bonus,
            failure_penalty,
            control_delta_penalty_weight,
            control_magnitude_penalty_weight,
        )
        if (
            success_hold_steps < 1
            or not 0.0 <= grasp_gate_floor <= 1.0
            or not 0.0 <= target_min_fraction < target_max_fraction <= 1.0
        ):
            raise ValueError("invalid scissors reward configuration")
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("scissors reward weights must be finite and non-negative")
        self.success_hold_steps = int(success_hold_steps)
        self.target_min_fraction = float(target_min_fraction)
        self.target_max_fraction = float(target_max_fraction)
        self._rng = np.random.default_rng(seed)
        self.coarse_tracking_reward_weight = float(coarse_tracking_reward_weight)
        self.fine_tracking_reward_weight = float(fine_tracking_reward_weight)
        self.coarse_tracking_sigma = float(coarse_tracking_sigma)
        self.progress_reward_weight = float(progress_reward_weight)
        self.grasp_reward_weight = float(grasp_reward_weight)
        self.interaction_reward_weight = float(interaction_reward_weight)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.grasp_gate_floor = float(grasp_gate_floor)
        self.control_delta_penalty_weight = float(control_delta_penalty_weight)
        self.control_magnitude_penalty_weight = float(control_magnitude_penalty_weight)
        self._success_streak = 0
        self._previous_action: np.ndarray | None = None
        self._initial_achieved = 0.0
        self._previous_achieved = 0.0

    def set_target_max_fraction(self, maximum: float):
        if not self.target_min_fraction < maximum <= 1:
            raise ValueError("invalid curriculum maximum")
        self.target_max_fraction = float(maximum)
        return self.target_max_fraction

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        reset_options = dict(options or {})
        if "target_angle" not in reset_options:
            low, high = (float(value) for value in self.env.unwrapped.target_angle_range)
            reset_options["target_angle"] = float(
                low
                + self._rng.uniform(self.target_min_fraction, self.target_max_fraction)
                * (high - low)
            )
        observation, info = self.env.reset(seed=seed, options=reset_options)
        self._success_streak = 0
        self._previous_action = None
        self._initial_achieved = float(
            np.asarray(observation["achieved_goal"]).reshape(-1)[0]
        )
        self._previous_achieved = self._initial_achieved
        return observation, self._augment_info(info)

    def step(self, action: np.ndarray):
        observation, _, terminated, truncated, info = self.env.step(action)
        current = np.asarray(action, dtype=np.float32)
        previous = current if self._previous_action is None else self._previous_action
        info = dict(info)
        info["reward_control_delta_cost"] = float(
            np.mean(np.square(current - previous)) / 4.0
        )
        info["reward_control_magnitude_cost"] = float(np.mean(np.square(current)))
        self._previous_action = current.copy()
        info["previous_achieved_goal"] = self._previous_achieved
        info = self._augment_info(info)
        # Push and Spray terminate on loss of the tool. Do the same here so the
        # common failure penalty is charged once instead of at every later step.
        terminated = bool(terminated or not info["grasp_maintained"])
        terms = self._reward_terms(
            observation["achieved_goal"], observation["desired_goal"], info
        )
        for name, values in terms.items():
            info[f"reward_{name}"] = float(np.asarray(values).reshape(-1)[0])
        self._previous_achieved = float(
            np.asarray(observation["achieved_goal"]).reshape(-1)[0]
        )
        return observation, info["reward_total"], terminated, truncated, info

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        result = dict(info)
        grasped = bool(result.get("is_in_grasp_space", False))
        success = bool(result.get("is_success", False)) and grasped
        self._success_streak = self._success_streak + 1 if success else 0
        result["grasp_maintained"] = float(grasped)
        result["success_streak"] = self._success_streak
        result["is_stable_success"] = float(
            self._success_streak >= self.success_hold_steps
        )
        result["success_hold_steps"] = self.success_hold_steps
        result["initial_achieved_goal"] = self._initial_achieved
        return result

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._reward_terms(achieved_goal, desired_goal, info)["total"]

    def _reward_terms(self, achieved_goal, desired_goal, info):
        achieved = np.asarray(achieved_goal, dtype=np.float32).reshape(-1)
        desired = np.asarray(desired_goal, dtype=np.float32).reshape(-1)
        infos = [info] * len(achieved) if isinstance(info, dict) else list(info)
        tolerance = max(float(self.env.unwrapped.angle_threshold), np.finfo(np.float32).eps)
        error = np.abs(achieved - desired)
        previous = np.asarray(
            [
                float(item.get("previous_achieved_goal", achieved[index]))
                for index, item in enumerate(infos)
            ],
            dtype=np.float32,
        )
        grasp_score = np.asarray(
            [float(item.get("is_in_grasp_space", 0.0)) for item in infos], dtype=np.float32
        )
        interaction_score = np.asarray(
            [float(item.get("keep_position", 0.0)) for item in infos], dtype=np.float32
        )
        delta_cost = np.asarray(
            [float(item.get("reward_control_delta_cost", 0.0)) for item in infos], dtype=np.float32
        )
        magnitude_cost = np.asarray(
            [
                float(item.get("reward_control_magnitude_cost", 0.0))
                for item in infos
            ],
            dtype=np.float32,
        )
        fine_tracking_score = np.exp(-0.5 * np.square(error / tolerance))
        coarse_tracking_score = np.exp(
            -0.5 * np.square(error / self.coarse_tracking_sigma)
        )
        previous_error = np.abs(previous - desired)
        target_low, target_high = (
            float(value) for value in self.env.unwrapped.target_angle_range
        )
        target_span = max(target_high - target_low, tolerance)
        # Signed potential difference: partial progress is rewarded once, while
        # regression is penalised.  This remains HER-compatible because the
        # previous achieved state is copied in ``info``.
        progress_score = np.clip((previous_error - error) / target_span, -1.0, 1.0)
        grasp_gate = self.grasp_gate_floor + (1.0 - self.grasp_gate_floor) * grasp_score
        success = (error <= tolerance) & (grasp_score > 0.5)
        failed = grasp_score <= 0.5
        coarse_tracking = (
            self.coarse_tracking_reward_weight * coarse_tracking_score * grasp_gate
        )
        fine_tracking = (
            self.fine_tracking_reward_weight * fine_tracking_score * grasp_gate
        )
        tracking = coarse_tracking + fine_tracking
        progress = self.progress_reward_weight * progress_score * grasp_gate
        grasp = self.grasp_reward_weight * grasp_score
        interaction = self.interaction_reward_weight * interaction_score
        success_reward = self.success_bonus * success.astype(np.float32)
        control_delta = -self.control_delta_penalty_weight * delta_cost
        control_magnitude = -self.control_magnitude_penalty_weight * magnitude_cost
        failure = -self.failure_penalty * failed.astype(np.float32)
        total = (
            tracking
            + progress
            + grasp
            + interaction
            + success_reward
            + control_delta
            + control_magnitude
        )
        total = np.where(failed, failure, total)
        return {
            "tracking": np.asarray(tracking, dtype=np.float32),
            "coarse_tracking": np.asarray(coarse_tracking, dtype=np.float32),
            "fine_tracking": np.asarray(fine_tracking, dtype=np.float32),
            "progress": np.asarray(progress, dtype=np.float32),
            "grasp": np.asarray(grasp, dtype=np.float32),
            "interaction": np.asarray(interaction, dtype=np.float32),
            "success_bonus": np.asarray(success_reward, dtype=np.float32),
            "control_delta_penalty": np.asarray(control_delta, dtype=np.float32),
            "control_magnitude_penalty": np.asarray(control_magnitude, dtype=np.float32),
            "failure_penalty": np.asarray(failure, dtype=np.float32),
            "total": np.asarray(total, dtype=np.float32),
        }
