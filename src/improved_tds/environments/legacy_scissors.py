from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from improved_tds.environments.mujoco_utils import (
    actuator_names,
    get_joint_qpos,
    name2id,
    robot_get_obs,
    set_joint_qpos,
)


INIT_JOINT_NAMES = (
    "robot0:WRJ0",
    "robot0:FFJ3",
    "robot0:FFJ2",
    "robot0:FFJ1",
    "robot0:FFJ0",
    "robot0:MFJ3",
    "robot0:MFJ2",
    "robot0:MFJ1",
    "robot0:MFJ0",
    "robot0:RFJ3",
    "robot0:RFJ2",
    "robot0:RFJ1",
    "robot0:RFJ0",
    "robot0:THJ4",
    "robot0:THJ3",
    "robot0:THJ2",
    "robot0:THJ1",
    "robot0:THJ0",
)

INIT_ANGLES_4_FINGER = np.array(
    [
        0.0,
        0.0,
        1.57,
        0.0,
        0.0,
        0.0,
        1.57,
        0.0,
        0.0,
        0.0,
        1.57,
        0.0,
        0.0,
        0.115,
        1.22,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
INIT_ANGLES_3_FINGER = INIT_ANGLES_4_FINGER.copy()
INIT_ANGLES_3_FINGER[9:13] = 0.0
INIT_ANGLES_2_FINGER = INIT_ANGLES_3_FINGER.copy()
INIT_ANGLES_2_FINGER[5:9] = 0.0


@dataclass(frozen=True)
class ScissorVariant:
    xml_name: str
    init_angles: np.ndarray
    grasp_threshold: float = 0.05
    grasp_radius: float = 0.05


SCISSOR_VARIANTS = {
    1: ScissorVariant("exp_scissors_1.xml", INIT_ANGLES_4_FINGER, grasp_threshold=0.04),
    2: ScissorVariant("exp_scissors_2.xml", INIT_ANGLES_4_FINGER, grasp_threshold=0.04),
    3: ScissorVariant("exp_scissors_3.xml", INIT_ANGLES_4_FINGER, grasp_threshold=0.04),
    4: ScissorVariant("exp_scissors_4.xml", INIT_ANGLES_3_FINGER, grasp_threshold=0.04),
    5: ScissorVariant("exp_scissors_5.xml", INIT_ANGLES_2_FINGER, grasp_threshold=0.04),
}


class ExpScissorEnv(gym.Env):
    """Shadow Hand Lite scissors task ported from mujoco-py to MuJoCo's official API."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        variant: int = 1,
        render_mode: str | None = None,
        reward_type: str = "standard",
        n_substeps: int = 20,
        angle_threshold: float = 0.01,
        relative_control: bool = False,
        target_angle_range: tuple[float, float] = (0.0, 1.0),
        initial_object_qpos: tuple[float, ...] = (1.07, 0.892, 0.4, 1.0, 0.0, 0.0, 0.0),
        hold_initial_steps: int = 30,
        success_hold_steps: int = 5,
        grasp_threshold: float | None = None,
        grasp_radius: float | None = None,
        show_goal_marker: bool = True,
        show_grasp_marker: bool = True,
    ):
        if variant not in SCISSOR_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(SCISSOR_VARIANTS)}")
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None")
        valid_reward_types = {"standard", "sparse", "dense", "synergy", "synergy2"}
        if reward_type not in valid_reward_types:
            raise ValueError(f"reward_type must be one of {sorted(valid_reward_types)}")
        if hold_initial_steps < 0:
            raise ValueError("hold_initial_steps must be non-negative")
        if success_hold_steps < 1:
            raise ValueError("success_hold_steps must be positive")

        self.variant = variant
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.n_substeps = int(n_substeps)
        self.angle_threshold = float(angle_threshold)
        self.relative_control = bool(relative_control)
        self.target_angle_range = np.asarray(target_angle_range, dtype=np.float64)
        self.init_object_qpos = np.asarray(initial_object_qpos, dtype=np.float64)
        self.hold_initial_steps = int(hold_initial_steps)
        self.success_hold_steps = int(success_hold_steps)
        self.grasp_threshold = float(
            SCISSOR_VARIANTS[variant].grasp_threshold
            if grasp_threshold is None
            else grasp_threshold
        )
        self.grasp_radius = float(
            SCISSOR_VARIANTS[variant].grasp_radius if grasp_radius is None else grasp_radius
        )
        self.show_goal_marker = bool(show_goal_marker)
        self.show_grasp_marker = bool(show_grasp_marker)
        self.step_n = 0
        self.success_streak = 0
        self.goal = np.zeros(1, dtype=np.float32)
        self.initial_qpos: np.ndarray | None = None
        self._fixed_grasp_center = self.init_object_qpos[:3].copy()
        self._viewer = None
        self._renderer = None

        xml_path = self._asset_path("exp_scissors", SCISSOR_VARIANTS[variant].xml_name)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._actuator_names = actuator_names(self.model)
        self._setup_initial_configuration()
        self._initial_data_qpos = self.data.qpos.copy()
        self._initial_data_qvel = self.data.qvel.copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32
                ),
            }
        )

    @staticmethod
    def _asset_path(*parts: str) -> Path:
        return Path(resources.files("improved_tds").joinpath("assets", *parts))

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.n_substeps)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        self._reset_sim()
        self._settle_initial_pose()
        self._capture_fixed_grasp_center()
        if "target" in options:
            target = float(options["target"])
            if not self.target_angle_range[0] <= target <= self.target_angle_range[1]:
                raise ValueError("target is outside target_angle_range")
            self.goal = np.asarray([target], dtype=np.float32)
        else:
            self.goal = self._sample_goal().astype(np.float32)
        obs = self._get_obs()
        info = self._get_info(obs)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: np.ndarray):
        self.step_n += 1
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_callback()
        obs = self._get_obs()
        info = self._get_info(obs)
        if bool(info["is_success"]):
            self.success_streak += 1
        else:
            self.success_streak = 0
        stable_success = self.success_streak >= self.success_hold_steps
        info.update(
            {
                "success_streak": self.success_streak,
                "is_stable_success": float(stable_success),
                "termination_reason": "running",
            }
        )
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self.render()
        return obs, float(np.asarray(reward).reshape(-1)[0]), terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        angle_error = np.abs(achieved - desired)
        success = (angle_error < self.angle_threshold).astype(np.float32)

        batch_size = success.reshape(-1).shape[0]
        grasp = self._info_values(info, "is_in_grasp_space", batch_size, default=1.0)
        success = success.reshape(-1) * grasp

        if self.reward_type in {"standard", "sparse"}:
            reward = success - 1.0
        elif self.reward_type == "synergy":
            error = self._info_values(info, "synergy_error", batch_size, default=0.0)
            scale = self._info_values(info, "lambda", batch_size, default=1.0)
            reward = success * np.exp(-(scale * error * error)) - 1.0
        elif self.reward_type == "synergy2":
            error = self._info_values(info, "synergy_error", batch_size, default=0.0)
            scale = self._info_values(info, "lambda", batch_size, default=1.0)
            reward = success + np.exp(-(scale * error * error)) - 2.0
        else:
            reward = -angle_error.reshape(-1)
            reward -= self._info_values(info, "contact_penalty", batch_size, default=0.0)
        return reward.astype(np.float32)

    @staticmethod
    def _info_values(info, key: str, batch_size: int, default: float) -> np.ndarray:
        if isinstance(info, dict):
            value = info.get(key, info.get("e" if key == "synergy_error" else key, default))
            values = np.asarray(value, dtype=np.float32).reshape(-1)
        elif isinstance(info, (list, tuple, np.ndarray)):
            extracted = []
            for item in info:
                if isinstance(item, dict):
                    extracted.append(item.get(key, item.get("e" if key == "synergy_error" else key, default)))
                else:
                    extracted.append(default)
            values = np.asarray(extracted, dtype=np.float32).reshape(-1)
        else:
            values = np.asarray([default], dtype=np.float32)

        if values.size == 1 and batch_size != 1:
            values = np.full(batch_size, float(values[0]), dtype=np.float32)
        return values[:batch_size]

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=500, width=500)
            self._renderer.update_scene(self.data, camera="side")
            self._add_grasp_threshold_marker_to_scene(self._renderer.scene)
            self._add_goal_angle_marker_to_scene(self._renderer.scene)
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mujoco_viewer

                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
                self._setup_viewer_camera()
            self._update_viewer_user_scene()
            self._viewer.sync()
            return None
        return None

    def _update_viewer_user_scene(self) -> None:
        if self._viewer is None:
            return
        scene = getattr(self._viewer, "user_scn", None)
        if scene is None:
            return

        def update_scene() -> None:
            scene.ngeom = 0
            self._add_grasp_threshold_marker_to_scene(scene)
            self._add_goal_angle_marker_to_scene(scene)

        lock = getattr(self._viewer, "lock", None)
        if callable(lock):
            with lock():
                update_scene()
        else:
            update_scene()

    def _add_goal_angle_marker_to_scene(self, scene) -> None:
        if not self.show_goal_marker:
            return
        if scene.ngeom >= scene.maxgeom:
            return

        base_site = name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "scissors2:center")
        base_pos = self.data.site_xpos[base_site].copy()
        rot_scissor = Rotation.from_matrix(self.data.site_xmat[base_site].reshape(3, 3))
        target_angle = float(self.goal.reshape(-1)[0])
        arrow_vec_local = np.array([np.cos(target_angle), -np.sin(target_angle), 0.0])
        arrow_vec_world = rot_scissor.apply(arrow_vec_local)
        norm = np.linalg.norm(arrow_vec_world)
        if norm <= 1e-8:
            return
        arrow_vec_world = arrow_vec_world / norm
        marker_rot = Rotation.align_vectors([arrow_vec_world], [[0.0, 0.0, 1.0]])[0]

        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.array([0.002, 0.002, 0.2], dtype=np.float64),
            base_pos.astype(np.float64),
            marker_rot.as_matrix().reshape(-1).astype(np.float64),
            np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32),
        )
        scene.ngeom += 1

    def _add_grasp_threshold_marker_to_scene(self, scene) -> None:
        if not self.show_grasp_marker:
            return
        if scene.ngeom >= scene.maxgeom:
            return

        grasp_pos = self._grasp_center_space(radius=self.grasp_radius)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array(
                [self.grasp_threshold, self.grasp_threshold, self.grasp_threshold],
                dtype=np.float64,
            ),
            grasp_pos.astype(np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.array([0.0, 1.0, 0.0, 0.18], dtype=np.float32),
        )
        scene.ngeom += 1

    def _setup_initial_configuration(self) -> None:
        self._set_hand_initial_angles()
        mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_data_qpos
        self.data.qvel[:] = self._initial_data_qvel
        self.data.ctrl[:] = 0.0
        self._set_hand_initial_angles()
        self._set_object_qpos(self.init_object_qpos)
        self._set_hinge_angles(0.0)
        self.step_n = 0
        self.success_streak = 0
        mujoco.mj_forward(self.model, self.data)

    def _settle_initial_pose(self) -> None:
        self._hold_initial_pose()
        for _ in range(self.hold_initial_steps):
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _sample_goal(self) -> np.ndarray:
        low, high = self.target_angle_range
        offset = self.np_random.uniform(max(0.25, low), min(1.0, high))
        return np.array([1.0 - offset], dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        robot_qpos, robot_qvel = robot_get_obs(self.model, self.data)
        achieved_goal = self._get_achieved_goal().astype(np.float32).reshape(1)
        touch_forces = self._touch_forces()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal, touch_forces]).astype(
            np.float32
        )
        return {
            "observation": observation,
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.astype(np.float32).reshape(1).copy(),
        }

    def _get_info(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        grasp_distance = self._grasp_distance()
        is_in_grasp = float(grasp_distance < self.grasp_threshold)
        angle_error = float(np.abs(obs["achieved_goal"] - self.goal).reshape(-1)[0])
        is_angle_success = float(angle_error < self.angle_threshold)
        return {
            "is_success": float(self._is_success(obs["achieved_goal"], self.goal, is_in_grasp)[0]),
            "is_angle_success": is_angle_success,
            "contact_penalty": float(self._contact_penalty()),
            "is_in_grasp_space": is_in_grasp,
            "grasp_distance": grasp_distance,
            "grasp_threshold": self.grasp_threshold,
            "angle_error": angle_error,
            "achieved_goal": obs["achieved_goal"].copy(),
            "desired_goal": self.goal.copy(),
            "keep_position": float(self._keep_position()),
        }

    def _get_achieved_goal(self) -> np.ndarray:
        return get_joint_qpos(self.model, self.data, "scissors_hinge_2:joint")

    def _is_success(self, achieved_goal, desired_goal, is_in_grasp_space=1.0) -> np.ndarray:
        return (
            (np.abs(np.asarray(achieved_goal) - np.asarray(desired_goal)) < self.angle_threshold)
            .astype(np.float32)
            .reshape(-1)
            * float(is_in_grasp_space)
        )

    def _set_action(self, action: np.ndarray) -> None:
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")

        ctrlrange = self.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        if self.relative_control:
            center = np.zeros_like(action, dtype=np.float64)
            for idx, actuator_name in enumerate(self._actuator_names):
                joint_name = actuator_name.replace(":A_", ":")
                center[idx] = get_joint_qpos(self.model, self.data, joint_name)[0]
        else:
            center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0

        self.data.ctrl[:] = np.clip(center + action * actuation_range, ctrlrange[:, 0], ctrlrange[:, 1])

    def _set_hand_initial_angles(self) -> None:
        angles = SCISSOR_VARIANTS[self.variant].init_angles
        for joint_name, angle in zip(INIT_JOINT_NAMES, angles, strict=True):
            set_joint_qpos(self.model, self.data, joint_name, angle)

    def _set_object_qpos(self, qpos: np.ndarray) -> None:
        self.initial_qpos = np.asarray(qpos, dtype=np.float64).copy()
        set_joint_qpos(self.model, self.data, "scissors:joint", self.initial_qpos)

    def _set_hinge_angles(self, angle: float) -> None:
        set_joint_qpos(self.model, self.data, "scissors_hinge_1:joint", 0.0)
        set_joint_qpos(self.model, self.data, "scissors_hinge_2:joint", angle)

    def _hold_initial_pose(self) -> None:
        if self.initial_qpos is not None:
            self._set_object_qpos(self.initial_qpos)
        self._set_hinge_angles(0.0)
        self._set_hand_initial_angles()
        self.data.qvel[:] = 0.0
        for index, actuator_name in enumerate(self._actuator_names):
            joint_name = actuator_name.replace(":A_", ":")
            self.data.ctrl[index] = get_joint_qpos(self.model, self.data, joint_name)[0]
        mujoco.mj_forward(self.model, self.data)

    def _capture_fixed_grasp_center(self) -> None:
        self._fixed_grasp_center = self._scissors_marker_pos()

    def _step_callback(self) -> None:
        pass

    def _touch_forces(self) -> np.ndarray:
        if self.data.sensordata.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.data.sensordata.copy().astype(np.float32)

    def _contact_penalty(self) -> float:
        forces = self._touch_forces()
        return 0.1 if forces.size and np.max(forces) > 1.5 else 0.0

    def _keep_position(self, distance_threshold: float = 0.05) -> bool:
        obj_pos = self._scissors_center_pos()
        goal_pos = self.init_object_qpos[:3]
        return bool(np.linalg.norm(obj_pos - goal_pos) <= distance_threshold)

    def _is_in_grasp_space(self, radius: float | None = None) -> bool:
        return bool(self._grasp_distance(radius=radius) < self.grasp_threshold)

    def _grasp_distance(self, radius: float | None = None) -> float:
        grasp_pos = self._grasp_center_space(radius=self.grasp_radius if radius is None else radius)
        object_pos = self._scissors_marker_pos()
        return float(np.linalg.norm(grasp_pos - object_pos))

    def _grasp_center_space(self, radius: float = 0.07) -> np.ndarray:
        del radius
        return self._fixed_grasp_center.copy()

    def _scissors_center_pos(self) -> np.ndarray:
        scissors_site = name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "scissors:center")
        return self.data.site_xpos[scissors_site].copy()

    def _scissors_joint_pos(self) -> np.ndarray:
        scissors_site = name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "scissors2:center")
        return self.data.site_xpos[scissors_site].copy()

    def _scissors_marker_pos(self) -> np.ndarray:
        scissors_geom = name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "scissors")
        return self.data.geom_xpos[scissors_geom].copy()

    def _setup_viewer_camera(self) -> None:
        if self._viewer is None:
            return
        palm_body = name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:palm")
        self._viewer.cam.lookat[:] = self.data.xpos[palm_body]
        self._viewer.cam.distance = 0.5
        self._viewer.cam.azimuth = 55.0
        self._viewer.cam.elevation = -25.0
