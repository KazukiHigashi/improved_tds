from __future__ import annotations

from dataclasses import asdict
from typing import Any

from gymnasium import spaces
import mujoco
import numpy as np

from improved_tds.environments.base import ArticulatedToolEnv


class OneDoFToolEnv(ArticulatedToolEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        family: str,
        instance_id: str,
        joint_type: str,
        travel: float,
        stiffness: float,
        damping: float,
        friction: float,
        preload: float,
        mass: float,
        force_limit: float,
        actuation_threshold: float,
        task: str,
        target_range: tuple[float, float],
        max_episode_steps: int,
        n_substeps: int,
        render_mode: str | None,
        geom_size: tuple[float, float, float],
    ):
        if joint_type not in {"hinge", "slide"}:
            raise ValueError("joint_type must be hinge or slide")
        if travel <= 0.0 or stiffness < 0.0 or damping < 0.0 or friction < 0.0:
            raise ValueError("invalid one-DoF physical parameters")
        if force_limit <= 0.0 or mass <= 0.0 or n_substeps <= 0:
            raise ValueError("mass, force limit, and substeps must be positive")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("unsupported render mode")
        self.tool_family = family
        self.tool_instance_id = instance_id
        self.joint_type = joint_type
        self.travel = float(travel)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.friction = float(friction)
        self.preload = float(preload)
        self.mass = float(mass)
        self.force_limit = float(force_limit)
        self.actuation_threshold = float(actuation_threshold)
        self.task = str(task)
        self.target_range = np.asarray(target_range, dtype=np.float64)
        self.max_episode_steps = int(max_episode_steps)
        self.n_substeps = int(n_substeps)
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None
        self._geom_size = geom_size

        self.model = mujoco.MjModel.from_xml_string(self._xml())
        self.data = mujoco.MjData(self.model)
        self._joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tool_joint")
        self._dof = int(self.model.jnt_dofadr[self._joint_id])
        self._qpos = int(self.model.jnt_qposadr[self._joint_id])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.goal = np.zeros(1, dtype=np.float32)
        self.step_count = 0
        self._last_force = 0.0

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.n_substeps)

    def _xml(self) -> str:
        axis = "0 1 0" if self.joint_type == "hinge" else "0 0 -1"
        geom_type = "capsule" if self.joint_type == "hinge" else "cylinder"
        size_count = 2 if geom_type in {"capsule", "cylinder"} else 3
        size = " ".join(str(value) for value in self._geom_size[:size_count])
        return f"""
<mujoco model="{self.tool_family}">
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 -1 1"/>
    <body name="tool_body" pos="0 0 0.1">
      <joint name="tool_joint" type="{self.joint_type}" axis="{axis}"
             range="0 {self.travel}" limited="true" damping="{self.damping}"
             frictionloss="{self.friction}" stiffness="{self.stiffness}" springref="0"/>
      <geom name="tool_geom" type="{geom_type}" size="{size}" mass="{self.mass}" rgba="0.2 0.5 0.8 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="tool_motor" joint="tool_joint" ctrllimited="true"
           ctrlrange="{-self.force_limit} {self.force_limit}"/>
  </actuator>
</mujoco>
"""

    @property
    def tool_state_limits(self) -> np.ndarray:
        return np.asarray([0.0, self.travel], dtype=np.float64)

    @property
    def phase(self) -> str:
        rate = float(self.data.qvel[self._dof])
        if abs(rate) < 1e-5:
            return "hold"
        return "press" if rate > 0.0 else "release"

    def get_tool_state(self) -> np.ndarray:
        return np.asarray([self.data.qpos[self._qpos]], dtype=np.float64)

    def get_tool_state_rate(self) -> np.ndarray:
        return np.asarray([self.data.qvel[self._dof]], dtype=np.float64)

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_tool_state(), self.get_tool_state_rate()

    def get_force_observation(self) -> dict[str, np.ndarray | float]:
        actuator = float(self.data.qfrc_actuator[self._dof])
        constraint = float(self.data.qfrc_constraint[self._dof])
        return {
            "joint_torque": np.asarray([actuator], dtype=np.float64),
            "mujoco_generalized_force": np.asarray([actuator + constraint], dtype=np.float64),
            "synergy_force": actuator + constraint,
            "contact_flags": np.asarray([self.data.ncon > 0], dtype=np.bool_),
            "contact_forces": np.asarray([abs(constraint)], dtype=np.float64),
        }

    def set_tool_parameters(self, params: dict[str, float]) -> None:
        allowed = {"stiffness", "damping", "friction", "preload", "mass", "actuation_threshold"}
        unknown = set(params).difference(allowed)
        if unknown:
            raise KeyError(f"unsupported runtime tool parameters: {sorted(unknown)}")
        if "stiffness" in params:
            self.model.jnt_stiffness[self._joint_id] = float(params["stiffness"])
        if "damping" in params:
            self.model.dof_damping[self._dof] = float(params["damping"])
        if "friction" in params:
            self.model.dof_frictionloss[self._dof] = float(params["friction"])
        if "mass" in params:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool_body")
            self.model.body_mass[body] = float(params["mass"])
        if "preload" in params:
            self.preload = float(params["preload"])
        if "actuation_threshold" in params:
            self.actuation_threshold = float(params["actuation_threshold"])
        mujoco.mj_setConst(self.model, self.data)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        options = options or {}
        target = (
            float(options["target"])
            if "target" in options
            else float(self.np_random.uniform(*self.target_range))
        )
        if not self.tool_state_limits[0] <= target <= self.tool_state_limits[1]:
            raise ValueError("target is outside tool-state limits")
        self.goal[:] = target
        self._last_force = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        command = np.asarray(action, dtype=np.float32).reshape(-1)
        if command.shape != (1,):
            raise ValueError("action must have shape (1,)")
        self.data.ctrl[0] = float(np.clip(command[0], -1.0, 1.0)) * self.force_limit + self.preload
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self._last_force = float(self.get_force_observation()["synergy_force"])
        obs = self._observation()
        info = self._info()
        error = abs(float(obs["achieved_goal"][0] - obs["desired_goal"][0]))
        reward = -error
        success = bool(info["is_success"])
        repeated = self.task in {"repeated", "repeated_press_release", "repeated_squeeze_release"}
        terminated = success and not repeated
        if success and repeated:
            self.goal[:] = 0.0 if self.goal[0] > self.travel / 2.0 else self.target_range[1]
        truncated = self.step_count >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def _observation(self) -> dict[str, np.ndarray]:
        state = self.get_tool_state().astype(np.float32)
        rate = self.get_tool_state_rate().astype(np.float32)
        force = np.asarray([self._last_force], dtype=np.float32)
        return {
            "observation": np.concatenate([state, rate, force]),
            "achieved_goal": state.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _info(self) -> dict[str, Any]:
        state = float(self.get_tool_state()[0])
        error = abs(state - float(self.goal[0]))
        return {
            "is_success": float(error < max(self.travel * 0.02, 1e-4) and abs(self._last_force) <= self.force_limit),
            "tool_state": self.get_tool_state().copy(),
            "tool_state_rate": self.get_tool_state_rate().copy(),
            "tool_state_error": error,
            "actuation_threshold_reached": float(state >= self.actuation_threshold),
            "actuation_force_proxy": self._last_force,
            "phase": self.phase,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=320, width=320)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mujoco_viewer

                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def parameter_dict(self) -> dict[str, Any]:
        parameters = getattr(self, "parameters", None)
        return asdict(parameters) if parameters is not None else {}

