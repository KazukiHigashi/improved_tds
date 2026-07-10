from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from improved_tds.environments.base import ArticulatedToolEnv
from improved_tds.environments.legacy_scissors import ExpScissorEnv as LegacyExpScissorEnv
from improved_tds.environments.mujoco_utils import name2id, robot_get_obs


class ExpScissorEnv(LegacyExpScissorEnv, ArticulatedToolEnv):
    """Backward-compatible Shadow Hand scissors with articulated-tool feedback API."""

    def __init__(
        self,
        *args: Any,
        hinge_friction: float | None = None,
        hinge_damping: float | None = None,
        resistance_torque: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tool_family = "scissors"
        self.tool_instance_id = f"scissors-{self.variant}"
        self.resistance_torque = float(resistance_torque)
        parameters: dict[str, float] = {}
        if hinge_friction is not None:
            parameters["hinge_friction"] = float(hinge_friction)
        if hinge_damping is not None:
            parameters["hinge_damping"] = float(hinge_damping)
        if parameters:
            self.set_tool_parameters(parameters)

    @property
    def tool_state_limits(self) -> np.ndarray:
        joint = name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "scissors_hinge_2:joint")
        return self.model.jnt_range[joint].copy()

    @property
    def phase(self) -> str:
        rate = float(self.get_tool_state_rate()[0])
        if abs(rate) < 1e-5:
            return "hold"
        return "open" if rate > 0.0 else "close"

    def get_tool_state(self) -> np.ndarray:
        return self._get_achieved_goal().astype(np.float64).reshape(1)

    def get_tool_state_rate(self) -> np.ndarray:
        joint = name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "scissors_hinge_2:joint")
        dof = int(self.model.jnt_dofadr[joint])
        return np.asarray([self.data.qvel[dof]], dtype=np.float64)

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q, qd = robot_get_obs(self.model, self.data)
        return q.astype(np.float64), qd.astype(np.float64)

    def get_force_observation(self) -> dict[str, np.ndarray | float]:
        contact_forces = self._touch_forces().astype(np.float64)
        actuator_force = self.data.actuator_force.copy().astype(np.float64)
        center = np.mean(self.model.actuator_ctrlrange, axis=1)
        direction = np.sign(self.data.ctrl - center)
        synergy_force = float(np.mean(actuator_force * direction)) if actuator_force.size else 0.0
        return {
            "joint_torque": actuator_force,
            "mujoco_generalized_force": self.data.qfrc_actuator.copy(),
            "synergy_force": synergy_force,
            "contact_flags": contact_forces > 0.0,
            "contact_forces": contact_forces,
        }

    def set_tool_parameters(self, params: dict[str, float]) -> None:
        allowed = {"hinge_friction", "hinge_damping", "resistance_torque"}
        unknown = set(params).difference(allowed)
        if unknown:
            raise KeyError(f"unsupported runtime scissors parameters: {sorted(unknown)}")
        for joint_name in ("scissors_hinge_1:joint", "scissors_hinge_2:joint"):
            joint = name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            dof = int(self.model.jnt_dofadr[joint])
            if "hinge_friction" in params:
                self.model.dof_frictionloss[dof] = float(params["hinge_friction"])
            if "hinge_damping" in params:
                self.model.dof_damping[dof] = float(params["hinge_damping"])
        if "resistance_torque" in params:
            self.resistance_torque = float(params["resistance_torque"])

    def _set_action(self, action: np.ndarray) -> None:
        super()._set_action(action)
        if not hasattr(self, "resistance_torque"):
            return
        joint = name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "scissors_hinge_2:joint")
        dof = int(self.model.jnt_dofadr[joint])
        rate = float(self.data.qvel[dof])
        self.data.qfrc_applied[dof] = -self.resistance_torque * np.sign(rate)

    def _get_info(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        info = super()._get_info(obs)
        force_observation = self.get_force_observation()
        info.update(
            {
                "tool_state": self.get_tool_state(),
                "tool_state_rate": self.get_tool_state_rate(),
                "actuation_force_proxy": float(force_observation["synergy_force"]),
                "contact_flags": force_observation["contact_flags"],
                "phase": self.phase,
            }
        )
        return info

