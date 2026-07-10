from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ForceEstimator(Protocol):
    def estimate_joint_torque(self, observation: Any) -> np.ndarray | None: ...

    def estimate_synergy_force(
        self, observation: Any, direction: np.ndarray
    ) -> float | None: ...

    def reset_bias(self, observation: Any | None = None) -> None: ...


class DirectJointTorqueEstimator:
    def __init__(self, key: str = "joint_torque"):
        self.key = key
        self.bias: np.ndarray | None = None

    def estimate_joint_torque(self, observation: Any) -> np.ndarray | None:
        value = observation.get(self.key) if isinstance(observation, dict) else None
        if value is None:
            return None
        torque = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(torque)):
            return None
        return torque - (0.0 if self.bias is None else self.bias)

    def estimate_synergy_force(self, observation: Any, direction: np.ndarray) -> float | None:
        torque = self.estimate_joint_torque(observation)
        if torque is None:
            return None
        vector = np.asarray(direction, dtype=np.float64).reshape(-1)
        if torque.shape != vector.shape:
            raise ValueError("torque and TDS direction shapes do not match")
        return float(vector @ torque)

    def reset_bias(self, observation: Any | None = None) -> None:
        if observation is None:
            self.bias = None
            return
        value = observation.get(self.key) if isinstance(observation, dict) else None
        if value is None:
            raise ValueError("bias observation has no joint torque")
        self.bias = np.asarray(value, dtype=np.float64).reshape(-1).copy()


class MotorCurrentEstimator(DirectJointTorqueEstimator):
    """Joint torque from current using configured, never guessed, motor constants."""

    def __init__(
        self,
        torque_constants: np.ndarray,
        *,
        gear_ratios: np.ndarray | None = None,
        key: str = "motor_current",
    ):
        super().__init__(key=key)
        constants = np.asarray(torque_constants, dtype=np.float64).reshape(-1)
        if constants.size == 0 or np.any(constants <= 0.0):
            raise ValueError("positive motor torque constants must be provided")
        self.torque_constants = constants
        self.gear_ratios = (
            np.ones_like(constants)
            if gear_ratios is None
            else np.asarray(gear_ratios, dtype=np.float64).reshape(-1)
        )
        if self.gear_ratios.shape != constants.shape:
            raise ValueError("gear ratios must match torque constants")

    def estimate_joint_torque(self, observation: Any) -> np.ndarray | None:
        current = super().estimate_joint_torque(observation)
        if current is None:
            return None
        if current.shape != self.torque_constants.shape:
            raise ValueError("motor current shape does not match configured constants")
        return current * self.torque_constants * self.gear_ratios


class PDTrackingErrorEstimator(DirectJointTorqueEstimator):
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        super().__init__(key="unused")
        self.kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        self.kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        if self.kp.shape != self.kd.shape or np.any(self.kp < 0.0) or np.any(self.kd < 0.0):
            raise ValueError("PD gains must be non-negative arrays of equal shape")

    def estimate_joint_torque(self, observation: Any) -> np.ndarray | None:
        if not isinstance(observation, dict):
            return None
        required = ("q_command", "q", "qd_command", "qd")
        if any(key not in observation for key in required):
            return None
        values = [np.asarray(observation[key], dtype=np.float64).reshape(-1) for key in required]
        if any(value.shape != self.kp.shape for value in values):
            raise ValueError("PD observation shape does not match gains")
        q_command, q, qd_command, qd = values
        torque = self.kp * (q_command - q) + self.kd * (qd_command - qd)
        return torque - (0.0 if self.bias is None else self.bias)


class MockForceEstimator(DirectJointTorqueEstimator):
    def __init__(self, torque: np.ndarray | None = None):
        super().__init__()
        self.torque = None if torque is None else np.asarray(torque, dtype=np.float64).reshape(-1)

    def estimate_joint_torque(self, observation: Any) -> np.ndarray | None:
        del observation
        return None if self.torque is None else self.torque.copy()

