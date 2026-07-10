from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from improved_tds.control.base import ControlObservation, ControllerOutput
from improved_tds.control.feedback import ToolStateFeedbackController


@dataclass(frozen=True)
class AdmittanceConfig:
    mass: float = 1.0
    damping: float = 4.0
    tool_state_gain: float = 1.0
    force_gain: float = 0.1
    desired_reaction: float = 0.0
    force_filter_time_constant: float = 0.02

    def __post_init__(self) -> None:
        if self.mass <= 0.0 or self.damping < 0.0:
            raise ValueError("admittance mass must be positive and damping non-negative")
        if self.force_filter_time_constant < 0.0:
            raise ValueError("force filter time constant must be non-negative")


class AdmittanceTDSController(ToolStateFeedbackController):
    """TDS-level admittance with safe degradation to tool-state feedback.

    With one actuation DoF, tool position and reaction force cannot be controlled
    independently. `desired_reaction` is therefore a compliant reaction target or
    safety preference, not a guaranteed force setpoint.
    """

    def __init__(self, *args: object, admittance: AdmittanceConfig | None = None, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.admittance = admittance or AdmittanceConfig()
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.admittance_rho: float | None = None
        self.admittance_rate = 0.0
        self.filtered_force: float | None = None

    def _synergy_force(self, observation: ControlObservation) -> float | None:
        if observation.synergy_force is not None:
            return float(observation.synergy_force)
        if observation.joint_torque is None:
            return None
        torque = np.asarray(observation.joint_torque, dtype=np.float64).reshape(-1)
        direction = np.asarray(self.estimator.direction_, dtype=np.float64).reshape(-1)
        if torque.shape != direction.shape:
            return None
        return float(direction @ torque)

    def step(
        self,
        target_tool_state: float,
        observation: ControlObservation,
        dt: float,
        *,
        phase: str | None = None,
    ) -> ControllerOutput:
        force = self._synergy_force(observation)
        if force is None or not np.isfinite(force):
            output = super().step(target_tool_state, observation, dt, phase=phase)
            output.mode = "admittance_force_dropout_feedback"
            output.diagnostics["force_dropout"] = True
            return output
        violation = self.safety.violation(observation)
        if violation is not None:
            return self.release_output(violation)
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")

        tau = self.admittance.force_filter_time_constant
        alpha = 1.0 if tau == 0.0 else dt / (tau + dt)
        if self.filtered_force is None:
            self.filtered_force = force
        else:
            self.filtered_force += alpha * (force - self.filtered_force)
        rho_ff = self.feedforward_rho(target_tool_state, phase)
        if self.admittance_rho is None:
            self.admittance_rho = rho_ff
        state_error = (
            0.0
            if not observation.sensor_valid or observation.tool_state is None
            else target_tool_state - observation.tool_state
        )
        reaction_error = self.admittance.desired_reaction - self.filtered_force
        acceleration = (
            self.admittance.tool_state_gain * state_error
            + self.admittance.force_gain * reaction_error
            - self.admittance.damping * self.admittance_rate
        ) / self.admittance.mass
        # Semi-implicit Euler is dissipative for the velocity term and more robust
        # than explicit position-first integration at the same sampling period.
        self.admittance_rate += acceleration * dt
        desired = self.admittance_rho + self.admittance_rate * dt
        rho, rho_saturated = self.safety.limit_rho(desired, dt)
        self.admittance_rate = (rho - self.admittance_rho) / dt
        self.admittance_rho = rho
        q, q_saturated = self.command_from_rho(rho)
        return ControllerOutput(
            q_command=q,
            rho_command=rho,
            mode="tds_admittance",
            saturated=rho_saturated or q_saturated,
            diagnostics={
                "tool_state_error": state_error,
                "synergy_force": self.filtered_force,
                "reaction_error": reaction_error,
                "rho_rate": self.admittance_rate,
                "rho_feedforward": rho_ff,
            },
        )

