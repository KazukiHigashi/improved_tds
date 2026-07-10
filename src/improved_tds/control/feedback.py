from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from improved_tds.control.base import ControlObservation, ControllerOutput
from improved_tds.control.feedforward import FeedforwardTDSController


@dataclass(frozen=True)
class FeedbackConfig:
    kp: float = 0.5
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.25
    derivative_filter_time_constant: float = 0.02

    def __post_init__(self) -> None:
        if self.ki < 0.0 or self.integral_limit < 0.0:
            raise ValueError("ki and integral_limit must be non-negative")
        if self.derivative_filter_time_constant < 0.0:
            raise ValueError("derivative filter time constant must be non-negative")


class ToolStateFeedbackController(FeedforwardTDSController):
    """Tool-state PID correction with filtered derivative and anti-windup.

    A missing/invalid tool-state sensor degrades to the feedforward controller.
    Integral action is disabled by default.
    """

    def __init__(self, *args: object, config: FeedbackConfig | None = None, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.config = config or FeedbackConfig()
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.integral_error = 0.0
        self.previous_error: float | None = None
        self.filtered_derivative = 0.0

    def step(
        self,
        target_tool_state: float,
        observation: ControlObservation,
        dt: float,
        *,
        phase: str | None = None,
    ) -> ControllerOutput:
        violation = self.safety.violation(observation)
        if violation is not None:
            return self.release_output(violation)
        if not observation.sensor_valid or observation.tool_state is None:
            output = super().step(target_tool_state, observation, dt, phase=phase)
            output.mode = "feedback_dropout_feedforward"
            output.diagnostics["sensor_dropout"] = True
            return output
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")

        rho_ff = self.feedforward_rho(target_tool_state, phase)
        error = float(target_tool_state - observation.tool_state)
        raw_derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        tau = self.config.derivative_filter_time_constant
        alpha = 1.0 if tau == 0.0 else dt / (tau + dt)
        self.filtered_derivative += alpha * (raw_derivative - self.filtered_derivative)
        candidate_integral = float(
            np.clip(
                self.integral_error + error * dt,
                -self.config.integral_limit,
                self.config.integral_limit,
            )
        )
        desired = (
            rho_ff
            + self.config.kp * error
            + self.config.ki * candidate_integral
            + self.config.kd * self.filtered_derivative
        )
        rho, rho_saturated = self.safety.limit_rho(desired, dt)
        # Conditional integration: keep integrating when unsaturated, or when the
        # error drives the command back toward the reachable range.
        if not rho_saturated or (desired - rho) * error <= 0.0:
            self.integral_error = candidate_integral
        self.previous_error = error
        q, q_saturated = self.command_from_rho(rho)
        return ControllerOutput(
            q_command=q,
            rho_command=rho,
            mode="tool_state_feedback",
            saturated=rho_saturated or q_saturated,
            diagnostics={
                "tool_state_error": error,
                "rho_feedforward": rho_ff,
                "integral_error": self.integral_error,
                "filtered_derivative": self.filtered_derivative,
            },
        )

