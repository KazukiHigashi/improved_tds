from __future__ import annotations

import numpy as np

from improved_tds.control.base import ControlObservation, ControllerOutput
from improved_tds.control.safety import SafetyLimiter
from improved_tds.synergy.base import TDSEstimator
from improved_tds.synergy.calibration import ToolStateCalibrator


class FeedforwardTDSController:
    """Baseline `rho=f^-1(c_d); q=q_mean+s*rho` controller."""

    def __init__(
        self,
        estimator: TDSEstimator,
        calibrator: ToolStateCalibrator,
        safety: SafetyLimiter,
    ):
        self.estimator = estimator
        self.calibrator = calibrator
        self.safety = safety

    def reset(self) -> None:
        self.safety.reset()

    def feedforward_rho(self, target_tool_state: float, phase: str | None = None) -> float:
        return float(np.asarray(self.calibrator.inverse(target_tool_state, phase)).item())

    def command_from_rho(self, rho: float) -> tuple[np.ndarray, bool]:
        q = np.asarray(self.estimator.decode(np.asarray([rho]))[0], dtype=np.float64)
        return self.safety.limit_q(q)

    def release_output(self, reason: str) -> ControllerOutput:
        q_release = np.asarray(self.estimator.q_mean_, dtype=np.float64).copy()
        q_release, _ = self.safety.limit_q(q_release)
        return ControllerOutput(
            q_command=q_release,
            rho_command=0.0,
            mode="emergency_release",
            emergency_release=True,
            diagnostics={"reason": reason},
        )

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
        desired = self.feedforward_rho(target_tool_state, phase)
        rho, rho_saturated = self.safety.limit_rho(desired, dt)
        q, q_saturated = self.command_from_rho(rho)
        return ControllerOutput(
            q_command=q,
            rho_command=rho,
            mode="feedforward",
            saturated=rho_saturated or q_saturated,
            diagnostics={"rho_feedforward": desired},
        )

