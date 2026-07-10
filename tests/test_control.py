from __future__ import annotations

import numpy as np
import pytest

from improved_tds.control.admittance import AdmittanceConfig, AdmittanceTDSController
from improved_tds.control.base import ControlObservation
from improved_tds.control.feedback import FeedbackConfig, ToolStateFeedbackController
from improved_tds.control.feedforward import FeedforwardTDSController
from improved_tds.control.safety import SafetyLimiter, SafetyLimits
from improved_tds.control.stabilization import StabilizationConfig, StabilizationSynergy
from improved_tds.synergy.calibration import ToolStateCalibrator
from improved_tds.synergy.pca import PCATDS


def components():
    rho = np.linspace(-1.0, 1.0, 50)
    q = np.column_stack([rho, np.zeros_like(rho)])
    estimator = PCATDS().fit(q, rho)
    calibrator = ToolStateCalibrator("linear").fit(rho, rho)
    limits = SafetyLimits(
        rho_min=-1.0,
        rho_max=1.0,
        rho_rate_max=100.0,
        rho_acceleration_max=1000.0,
        q_min=np.array([-1.1, -0.1]),
        q_max=np.array([1.1, 0.1]),
        force_limit=5.0,
    )
    return estimator, calibrator, limits


def test_feedforward_saturation_and_emergency_release() -> None:
    estimator, calibrator, limits = components()
    controller = FeedforwardTDSController(estimator, calibrator, SafetyLimiter(limits))
    output = controller.step(0.8, ControlObservation(tool_state=None), 0.01)
    assert output.rho_command == pytest.approx(0.8)
    emergency = controller.step(
        0.8, ControlObservation(tool_state=0.0, synergy_force=10.0), 0.01
    )
    assert emergency.emergency_release


def test_feedback_reduces_model_mismatch_and_handles_dropout() -> None:
    estimator, calibrator, limits = components()
    controller = ToolStateFeedbackController(
        estimator,
        calibrator,
        SafetyLimiter(limits),
        config=FeedbackConfig(kp=1.0, ki=0.2, integral_limit=0.2),
    )
    measured = 0.0
    for _ in range(80):
        output = controller.step(0.8, ControlObservation(tool_state=measured), 0.01)
        measured = 0.7 * output.rho_command
    assert abs(0.8 - measured) < abs(0.8 - 0.7 * 0.8)
    assert abs(controller.integral_error) <= 0.2
    dropout = controller.step(
        0.5, ControlObservation(tool_state=None, sensor_valid=False), 0.01
    )
    assert "dropout" in dropout.mode


def test_admittance_degrades_and_reduces_command_under_reaction() -> None:
    estimator, calibrator, limits = components()
    controller = AdmittanceTDSController(
        estimator,
        calibrator,
        SafetyLimiter(limits),
        admittance=AdmittanceConfig(
            mass=0.2, damping=2.0, tool_state_gain=1.0, force_gain=0.5
        ),
    )
    dropout = controller.step(0.8, ControlObservation(tool_state=0.0), 0.01)
    assert "dropout" in dropout.mode
    controller.reset()
    first = controller.step(
        0.8, ControlObservation(tool_state=0.0, synergy_force=2.0), 0.01
    )
    assert first.rho_command < 0.8


def test_stabilization_is_bounded_and_releases_on_force() -> None:
    stabilizer = StabilizationSynergy(
        np.array([0.0, 1.0]),
        config=StabilizationConfig(desired_force=1.0, phi_max=0.05, force_limit=3.0),
    )
    q, info = stabilizer.apply(
        np.zeros(2), force_proxy=0.0, tool_state_deviation=0.0, dt=1.0
    )
    assert abs(q[1]) <= 0.05
    _, emergency = stabilizer.apply(
        np.zeros(2), force_proxy=4.0, tool_state_deviation=0.0, dt=0.1
    )
    assert emergency["emergency_release"]

