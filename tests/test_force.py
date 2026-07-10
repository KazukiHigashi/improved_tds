from __future__ import annotations

import numpy as np

from improved_tds.tools.force import DirectJointTorqueEstimator, MotorCurrentEstimator


def test_force_projection_and_bias() -> None:
    estimator = DirectJointTorqueEstimator()
    estimator.reset_bias({"joint_torque": [0.5, 0.5]})
    torque = estimator.estimate_joint_torque({"joint_torque": [1.5, -0.5]})
    np.testing.assert_allclose(torque, [1.0, -1.0])
    assert estimator.estimate_synergy_force(
        {"joint_torque": [1.5, -0.5]}, np.array([1.0, 0.0])
    ) == 1.0


def test_motor_current_requires_explicit_constants() -> None:
    estimator = MotorCurrentEstimator(np.array([0.1, 0.2]), gear_ratios=np.array([2.0, 3.0]))
    np.testing.assert_allclose(
        estimator.estimate_joint_torque({"motor_current": [1.0, 2.0]}), [0.2, 1.2]
    )

