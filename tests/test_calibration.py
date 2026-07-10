from __future__ import annotations

import numpy as np
import pytest

from improved_tds.synergy.calibration import ToolStateCalibrator


@pytest.mark.parametrize("method", ["linear", "piecewise_linear", "isotonic", "pchip"])
def test_calibration_forward_inverse(method, tmp_path) -> None:
    rho = np.linspace(-1.0, 1.0, 21)
    state = 0.2 + 0.5 * rho
    model = ToolStateCalibrator(method).fit(rho, state)
    recovered = model.inverse(model.forward(rho))
    np.testing.assert_allclose(recovered, rho, atol=1e-8)
    with pytest.raises(ValueError, match="outside"):
        model.inverse(2.0)
    path = tmp_path / f"{method}.npz"
    model.save(path)
    loaded = ToolStateCalibrator.load(path)
    np.testing.assert_allclose(loaded.forward(rho), state, atol=1e-8)


def test_isotonic_repairs_noise_and_phase_hysteresis() -> None:
    rho = np.tile(np.linspace(0.0, 1.0, 8), 2)
    phase = np.repeat(["open", "close"], 8)
    state = np.concatenate([rho[:8] + 0.01 * np.sin(20 * rho[:8]), rho[:8] + 0.05])
    model = ToolStateCalibrator("isotonic").fit(rho, state, phase)
    assert model.hysteresis() == pytest.approx(0.05, abs=0.015)


def test_piecewise_rejects_non_monotone_data() -> None:
    with pytest.raises(ValueError, match="not monotone"):
        ToolStateCalibrator("piecewise_linear").fit([0, 1, 2], [0, 2, 1])

