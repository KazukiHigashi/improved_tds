from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression


CalibrationMethod = Literal["linear", "piecewise_linear", "isotonic", "pchip"]


@dataclass
class _Curve:
    rho: np.ndarray
    state: np.ndarray

    @property
    def rho_range(self) -> tuple[float, float]:
        return float(self.rho[0]), float(self.rho[-1])

    @property
    def state_range(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[-1])


class ToolStateCalibrator:
    """Monotone and range-safe mapping between TDS coordinate and tool state."""

    def __init__(
        self,
        method: CalibrationMethod = "pchip",
        *,
        allow_extrapolation: bool = False,
        increasing: bool | None = None,
    ):
        if method not in {"linear", "piecewise_linear", "isotonic", "pchip"}:
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.allow_extrapolation = bool(allow_extrapolation)
        self.increasing = increasing
        self.curves_: dict[str, _Curve] = {}

    def fit(
        self,
        rho: np.ndarray,
        tool_state: np.ndarray,
        phase: np.ndarray | None = None,
    ) -> "ToolStateCalibrator":
        coordinate = np.asarray(rho, dtype=np.float64).reshape(-1)
        state = np.asarray(tool_state, dtype=np.float64).reshape(-1)
        if coordinate.size < 2 or coordinate.shape != state.shape:
            raise ValueError("rho and tool_state must be equal-length arrays with at least 2 samples")
        if not np.all(np.isfinite(coordinate)) or not np.all(np.isfinite(state)):
            raise ValueError("calibration data contains NaN or infinity")
        labels = (
            np.full(coordinate.shape, "default", dtype=str)
            if phase is None
            else np.asarray(phase, dtype=str).reshape(-1)
        )
        if labels.shape != coordinate.shape:
            raise ValueError("phase must match calibration sample count")
        self.curves_.clear()
        for label in np.unique(labels):
            mask = labels == label
            self.curves_[str(label)] = self._fit_curve(coordinate[mask], state[mask])
        return self

    def _fit_curve(self, rho: np.ndarray, state: np.ndarray) -> _Curve:
        if rho.size < 2:
            raise ValueError("each calibration phase requires at least two samples")
        order = np.argsort(rho)
        rho = rho[order]
        state = state[order]
        unique_rho, inverse = np.unique(rho, return_inverse=True)
        averaged = np.zeros_like(unique_rho)
        counts = np.zeros_like(unique_rho)
        np.add.at(averaged, inverse, state)
        np.add.at(counts, inverse, 1.0)
        state = averaged / counts
        rho = unique_rho
        if rho.size < 2:
            raise ValueError("calibration requires at least two distinct rho values")
        increasing = self.increasing
        if increasing is None:
            increasing = float(np.corrcoef(rho, state)[0, 1]) >= 0.0
        if self.method == "linear":
            slope, intercept = np.polyfit(rho, state, 1)
            if (slope >= 0.0) != increasing or abs(slope) <= 1e-12:
                raise ValueError("linear calibration is not invertibly monotone")
            state = intercept + slope * np.asarray([rho[0], rho[-1]])
            rho = np.asarray([rho[0], rho[-1]])
        elif self.method == "isotonic":
            state = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit_transform(
                rho, state
            )
        else:
            differences = np.diff(state)
            valid = np.all(differences >= -1e-10) if increasing else np.all(differences <= 1e-10)
            if not valid:
                raise ValueError(
                    f"{self.method} calibration data is not monotone; use method='isotonic'"
                )
        # Remove flat spans: they are not uniquely invertible. Keep endpoints of the usable range.
        state_oriented = state if increasing else -state
        keep = np.concatenate(([True], np.diff(state_oriented) > 1e-12))
        rho = rho[keep]
        state = state[keep]
        if rho.size < 2:
            raise ValueError("calibration has no invertible interval")
        return _Curve(rho=rho.astype(np.float64), state=state.astype(np.float64))

    def _curve(self, phase: str | None) -> _Curve:
        if not self.curves_:
            raise RuntimeError("calibrator has not been fitted")
        key = "default" if phase is None else str(phase)
        if key not in self.curves_:
            if phase is None and len(self.curves_) == 1:
                return next(iter(self.curves_.values()))
            raise KeyError(f"no calibration curve for phase {key!r}")
        return self.curves_[key]

    def forward(self, rho: np.ndarray | float, phase: str | None = None) -> np.ndarray:
        curve = self._curve(phase)
        values = np.asarray(rho, dtype=np.float64)
        self._check_domain(values, curve.rho_range, "rho")
        return self._interpolate(values, curve.rho, curve.state)

    def inverse(self, tool_state: np.ndarray | float, phase: str | None = None) -> np.ndarray:
        curve = self._curve(phase)
        values = np.asarray(tool_state, dtype=np.float64)
        xp = curve.state
        fp = curve.rho
        if xp[0] > xp[-1]:
            xp = xp[::-1]
            fp = fp[::-1]
        self._check_domain(values, (float(xp[0]), float(xp[-1])), "tool_state")
        return self._interpolate(values, xp, fp)

    def _interpolate(self, values: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        if self.method == "pchip":
            return np.asarray(
                PchipInterpolator(xp, fp, extrapolate=self.allow_extrapolation)(values),
                dtype=np.float64,
            )
        return np.asarray(np.interp(values, xp, fp), dtype=np.float64)

    def _check_domain(
        self,
        values: np.ndarray,
        domain: tuple[float, float],
        name: str,
    ) -> None:
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity")
        if not self.allow_extrapolation and (
            np.any(values < domain[0] - 1e-12) or np.any(values > domain[1] + 1e-12)
        ):
            raise ValueError(f"{name} is outside calibrated domain {domain}")

    def safe_ranges(self, phase: str | None = None) -> dict[str, tuple[float, float]]:
        curve = self._curve(phase)
        return {
            "rho": tuple(sorted(curve.rho_range)),
            "tool_state": tuple(sorted(curve.state_range)),
        }

    def hysteresis(self, open_phase: str = "open", close_phase: str = "close") -> float:
        first = self._curve(open_phase)
        second = self._curve(close_phase)
        low = max(min(first.rho_range), min(second.rho_range))
        high = min(max(first.rho_range), max(second.rho_range))
        if high <= low:
            raise ValueError("phase calibrations have no overlapping rho range")
        samples = np.linspace(low, high, 128)
        return float(np.mean(np.abs(self.forward(samples, open_phase) - self.forward(samples, close_phase))))

    def save(self, path: Path | str) -> None:
        if not self.curves_:
            raise RuntimeError("calibrator has not been fitted")
        payload: dict[str, np.ndarray] = {
            "calibrator_type": np.asarray("tool_state_calibrator"),
            "method": np.asarray(self.method),
            "allow_extrapolation": np.asarray(self.allow_extrapolation),
            "increasing": np.asarray("auto" if self.increasing is None else str(self.increasing)),
            "phases": np.asarray(sorted(self.curves_)),
        }
        for index, phase in enumerate(sorted(self.curves_)):
            payload[f"rho_{index}"] = self.curves_[phase].rho
            payload[f"state_{index}"] = self.curves_[phase].state
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load(cls, path: Path | str) -> "ToolStateCalibrator":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["calibrator_type"].item()) != "tool_state_calibrator":
                raise ValueError("file is not a ToolStateCalibrator")
            increasing_raw = str(data["increasing"].item())
            increasing = None if increasing_raw == "auto" else increasing_raw == "True"
            model = cls(
                method=str(data["method"].item()),
                allow_extrapolation=bool(data["allow_extrapolation"].item()),
                increasing=increasing,
            )
            for index, phase in enumerate(np.asarray(data["phases"], dtype=str)):
                model.curves_[str(phase)] = _Curve(
                    rho=np.asarray(data[f"rho_{index}"], dtype=np.float64),
                    state=np.asarray(data[f"state_{index}"], dtype=np.float64),
                )
        return model

