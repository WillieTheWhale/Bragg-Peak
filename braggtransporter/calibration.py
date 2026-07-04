"""Uncertainty calibration helpers for BraggTransporter.

The wrapper is intentionally model-agnostic: it only rescales an existing
predicted ``sigma`` array once later phases provide one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _as_arrays(
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if pred_arr.shape != ref_arr.shape or pred_arr.shape != sigma_arr.shape:
        raise ValueError("pred, ref, and sigma must have identical shapes.")
    if np.any(sigma_arr < 0.0):
        raise ValueError("sigma must be nonnegative.")
    return pred_arr, ref_arr, sigma_arr


def _z_for_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be a probability in (0, 1).")
    return float(NormalDist().inv_cdf((1.0 + level) / 2.0))


def coverage(
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    sigma: NDArray[np.float64],
    level: float,
) -> float:
    """Empirical fraction of reference values inside a Gaussian central band.

    ``level`` is the nominal central probability, for example ``0.6827`` for a
    one-sigma band or ``0.95`` for a 95 percent band.
    """
    pred_arr, ref_arr, sigma_arr = _as_arrays(pred, ref, sigma)
    band = _z_for_level(level) * sigma_arr
    return float(np.mean(np.abs(ref_arr - pred_arr) <= band))


@dataclass(frozen=True)
class CalibrationWrapper:
    """Scale predicted sigma values by a fitted global temperature.

    ``method="temperature"`` fits the RMS standardised residual. ``method=
    "quantile"`` fits the scale whose Gaussian ``level`` band matches the
    empirical absolute-error quantile.
    """

    method: Literal["temperature", "quantile"] = "temperature"
    temperature: float = 1.0
    level: float = 0.6826894921370859
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.method not in {"temperature", "quantile"}:
            raise ValueError("method must be 'temperature' or 'quantile'.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        _z_for_level(self.level)

    def fit(
        self,
        pred: NDArray[np.float64],
        ref: NDArray[np.float64],
        sigma: NDArray[np.float64],
        level: float | None = None,
    ) -> "CalibrationWrapper":
        """Return a new wrapper fitted to validation residuals."""
        pred_arr, ref_arr, sigma_arr = _as_arrays(pred, ref, sigma)
        target_level = self.level if level is None else level
        ratios = np.abs(ref_arr - pred_arr) / np.maximum(sigma_arr, self.eps)

        if self.method == "temperature":
            fitted = float(np.sqrt(np.mean(ratios**2)))
        else:
            fitted = float(np.quantile(ratios, target_level) / _z_for_level(target_level))

        return replace(self, temperature=max(fitted, self.eps), level=target_level)

    def transform_sigma(self, sigma: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the fitted temperature to predicted sigma values."""
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        if np.any(sigma_arr < 0.0):
            raise ValueError("sigma must be nonnegative.")
        return sigma_arr * self.temperature

    def __call__(self, sigma: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.transform_sigma(sigma)

    def state_dict(self) -> dict[str, float | str]:
        return {"method": self.method, "temperature": self.temperature, "level": self.level}

    @classmethod
    def from_state_dict(cls, state: dict[str, float | str]) -> "CalibrationWrapper":
        return cls(
            method=str(state.get("method", "temperature")),  # type: ignore[arg-type]
            temperature=float(state.get("temperature", 1.0)),
            level=float(state.get("level", 0.6826894921370859)),
        )
