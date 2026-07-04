"""Metric wrappers for BraggTransporter v3.1.

All depth grids are one-dimensional arrays in cm. Range-like outputs use the
unit stated in the function name. Dose arrays are physical dose values and are
only normalised inside the delegated scoring routines where explicitly named.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from braggpeak import scoring


def _as_1d(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    return arr


def _same_shape(*arrays: NDArray[np.float64]) -> None:
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError("All metric inputs must have the same shape.")


def gamma_index_1d(
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    z: NDArray[np.float64],
    dose_pct: float = 2.0,
    dta_mm: float = 2.0,
) -> float:
    """Return 1-D gamma pass rate in percent on a shared depth grid.

    Parameters
    ----------
    pred, ref:
        Predicted and reference dose arrays, shape ``(Nz,)``.
    z:
        Depth grid in cm, shape ``(Nz,)``.
    dose_pct, dta_mm:
        Global dose-difference tolerance in percent of reference peak and
        distance-to-agreement tolerance in millimetres.
    """
    pred_arr = _as_1d("pred", pred)
    ref_arr = _as_1d("ref", ref)
    z_arr = _as_1d("z", z)
    _same_shape(pred_arr, ref_arr, z_arr)
    pass_rate, _ = scoring.gamma_index_1d(
        z_arr,
        ref_arr,
        z_arr,
        pred_arr,
        dose_tol_pct=dose_pct,
        dta_mm=dta_mm,
    )
    return float(pass_rate * 100.0)


def r80_r90_r50(z: NDArray[np.float64], dose: NDArray[np.float64]) -> dict[str, float]:
    """Return distal R80/R90/R50 in millimetres."""
    z_arr = _as_1d("z", z)
    dose_arr = _as_1d("dose", dose)
    _same_shape(z_arr, dose_arr)
    m = scoring.compute_bragg_metrics(z_arr, dose_arr)
    return {"r80_mm": float(m.r80_mm), "r90_mm": float(m.r90_mm), "r50_mm": float(m.r50_mm)}


def peak_depth_cm(z: NDArray[np.float64], dose: NDArray[np.float64]) -> float:
    """Return peak depth in centimetres."""
    z_arr = _as_1d("z", z)
    dose_arr = _as_1d("dose", dose)
    _same_shape(z_arr, dose_arr)
    m = scoring.compute_bragg_metrics(z_arr, dose_arr)
    return float(m.peak_depth_mm / 10.0)


def peak_depth(z: NDArray[np.float64], dose: NDArray[np.float64]) -> float:
    """Compatibility alias for the interface contract: peak depth in cm."""
    return peak_depth_cm(z, dose)


def distal_80_20_mm(z: NDArray[np.float64], dose: NDArray[np.float64]) -> float:
    """Return distal 80-to-20 percent falloff width in millimetres."""
    z_arr = _as_1d("z", z)
    dose_arr = _as_1d("dose", dose)
    _same_shape(z_arr, dose_arr)
    return float(scoring.compute_bragg_metrics(z_arr, dose_arr).distal_falloff_80_20_mm)


def rmse_pct(pred: NDArray[np.float64], ref: NDArray[np.float64]) -> float:
    """Return RMSE normalised to the reference peak, in percent."""
    pred_arr = _as_1d("pred", pred)
    ref_arr = _as_1d("ref", ref)
    _same_shape(pred_arr, ref_arr)
    return float(scoring.normalized_rmse(ref_arr, pred_arr))


def distal_edge_error_mm(
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    z: NDArray[np.float64],
) -> float:
    """Return ``abs(R80_pred - R80_ref)`` in millimetres."""
    pred_arr = _as_1d("pred", pred)
    ref_arr = _as_1d("ref", ref)
    z_arr = _as_1d("z", z)
    _same_shape(pred_arr, ref_arr, z_arr)
    r80_pred = scoring.compute_bragg_metrics(z_arr, pred_arr).r80_mm
    r80_ref = scoring.compute_bragg_metrics(z_arr, ref_arr).r80_mm
    return float(abs(r80_pred - r80_ref))
