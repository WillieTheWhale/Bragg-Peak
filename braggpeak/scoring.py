"""Bragg-curve metrics and dose-comparison scoring.

Given a depth grid and a depth-dose curve, compute the standard proton range
and shape metrics (peak depth, R90/R80/R50 distal ranges, distal 80-20 width,
FWHM), plus curve-to-curve comparison metrics (RMSE, MAE) and a 1-D gamma
index. Every function is pure and unit-explicit: depths are cm in, mm out for
the reported scalar ranges (documented per function).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .units import cm_to_mm


@dataclass
class BraggMetrics:
    """Scalar descriptors of a single depth-dose curve. Ranges in mm."""

    peak_depth_mm: float
    peak_dose: float
    entrance_dose: float
    peak_to_entrance_ratio: float
    r90_mm: float
    r80_mm: float
    r50_mm: float
    proximal_r90_mm: float
    distal_falloff_80_20_mm: float
    fwhm_mm: float

    def as_dict(self) -> dict:
        return asdict(self)


def _interp_crossing_distal(
    z_cm: NDArray[np.float64],
    dose: NDArray[np.float64],
    peak_idx: int,
    level: float,
) -> float:
    """Depth (cm) where the distal side falls to ``level`` (fraction of peak).

    Linearly interpolates the first crossing on the distal (post-peak) side.
    Returns NaN if the curve never falls below the level after the peak.
    """
    peak = dose[peak_idx]
    target = level * peak
    for i in range(peak_idx, len(dose) - 1):
        if dose[i] >= target >= dose[i + 1]:
            d0, d1 = dose[i], dose[i + 1]
            z0, z1 = z_cm[i], z_cm[i + 1]
            if d0 == d1:
                return float(z0)
            frac = (d0 - target) / (d0 - d1)
            return float(z0 + frac * (z1 - z0))
    return float("nan")


def _interp_crossing_proximal(
    z_cm: NDArray[np.float64],
    dose: NDArray[np.float64],
    peak_idx: int,
    level: float,
) -> float:
    """Depth (cm) where the proximal side rises to ``level`` (fraction of peak)."""
    peak = dose[peak_idx]
    target = level * peak
    for i in range(peak_idx, 0, -1):
        if dose[i] >= target >= dose[i - 1]:
            d0, d1 = dose[i - 1], dose[i]
            z0, z1 = z_cm[i - 1], z_cm[i]
            if d0 == d1:
                return float(z1)
            frac = (target - d0) / (d1 - d0)
            return float(z0 + frac * (z1 - z0))
    return float("nan")


def compute_bragg_metrics(
    z_cm: NDArray[np.float64],
    dose: NDArray[np.float64],
) -> BraggMetrics:
    """Compute range/shape metrics for one depth-dose curve.

    Distal ranges (R90/R80/R50) are the depths on the *distal* edge where the
    dose falls to 90/80/50 percent of the peak -- the clinically used range
    surrogates. All returned distances are in millimetres.
    """
    z_cm = np.asarray(z_cm, dtype=np.float64)
    dose = np.asarray(dose, dtype=np.float64)
    if z_cm.shape != dose.shape or z_cm.size < 3:
        raise ValueError("z_cm and dose must be 1-D arrays of equal length >= 3.")

    peak_idx = int(np.argmax(dose))
    peak = float(dose[peak_idx])
    entrance = float(dose[0])

    r90 = _interp_crossing_distal(z_cm, dose, peak_idx, 0.90)
    r80 = _interp_crossing_distal(z_cm, dose, peak_idx, 0.80)
    r50 = _interp_crossing_distal(z_cm, dose, peak_idx, 0.50)
    r20 = _interp_crossing_distal(z_cm, dose, peak_idx, 0.20)
    prox90 = _interp_crossing_proximal(z_cm, dose, peak_idx, 0.90)
    prox_half = _interp_crossing_proximal(z_cm, dose, peak_idx, 0.50)

    distal_80_20 = cm_to_mm(r20 - r80) if np.isfinite(r20) and np.isfinite(r80) else float("nan")
    fwhm = cm_to_mm(r50 - prox_half) if np.isfinite(r50) and np.isfinite(prox_half) else float("nan")

    return BraggMetrics(
        peak_depth_mm=cm_to_mm(z_cm[peak_idx]),
        peak_dose=peak,
        entrance_dose=entrance,
        peak_to_entrance_ratio=peak / entrance if entrance > 0 else float("inf"),
        r90_mm=cm_to_mm(r90),
        r80_mm=cm_to_mm(r80),
        r50_mm=cm_to_mm(r50),
        proximal_r90_mm=cm_to_mm(prox90),
        distal_falloff_80_20_mm=distal_80_20,
        fwhm_mm=fwhm,
    )


# --- Curve comparison ----------------------------------------------------


def normalized_rmse(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    *,
    low_dose_threshold: float = 0.0,
) -> float:
    """RMSE between two curves, normalised to the reference peak, in percent.

    Points where the (peak-normalised) reference is below ``low_dose_threshold``
    are excluded -- use this to drop clearly documented low-dose tails. The
    threshold is a fraction of the reference peak (e.g. 0.01 for 1 percent).
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    peak = float(ref.max())
    if peak <= 0:
        raise ValueError("Reference peak must be positive for normalisation.")
    ref_n = ref / peak
    cand_n = cand / peak
    mask = ref_n >= low_dose_threshold
    diff = cand_n[mask] - ref_n[mask]
    return float(np.sqrt(np.mean(diff**2)) * 100.0)


def normalized_mae(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    *,
    low_dose_threshold: float = 0.0,
) -> float:
    """Mean absolute error normalised to the reference peak, in percent."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    peak = float(ref.max())
    if peak <= 0:
        raise ValueError("Reference peak must be positive for normalisation.")
    ref_n = ref / peak
    cand_n = cand / peak
    mask = ref_n >= low_dose_threshold
    return float(np.mean(np.abs(cand_n[mask] - ref_n[mask])) * 100.0)


def gamma_index_1d(
    z_ref_cm: NDArray[np.float64],
    dose_ref: NDArray[np.float64],
    z_cand_cm: NDArray[np.float64],
    dose_cand: NDArray[np.float64],
    *,
    dose_tol_pct: float = 2.0,
    dta_mm: float = 2.0,
    low_dose_threshold: float = 0.1,
) -> tuple[float, NDArray[np.float64]]:
    """1-D global gamma index and pass rate.

    Standard global-normalisation gamma: dose difference normalised to the
    reference peak (``dose_tol_pct`` percent) and distance-to-agreement
    (``dta_mm`` mm). Reference points below ``low_dose_threshold`` * peak are
    excluded from the pass-rate denominator (documented low-dose tail cut).

    Returns ``(pass_rate_fraction, gamma_per_point)``.
    """
    z_ref_cm = np.asarray(z_ref_cm, dtype=np.float64)
    dose_ref = np.asarray(dose_ref, dtype=np.float64)
    z_cand_mm = np.asarray(z_cand_cm, dtype=np.float64) * 10.0
    dose_cand = np.asarray(dose_cand, dtype=np.float64)
    z_ref_mm = z_ref_cm * 10.0

    peak = float(dose_ref.max())
    if peak <= 0:
        raise ValueError("Reference peak must be positive.")
    dose_norm = dose_tol_pct / 100.0 * peak

    gamma = np.full(z_ref_mm.shape, np.inf)
    for i, (zr, dr) in enumerate(zip(z_ref_mm, dose_ref)):
        dz = (z_cand_mm - zr) / dta_mm
        dd = (dose_cand - dr) / dose_norm
        gamma[i] = np.sqrt(np.min(dz**2 + dd**2))

    mask = dose_ref >= low_dose_threshold * peak
    if not np.any(mask):
        return float("nan"), gamma
    pass_rate = float(np.mean(gamma[mask] <= 1.0))
    return pass_rate, gamma
