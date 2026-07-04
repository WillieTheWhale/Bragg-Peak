"""BraggTransporter metric wrappers against an analytic Bragg curve."""

from __future__ import annotations

import numpy as np

from braggpeak.analytic_bragg import bortfeld_depth_dose
from braggpeak.scoring import compute_bragg_metrics
from braggtransporter.metrics import (
    distal_80_20_mm,
    distal_edge_error_mm,
    gamma_index_1d,
    peak_depth_cm,
    r80_r90_r50,
    rmse_pct,
)


def _analytic_curve() -> tuple[np.ndarray, np.ndarray]:
    z = np.arange(0.05, 22.0, 0.02, dtype=np.float64)
    curve = bortfeld_depth_dose(150.0, z, energy_spread_pct=0.8)
    return curve.z_cm, curve.dose


def test_r80_r90_r50_match_braggpeak_scoring_within_one_voxel():
    z, dose = _analytic_curve()
    wrapped = r80_r90_r50(z, dose)
    direct = compute_bragg_metrics(z, dose)
    voxel_mm = float((z[1] - z[0]) * 10.0)

    assert abs(wrapped["r80_mm"] - direct.r80_mm) <= voxel_mm
    assert abs(wrapped["r90_mm"] - direct.r90_mm) <= voxel_mm
    assert abs(wrapped["r50_mm"] - direct.r50_mm) <= voxel_mm


def test_peak_depth_and_distal_falloff_match_scoring():
    z, dose = _analytic_curve()
    wrapped_peak_cm = peak_depth_cm(z, dose)
    wrapped_falloff_mm = distal_80_20_mm(z, dose)
    direct = compute_bragg_metrics(z, dose)

    assert abs(wrapped_peak_cm - direct.peak_depth_mm / 10.0) <= (z[1] - z[0])
    assert wrapped_falloff_mm == direct.distal_falloff_80_20_mm


def test_identical_curve_has_full_gamma_and_zero_errors():
    z, dose = _analytic_curve()

    assert gamma_index_1d(dose, dose, z) == 100.0
    assert rmse_pct(dose, dose) == 0.0
    assert distal_edge_error_mm(dose, dose, z) == 0.0


def test_distal_edge_error_detects_shifted_edge():
    z, dose = _analytic_curve()
    shifted = np.interp(z - 0.10, z, dose, left=dose[0], right=0.0)

    assert distal_edge_error_mm(shifted, dose, z) > 0.0
    assert rmse_pct(shifted, dose) > 0.0
