"""Tests for Bragg metrics and dose-comparison scoring."""

from __future__ import annotations

import numpy as np

from braggpeak.scoring import (
    compute_bragg_metrics,
    normalized_rmse,
    normalized_mae,
    gamma_index_1d,
)


def _gaussian_curve(peak_cm=15.0, sigma_cm=0.3, zmax=20.0, dz=0.01):
    z = np.arange(0.0, zmax, dz)
    dose = np.exp(-0.5 * ((z - peak_cm) / sigma_cm) ** 2)
    return z, dose


def test_metrics_on_symmetric_gaussian():
    z, dose = _gaussian_curve(peak_cm=15.0, sigma_cm=0.3)
    m = compute_bragg_metrics(z, dose)
    assert abs(m.peak_depth_mm - 150.0) < 1.0
    # FWHM of a Gaussian = 2.3548 * sigma.
    expected_fwhm_mm = 2.3548 * 0.3 * 10.0
    assert abs(m.fwhm_mm - expected_fwhm_mm) < 1.0
    # Distal R50 must be beyond the peak.
    assert m.r50_mm > m.peak_depth_mm


def test_identical_curves_zero_error_full_gamma():
    z, dose = _gaussian_curve()
    assert normalized_rmse(dose, dose) == 0.0
    assert normalized_mae(dose, dose) == 0.0
    pass_rate, gamma = gamma_index_1d(z, dose, z, dose)
    assert pass_rate == 1.0
    assert np.all(gamma[dose >= 0.1 * dose.max()] <= 1.0)


def test_rmse_grows_with_perturbation():
    z, dose = _gaussian_curve()
    small = dose * 1.01
    large = dose * 1.10
    assert normalized_rmse(dose, small) < normalized_rmse(dose, large)


def test_gamma_shifted_curve_fails_more():
    z, dose = _gaussian_curve(peak_cm=15.0, sigma_cm=0.3)
    _, shifted = _gaussian_curve(peak_cm=15.5, sigma_cm=0.3)  # 5 mm shift
    pass_close, _ = gamma_index_1d(z, dose, z, dose, dose_tol_pct=2.0, dta_mm=2.0)
    pass_shift, _ = gamma_index_1d(z, dose, z, shifted, dose_tol_pct=2.0, dta_mm=2.0)
    assert pass_shift < pass_close


def test_low_dose_threshold_excludes_tail():
    z, dose = _gaussian_curve()
    cand = dose.copy()
    cand[dose < 0.01 * dose.max()] += 0.005  # perturb only the tail
    err_all = normalized_rmse(dose, cand, low_dose_threshold=0.0)
    err_cut = normalized_rmse(dose, cand, low_dose_threshold=0.05)
    assert err_cut < err_all
