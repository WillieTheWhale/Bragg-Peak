"""Tests for calibration, the Bortfeld analytic curve, and the SDE model."""

from __future__ import annotations

import numpy as np
import pytest

from braggpeak.analytic_bragg import bortfeld_depth_dose
from braggpeak.calibrate import fit_stopping_scale, nist_range_cm, load_nist_water_ranges
from braggpeak.materials import WATER
from braggpeak.scoring import compute_bragg_metrics
from braggpeak.sde_model import simulate_depth_dose_sde, dose_uncertainty
from braggpeak.transport import Slab
from braggpeak.stopping_power import BetheStoppingPower


def test_nist_reference_monotonic_and_reasonable():
    e, r = load_nist_water_ranges()
    assert np.all(np.diff(r) > 0)
    # 150 MeV proton reaches ~15.8 cm in water.
    assert abs(nist_range_cm(150.0) - 15.77) < 0.1


def test_calibration_reduces_range_error():
    cal = fit_stopping_scale()
    assert cal.calibrated_rms_mm < cal.baseline_rms_mm
    # Calibrated range agrees with NIST to well under the 0.7 mm criterion.
    assert cal.calibrated_max_abs_mm < 0.7
    # The fitted scale is a small correction near unity.
    assert 1.0 < cal.stopping_scale < 1.02


def test_calibration_scale_shortens_range():
    r_base = BetheStoppingPower(WATER).csda_range_cm(150.0)
    r_scaled = BetheStoppingPower(WATER, stopping_scale=1.01).csda_range_cm(150.0)
    assert r_scaled < r_base


def test_bortfeld_curve_physical():
    z = np.arange(0.05, 20.0, 0.02)
    c = bortfeld_depth_dose(150.0, z)
    m = compute_bragg_metrics(c.z_cm, c.dose)
    # Peak near the range, sharp distal falloff, peak above plateau.
    assert 14.0 < m.peak_depth_mm / 10.0 < 16.0
    assert 1.0 < m.distal_falloff_80_20_mm < 6.0
    assert m.peak_to_entrance_ratio > 2.0
    assert np.all(c.dose >= 0.0)


def test_bortfeld_distal_falloff_widens_with_energy():
    z = np.arange(0.05, 30.0, 0.02)
    d100 = compute_bragg_metrics(z, bortfeld_depth_dose(100.0, z).dose).distal_falloff_80_20_mm
    d200 = compute_bragg_metrics(z, bortfeld_depth_dose(200.0, z).dose).distal_falloff_80_20_mm
    assert d200 > d100


def test_sde_deterministic_replay():
    a = simulate_depth_dose_sde(150.0, [Slab(WATER, 20.0)], dz_cm=0.05, n_histories=2000, seed=5)
    b = simulate_depth_dose_sde(150.0, [Slab(WATER, 20.0)], dz_cm=0.05, n_histories=2000, seed=5)
    assert np.array_equal(a.dose, b.dose)


def test_sde_nuclear_removal_lowers_peak_ratio():
    """Nuclear removal must reduce peak-to-entrance ratio vs no removal."""
    common = dict(dz_cm=0.05, n_histories=8000, seed=3)
    with_nuc = simulate_depth_dose_sde(150.0, [Slab(WATER, 20.0)], nuclear_removal=True, **common)
    no_nuc = simulate_depth_dose_sde(150.0, [Slab(WATER, 20.0)], nuclear_removal=False, **common)
    m_with = compute_bragg_metrics(with_nuc.z_cm, with_nuc.dose)
    m_no = compute_bragg_metrics(no_nuc.z_cm, no_nuc.dose)
    assert m_with.peak_to_entrance_ratio < m_no.peak_to_entrance_ratio


def test_sde_range_matches_bethe_within_mm():
    """SDE Bragg peak range agrees with the deterministic Bethe range."""
    sde = simulate_depth_dose_sde(150.0, [Slab(WATER, 20.0)], dz_cm=0.02, n_histories=20000, seed=9)
    m = compute_bragg_metrics(sde.z_cm, sde.dose)
    r_bethe_mm = BetheStoppingPower(WATER).csda_range_cm(150.0) * 10.0
    assert abs(m.r80_mm - r_bethe_mm) < 2.0


def test_sde_uncertainty_positive_and_shrinks_with_histories():
    _, se_small = dose_uncertainty(150.0, [Slab(WATER, 20.0)], n_replicas=3, base_seed=0,
                                   dz_cm=0.1, n_histories=1000)
    _, se_large = dose_uncertainty(150.0, [Slab(WATER, 20.0)], n_replicas=3, base_seed=0,
                                   dz_cm=0.1, n_histories=8000)
    # Peak-region statistical error decreases with more histories.
    assert np.nanmax(se_large) <= np.nanmax(se_small)
