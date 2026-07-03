"""Tests for the CT/material phantom and the OpenGATE reference adapter.

The OpenGATE tests are skipped automatically when the optional Geant4-backed
dependency is not installed, so the suite stays green in a pure-Python CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from braggpeak.ct_materials import (
    hu_to_density,
    hu_to_rsp,
    hu_to_material,
    synthetic_head_profile,
    AIR,
)
from braggpeak.materials import WATER, CORTICAL_BONE, LUNG
from braggpeak.sde_model import simulate_depth_dose_ct
from braggpeak.calibrate import fit_stopping_scale
from braggpeak.validation import ct_wepl_reference, compare_curves
from braggpeak.monte_carlo_gate import opengate_available, simulate_depth_dose_gate
from braggpeak.transport import Slab
from braggpeak.scoring import compute_bragg_metrics


def test_hu_calibration_monotonic_and_anchored():
    hu = np.linspace(-1000, 3000, 200)
    assert np.all(np.diff(hu_to_density(hu)) >= 0)
    assert np.all(np.diff(hu_to_rsp(hu)) >= 0)
    # Water at HU=0 has density ~1 and RSP ~1.
    assert abs(float(hu_to_density(0.0)) - 1.0) < 1e-6
    assert abs(float(hu_to_rsp(0.0)) - 1.0) < 1e-6


def test_hu_material_bands():
    assert hu_to_material(-900.0) is AIR
    assert hu_to_material(-300.0) is LUNG
    assert hu_to_material(0.0) is WATER
    assert hu_to_material(900.0) is CORTICAL_BONE


def test_ct_head_range_within_1mm():
    scale = fit_stopping_scale().stopping_scale
    prof = synthetic_head_profile(dz_cm=0.02)
    sde = simulate_depth_dose_ct(150.0, prof.hu, 0.02, n_histories=60000, seed=21,
                                 stopping_scale=scale)
    zr, dr = ct_wepl_reference(150.0, prof.hu, 0.02, scale=scale)
    c = compare_curves(sde.z_cm, dr, sde.z_cm, sde.dose, low_dose_threshold=0.01)
    assert abs(c.peak_depth_err_mm) < 1.0
    assert abs(c.r80_err_mm) < 1.0


# --- OpenGATE reference (optional) ---------------------------------------

gate_required = pytest.mark.skipif(
    not opengate_available(), reason="opengate (Geant4 backend) not installed"
)


@gate_required
def test_gate_water_bragg_peak_physical():
    r = simulate_depth_dose_gate(150.0, [Slab(WATER, 20.0)], dz_cm=0.1,
                                 n_primaries=3000, seed=1)
    m = compute_bragg_metrics(r.z_cm, r.dose)
    # 150 MeV proton peak lands near 15.5-16.0 cm with a sharp distal edge.
    assert 15.0 < m.peak_depth_mm / 10.0 < 16.2
    assert 1.0 < m.distal_falloff_80_20_mm < 7.0
    assert r.metadata["model"] == "opengate_geant4"


@gate_required
def test_sde_matches_gate_range_within_criteria():
    from braggpeak.calibrate import calibrated_stopping_factory
    from braggpeak.sde_model import simulate_depth_dose_sde

    scale = fit_stopping_scale().stopping_scale
    g = simulate_depth_dose_gate(150.0, [Slab(WATER, 20.0)], dz_cm=0.05,
                                 n_primaries=60000, seed=1)
    s = simulate_depth_dose_sde(150.0, [Slab(WATER, 20.0)], dz_cm=0.05,
                                n_histories=60000, seed=1234,
                                stopping_model_factory=calibrated_stopping_factory(scale))
    c = compare_curves(g.z_cm, g.dose, s.z_cm, s.dose, low_dose_threshold=0.01)
    # Range agreement with the Geant4 reference meets the water criteria.
    assert abs(c.peak_depth_err_mm) <= 1.0
    assert abs(c.r80_err_mm) <= 0.7


@gate_required
def test_sde_matches_gate_heterogeneous_bone_within_1mm():
    """Water-bone-water range vs Geant4 meets the 1.0 mm heterogeneous criterion."""
    from braggpeak.calibrate import calibrated_stopping_factory
    from braggpeak.sde_model import simulate_depth_dose_sde
    from braggpeak.materials import CORTICAL_BONE

    scale = fit_stopping_scale().stopping_scale
    slabs = [Slab(WATER, 5.0), Slab(CORTICAL_BONE, 2.0), Slab(WATER, 15.0)]
    g = simulate_depth_dose_gate(150.0, slabs, dz_cm=0.05, n_primaries=150000, seed=1)
    s = simulate_depth_dose_sde(150.0, slabs, dz_cm=0.05, n_histories=120000, seed=1234,
                                stopping_model_factory=calibrated_stopping_factory(scale))
    c = compare_curves(g.z_cm, g.dose, s.z_cm, s.dose, low_dose_threshold=0.01)
    assert abs(c.peak_depth_err_mm) <= 1.0
    assert abs(c.r80_err_mm) <= 1.0


@gate_required
def test_sde_matches_gate_patient_head_within_1mm():
    """Patient-like head geometry range vs Geant4 meets the 1.0 mm criterion."""
    from braggpeak.calibrate import calibrated_stopping_factory
    from braggpeak.sde_model import simulate_depth_dose_sde
    from braggpeak.ct_materials import synthetic_head_slabs

    scale = fit_stopping_scale().stopping_scale
    head = synthetic_head_slabs()
    g = simulate_depth_dose_gate(150.0, head, dz_cm=0.05, n_primaries=150000, seed=1)
    s = simulate_depth_dose_sde(150.0, head, dz_cm=0.05, n_histories=120000, seed=1234,
                                stopping_model_factory=calibrated_stopping_factory(scale))
    c = compare_curves(g.z_cm, g.dose, s.z_cm, s.dose, low_dose_threshold=0.01)
    assert abs(c.peak_depth_err_mm) <= 1.0
    assert abs(c.r80_err_mm) <= 1.0
