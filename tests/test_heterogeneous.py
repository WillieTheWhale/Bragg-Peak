"""Heterogeneous-slab range-accuracy tests against the WEPL reference."""

from __future__ import annotations

import numpy as np
import pytest

from braggpeak.materials import WATER, CORTICAL_BONE, LUNG
from braggpeak.transport import Slab
from braggpeak.sde_model import simulate_depth_dose_sde
from braggpeak.calibrate import calibrated_stopping_factory, fit_stopping_scale
from braggpeak.validation import wepl_reference, compare_curves


@pytest.fixture(scope="module")
def scale() -> float:
    return fit_stopping_scale().stopping_scale


def _run(slabs, scale, energy=150.0, seed=11):
    fac = calibrated_stopping_factory(scale)
    sde = simulate_depth_dose_sde(energy, slabs, dz_cm=0.02, n_histories=60000,
                                  seed=seed, stopping_model_factory=fac)
    z = sde.z_cm
    zr, dr = wepl_reference(energy, slabs, z, scale=scale)
    return compare_curves(z, dr, z, sde.dose, low_dose_threshold=0.01)


def test_bone_slab_range_within_1mm(scale):
    """Water-bone-water: peak and R80 within the 1.0 mm heterogeneous criterion."""
    slabs = [Slab(WATER, 5.0), Slab(CORTICAL_BONE, 2.0), Slab(WATER, 15.0)]
    c = _run(slabs, scale)
    assert abs(c.peak_depth_err_mm) < 1.0
    assert abs(c.r80_err_mm) < 1.0


def test_lung_slab_range_within_1mm(scale):
    slabs = [Slab(WATER, 5.0), Slab(LUNG, 5.0), Slab(WATER, 15.0)]
    c = _run(slabs, scale, seed=12)
    assert abs(c.peak_depth_err_mm) < 1.0
    assert abs(c.r80_err_mm) < 1.0


def test_bone_pulls_peak_shallower_than_water(scale):
    """A bone slab must shorten the physical range vs pure water."""
    fac = calibrated_stopping_factory(scale)
    het = simulate_depth_dose_sde(150.0,
        [Slab(WATER, 5.0), Slab(CORTICAL_BONE, 2.0), Slab(WATER, 15.0)],
        dz_cm=0.05, n_histories=20000, seed=1, stopping_model_factory=fac)
    pure = simulate_depth_dose_sde(150.0, [Slab(WATER, 22.0)],
        dz_cm=0.05, n_histories=20000, seed=1, stopping_model_factory=fac)
    from braggpeak.scoring import compute_bragg_metrics
    peak_het = compute_bragg_metrics(het.z_cm, het.dose).peak_depth_mm
    peak_pure = compute_bragg_metrics(pure.z_cm, pure.dose).peak_depth_mm
    # 2 cm bone (RSP ~1.7) replaces ~3.4 cm water-equivalent -> peak ~1.4 cm shallower.
    assert (peak_pure - peak_het) > 10.0  # mm
