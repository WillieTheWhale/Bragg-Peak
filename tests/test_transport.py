"""Physics-sanity and reproducibility tests for the CSDA transport model."""

from __future__ import annotations

import numpy as np
import pytest

from braggpeak.materials import WATER
from braggpeak.transport import Slab, simulate_depth_dose
from braggpeak.scoring import compute_bragg_metrics


def _run(energy, dz=0.02, depth=None):
    depth = depth if depth is not None else max(5.0, energy / 6.0)
    return simulate_depth_dose(energy, [Slab(WATER, depth)], dz_cm=dz)


def test_peak_depth_increases_with_energy():
    depths = []
    for e in [80.0, 120.0, 160.0, 200.0]:
        r = _run(e, dz=0.05, depth=30.0)
        m = compute_bragg_metrics(r.z_cm, r.dose)
        depths.append(m.peak_depth_mm)
    assert all(np.diff(depths) > 0), f"peak depth not monotonic: {depths}"


def test_peak_near_pstar_range():
    """Peak depth should sit close to (just proximal of) the CSDA range."""
    r = _run(150.0, dz=0.01, depth=20.0)
    m = compute_bragg_metrics(r.z_cm, r.dose)
    range_mm = r.metadata["water_csda_range_cm"] * 10.0
    # Peak forms slightly before the CSDA range; within a few mm.
    assert abs(m.peak_depth_mm - range_mm) < 4.0


def test_distal_falloff_is_sharp():
    """Distal 80-20 falloff for a clinical energy is a few mm, not tens."""
    r = _run(150.0, dz=0.01, depth=20.0)
    m = compute_bragg_metrics(r.z_cm, r.dose)
    assert 0.5 < m.distal_falloff_80_20_mm < 8.0


def test_peak_to_entrance_ratio_above_one():
    """A pristine Bragg peak deposits far more at the peak than at entrance."""
    r = _run(150.0, dz=0.01, depth=20.0)
    m = compute_bragg_metrics(r.z_cm, r.dose)
    assert m.peak_to_entrance_ratio > 3.0


def test_energy_nonnegative_everywhere():
    r = _run(150.0, dz=0.02, depth=20.0)
    assert np.all(r.energy_mev >= 0.0)
    assert np.all(r.dose >= 0.0)


def test_energy_conservation_bound():
    """Broadening must not create or destroy total deposited energy.

    Integral of ideal dose (MeV/g) over mass equals broadened integral, since
    the variable-sigma kernel is per-source normalised.
    """
    r = _run(150.0, dz=0.01, depth=20.0)
    dz = r.metadata["dz_cm"]
    rho = WATER.density_g_cm3
    total_ideal = np.sum(r.dose_ideal) * dz * rho
    total_broad = np.sum(r.dose) * dz * rho
    assert np.isclose(total_ideal, total_broad, rtol=1e-6)


def test_deposited_energy_within_beam_energy():
    """Total deposited energy per proton cannot exceed the initial energy."""
    r = _run(150.0, dz=0.01, depth=20.0)
    dz = r.metadata["dz_cm"]
    rho = WATER.density_g_cm3
    # dose is MeV/g; multiply by mass per unit area (rho * dz) and sum.
    deposited = np.sum(r.dose_ideal) * dz * rho
    assert deposited <= 150.0 * 1.0000001
    # And most of the energy is accounted for (CSDA, no escape).
    assert deposited > 0.9 * 150.0


def test_deterministic_replay():
    r1 = simulate_depth_dose(150.0, [Slab(WATER, 20.0)], dz_cm=0.02, seed=42)
    r2 = simulate_depth_dose(150.0, [Slab(WATER, 20.0)], dz_cm=0.02, seed=42)
    assert np.array_equal(r1.dose, r2.dose)
    assert np.array_equal(r1.energy_mev, r2.energy_mev)


def test_let_rises_toward_distal_edge():
    """Dose/track-averaged LET must increase near the Bragg peak."""
    r = _run(150.0, dz=0.01, depth=20.0)
    peak_idx = int(np.argmax(r.dose))
    entrance_let = r.let_kev_um[5]
    peak_let = r.let_kev_um[peak_idx]
    assert peak_let > entrance_let


def test_rejects_energy_below_cut():
    with pytest.raises(ValueError):
        simulate_depth_dose(0.5, [Slab(WATER, 5.0)], e_cut_mev=1.0)
