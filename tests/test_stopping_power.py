"""Physics-sanity tests for stopping-power and range models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from braggpeak.materials import WATER, CORTICAL_BONE
from braggpeak.stopping_power import (
    BetheStoppingPower,
    BortfeldRange,
    PstarTable,
    bortfeld_range_cm,
    bortfeld_energy_mev,
)

CLINICAL_ENERGIES = [60.0, 100.0, 150.0, 200.0, 230.0]

# NIST PSTAR liquid-water CSDA ranges (g/cm^2) at reference energies.
PSTAR_WATER_RANGE_G_CM2 = {
    60.0: 3.093,
    100.0: 7.718,
    150.0: 15.77,
    200.0: 25.96,
    230.0: 32.99,
}


def test_range_monotonic_with_energy_bethe():
    bethe = BetheStoppingPower(WATER)
    ranges = [bethe.csda_range_cm(e) for e in CLINICAL_ENERGIES]
    assert all(np.diff(ranges) > 0), "CSDA range must increase with energy."


def test_stopping_power_positive_and_rises_at_low_energy():
    bethe = BetheStoppingPower(WATER)
    s_high = bethe.mass_stopping_power(150.0)[0]
    s_low = bethe.mass_stopping_power(10.0)[0]
    assert s_high > 0 and s_low > 0
    # Stopping power rises steeply as energy falls (Bragg peak mechanism).
    assert s_low > s_high


def test_bethe_matches_bortfeld_within_2pct():
    bethe = BetheStoppingPower(WATER)
    for e in CLINICAL_ENERGIES:
        r_bethe = bethe.csda_range_cm(e)
        r_bortfeld = float(bortfeld_range_cm(e))
        rel = abs(r_bethe - r_bortfeld) / r_bortfeld
        assert rel < 0.02, f"{e} MeV: Bethe {r_bethe:.3f} vs Bortfeld {r_bortfeld:.3f}"


def test_bethe_matches_nist_pstar_within_2pct():
    """Independent check against tabulated NIST PSTAR water ranges."""
    bethe = BetheStoppingPower(WATER)
    for e, r_pstar in PSTAR_WATER_RANGE_G_CM2.items():
        r_model = bethe.csda_range_cm(e) * WATER.density_g_cm3  # g/cm^2
        rel = abs(r_model - r_pstar) / r_pstar
        assert rel < 0.02, f"{e} MeV: model {r_model:.3f} vs PSTAR {r_pstar:.3f} g/cm2"


def test_bortfeld_roundtrip():
    for e in CLINICAL_ENERGIES:
        r = float(bortfeld_range_cm(e))
        e_back = float(bortfeld_energy_mev(r))
        assert abs(e_back - e) / e < 1e-9


def test_denser_material_shorter_range():
    """Bone (higher density) must stop protons in a shorter physical distance."""
    r_water = BetheStoppingPower(WATER).csda_range_cm(150.0)
    r_bone = BetheStoppingPower(CORTICAL_BONE).csda_range_cm(150.0)
    assert r_bone < r_water


def test_pstar_table_interpolation_recovers_grid():
    csv = Path(__file__).resolve().parents[1] / "braggpeak" / "data" / "water_stopping_analytic.csv"
    table = PstarTable.from_csv(csv, WATER)
    # Interpolating exactly at a tabulated energy returns the tabulated value.
    idx = 5
    e = table.energy_mev[idx]
    s = table.mass_stopping_power(e)[0]
    assert np.isclose(s, table.mass_stopping[idx], rtol=1e-6)


def test_pstar_table_range_monotonic_and_matches_bethe():
    csv = Path(__file__).resolve().parents[1] / "braggpeak" / "data" / "water_stopping_analytic.csv"
    table = PstarTable.from_csv(csv, WATER)
    bethe = BetheStoppingPower(WATER)
    ranges = [table.csda_range_cm(e) for e in CLINICAL_ENERGIES]
    assert all(np.diff(ranges) > 0)
    for e in CLINICAL_ENERGIES:
        rel = abs(table.csda_range_cm(e) - bethe.csda_range_cm(e)) / bethe.csda_range_cm(e)
        assert rel < 0.02


def test_bethe_rejects_nonpositive_energy():
    bethe = BetheStoppingPower(WATER)
    with pytest.raises(ValueError):
        bethe.mass_stopping_power(-5.0)
