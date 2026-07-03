"""Model calibration against a tabulated range reference.

The baseline Bethe-Bloch model carries a small, nearly energy-independent range
bias (omitted shell corrections + CSDA-to-projected-range detour). This module
fits a single multiplicative stopping-power scale so the model's CSDA range
matches the NIST PSTAR water reference across the clinical energy ladder.

The fit is transparent and logged: one scalar, its residual range errors, and
the reference it was fit against. No per-energy fudge factors, no hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .materials import Material, WATER
from .stopping_power import BetheStoppingPower


def load_nist_water_ranges() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load bundled NIST PSTAR water CSDA ranges: (energy_MeV, range_g_cm2)."""
    with resources.files("braggpeak.data").joinpath("nist_pstar_water_range.csv").open() as fh:
        rows = np.loadtxt(fh, delimiter=",", comments="#")
    return rows[:, 0], rows[:, 1]


def nist_range_cm(energy_mev: float, material: Material = WATER) -> float:
    """Reference CSDA range (cm) at ``energy_mev`` by log-log interpolation."""
    e, r_mass = load_nist_water_ranges()
    r = float(np.exp(np.interp(np.log(energy_mev), np.log(e), np.log(r_mass))))
    return r / material.density_g_cm3


@dataclass
class Calibration:
    """Result of a stopping-power scale fit."""

    stopping_scale: float
    energies_mev: list[float]
    baseline_range_err_mm: list[float]
    calibrated_range_err_mm: list[float]
    reference: str

    @property
    def baseline_rms_mm(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.baseline_range_err_mm))))

    @property
    def calibrated_rms_mm(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.calibrated_range_err_mm))))

    @property
    def calibrated_max_abs_mm(self) -> float:
        return float(np.max(np.abs(self.calibrated_range_err_mm)))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["baseline_rms_mm"] = self.baseline_rms_mm
        d["calibrated_rms_mm"] = self.calibrated_rms_mm
        d["calibrated_max_abs_mm"] = self.calibrated_max_abs_mm
        return d


def fit_stopping_scale(
    energies_mev: list[float] | None = None,
    *,
    material: Material = WATER,
) -> Calibration:
    """Fit one multiplicative stopping-power scale to the NIST range reference.

    Because a uniform stopping-power scale ``s`` maps range ``R -> R/s`` at every
    energy, the least-squares optimum in log-range is the geometric mean of the
    per-energy ratios ``R_model / R_ref`` -- a closed form, no iteration needed.
    """
    if energies_mev is None:
        energies_mev = [60.0, 80.0, 100.0, 120.0, 150.0, 175.0, 200.0, 230.0]

    baseline = BetheStoppingPower(material)
    ratios = []
    baseline_err = []
    ref_ranges = []
    for e in energies_mev:
        r_model = baseline.csda_range_cm(e)
        r_ref = nist_range_cm(e, material)
        ref_ranges.append(r_ref)
        ratios.append(r_model / r_ref)
        baseline_err.append((r_model - r_ref) * 10.0)  # mm

    scale = float(np.exp(np.mean(np.log(ratios))))  # geometric mean

    calibrated = BetheStoppingPower(material, stopping_scale=scale)
    calibrated_err = [
        (calibrated.csda_range_cm(e) - r_ref) * 10.0
        for e, r_ref in zip(energies_mev, ref_ranges)
    ]

    return Calibration(
        stopping_scale=scale,
        energies_mev=list(energies_mev),
        baseline_range_err_mm=baseline_err,
        calibrated_range_err_mm=calibrated_err,
        reference="NIST PSTAR water CSDA range",
    )


def calibrated_stopping_factory(scale: float):
    """Return a ``material -> BetheStoppingPower`` factory with a fixed scale."""
    return lambda m: BetheStoppingPower(m, stopping_scale=scale)


def relative_stopping_power(
    material: Material,
    *,
    energy_mev: float = 100.0,
    scale: float = 1.0,
) -> float:
    """Relative (to water) linear stopping power of ``material`` -- the RSP.

    RSP = (rho_mat S_mass_mat) / (rho_water S_mass_water), evaluated at a
    representative proton energy. This is the water-equivalent thickness per
    unit physical thickness used to compute WEPL for heterogeneous geometry.
    """
    s_mat = BetheStoppingPower(material, stopping_scale=scale).linear_stopping_power(energy_mev)[0]
    s_water = BetheStoppingPower(WATER, stopping_scale=scale).linear_stopping_power(energy_mev)[0]
    return float(s_mat / s_water)
