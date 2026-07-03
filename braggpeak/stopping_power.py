"""Proton stopping-power and range models.

Three interchangeable backends share a common interface so the transport code
and the validation harness can swap them without knowing the internals:

- :class:`BetheStoppingPower` -- first-principles electronic mass stopping
  power from the Bethe formula. Physically transparent; the primary candidate
  model whose range predictions we validate.
- :func:`bortfeld_range_cm` / :class:`BortfeldRange` -- the Bortfeld (1997)
  analytic range-energy relation ``R = alpha * E^p`` for water, used as an
  independent cross-check of the Bethe range integral.
- :class:`PstarTable` -- log-log interpolation over a tabulated stopping
  power / CSDA range file (NIST PSTAR export or equivalent).

All energies are MeV, mass stopping power is MeV cm^2/g, linear stopping power
is MeV/cm, ranges are cm (unless mass units are explicit). Nothing here is
normalised silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import Material, WATER
from .units import (
    BETHE_K_MEV_CM2_PER_MOL,
    ELECTRON_MASS_MEV,
    MEV_PER_EV,
    PROTON_MASS_MEV,
)


class StoppingPowerModel(Protocol):
    """Common interface for stopping-power backends."""

    def mass_stopping_power(self, energy_mev: ArrayLike) -> NDArray[np.float64]:
        """Total mass stopping power in MeV cm^2/g at the given energy/energies."""
        ...

    def linear_stopping_power(self, energy_mev: ArrayLike) -> NDArray[np.float64]:
        """Linear stopping power in MeV/cm at the given energy/energies."""
        ...

    def csda_range_cm(self, energy_mev: float) -> float:
        """Continuous-slowing-down range in cm for a proton of ``energy_mev``."""
        ...


# --- Bethe-Bloch electronic stopping power -------------------------------


@dataclass
class BetheStoppingPower:
    """Electronic mass stopping power from the Bethe formula.

    Valid for clinical proton energies (roughly 1-300 MeV). Below ~1 MeV the
    Bethe formula loses validity (shell corrections and charge-exchange
    dominate); callers should not rely on it there. Nuclear stopping is
    negligible above a few MeV and is omitted.

    Parameters
    ----------
    material:
        Medium whose ``z_over_a``, ``i_value_ev`` and ``density_g_cm3`` are used.
    e_min_mev:
        Lower energy bound for range integration and validity guards.
    stopping_scale:
        Multiplicative correction applied to the mass stopping power. Defaults
        to 1.0 (pure Bethe). Calibration fits this factor so the model range
        matches a tabulated reference; a value >1 shortens the predicted range.
        It absorbs the small, energy-independent bias from omitted shell
        corrections and the CSDA-to-projected-range detour factor.
    """

    material: Material = WATER
    e_min_mev: float = 1.0
    stopping_scale: float = 1.0

    def _beta2_gamma(self, energy_mev: NDArray[np.float64]):
        gamma = 1.0 + energy_mev / PROTON_MASS_MEV
        beta2 = 1.0 - 1.0 / (gamma * gamma)
        return beta2, gamma

    def mass_stopping_power(self, energy_mev: ArrayLike) -> NDArray[np.float64]:
        """Mass stopping power ``-1/rho dE/dx`` in MeV cm^2/g.

        Implements ``S/rho = K (Z/A) (1/beta^2) [0.5 ln(2 m_e c^2 beta^2
        gamma^2 T_max / I^2) - beta^2]`` with the exact kinematic maximum
        energy transfer ``T_max``.
        """
        e = np.atleast_1d(np.asarray(energy_mev, dtype=np.float64))
        if np.any(e <= 0):
            raise ValueError("Energy must be strictly positive for Bethe formula.")
        beta2, gamma = self._beta2_gamma(e)

        i_mev = self.material.i_value_ev * MEV_PER_EV
        me_over_mp = ELECTRON_MASS_MEV / PROTON_MASS_MEV
        # Exact maximum kinetic energy transferable to a free electron.
        t_max = (
            2.0 * ELECTRON_MASS_MEV * beta2 * gamma * gamma
            / (1.0 + 2.0 * gamma * me_over_mp + me_over_mp * me_over_mp)
        )
        log_arg = 2.0 * ELECTRON_MASS_MEV * beta2 * gamma * gamma * t_max / (i_mev * i_mev)
        coeff = BETHE_K_MEV_CM2_PER_MOL * self.material.z_over_a / beta2
        s_mass = coeff * (0.5 * np.log(log_arg) - beta2)
        # Guard against the formula going negative at very low energy.
        s_mass = np.maximum(s_mass, 0.0) * self.stopping_scale
        return s_mass

    def linear_stopping_power(self, energy_mev: ArrayLike) -> NDArray[np.float64]:
        """Linear stopping power ``-dE/dx`` in MeV/cm."""
        return self.mass_stopping_power(energy_mev) * self.material.density_g_cm3

    def csda_range_cm(self, energy_mev: float, n_steps: int = 4000) -> float:
        """CSDA range in cm: ``integral_{e_min}^{E} dE' / S_lin(E')``.

        Uses log-spaced quadrature in energy, which resolves the rapidly
        rising stopping power near the track end far better than a linear grid.
        The residual range below ``e_min`` is added from the Bragg-Kleeman-like
        low-energy tail approximated as ``R ~ E^2`` scaling of the last cell.
        """
        e_hi = float(energy_mev)
        if e_hi <= self.e_min_mev:
            raise ValueError(
                f"Energy {e_hi} MeV must exceed e_min={self.e_min_mev} MeV."
            )
        e_grid = np.logspace(np.log10(self.e_min_mev), np.log10(e_hi), n_steps)
        s_lin = self.linear_stopping_power(e_grid)
        integrand = 1.0 / s_lin
        range_cm = float(np.trapezoid(integrand, e_grid))
        # Low-energy residual: below e_min approximate R(e_min) via R ~ E^(1+p)
        # calibration is unnecessary for clinical energies (<0.01 mm at 1 MeV);
        # add the CSDA cell explicitly so the value is never silently dropped.
        residual = self.e_min_mev / self.linear_stopping_power(self.e_min_mev)[0]
        return range_cm + residual


# --- Bortfeld analytic range-energy relation (water) ---------------------

# Bortfeld, Med. Phys. 24(12), 1997: R0 = alpha * E0^p for liquid water,
# with alpha in cm / MeV^p and p dimensionless. These reproduce NIST PSTAR
# water CSDA ranges to better than ~1% over 10-250 MeV.
BORTFELD_ALPHA_CM: float = 0.0022
BORTFELD_P: float = 1.77


def bortfeld_range_cm(energy_mev: ArrayLike) -> NDArray[np.float64]:
    """Bortfeld CSDA range in water (cm) for proton energy in MeV."""
    e = np.asarray(energy_mev, dtype=np.float64)
    return BORTFELD_ALPHA_CM * np.power(e, BORTFELD_P)


def bortfeld_energy_mev(range_cm: ArrayLike) -> NDArray[np.float64]:
    """Inverse Bortfeld relation: energy (MeV) needed to reach ``range_cm``."""
    r = np.asarray(range_cm, dtype=np.float64)
    return np.power(r / BORTFELD_ALPHA_CM, 1.0 / BORTFELD_P)


@dataclass
class BortfeldRange:
    """Range/stopping backend from the Bortfeld water parametrisation.

    Only strictly valid for water; used as an independent analytic reference.
    Linear stopping power is derived analytically as ``dE/dR``.
    """

    material: Material = WATER

    def csda_range_cm(self, energy_mev: float) -> float:
        scale = WATER.density_g_cm3 / self.material.density_g_cm3
        return float(bortfeld_range_cm(energy_mev)) * scale

    def mass_stopping_power(self, energy_mev: ArrayLike) -> NDArray[np.float64]:
        return self.linear_stopping_power(energy_mev) / self.material.density_g_cm3

    def linear_stopping_power(self, energy_mev: ArrayLike) -> NDArray[np.float64]:
        # R = alpha E^p  =>  dR/dE = alpha p E^(p-1)  =>  S = dE/dR = 1/(alpha p E^(p-1)).
        e = np.atleast_1d(np.asarray(energy_mev, dtype=np.float64))
        dr_de = BORTFELD_ALPHA_CM * BORTFELD_P * np.power(e, BORTFELD_P - 1.0)
        s_water = 1.0 / dr_de
        return s_water * (self.material.density_g_cm3 / WATER.density_g_cm3)


# --- Tabulated (PSTAR-style) backend -------------------------------------


@dataclass
class PstarTable:
    """Log-log interpolation over a tabulated stopping-power / range file.

    The file is CSV with columns ``energy_MeV, stopping_MeV_cm2_g,
    csda_range_g_cm2`` (a direct NIST PSTAR export layout). Density scaling
    converts mass range (g/cm^2) to physical range (cm).
    """

    energy_mev: NDArray[np.float64]
    mass_stopping: NDArray[np.float64]
    csda_mass_range: NDArray[np.float64]
    material: Material = WATER

    @classmethod
    def from_csv(cls, path: str | Path, material: Material = WATER) -> "PstarTable":
        rows = np.loadtxt(path, delimiter=",", comments="#")
        if rows.ndim != 2 or rows.shape[1] < 3:
            raise ValueError(
                f"PSTAR CSV {path} must have >=3 columns "
                "(energy_MeV, stopping_MeV_cm2_g, csda_range_g_cm2)."
            )
        e, s, r = rows[:, 0], rows[:, 1], rows[:, 2]
        order = np.argsort(e)
        e, s, r = e[order], s[order], r[order]
        keep = np.concatenate(([True], np.diff(e) > 0))
        return cls(e[keep], s[keep], r[keep], material)

    def _log_interp(self, x_new: ArrayLike, x: NDArray, y: NDArray) -> NDArray:
        xn = np.atleast_1d(np.asarray(x_new, dtype=np.float64))
        return np.exp(np.interp(np.log(xn), np.log(x), np.log(y)))

    def mass_stopping_power(self, energy_mev: ArrayLike) -> NDArray[np.float64]:
        return self._log_interp(energy_mev, self.energy_mev, self.mass_stopping)

    def linear_stopping_power(self, energy_mev: ArrayLike) -> NDArray[np.float64]:
        return self.mass_stopping_power(energy_mev) * self.material.density_g_cm3

    def csda_range_cm(self, energy_mev: float) -> float:
        mass_range = float(self._log_interp(energy_mev, self.energy_mev, self.csda_mass_range)[0])
        return mass_range / self.material.density_g_cm3
