"""Bortfeld (1997) closed-form analytic Bragg curve for water.

Reference: T. Bortfeld, "An analytical approximation of the Bragg curve for
therapeutic proton beams", Med. Phys. 24(12):2024-2033, 1997.

This provides an established analytic depth-dose ``D(z)`` that includes range
straggling (parabolic-cylinder broadening) and the nuclear fluence reduction,
which we use as an independent *shape* reference for the candidate transport
models -- not merely a range checkpoint. Valid for liquid water and clinical
proton energies (roughly 10-250 MeV).

Units: depth cm, energy MeV, dose in MeV/g (per unit primary fluence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from .stopping_power import BORTFELD_ALPHA_CM, BORTFELD_P

# Bortfeld model constants (his Table/eqs, liquid water).
_BETA_PER_CM = 0.012      # slope of primary-fluence reduction with depth (1/cm)
_GAMMA = 0.6              # fraction of nuclear-released energy absorbed locally
_EPSILON = 0.2           # low-energy tail fraction of primary fluence
_RHO_WATER = 1.0         # g/cm^3


def range_straggling_sigma_cm(range_cm: float) -> float:
    """Monoenergetic range-straggling width sigma (cm) for water (Bortfeld)."""
    return 0.012 * range_cm**0.935


def _sigma_total_cm(range_cm: float, energy_mev: float, energy_spread_pct: float) -> float:
    """Total straggling width folding in beam energy spread."""
    sigma_mono = range_straggling_sigma_cm(range_cm)
    # d R / d E = alpha p E^(p-1); range spread from energy spread.
    sigma_e = energy_spread_pct / 100.0 * energy_mev
    sigma_range_from_e = BORTFELD_ALPHA_CM * BORTFELD_P * energy_mev ** (BORTFELD_P - 1.0) * sigma_e
    return float(np.sqrt(sigma_mono**2 + sigma_range_from_e**2))


@dataclass
class BortfeldCurve:
    """Analytic Bragg curve on a depth grid with its defining parameters."""

    z_cm: NDArray[np.float64]
    dose: NDArray[np.float64]
    range_cm: float
    sigma_cm: float
    energy_mev: float


def bortfeld_depth_dose(
    energy_mev: float,
    z_cm: ArrayLike,
    *,
    energy_spread_pct: float = 0.8,
    fluence0: float = 1.0,
) -> BortfeldCurve:
    """Evaluate the Bortfeld analytic Bragg curve at depths ``z_cm``.

    Implements Bortfeld eq. (28): a superposition of two parabolic-cylinder
    functions ``D_{-1/p}`` and ``D_{-1/p-1}`` of the scaled depth
    ``xi = (z - R0)/sigma``, weighted by the straggling width and the nuclear
    fluence terms. Proximal to the straggling region it reduces to the smooth
    power-law build-up; distal to the peak it falls off sharply.
    """
    z = np.asarray(z_cm, dtype=np.float64)
    p = BORTFELD_P
    alpha = BORTFELD_ALPHA_CM
    r0 = alpha * energy_mev**p
    sigma = _sigma_total_cm(r0, energy_mev, energy_spread_pct)

    xi = (z - r0) / sigma
    coef = _BETA_PER_CM / p + _GAMMA * _BETA_PER_CM + _EPSILON / r0
    dose = np.zeros_like(z)

    # Near-peak region: parabolic-cylinder form (Bortfeld eq. 28). Restricted to
    # |xi| <= 8 so the e^{-xi^2/4} prefactor and D_v(xi) stay in float64 range.
    near = np.abs(xi) <= 8.0
    if np.any(near):
        xn = xi[near]
        dv1 = special.pbdv(-1.0 / p, xn)[0]
        dv2 = special.pbdv(-1.0 / p - 1.0, xn)[0]
        gauss = np.exp(-(xn**2) / 4.0)
        prefactor = (
            fluence0
            * sigma ** (1.0 / p)
            * special.gamma(1.0 / p)
            / (np.sqrt(2.0 * np.pi) * _RHO_WATER * p * alpha ** (1.0 / p) * (1.0 + _BETA_PER_CM * r0))
        )
        dose[near] = prefactor * gauss * ((1.0 / sigma) * dv1 + coef * dv2)

    # Far proximal region (xi < -8): straggling negligible -> exact power-law
    # limit of eq. (28), i.e. Bortfeld eq. (29). Distal region (xi > 8) stays 0.
    far = xi < -8.0
    if np.any(far):
        depth_to_go = r0 - z[far]  # > 0
        dose[far] = (
            fluence0
            / (_RHO_WATER * alpha ** (1.0 / p) * (1.0 + _BETA_PER_CM * r0))
            * (depth_to_go ** (1.0 / p - 1.0) / p + coef * depth_to_go ** (1.0 / p))
        )

    dose = np.maximum(dose, 0.0)  # clamp tiny negative numerical wings
    return BortfeldCurve(z_cm=z, dose=dose, range_cm=r0, sigma_cm=sigma, energy_mev=energy_mev)
