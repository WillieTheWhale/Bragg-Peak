"""Explicit unit conventions for the braggpeak platform.

The whole package uses a single, explicit set of base units so that no
quantity is ever silently rescaled. Every physical value passed between
modules is documented in these units:

- length / depth:        centimetre (cm)
- energy:                mega-electron-volt (MeV)
- mass stopping power:   MeV cm^2 / g
- linear stopping power: MeV / cm
- density:               g / cm^3
- mean excitation (I):   eV (converted to MeV internally where needed)
- LET:                   keV / um
- dose:                  arbitrary units unless a mass is supplied, then MeV/g

Coordinate frame: depth ``z`` increases along the beam direction, with the
phantom entrance face at ``z = 0``. There is no implicit source-to-surface
offset. Callers must state any offset explicitly.
"""

from __future__ import annotations

# Physical constants (PDG / CODATA rounded to the precision this package needs).
ELECTRON_MASS_MEV: float = 0.510_998_95  # electron rest energy, MeV
PROTON_MASS_MEV: float = 938.272_089  # proton rest energy, MeV
# K = 4 pi N_A r_e^2 m_e c^2, the Bethe coefficient, in MeV cm^2 / mol.
BETHE_K_MEV_CM2_PER_MOL: float = 0.307_075

# Unit conversion helpers -------------------------------------------------
MEV_PER_EV: float = 1.0e-6
CM_PER_MM: float = 0.1
MM_PER_CM: float = 10.0
KEV_PER_MEV: float = 1.0e3
UM_PER_CM: float = 1.0e4


def mm_to_cm(x_mm: float) -> float:
    """Convert millimetres to centimetres."""
    return x_mm * CM_PER_MM


def cm_to_mm(x_cm: float) -> float:
    """Convert centimetres to millimetres."""
    return x_cm * MM_PER_CM


def ev_to_mev(x_ev: float) -> float:
    """Convert electron-volts to mega-electron-volts."""
    return x_ev * MEV_PER_EV


def mev_cm_to_kev_um(let_mev_per_cm: float) -> float:
    """Convert a linear energy loss in MeV/cm to keV/um."""
    return let_mev_per_cm * KEV_PER_MEV / UM_PER_CM
