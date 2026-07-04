"""Physics-prior feature construction for BraggTransporter.

Inputs and outputs use the canonical units in :mod:`braggpeak.units`: depth in
cm, energy in MeV, density in g/cm^3, stopping power in MeV cm^2/g.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from braggpeak.analytic_bragg import bortfeld_depth_dose
from braggpeak.materials import WATER
from braggpeak.stopping_power import BetheStoppingPower, bortfeld_energy_mev

from .schema import PRIOR_FIELDS


def _cell_widths_cm(z_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.asarray(z_cm, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        raise ValueError("z_cm must be a nonempty 1-D depth grid in cm.")
    if z.size == 1:
        return np.asarray([float(z[0]) * 2.0 if z[0] > 0 else 1.0], dtype=np.float64)
    edges = np.empty(z.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = max(0.0, z[0] - 0.5 * (z[1] - z[0]))
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("z_cm must be strictly increasing.")
    return widths


def compute_prior(
    z_cm: NDArray[np.float64],
    material_profile: dict[str, NDArray[np.float64]],
    beam: dict[str, float],
) -> dict[str, NDArray[np.float64]]:
    """Return the five canonical physics-prior arrays.

    ``wepl_cm`` is integrated to voxel centres using the material RSP field.
    Residual energy is inferred from the Bortfeld water range-energy inverse at
    the remaining WEPL. ``bortfeld_dose`` is the analytic water curve evaluated
    against WEPL and explicitly normalised to unit peak.
    """

    z = np.asarray(z_cm, dtype=np.float64)
    density = np.asarray(material_profile["density_g_cm3"], dtype=np.float64)
    rsp = np.asarray(material_profile["rsp"], dtype=np.float64)
    if density.shape != z.shape or rsp.shape != z.shape:
        raise ValueError("material_profile fields must match z_cm shape.")

    energy_mev = float(beam["energy_mev"])
    energy_spread_pct = float(beam.get("energy_spread_pct", 0.8))
    widths = _cell_widths_cm(z)

    # Water-equivalent path length at voxel centres.
    wepl_edges = np.concatenate(([0.0], np.cumsum(rsp * widths)))
    wepl_cm = wepl_edges[:-1] + 0.5 * rsp * widths

    water_model = BetheStoppingPower(WATER)
    r0_cm = float(water_model.csda_range_cm(energy_mev))
    remaining_range_cm = np.maximum(r0_cm - wepl_cm, 0.0)
    resid_energy_mev = np.where(
        remaining_range_cm > 0.0,
        bortfeld_energy_mev(np.maximum(remaining_range_cm, 1e-12)),
        0.0,
    ).astype(np.float64)
    resid_energy_mev = np.minimum(resid_energy_mev, energy_mev)

    csda_stopping = np.zeros_like(z, dtype=np.float64)
    active = resid_energy_mev > water_model.e_min_mev
    if np.any(active):
        water_mass = water_model.mass_stopping_power(resid_energy_mev[active])
        # RSP scales water linear stopping. Convert back to local mass stopping.
        csda_stopping[active] = water_mass * rsp[active] * WATER.density_g_cm3 / density[active]

    bortfeld = bortfeld_depth_dose(
        energy_mev,
        wepl_cm,
        energy_spread_pct=energy_spread_pct,
    ).dose
    peak = float(np.max(bortfeld)) if bortfeld.size else 0.0
    bortfeld_dose = bortfeld / peak if peak > 0.0 else bortfeld

    prior = {
        "wepl_cm": wepl_cm.astype(np.float64),
        "csda_stopping": np.maximum(csda_stopping, 0.0).astype(np.float64),
        "bortfeld_dose": np.maximum(bortfeld_dose, 0.0).astype(np.float64),
        "depth_over_r0": (wepl_cm / r0_cm).astype(np.float64),
        "resid_energy_mev": resid_energy_mev.astype(np.float64),
    }
    return {k: prior[k] for k in PRIOR_FIELDS}
