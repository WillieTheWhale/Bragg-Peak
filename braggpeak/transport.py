"""Deterministic CSDA depth-dose transport (candidate analytic model).

This is the first candidate model: a 1-D continuous-slowing-down-approximation
pencil beam that steps a proton through a (possibly layered) medium using a
:class:`~braggpeak.stopping_power.StoppingPowerModel`, deposits energy locally,
and then applies physically motivated range straggling and beam energy spread
as a depth-dependent Gaussian broadening.

It is fully deterministic given its inputs; the RNG seed is recorded only for
provenance parity with the stochastic models. Nothing is normalised unless the
caller explicitly asks for it via :func:`DepthDose.normalised`.

Coordinate frame: ``z = 0`` at the phantom entrance, increasing along the beam.
Units follow :mod:`braggpeak.units` (depth cm, energy MeV, dose MeV/g).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .materials import Material, WATER
from .stopping_power import StoppingPowerModel, BetheStoppingPower


@dataclass(frozen=True)
class Slab:
    """A homogeneous layer of thickness ``thickness_cm`` of ``material``."""

    material: Material
    thickness_cm: float

    def __post_init__(self) -> None:
        if self.thickness_cm <= 0:
            raise ValueError("Slab thickness must be positive.")


@dataclass
class DepthDose:
    """Result of a depth-dose transport run with full provenance.

    All arrays are indexed on the same depth grid ``z_cm``.
    """

    z_cm: NDArray[np.float64]
    dose: NDArray[np.float64]  # broadened deposited energy per unit mass, MeV/g
    dose_ideal: NDArray[np.float64]  # pre-broadening (CSDA) dose, MeV/g
    energy_mev: NDArray[np.float64]  # residual proton energy vs depth, MeV
    lin_stopping: NDArray[np.float64]  # linear stopping power vs depth, MeV/cm
    let_kev_um: NDArray[np.float64]  # track-averaged LET vs depth, keV/um
    metadata: dict = field(default_factory=dict)

    def normalised(self) -> NDArray[np.float64]:
        """Dose divided by its peak (explicit, never applied silently)."""
        peak = float(self.dose.max())
        if peak <= 0:
            return self.dose.copy()
        return self.dose / peak


def _straggling_sigma_cm(residual_range_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Range-straggling standard deviation in water (cm).

    Bortfeld (1997) parametrisation of the monoenergetic straggling width,
    ``sigma = 0.012 * R0^0.935`` (cm) with R0 the CSDA range in water. Applied
    as a function of the local residual range so the Bragg peak broadens
    correctly with depth.
    """
    return 0.012 * np.power(np.maximum(residual_range_cm, 1e-6), 0.935)


def _build_depth_grid(slabs: Sequence[Slab], dz_cm: float) -> tuple[NDArray, NDArray]:
    """Return (z_cm edges-as-centres, per-cell material index)."""
    total = sum(s.thickness_cm for s in slabs)
    n = int(np.ceil(total / dz_cm))
    z = (np.arange(n) + 0.5) * dz_cm
    mat_idx = np.zeros(n, dtype=np.int64)
    boundaries = np.cumsum([s.thickness_cm for s in slabs])
    for i, zc in enumerate(z):
        mat_idx[i] = int(np.searchsorted(boundaries, zc, side="right"))
        if mat_idx[i] >= len(slabs):
            mat_idx[i] = len(slabs) - 1
    return z, mat_idx


def simulate_depth_dose(
    energy_mev: float,
    slabs: Sequence[Slab],
    *,
    dz_cm: float = 0.01,
    energy_spread_pct: float = 0.8,
    e_cut_mev: float = 1.0,
    fluence0: float = 1.0,
    seed: int = 0,
    stopping_model_factory=None,
) -> DepthDose:
    """Transport a monoenergetic pencil beam through layered slabs.

    Parameters
    ----------
    energy_mev:
        Initial proton kinetic energy (MeV).
    slabs:
        Ordered layers the beam traverses (entrance first).
    dz_cm:
        Depth step / voxel size (cm). Logged in metadata.
    energy_spread_pct:
        1-sigma Gaussian spread of the initial beam energy, as a percent of
        ``energy_mev``. Converted to an extra range-broadening term.
    e_cut_mev:
        Energy below which the proton is considered stopped.
    fluence0:
        Entrance fluence (a.u.); dose scales linearly with it.
    seed:
        Recorded for provenance parity with stochastic models (unused here).
    stopping_model_factory:
        Callable ``material -> StoppingPowerModel``. Defaults to Bethe-Bloch.

    Returns
    -------
    DepthDose
        Depth grid, broadened and ideal dose, residual energy, stopping power,
        LET, and a metadata dict recording every input.
    """
    if energy_mev <= e_cut_mev:
        raise ValueError("Initial energy must exceed the cut energy.")
    if not slabs:
        raise ValueError("At least one slab is required.")
    if stopping_model_factory is None:
        stopping_model_factory = lambda m: BetheStoppingPower(m)  # noqa: E731

    z_cm, mat_idx = _build_depth_grid(slabs, dz_cm)
    models: list[StoppingPowerModel] = [stopping_model_factory(s.material) for s in slabs]
    densities = np.array([s.material.density_g_cm3 for s in slabs])

    n = len(z_cm)
    energy = np.zeros(n)
    s_lin = np.zeros(n)
    dose_ideal = np.zeros(n)
    fluence = np.full(n, fluence0)  # no nuclear attenuation in this candidate

    e = float(energy_mev)
    for i in range(n):
        if e <= e_cut_mev:
            break
        model = models[mat_idx[i]]
        s = float(model.linear_stopping_power(e)[0])
        energy[i] = e
        s_lin[i] = s
        rho = densities[mat_idx[i]]
        # Energy actually deposited in this cell, capped at the residual energy
        # so the final partial step never deposits more than the proton carries.
        deposit_mev = min(s * dz_cm, e)
        dose_ideal[i] = fluence[i] * deposit_mev / (rho * dz_cm)  # MeV/g
        e = max(e - deposit_mev, 0.0)

    # LET (track-averaged here) in keV/um from linear stopping power.
    let_kev_um = s_lin * 1.0e3 / 1.0e4  # MeV/cm -> keV/um

    # Depth-dependent straggling: convert residual range at each depth to sigma.
    water_model = stopping_model_factory(WATER)
    r0_cm = water_model.csda_range_cm(energy_mev)
    residual_range = np.maximum(r0_cm - z_cm * (densities[mat_idx] / WATER.density_g_cm3), 0.0)
    sigma_strag = _straggling_sigma_cm(residual_range)
    # Beam energy spread -> range spread via dR/dE near end of range.
    sigma_espread = (energy_spread_pct / 100.0) * r0_cm * 1.77  # p * relative range spread
    sigma_cm = np.sqrt(sigma_strag**2 + sigma_espread**2)

    dose = _variable_gaussian_broaden(dose_ideal, z_cm, sigma_cm, dz_cm)

    metadata = {
        "model": "csda_transport",
        "initial_energy_mev": energy_mev,
        "dz_cm": dz_cm,
        "energy_spread_pct": energy_spread_pct,
        "e_cut_mev": e_cut_mev,
        "fluence0": fluence0,
        "seed": seed,
        "slabs": [
            {"material": s.material.name, "thickness_cm": s.thickness_cm,
             "density_g_cm3": s.material.density_g_cm3, "i_value_ev": s.material.i_value_ev}
            for s in slabs
        ],
        "water_csda_range_cm": r0_cm,
        "stopping_model": type(models[0]).__name__,
        "units": {"z": "cm", "dose": "MeV/g", "energy": "MeV", "let": "keV/um"},
    }
    return DepthDose(
        z_cm=z_cm,
        dose=dose,
        dose_ideal=dose_ideal,
        energy_mev=energy,
        lin_stopping=s_lin,
        let_kev_um=let_kev_um,
        metadata=metadata,
    )


def _variable_gaussian_broaden(
    dose: NDArray[np.float64],
    z_cm: NDArray[np.float64],
    sigma_cm: NDArray[np.float64],
    dz_cm: float,
) -> NDArray[np.float64]:
    """Convolve ``dose`` with a depth-dependent Gaussian kernel.

    Because the straggling width varies with depth, each source cell scatters
    dose into neighbours with its own local sigma. This preserves total
    deposited energy (the kernel is normalised per source cell).
    """
    n = len(dose)
    out = np.zeros(n)
    sigma_cm = np.maximum(sigma_cm, dz_cm)  # floor at one voxel
    for i in range(n):
        d = dose[i]
        if d == 0.0:
            continue
        s = sigma_cm[i]
        half = int(np.ceil(4.0 * s / dz_cm))
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        offsets = z_cm[lo:hi] - z_cm[i]
        kernel = np.exp(-0.5 * (offsets / s) ** 2)
        ksum = kernel.sum()
        if ksum > 0:
            out[lo:hi] += d * kernel / ksum
    return out
