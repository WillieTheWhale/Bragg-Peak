"""Stochastic differential equation (SDE) proton transport (candidate model 2).

A lightweight Monte Carlo that integrates each proton's energy along depth as
an SDE:

    dE = -S(E) dz + sqrt(Omega^2(E) dz) dW

where ``S`` is the linear stopping power and ``Omega^2`` is the Bohr
energy-loss straggling variance per unit length. Energy deposited per voxel is
scored directly, so the Bragg-peak broadening emerges from the stochastic
straggling rather than from an analytic convolution (contrast with
:mod:`braggpeak.transport`). This mirrors the SDE-based fast dose approach
benchmarked against Geant4 in the 2026 literature.

Deterministic given a seed: the same seed reproduces the histories bit-for-bit.
Units follow :mod:`braggpeak.units`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .materials import WATER
from .stopping_power import BetheStoppingPower
from .transport import DepthDose, Slab, _build_depth_grid


# Bohr energy-loss straggling constant: Omega^2 = XI * (Z/A) * rho * dz  [MeV^2],
# with XI = 4 pi N_A r_e^2 (m_e c^2)^2 z^2 = 0.1569 MeV^2 cm^2 / g for protons.
BOHR_XI_MEV2_CM2_PER_G: float = 0.1569


def _straggling_variance_per_cm(z_over_a: float, density: float, beta2: NDArray) -> NDArray:
    """Bohr straggling variance per cm (MeV^2/cm), with relativistic factor."""
    rel = (1.0 - 0.5 * beta2) / np.maximum(1.0 - beta2, 1e-9)
    return BOHR_XI_MEV2_CM2_PER_G * z_over_a * density * rel


def simulate_depth_dose_sde(
    energy_mev: float,
    slabs: Sequence[Slab],
    *,
    dz_cm: float = 0.02,
    n_histories: int = 20000,
    energy_spread_pct: float = 0.8,
    e_cut_mev: float = 1.0,
    seed: int = 0,
    nuclear_removal: bool = True,
    stopping_model_factory=None,
) -> DepthDose:
    """Monte Carlo SDE transport of a monoenergetic pencil beam.

    Parameters mirror :func:`braggpeak.transport.simulate_depth_dose` where they
    overlap. ``n_histories`` protons are transported; ``nuclear_removal`` applies
    an approximate proton-loss probability (~1%/cm in water) that lowers the
    Bragg peak relative to entrance, as observed experimentally.

    Returns a :class:`~braggpeak.transport.DepthDose` on the same voxel grid as
    the deterministic model, so both feed the identical scorer.
    """
    if energy_mev <= e_cut_mev:
        raise ValueError("Initial energy must exceed the cut energy.")
    if not slabs:
        raise ValueError("At least one slab is required.")
    if stopping_model_factory is None:
        stopping_model_factory = lambda m: BetheStoppingPower(m)  # noqa: E731

    rng = np.random.default_rng(seed)
    z_cm, mat_idx = _build_depth_grid(slabs, dz_cm)
    n = len(z_cm)
    models = [stopping_model_factory(s.material) for s in slabs]
    densities = np.array([s.material.density_g_cm3 for s in slabs])
    z_over_a = np.array([s.material.z_over_a for s in slabs])
    from .units import PROTON_MASS_MEV

    edep = np.zeros(n)  # total energy deposited per voxel (MeV, summed over histories)
    alive_count = np.zeros(n)  # protons entering each voxel (for LET averaging)
    lin_stop_accum = np.zeros(n)

    # Nuclear inelastic removal: ~0.01 /cm in water, scaled by density.
    nuclear_mu_per_cm = 0.0100 if nuclear_removal else 0.0

    # Initial energies with Gaussian spread.
    sigma_e0 = energy_spread_pct / 100.0 * energy_mev
    e_hist = rng.normal(energy_mev, sigma_e0, size=n_histories)
    e_hist = np.maximum(e_hist, e_cut_mev + 1e-6)
    alive = np.ones(n_histories, dtype=bool)

    for i in range(n):
        if not np.any(alive):
            break
        mi = mat_idx[i]
        model = models[mi]
        rho = densities[mi]
        za = z_over_a[mi]

        e_active = e_hist[alive]
        s_lin = model.linear_stopping_power(e_active)  # MeV/cm
        gamma = 1.0 + e_active / PROTON_MASS_MEV
        beta2 = 1.0 - 1.0 / (gamma * gamma)
        omega2 = _straggling_variance_per_cm(za, rho, beta2)  # MeV^2/cm

        mean_loss = s_lin * dz_cm
        std_loss = np.sqrt(np.maximum(omega2 * dz_cm, 0.0))
        loss = rng.normal(mean_loss, std_loss)
        loss = np.clip(loss, 0.0, e_active)  # cannot deposit more than carried

        # Score deposition into this voxel.
        idx_alive = np.where(alive)[0]
        np.add.at(edep, i, loss.sum())
        alive_count[i] += e_active.size
        lin_stop_accum[i] += s_lin.sum()

        e_hist[idx_alive] = e_active - loss

        # Nuclear removal this step.
        if nuclear_mu_per_cm > 0:
            p_remove = 1.0 - np.exp(-nuclear_mu_per_cm * rho * dz_cm)
            removed = rng.random(idx_alive.size) < p_remove
            alive[idx_alive[removed]] = False

        # Stop protons below the cut.
        stopped = e_hist[idx_alive] <= e_cut_mev
        alive[idx_alive[stopped]] = False

    # Convert scored energy to dose (MeV/g): edep / (rho * dz) per history-average.
    rho_vox = densities[mat_idx]
    dose = edep / (rho_vox * dz_cm * n_histories)
    # Mean linear stopping power per voxel (for LET), guarding empty voxels.
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_slin = np.where(alive_count > 0, lin_stop_accum / np.maximum(alive_count, 1), 0.0)
    let_kev_um = mean_slin * 1.0e3 / 1.0e4
    energy_profile = np.zeros(n)  # per-voxel mean residual energy not tracked; leave zeros

    water_model = stopping_model_factory(WATER)
    metadata = {
        "model": "sde_monte_carlo",
        "initial_energy_mev": energy_mev,
        "dz_cm": dz_cm,
        "n_histories": n_histories,
        "energy_spread_pct": energy_spread_pct,
        "e_cut_mev": e_cut_mev,
        "seed": seed,
        "nuclear_removal": nuclear_removal,
        "slabs": [
            {"material": s.material.name, "thickness_cm": s.thickness_cm,
             "density_g_cm3": s.material.density_g_cm3, "i_value_ev": s.material.i_value_ev}
            for s in slabs
        ],
        "water_csda_range_cm": water_model.csda_range_cm(energy_mev),
        "stopping_model": type(models[0]).__name__,
        "units": {"z": "cm", "dose": "MeV/g", "energy": "MeV", "let": "keV/um"},
    }
    return DepthDose(
        z_cm=z_cm,
        dose=dose,
        dose_ideal=dose.copy(),
        energy_mev=energy_profile,
        lin_stopping=mean_slin,
        let_kev_um=let_kev_um,
        metadata=metadata,
    )


def dose_uncertainty(
    energy_mev: float,
    slabs: Sequence[Slab],
    *,
    n_replicas: int = 5,
    base_seed: int = 0,
    **kwargs,
) -> tuple[DepthDose, NDArray[np.float64]]:
    """Run several independent SDE replicas to estimate per-voxel 1-sigma dose.

    Returns the mean :class:`DepthDose` and a per-voxel standard-error array
    (statistical uncertainty of the mean), so runs report uncertainty intervals.
    """
    runs = [
        simulate_depth_dose_sde(energy_mev, slabs, seed=base_seed + k, **kwargs)
        for k in range(n_replicas)
    ]
    stack = np.vstack([r.dose for r in runs])
    mean_dose = stack.mean(axis=0)
    stderr = stack.std(axis=0, ddof=1) / np.sqrt(n_replicas)
    mean = runs[0]
    mean.dose = mean_dose
    mean.dose_ideal = mean_dose.copy()
    mean.metadata = dict(mean.metadata)
    mean.metadata["n_replicas"] = n_replicas
    return mean, stderr
