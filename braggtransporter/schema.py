"""Core data contract for BraggTransporter (v3.1).

This module is the *fixed interface* every other module depends on. It is written
by hand (not by a code agent) so parallel implementation work cannot silently
diverge on field names, units, or shapes.

Units (see braggpeak/units.py): depth cm, energy MeV, mass stopping power
MeV cm^2/g, density g/cm^3, LET keV/um. Depth z increases along the beam with the
entrance face at z = 0. Dose is MeV/g per voxel and is never implicitly normalised.

Phase 1 is strictly 1-D (depth-dose + energy). The schema is written so the
transverse axis can be added later without breaking the 1-D contract: 1-D samples
carry ``ndim == 1`` and profiles of shape ``(Nz,)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Fidelity(IntEnum):
    """Data-fidelity tier (plan section 3). Lower = cheaper/less trusted.

    The fidelity tag is an explicit model input so the network always knows how
    much to trust a target (plan section 5, Stage 1).
    """

    ANALYTIC = 0   # Bortfeld / PSTAR closed form
    SDE = 1        # braggpeak stochastic SDE Monte Carlo
    FAST_MC = 2    # MCsquare / FRED-style (external)
    GEANT4 = 3     # OpenGATE / Geant4 / TOPAS high fidelity
    MEASUREMENT = 4  # real measured data (the only tier that licenses Claim B)


# Canonical per-depth material-profile field order for the input tensor.
MATERIAL_FIELDS: tuple[str, ...] = (
    "density_g_cm3",
    "rsp",          # relative stopping power (water = 1)
    "i_value_ev",   # mean excitation energy
    "material_class",  # integer code cast to float
)

# Canonical scalar beam-condition field order.
BEAM_FIELDS: tuple[str, ...] = (
    "energy_mev",
    "energy_spread_pct",
    "spot_sigma_cm",
)

# Physics-prior feature field order (plan section 2, layer 1). Precomputed known
# structure so the network learns a *correction*, not range-energy physics.
PRIOR_FIELDS: tuple[str, ...] = (
    "wepl_cm",              # cumulative water-equivalent path length to each z
    "csda_stopping",        # CSDA mass stopping power at each z
    "bortfeld_dose",        # analytic Bortfeld depth-dose (normalised peak=1)
    "depth_over_r0",        # z / expected CSDA range (dimensionless)
    "resid_energy_mev",     # estimated residual energy at each z
)

# Query-able output quantities (plan section 2, layer 3: coordinate-query decoder).
QUANTITIES: tuple[str, ...] = ("dose", "letd", "lett", "fluence")


@dataclass
class Sample:
    """One training example.

    All arrays are float64 numpy in physical units. Torch conversion + batching
    happens in the Dataset layer, never here. For 1-D, ``z_cm`` has shape (Nz,)
    and every per-depth array matches it.
    """

    # --- geometry / grid ---
    z_cm: NDArray[np.float64]                 # (Nz,) depth grid, entrance at 0
    dz_cm: float                              # voxel spacing
    ndim: int = 1

    # --- inputs ---
    material_profile: dict[str, NDArray[np.float64]] = field(default_factory=dict)  # keys = MATERIAL_FIELDS, each (Nz,)
    beam: dict[str, float] = field(default_factory=dict)                            # keys = BEAM_FIELDS
    prior: dict[str, NDArray[np.float64]] = field(default_factory=dict)             # keys = PRIOR_FIELDS, each (Nz,)
    fidelity: int = int(Fidelity.SDE)

    # --- targets (any subset may be present; dose is required) ---
    targets: dict[str, NDArray[np.float64]] = field(default_factory=dict)          # keys subset of QUANTITIES, each (Nz,)
    range_r80_cm: float = float("nan")        # scalar distal R80

    # --- provenance / reproducibility (echoed into every artifact) ---
    seed: int = 0
    source: str = ""                          # e.g. "braggpeak.sde v0.1"
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Fail loud on any contract violation. Called by the Dataset loader."""
        nz = self.z_cm.shape[0]
        assert self.z_cm.ndim == 1, "z_cm must be 1-D"
        for k in MATERIAL_FIELDS:
            assert k in self.material_profile, f"missing material field {k!r}"
            assert self.material_profile[k].shape == (nz,), f"{k} shape != (Nz,)"
        for k in BEAM_FIELDS:
            assert k in self.beam, f"missing beam field {k!r}"
        assert "dose" in self.targets, "dose target is required"
        for k, v in self.targets.items():
            assert k in QUANTITIES, f"unknown target quantity {k!r}"
            assert v.shape == (nz,), f"target {k} shape != (Nz,)"
        assert np.all(self.targets["dose"] >= -1e-9), "dose must be nonnegative"

    def to_serializable(self) -> dict[str, Any]:
        d = asdict(self)
        for grp in ("material_profile", "prior", "targets"):
            d[grp] = {k: np.asarray(v, dtype=np.float64) for k, v in d[grp].items()}
        d["z_cm"] = np.asarray(self.z_cm, dtype=np.float64)
        return d


# --- Tensor packing contract (used by Dataset + all models) --------------------
# Input feature channels, in fixed order, concatenated along a channel axis to
# form an (Nz, C_in) per-sample tensor:
#     [ material_profile[MATERIAL_FIELDS] , prior[PRIOR_FIELDS] ]  -> C_in = 4 + 5 = 9
# Beam scalars (BEAM_FIELDS, len 3) and the fidelity tag (1) are provided
# separately as an (C_scalar,) vector = 4, to be broadcast/conditioned by models.
C_IN_PERDEPTH: int = len(MATERIAL_FIELDS) + len(PRIOR_FIELDS)   # 9
C_SCALAR: int = len(BEAM_FIELDS) + 1                            # 4 (+fidelity)


def pack_perdepth(sample: Sample) -> NDArray[np.float64]:
    """(Nz, C_IN_PERDEPTH) input feature matrix in the canonical channel order."""
    cols = [sample.material_profile[k] for k in MATERIAL_FIELDS]
    cols += [sample.prior[k] for k in PRIOR_FIELDS]
    return np.stack(cols, axis=-1).astype(np.float64)


def pack_scalars(sample: Sample) -> NDArray[np.float64]:
    """(C_SCALAR,) beam + fidelity conditioning vector."""
    vals = [sample.beam[k] for k in BEAM_FIELDS] + [float(sample.fidelity)]
    return np.asarray(vals, dtype=np.float64)
