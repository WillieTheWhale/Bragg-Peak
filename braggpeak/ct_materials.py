"""CT Hounsfield-unit to material / stopping-power mapping.

Maps HU values to a density and a relative stopping power (RSP, water = 1) via a
piecewise-linear stoichiometric-style calibration, and assigns each voxel a
reference :class:`~braggpeak.materials.Material`. This makes CT-to-material
conversion a pluggable strategy (as the DECT/PCT literature recommends) rather
than a hard-coded lookup, and lets a synthetic patient-like profile feed the
same transport and WEPL machinery as the slab phantoms.

Nothing is normalised implicitly: the calibration nodes, densities, and RSPs are
explicit and logged. HU is dimensionless; density is g/cm^3; RSP is unitless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import (
    Material,
    WATER,
    SOFT_TISSUE,
    LUNG,
    CORTICAL_BONE,
)

# Air as a low-density medium for CT completeness (kept out of the main
# registry). Composition uses the two elements available in the atomic tables;
# argon/CO2 traces are folded into N/O since they negligibly affect protons.
AIR = Material(
    name="air",
    density_g_cm3=1.205e-3,
    i_value_ev=85.7,
    composition={"N": 0.7553, "O": 0.2447},
)

# Piecewise-linear HU -> density (g/cm^3) calibration nodes (schneider-like).
_HU_DENSITY_NODES = np.array([
    [-1000.0, 0.001],
    [-800.0, 0.20],
    [-100.0, 0.93],
    [0.0, 1.00],
    [100.0, 1.06],
    [400.0, 1.28],
    [1000.0, 1.60],
    [2000.0, 1.90],
    [3000.0, 2.30],
])

# Piecewise-linear HU -> relative stopping power (water = 1) calibration nodes.
_HU_RSP_NODES = np.array([
    [-1000.0, 0.001],
    [-800.0, 0.21],
    [-100.0, 0.94],
    [0.0, 1.00],
    [100.0, 1.05],
    [400.0, 1.24],
    [1000.0, 1.48],
    [2000.0, 1.70],
    [3000.0, 2.00],
])


def hu_to_density(hu: ArrayLike) -> NDArray[np.float64]:
    """Piecewise-linear HU -> mass density (g/cm^3)."""
    hu = np.asarray(hu, dtype=np.float64)
    return np.interp(hu, _HU_DENSITY_NODES[:, 0], _HU_DENSITY_NODES[:, 1])


def hu_to_rsp(hu: ArrayLike) -> NDArray[np.float64]:
    """Piecewise-linear HU -> relative stopping power (water = 1)."""
    hu = np.asarray(hu, dtype=np.float64)
    return np.interp(hu, _HU_RSP_NODES[:, 0], _HU_RSP_NODES[:, 1])


# HU thresholds for assigning a representative reference material to a voxel.
_MATERIAL_BANDS = [
    (-1000.0, -500.0, AIR),
    (-500.0, -150.0, LUNG),
    (-150.0, 150.0, WATER),
    (150.0, 300.0, SOFT_TISSUE),
    (300.0, 4000.0, CORTICAL_BONE),
]


def hu_to_material(hu: float) -> Material:
    """Assign a reference material to a single HU value by band."""
    for lo, hi, mat in _MATERIAL_BANDS:
        if lo <= hu < hi:
            return mat
    return CORTICAL_BONE if hu >= 300.0 else AIR


@dataclass
class CTProfile:
    """A 1-D CT depth profile: per-voxel HU with explicit voxel size."""

    hu: NDArray[np.float64]
    dz_cm: float

    @property
    def density_g_cm3(self) -> NDArray[np.float64]:
        return hu_to_density(self.hu)

    @property
    def rsp(self) -> NDArray[np.float64]:
        return hu_to_rsp(self.hu)

    def wepl_cm(self) -> NDArray[np.float64]:
        """Cumulative water-equivalent path length (cm) along the profile."""
        return np.cumsum(self.rsp * self.dz_cm)


def synthetic_head_profile(dz_cm: float = 0.02) -> CTProfile:
    """A patient-like 1-D CT profile through a head: skin/bone/brain/bone.

    Layout along the beam (cm): 0.5 scalp soft tissue, 0.8 cortical bone (skull),
    7.0 brain (water-like), 0.8 cortical bone (far skull), 1.0 soft tissue,
    then air. Depths and HU are explicit so the geometry is fully reproducible.
    """
    segments = [
        (0.5, 40.0),     # scalp / soft tissue
        (0.8, 900.0),    # skull (cortical bone)
        (7.0, 30.0),     # brain (~water)
        (0.8, 900.0),    # far skull
        (1.0, 40.0),     # soft tissue
        (6.0, 20.0),     # residual water-like backing to catch the peak
    ]
    hu = []
    for thickness, value in segments:
        n = int(round(thickness / dz_cm))
        hu.extend([value] * n)
    return CTProfile(hu=np.array(hu, dtype=np.float64), dz_cm=dz_cm)
