"""braggpeak: reproducible proton Bragg-peak simulation and validation.

Research software only. Nothing in this package is validated for clinical use
or patient decision-making.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .materials import (
    Material,
    WATER,
    SOFT_TISSUE,
    CORTICAL_BONE,
    LUNG,
    ALUMINIUM,
    get_material,
)
from .stopping_power import (
    BetheStoppingPower,
    BortfeldRange,
    PstarTable,
    bortfeld_range_cm,
    bortfeld_energy_mev,
)
from .transport import Slab, DepthDose, simulate_depth_dose
from .scoring import (
    BraggMetrics,
    compute_bragg_metrics,
    normalized_rmse,
    normalized_mae,
    gamma_index_1d,
)

__all__ = [
    "__version__",
    "Material",
    "WATER",
    "SOFT_TISSUE",
    "CORTICAL_BONE",
    "LUNG",
    "ALUMINIUM",
    "get_material",
    "BetheStoppingPower",
    "BortfeldRange",
    "PstarTable",
    "bortfeld_range_cm",
    "bortfeld_energy_mev",
    "Slab",
    "DepthDose",
    "simulate_depth_dose",
    "BraggMetrics",
    "compute_bragg_metrics",
    "normalized_rmse",
    "normalized_mae",
    "gamma_index_1d",
]
