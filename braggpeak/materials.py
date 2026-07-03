"""Material definitions for proton transport.

A :class:`Material` carries everything the stopping-power models need:
mass density, mean excitation energy ``I`` (eV), and elemental composition
expressed as mass fractions. The effective ``Z/A`` used by the Bethe formula
is derived from the composition so that no material hard-codes a value that
could drift out of sync with its atomic makeup.

Reference values follow ICRU Report 37 / NIST STAR conventions. These are
research inputs, not a clinical material library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Standard atomic weights (g/mol) for the elements used in tissue-like media.
_ATOMIC_WEIGHT: Dict[str, float] = {
    "H": 1.007_94,
    "C": 12.010_7,
    "N": 14.006_7,
    "O": 15.999_4,
    "Na": 22.989_77,
    "Mg": 24.305,
    "P": 30.973_76,
    "S": 32.065,
    "Cl": 35.453,
    "K": 39.098_3,
    "Ca": 40.078,
    "Al": 26.981_54,
}
_ATOMIC_NUMBER: Dict[str, int] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "Na": 11,
    "Mg": 12,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
    "Al": 13,
}


@dataclass(frozen=True)
class Material:
    """A homogeneous medium for stopping-power evaluation.

    Attributes
    ----------
    name:
        Human-readable identifier used in logs and reports.
    density_g_cm3:
        Mass density in g/cm^3.
    i_value_ev:
        Mean excitation energy in eV.
    composition:
        Mapping of element symbol to mass fraction. Fractions are normalised
        on construction; they need not sum to exactly 1 on input.
    """

    name: str
    density_g_cm3: float
    i_value_ev: float
    composition: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.density_g_cm3 <= 0:
            raise ValueError(f"{self.name}: density must be positive.")
        if self.i_value_ev <= 0:
            raise ValueError(f"{self.name}: I-value must be positive.")
        if not self.composition:
            raise ValueError(f"{self.name}: composition must not be empty.")
        total = sum(self.composition.values())
        if total <= 0:
            raise ValueError(f"{self.name}: composition mass fractions sum to <= 0.")
        # Normalise mass fractions in place (dataclass is frozen -> object.__setattr__).
        normalised = {el: w / total for el, w in self.composition.items()}
        object.__setattr__(self, "composition", normalised)

    @property
    def z_over_a(self) -> float:
        """Effective ``Z/A`` (electrons per gram, in mol/g x Z) for the mixture.

        Computed as ``sum_i w_i * Z_i / A_i`` where ``w_i`` are mass fractions.
        This is the quantity that enters the Bethe stopping-power formula.
        """
        total = 0.0
        for el, w in self.composition.items():
            if el not in _ATOMIC_NUMBER:
                raise KeyError(f"Unknown element '{el}' in material '{self.name}'.")
            total += w * _ATOMIC_NUMBER[el] / _ATOMIC_WEIGHT[el]
        return total

    def describe(self) -> str:
        """One-line provenance string for logs and report headers."""
        comp = ", ".join(f"{el}:{w:.4f}" for el, w in sorted(self.composition.items()))
        return (
            f"{self.name} (rho={self.density_g_cm3:g} g/cm^3, "
            f"I={self.i_value_ev:g} eV, Z/A={self.z_over_a:.4f}, [{comp}])"
        )


# --- Reference materials (ICRU 37 / NIST STAR mean excitation energies) ---

WATER = Material(
    name="liquid_water",
    density_g_cm3=1.0,
    i_value_ev=78.0,
    composition={"H": 0.111_894, "O": 0.888_106},
)

# ICRP soft tissue (approximate elemental makeup), I = 72.3 eV.
SOFT_TISSUE = Material(
    name="icrp_soft_tissue",
    density_g_cm3=1.03,
    i_value_ev=72.3,
    composition={
        "H": 0.104_5,
        "C": 0.232_1,
        "N": 0.024_9,
        "O": 0.630_8,
        "Na": 0.001_1,
        "P": 0.001_3,
        "S": 0.001_9,
        "Cl": 0.001_3,
        "K": 0.001_9,
    },
)

# ICRU compact bone (cortical), I = 91.9 eV.
CORTICAL_BONE = Material(
    name="icru_cortical_bone",
    density_g_cm3=1.85,
    i_value_ev=91.9,
    composition={
        "H": 0.047_234,
        "C": 0.144_330,
        "N": 0.041_990,
        "O": 0.446_096,
        "Mg": 0.002_20,
        "P": 0.104_970,
        "S": 0.003_15,
        "Ca": 0.209_930,
    },
)

# ICRU lung (inflated), I = 75.3 eV.
LUNG = Material(
    name="icru_lung_inflated",
    density_g_cm3=0.26,
    i_value_ev=75.3,
    composition={
        "H": 0.101_3,
        "C": 0.102_3,
        "N": 0.028_0,
        "O": 0.757_2,
        "Na": 0.001_84,
        "P": 0.000_80,
        "S": 0.002_25,
        "Cl": 0.002_66,
        "K": 0.001_94,
    },
)

# Elemental aluminium, I = 166 eV (common range-shifter / QA material).
ALUMINIUM = Material(
    name="aluminium",
    density_g_cm3=2.699,
    i_value_ev=166.0,
    composition={"Al": 1.0},
)

REGISTRY: Dict[str, Material] = {
    m.name: m
    for m in (WATER, SOFT_TISSUE, CORTICAL_BONE, LUNG, ALUMINIUM)
}


def get_material(name: str) -> Material:
    """Look up a reference material by canonical name.

    Raises ``KeyError`` with the available names if not found.
    """
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown material '{name}'. Available: {sorted(REGISTRY)}"
        ) from exc
