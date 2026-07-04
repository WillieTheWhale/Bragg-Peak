"""Sample-level data generation backed by :mod:`braggpeak` physics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from braggpeak.materials import CORTICAL_BONE, LUNG, WATER, Material
from braggpeak.scoring import compute_bragg_metrics
from braggpeak.sde_model import simulate_depth_dose_sde
from braggpeak.stopping_power import BetheStoppingPower
from braggpeak.transport import Slab, simulate_depth_dose
from braggpeak.units import KEV_PER_MEV, UM_PER_CM

from braggtransporter.physics_prior import compute_prior
from braggtransporter.schema import BEAM_FIELDS, Fidelity, MATERIAL_FIELDS, Sample


AIR = Material(
    name="dry_air",
    density_g_cm3=0.001205,
    i_value_ev=85.7,
    composition={"N": 0.755, "O": 0.232, "C": 0.013},
)

MATERIALS: dict[str, Material] = {
    "water": WATER,
    "liquid_water": WATER,
    "bone": CORTICAL_BONE,
    "cortical_bone": CORTICAL_BONE,
    "icru_cortical_bone": CORTICAL_BONE,
    "lung": LUNG,
    "icru_lung_inflated": LUNG,
    "air": AIR,
    "dry_air": AIR,
}

MATERIAL_CLASS: dict[str, int] = {
    WATER.name: 0,
    LUNG.name: 1,
    CORTICAL_BONE.name: 2,
    AIR.name: 3,
}


@dataclass(frozen=True)
class GeometrySpec:
    slabs: tuple[Slab, ...]
    max_depth_cm: float
    label: str


def _material_from_name(name: str) -> Material:
    key = name.lower()
    if key not in MATERIALS:
        raise KeyError(f"Unknown BraggTransporter material {name!r}.")
    return MATERIALS[key]


def _coerce_fidelity(fidelity: Fidelity | str | int) -> Fidelity:
    if isinstance(fidelity, Fidelity):
        return fidelity
    if isinstance(fidelity, str):
        return Fidelity[fidelity.upper()]
    return Fidelity(int(fidelity))


def _geometry_to_slabs(geometry: str | dict[str, Any] | list[dict[str, Any]]) -> GeometrySpec:
    if isinstance(geometry, str):
        geometry = {"type": geometry}
    if isinstance(geometry, list):
        geometry = {"type": "layers", "layers": geometry}

    if not isinstance(geometry, dict):
        raise TypeError("geometry must be a string, dict, or list of layer dicts.")

    max_depth_cm = float(geometry.get("max_depth_cm", 34.0))
    gtype = str(geometry.get("type", "water")).lower()
    label = str(geometry.get("label", gtype))

    if gtype == "water":
        slabs = [Slab(WATER, max_depth_cm)]
    else:
        layer_defs = geometry.get("layers") or geometry.get("slabs")
        if not layer_defs:
            raise ValueError("Layered geometries require a 'layers' list.")
        slabs = []
        for layer in layer_defs:
            material = _material_from_name(str(layer["material"]))
            slabs.append(Slab(material, float(layer["thickness_cm"])))
        total = sum(s.thickness_cm for s in slabs)
        if total < max_depth_cm:
            slabs.append(Slab(WATER, max_depth_cm - total))
        elif total > max_depth_cm:
            trimmed: list[Slab] = []
            acc = 0.0
            for slab in slabs:
                remaining = max_depth_cm - acc
                if remaining <= 0.0:
                    break
                trimmed.append(Slab(slab.material, min(slab.thickness_cm, remaining)))
                acc += trimmed[-1].thickness_cm
            slabs = trimmed
    return GeometrySpec(tuple(slabs), max_depth_cm, label)


def _material_profile(
    z_cm: NDArray[np.float64],
    slabs: tuple[Slab, ...],
    energy_mev: float,
) -> dict[str, NDArray[np.float64]]:
    boundaries = np.cumsum([s.thickness_cm for s in slabs])
    material_idx = np.searchsorted(boundaries, z_cm, side="right")
    material_idx = np.minimum(material_idx, len(slabs) - 1)

    water_s = float(BetheStoppingPower(WATER).linear_stopping_power(energy_mev)[0])
    density = np.zeros_like(z_cm, dtype=np.float64)
    rsp = np.zeros_like(z_cm, dtype=np.float64)
    i_value = np.zeros_like(z_cm, dtype=np.float64)
    material_class = np.zeros_like(z_cm, dtype=np.float64)
    for i, slab in enumerate(slabs):
        mask = material_idx == i
        mat = slab.material
        density[mask] = mat.density_g_cm3
        local_s = float(BetheStoppingPower(mat).linear_stopping_power(energy_mev)[0])
        rsp[mask] = local_s / water_s if water_s > 0.0 else 1.0
        i_value[mask] = mat.i_value_ev
        material_class[mask] = float(MATERIAL_CLASS.get(mat.name, 9))

    profile = {
        "density_g_cm3": density,
        "rsp": rsp,
        "i_value_ev": i_value,
        "material_class": material_class,
    }
    return {k: profile[k].astype(np.float64) for k in MATERIAL_FIELDS}


def _let_proxy_from_prior(
    prior: dict[str, NDArray[np.float64]],
    material_profile: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    lin_stop = prior["csda_stopping"] * material_profile["density_g_cm3"]
    return np.maximum(lin_stop * KEV_PER_MEV / UM_PER_CM, 0.0).astype(np.float64)


def simulate_sample(
    energy_mev: float,
    geometry: str | dict[str, Any] | list[dict[str, Any]],
    dz_cm: float,
    fidelity: Fidelity | str | int,
    seed: int,
) -> Sample:
    """Build one validated 1-D BraggTransporter sample.

    ``geometry`` may be ``"water"`` or a dict/list describing entrance-ordered
    slabs with material names and thicknesses in cm. All arrays are float64 in
    physical units; the Dataset layer handles torch conversion.
    """

    fid = _coerce_fidelity(fidelity)
    spec = _geometry_to_slabs(geometry)
    energy_spread_pct = float(geometry.get("energy_spread_pct", 0.8)) if isinstance(geometry, dict) else 0.8
    spot_sigma_cm = float(geometry.get("spot_sigma_cm", 0.1)) if isinstance(geometry, dict) else 0.1
    n_histories = int(geometry.get("n_histories", 512)) if isinstance(geometry, dict) else 512

    if fid == Fidelity.SDE:
        depth_dose = simulate_depth_dose_sde(
            float(energy_mev),
            spec.slabs,
            dz_cm=float(dz_cm),
            n_histories=n_histories,
            energy_spread_pct=energy_spread_pct,
            seed=int(seed),
        )
    elif fid == Fidelity.ANALYTIC:
        depth_dose = simulate_depth_dose(
            float(energy_mev),
            spec.slabs,
            dz_cm=float(dz_cm),
            energy_spread_pct=energy_spread_pct,
            seed=int(seed),
        )
    else:
        raise ValueError(f"Module A can generate ANALYTIC or SDE samples, not {fid.name}.")

    z_cm = np.asarray(depth_dose.z_cm, dtype=np.float64)
    material_profile = _material_profile(z_cm, spec.slabs, float(energy_mev))
    beam = {
        "energy_mev": float(energy_mev),
        "energy_spread_pct": energy_spread_pct,
        "spot_sigma_cm": spot_sigma_cm,
    }
    beam = {k: float(beam[k]) for k in BEAM_FIELDS}
    prior = compute_prior(z_cm, material_profile, beam)

    dose = np.maximum(np.asarray(depth_dose.dose, dtype=np.float64), 0.0)
    letd = np.asarray(getattr(depth_dose, "let_kev_um", np.zeros_like(dose)), dtype=np.float64)
    let_source = "braggpeak_depth_dose.let_kev_um"
    if letd.shape != dose.shape or not np.any(letd > 0.0):
        letd = _let_proxy_from_prior(prior, material_profile)
        let_source = "csda_stopping_proxy_monotone_to_distal"
    letd = np.maximum(letd, 0.0)

    metrics = compute_bragg_metrics(z_cm, dose)
    sample = Sample(
        z_cm=z_cm,
        dz_cm=float(dz_cm),
        ndim=1,
        material_profile=material_profile,
        beam=beam,
        prior=prior,
        fidelity=int(fid),
        targets={"dose": dose, "letd": letd},
        range_r80_cm=float(metrics.r80_mm) / 10.0,
        seed=int(seed),
        source=f"braggpeak.{depth_dose.metadata.get('model', 'unknown')}",
        meta={
            "geometry_label": spec.label,
            "geometry": {
                "max_depth_cm": spec.max_depth_cm,
                "slabs": [
                    {
                        "material": slab.material.name,
                        "thickness_cm": slab.thickness_cm,
                        "density_g_cm3": slab.material.density_g_cm3,
                        "i_value_ev": slab.material.i_value_ev,
                    }
                    for slab in spec.slabs
                ],
            },
            "letd_source": let_source,
            "physics_metadata": depth_dose.metadata,
            "units": {
                "z": "cm",
                "dz": "cm",
                "energy": "MeV",
                "dose": "MeV/g",
                "letd": "keV/um",
                "density": "g/cm^3",
                "i_value": "eV",
            },
        },
    )
    sample.validate()
    return sample
