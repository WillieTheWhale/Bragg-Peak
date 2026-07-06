"""Sharp 1-D BraggTransporter data generator.

This module deliberately stresses the distal edge.  Each sample mixes several
laterally distinct, entrance-ordered slab paths through water/bone/lung/air and
scores them on the same physical depth grid.  The target is still generated
from braggpeak CSDA transport, but with very small energy spread and fine dz so
the distal 80-20 falloff spans only a few voxels.  Inputs store the path-weighted
effective material profile and the usual physics prior, so the existing
BraggDataset/training code can read the HDF5 unchanged.

Units follow the fixed BraggTransporter contract: depth cm, energy MeV, density
g/cm^3, I-value eV, dose MeV/g per voxel, LET keV/um.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from braggpeak.materials import CORTICAL_BONE, LUNG, WATER, Material
from braggpeak.scoring import compute_bragg_metrics
from braggpeak.stopping_power import BetheStoppingPower
from braggpeak.transport import Slab, simulate_depth_dose
from braggtransporter.config import DataConfig
from braggtransporter.data.physics_engine import AIR, MATERIAL_CLASS
from braggtransporter.physics_prior import compute_prior
from braggtransporter.schema import (
    BEAM_FIELDS,
    MATERIAL_FIELDS,
    PRIOR_FIELDS,
    Fidelity,
    Sample,
    pack_perdepth,
    pack_scalars,
)


MATERIALS: dict[str, Material] = {
    "water": WATER,
    "bone": CORTICAL_BONE,
    "lung": LUNG,
    "air": AIR,
}


@dataclass(frozen=True)
class MixedPath:
    """One laterally distinct path through the 1-D slab stack."""

    weight: float
    slabs: tuple[Slab, ...]
    label: str


@dataclass
class SharpDataConfig:
    energies_mev: list[float]
    heldout_energies_mev: list[float]
    dz_cm: float = 0.02
    max_depth_cm: float = 34.0
    n_geometries_per_energy: int = 64
    seed: int = 0
    out_path: str = "data/generated/sharp_1d.h5"
    energy_spread_pct: float = 0.08
    n_paths: int = 3
    target_noise_pct: float = 0.0

    @classmethod
    def defaults(cls) -> "SharpDataConfig":
        cfg = DataConfig()
        return cls(
            energies_mev=list(cfg.energies_mev),
            heldout_energies_mev=list(cfg.heldout_energies_mev),
            dz_cm=0.02,
            max_depth_cm=cfg.max_depth_cm,
            n_geometries_per_energy=cfg.n_geometries_per_energy,
            seed=cfg.seed,
        )


def generate_sharp_dataset(
    cfg: SharpDataConfig,
    *,
    n_energies: int | None = None,
    n_heldout_energies: int | None = None,
    n_geometries: int | None = None,
) -> dict[str, Any]:
    """Write a sharp-edge HDF5 dataset with the Module-A schema."""

    if cfg.dz_cm <= 0.0:
        raise ValueError("dz_cm must be positive.")
    if cfg.dz_cm > 0.05:
        raise ValueError("sharp data requires dz_cm <= 0.05 cm.")

    energies = list(cfg.energies_mev[: n_energies or len(cfg.energies_mev)])
    heldout = list(cfg.heldout_energies_mev[: n_heldout_energies or len(cfg.heldout_energies_mev)])
    ngeom = int(n_geometries or cfg.n_geometries_per_energy)
    if not energies or not heldout:
        raise ValueError("at least one train and held-out energy is required.")
    if ngeom < 1:
        raise ValueError("n_geometries_per_energy must be >= 1.")

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "heldout_energy": []}
    val_count = max(1, int(round(0.2 * ngeom)))
    val_start = max(1, ngeom - val_count)

    for ei, energy in enumerate(energies):
        for gi in range(ngeom):
            sample_seed = int(cfg.seed + 100_000 * ei + gi)
            sample = simulate_sharp_sample(float(energy), cfg, gi, sample_seed)
            split = "train" if ngeom == 1 or gi < val_start else "val"
            rows[split].append(_sample_to_row(sample))
        if ngeom == 1:
            rows["val"].append(rows["train"][-1])

    for ei, energy in enumerate(heldout):
        for gi in range(ngeom):
            sample_seed = int(cfg.seed + 9_000_000 + 100_000 * ei + gi)
            rows["heldout_energy"].append(_sample_to_row(simulate_sharp_sample(float(energy), cfg, gi, sample_seed)))

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["schema"] = "braggtransporter_v3.1_module_a"
        h5.attrs["generator"] = "braggtransporter.data.sharp"
        h5.attrs["data_config_json"] = json.dumps(asdict(cfg), sort_keys=True)
        h5.attrs["material_fields"] = json.dumps(MATERIAL_FIELDS)
        h5.attrs["prior_fields"] = json.dumps(PRIOR_FIELDS)
        h5.attrs["beam_fields"] = json.dumps(BEAM_FIELDS)
        for split, split_rows in rows.items():
            _write_split(h5.create_group(split), split_rows)
        norm = _compute_norm(h5["train/x"][...], h5["train/scalars"][...])
        norm_group = h5.create_group("norm")
        for key, values in norm.items():
            norm_group.create_dataset(key, data=np.asarray(values, dtype=np.float64))

    norm_path = out_path.with_suffix(out_path.suffix + ".norm.json")
    norm_path.write_text(
        json.dumps({"source_hdf5": str(out_path), **norm}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "out_path": str(out_path),
        "norm_path": str(norm_path),
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "x_shape": rows["train"][0]["x"].shape,
        "scalar_shape": rows["train"][0]["scalars"].shape,
        "fidelity": Fidelity.ANALYTIC.name,
        "dz_cm": cfg.dz_cm,
        "max_depth_cm": cfg.max_depth_cm,
        "energy_spread_pct": cfg.energy_spread_pct,
    }


def simulate_sharp_sample(energy_mev: float, cfg: SharpDataConfig, geometry_idx: int, seed: int) -> Sample:
    """Generate one deterministic sharp-edge sample."""

    rng = np.random.default_rng(seed)
    paths = _sharp_paths(float(energy_mev), cfg.max_depth_cm, cfg.n_paths, geometry_idx, rng)
    dose_sum: NDArray[np.float64] | None = None
    let_weighted: NDArray[np.float64] | None = None
    z_cm: NDArray[np.float64] | None = None
    channel_meta: list[dict[str, Any]] = []

    for pi, path in enumerate(paths):
        depth_dose = simulate_depth_dose(
            energy_mev,
            path.slabs,
            dz_cm=cfg.dz_cm,
            energy_spread_pct=cfg.energy_spread_pct,
            seed=seed + 10_000 * pi,
        )
        channel_dose = 0.75 * depth_dose.dose + 0.25 * depth_dose.dose_ideal
        if dose_sum is None:
            z_cm = np.asarray(depth_dose.z_cm, dtype=np.float64)
            dose_sum = np.zeros_like(channel_dose, dtype=np.float64)
            let_weighted = np.zeros_like(channel_dose, dtype=np.float64)
        dose_sum += path.weight * channel_dose
        let_weighted += path.weight * channel_dose * np.maximum(depth_dose.let_kev_um, 0.0)
        channel_metrics = compute_bragg_metrics(depth_dose.z_cm, np.maximum(channel_dose, 0.0))
        channel_meta.append(
            {
                "label": path.label,
                "weight": path.weight,
                "r80_mm": channel_metrics.r80_mm,
                "falloff_80_20_mm": channel_metrics.distal_falloff_80_20_mm,
                "slabs": [_slab_meta(slab) for slab in path.slabs],
            }
        )

    assert z_cm is not None and dose_sum is not None and let_weighted is not None
    dose = np.maximum(dose_sum, 0.0)
    if cfg.target_noise_pct > 0.0:
        peak = max(float(dose.max()), 1.0e-12)
        noise = rng.normal(0.0, cfg.target_noise_pct / 100.0 * peak, size=dose.shape)
        distal = _distal_window_mask(z_cm, dose)
        dose = np.maximum(dose + distal * noise, 0.0)
    letd = np.divide(let_weighted, dose, out=np.zeros_like(dose), where=dose > 1.0e-12)

    material_profile = _mixed_material_profile(z_cm, paths, energy_mev)
    beam = {
        "energy_mev": float(energy_mev),
        "energy_spread_pct": float(cfg.energy_spread_pct),
        "spot_sigma_cm": 0.1,
    }
    beam = {k: float(beam[k]) for k in BEAM_FIELDS}
    prior = compute_prior(z_cm, material_profile, beam)
    metrics = compute_bragg_metrics(z_cm, dose)
    sample = Sample(
        z_cm=z_cm,
        dz_cm=float(cfg.dz_cm),
        ndim=1,
        material_profile=material_profile,
        beam=beam,
        prior=prior,
        fidelity=int(Fidelity.ANALYTIC),
        targets={"dose": dose.astype(np.float64), "letd": np.maximum(letd, 0.0).astype(np.float64)},
        range_r80_cm=float(metrics.r80_mm) / 10.0,
        seed=int(seed),
        source="braggpeak.csda_sharp_wepl_mix",
        meta={
            "geometry_label": f"sharp_mix_{geometry_idx:04d}",
            "sharp_design": {
                "path_count": len(paths),
                "energy_spread_pct": cfg.energy_spread_pct,
                "target_noise_pct": cfg.target_noise_pct,
                "dose_blend": "0.75*braggpeak_broadened + 0.25*braggpeak_ideal",
            },
            "channels": channel_meta,
            "metrics": {
                "r80_mm": metrics.r80_mm,
                "distal_falloff_80_20_mm": metrics.distal_falloff_80_20_mm,
            },
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


def _sharp_paths(
    energy_mev: float,
    max_depth_cm: float,
    n_paths: int,
    geometry_idx: int,
    rng: np.random.Generator,
) -> list[MixedPath]:
    n = max(2, int(n_paths))
    satellite = rng.dirichlet(np.full(n - 1, 2.0)) if n > 1 else np.asarray([], dtype=np.float64)
    raw_weights = np.zeros(n, dtype=np.float64)
    dominant_idx = geometry_idx % n
    raw_weights[dominant_idx] = 0.82
    raw_weights[np.arange(n) != dominant_idx] = 0.18 * satellite
    water_range = BetheStoppingPower(WATER).csda_range_cm(energy_mev)
    cluster_start = float(rng.uniform(0.35, 0.78) * min(water_range, max_depth_cm * 0.86))
    cluster_start = float(np.clip(cluster_start, 0.3, max_depth_cm - 1.0))

    patterns = [
        ("bone_lung", (("bone", 0.08, 0.34), ("lung", 0.06, 0.38), ("water", 0.04, 0.20))),
        ("lung_bone", (("lung", 0.10, 0.48), ("bone", 0.04, 0.22), ("water", 0.04, 0.18))),
        ("air_bone_lung", (("air", 0.03, 0.12), ("bone", 0.06, 0.26), ("lung", 0.08, 0.34))),
        ("bone_air", (("bone", 0.10, 0.42), ("air", 0.02, 0.10), ("lung", 0.05, 0.22))),
    ]

    paths: list[MixedPath] = []
    for i in range(n):
        label, pattern = patterns[(geometry_idx + i) % len(patterns)]
        start = float(np.clip(cluster_start + rng.normal(0.0, 0.08), 0.1, max_depth_cm - 0.8))
        slabs: list[Slab] = [Slab(WATER, start)]
        for material_name, lo, hi in pattern:
            thickness = float(rng.uniform(lo, hi))
            material = MATERIALS[material_name]
            remaining = max_depth_cm - sum(s.thickness_cm for s in slabs)
            if remaining <= 0.08:
                break
            slabs.append(Slab(material, min(thickness, remaining - 0.04)))
        total = sum(s.thickness_cm for s in slabs)
        if total < max_depth_cm:
            slabs.append(Slab(WATER, max_depth_cm - total))
        paths.append(MixedPath(float(raw_weights[i]), tuple(slabs), f"{label}_path_{i}"))
    weight_sum = sum(path.weight for path in paths)
    return [MixedPath(path.weight / weight_sum, path.slabs, path.label) for path in paths]


def _mixed_material_profile(
    z_cm: NDArray[np.float64],
    paths: list[MixedPath],
    energy_mev: float,
) -> dict[str, NDArray[np.float64]]:
    water_s = float(BetheStoppingPower(WATER).linear_stopping_power(energy_mev)[0])
    density = np.zeros_like(z_cm, dtype=np.float64)
    rsp = np.zeros_like(z_cm, dtype=np.float64)
    i_value = np.zeros_like(z_cm, dtype=np.float64)
    material_class = np.zeros_like(z_cm, dtype=np.float64)

    for path in paths:
        mats = _materials_for_grid(z_cm, path.slabs)
        unique: dict[str, Material] = {mat.name: mat for mat in mats}
        for mat in unique.values():
            mask = np.asarray([m.name == mat.name for m in mats], dtype=bool)
            local_s = float(BetheStoppingPower(mat).linear_stopping_power(energy_mev)[0])
            density[mask] += path.weight * mat.density_g_cm3
            rsp[mask] += path.weight * (local_s / water_s if water_s > 0.0 else 1.0)
            i_value[mask] += path.weight * mat.i_value_ev
            material_class[mask] += path.weight * float(MATERIAL_CLASS.get(mat.name, 9))

    profile = {
        "density_g_cm3": density,
        "rsp": rsp,
        "i_value_ev": i_value,
        "material_class": material_class,
    }
    return {k: profile[k].astype(np.float64) for k in MATERIAL_FIELDS}


def _materials_for_grid(z_cm: NDArray[np.float64], slabs: tuple[Slab, ...]) -> list[Material]:
    boundaries = np.cumsum([slab.thickness_cm for slab in slabs])
    idx = np.searchsorted(boundaries, z_cm, side="right")
    idx = np.minimum(idx, len(slabs) - 1)
    return [slabs[int(i)].material for i in idx]


def _distal_window_mask(z_cm: NDArray[np.float64], dose: NDArray[np.float64]) -> NDArray[np.float64]:
    peak_idx = int(np.argmax(dose))
    peak_z = z_cm[peak_idx]
    return ((z_cm >= peak_z - 0.4) & (z_cm <= peak_z + 0.8)).astype(np.float64)


def _sample_to_row(sample: Sample) -> dict[str, Any]:
    return {
        "x": pack_perdepth(sample),
        "scalars": pack_scalars(sample),
        "material_profile": np.stack([sample.material_profile[k] for k in MATERIAL_FIELDS], axis=-1),
        "prior": np.stack([sample.prior[k] for k in PRIOR_FIELDS], axis=-1),
        "z": sample.z_cm,
        "dose": sample.targets["dose"],
        "letd": sample.targets["letd"],
        "r80": np.asarray(sample.range_r80_cm, dtype=np.float64),
        "fidelity": np.asarray(sample.fidelity, dtype=np.int64),
        "energy_mev": np.asarray(sample.beam["energy_mev"], dtype=np.float64),
        "seed": np.asarray(sample.seed, dtype=np.int64),
        "meta_json": json.dumps(sample.meta, sort_keys=True),
    }


def _write_split(group: h5py.Group, rows: list[dict[str, Any]]) -> None:
    n_in = len(rows)
    rows = [r for r in rows if np.isfinite(r["r80"]) and np.all(np.isfinite(r["dose"]))]
    if len(rows) != n_in:
        print(f"[sharp] {group.name}: dropped {n_in - len(rows)} sample(s) with non-finite R80/dose")
    if not rows:
        raise ValueError(f"Split {group.name!r} has no rows.")
    for key in ("x", "scalars", "material_profile", "prior", "z", "dose", "letd"):
        group.create_dataset(key, data=np.stack([r[key] for r in rows]).astype(np.float64))
    for key in ("r80", "energy_mev"):
        group.create_dataset(key, data=np.asarray([r[key] for r in rows], dtype=np.float64))
    for key in ("fidelity", "seed"):
        group.create_dataset(key, data=np.asarray([r[key] for r in rows], dtype=np.int64))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("meta_json", data=[r["meta_json"] for r in rows], dtype=string_dtype)


def _compute_norm(train_x: NDArray[np.float64], train_scalars: NDArray[np.float64]) -> dict[str, list[float]]:
    x_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    x_std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    scalar_mean = train_scalars.mean(axis=0)
    scalar_std = train_scalars.std(axis=0)
    scalar_mean[-1] = 0.0
    scalar_std[-1] = 1.0
    return {
        "x_mean": x_mean.astype(float).tolist(),
        "x_std": np.maximum(x_std, 1.0e-8).astype(float).tolist(),
        "scalar_mean": scalar_mean.astype(float).tolist(),
        "scalar_std": np.maximum(scalar_std, 1.0e-8).astype(float).tolist(),
    }


def _slab_meta(slab: Slab) -> dict[str, Any]:
    return {
        "material": slab.material.name,
        "thickness_cm": slab.thickness_cm,
        "density_g_cm3": slab.material.density_g_cm3,
        "i_value_ev": slab.material.i_value_ev,
    }


def _parse_list(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/generated/sharp_1d.h5")
    parser.add_argument("--energies", default=None, help="Comma-separated train energies in MeV.")
    parser.add_argument("--heldout-energies", default=None, help="Comma-separated held-out energies in MeV.")
    parser.add_argument("--n-energies", type=int, default=None)
    parser.add_argument("--n-heldout-energies", type=int, default=None)
    parser.add_argument("--n-geometries", type=int, default=None)
    parser.add_argument("--dz-cm", type=float, default=0.02)
    parser.add_argument("--max-depth-cm", type=float, default=None)
    parser.add_argument("--energy-spread-pct", type=float, default=0.08)
    parser.add_argument("--n-paths", type=int, default=3)
    parser.add_argument("--noise-pct", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = SharpDataConfig.defaults()
    cfg.out_path = args.out
    cfg.dz_cm = args.dz_cm
    cfg.energy_spread_pct = args.energy_spread_pct
    cfg.n_paths = args.n_paths
    cfg.target_noise_pct = args.noise_pct
    cfg.seed = args.seed
    if args.max_depth_cm is not None:
        cfg.max_depth_cm = args.max_depth_cm
    if args.energies is not None:
        cfg.energies_mev = _parse_list(args.energies)
    if args.heldout_energies is not None:
        cfg.heldout_energies_mev = _parse_list(args.heldout_energies)

    summary = generate_sharp_dataset(
        cfg,
        n_energies=args.n_energies,
        n_heldout_energies=args.n_heldout_energies,
        n_geometries=args.n_geometries,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
