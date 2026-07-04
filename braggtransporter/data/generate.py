"""CLI for generating BraggTransporter v3.1 Module-A HDF5 datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from braggtransporter.config import DataConfig
from braggtransporter.data.physics_engine import simulate_sample
from braggtransporter.schema import (
    BEAM_FIELDS,
    MATERIAL_FIELDS,
    PRIOR_FIELDS,
    Fidelity,
    pack_perdepth,
    pack_scalars,
)


def _load_cfg(path: str | None) -> DataConfig:
    cfg = DataConfig()
    if path is None or path == "defaults":
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _fidelity(value: str) -> Fidelity:
    return Fidelity.SDE if value.lower() == "sde" else Fidelity.ANALYTIC


def _random_layered_geometry(
    rng: np.random.Generator,
    max_depth_cm: float,
    n_histories: int,
    idx: int,
) -> dict[str, Any]:
    mat = str(rng.choice(["bone", "lung", "air"]))
    start_cm = float(rng.uniform(1.0, max(1.1, max_depth_cm * 0.62)))
    thickness_max = max(0.25, min(4.0, max_depth_cm - start_cm - 0.25))
    thickness_cm = float(rng.uniform(0.25, thickness_max))
    return {
        "type": "layers",
        "label": f"{mat}_slab_{idx:04d}",
        "max_depth_cm": float(max_depth_cm),
        "n_histories": int(n_histories),
        "layers": [
            {"material": "water", "thickness_cm": start_cm},
            {"material": mat, "thickness_cm": thickness_cm},
        ],
    }


def _geometries(
    count: int,
    max_depth_cm: float,
    seed: int,
    n_histories: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = [
        {
            "type": "water",
            "label": "water",
            "max_depth_cm": float(max_depth_cm),
            "n_histories": int(n_histories),
        }
    ]
    required = ["bone", "lung", "air"]
    for i in range(1, count):
        if i <= len(required):
            mat = required[i - 1]
            start_cm = float(rng.uniform(1.0, max(1.1, max_depth_cm * 0.62)))
            thickness_cm = float(rng.uniform(0.25, min(2.0, max_depth_cm - start_cm - 0.25)))
            out.append(
                {
                    "type": "layers",
                    "label": f"{mat}_slab_{i:04d}",
                    "max_depth_cm": float(max_depth_cm),
                    "n_histories": int(n_histories),
                    "layers": [
                        {"material": "water", "thickness_cm": start_cm},
                        {"material": mat, "thickness_cm": thickness_cm},
                    ],
                }
            )
        else:
            out.append(_random_layered_geometry(rng, max_depth_cm, n_histories, i))
    return out


def _sample_to_row(sample) -> dict[str, Any]:
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


def _compute_norm(train_x: np.ndarray, train_scalars: np.ndarray) -> dict[str, list[float]]:
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


def generate_dataset(
    cfg: DataConfig,
    *,
    n_energies: int | None = None,
    n_heldout_energies: int | None = None,
    n_geometries: int | None = None,
    n_histories: int = 512,
) -> dict[str, Any]:
    energies = list(cfg.energies_mev[: n_energies or len(cfg.energies_mev)])
    heldout_energies = list(cfg.heldout_energies_mev[: n_heldout_energies or len(cfg.heldout_energies_mev)])
    ngeom = int(n_geometries or cfg.n_geometries_per_energy)
    fid = _fidelity(cfg.fidelity)

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "heldout_energy": []}
    geometry_bank = _geometries(ngeom, cfg.max_depth_cm, cfg.seed, n_histories)
    val_count = max(1, int(round(0.2 * ngeom)))
    val_start = max(1, ngeom - val_count)

    for ei, energy in enumerate(energies):
        for gi, geom in enumerate(geometry_bank):
            sample_seed = int(cfg.seed + 100_000 * ei + gi)
            row = _sample_to_row(simulate_sample(energy, geom, cfg.dz_cm, fid, sample_seed))
            if ngeom == 1:
                rows["train"].append(row)
                rows["val"].append(row)
            else:
                split = "val" if gi >= val_start else "train"
                rows[split].append(row)

    for ei, energy in enumerate(heldout_energies):
        for gi, geom in enumerate(geometry_bank):
            sample_seed = int(cfg.seed + 9_000_000 + 100_000 * ei + gi)
            rows["heldout_energy"].append(
                _sample_to_row(simulate_sample(energy, geom, cfg.dz_cm, fid, sample_seed))
            )

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["schema"] = "braggtransporter_v3.1_module_a"
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
    norm_payload = {"source_hdf5": str(out_path), **norm}
    norm_path.write_text(json.dumps(norm_payload, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "out_path": str(out_path),
        "norm_path": str(norm_path),
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "x_shape": rows["train"][0]["x"].shape,
        "scalar_shape": rows["train"][0]["scalars"].shape,
        "fidelity": fid.name,
        "dz_cm": cfg.dz_cm,
        "max_depth_cm": cfg.max_depth_cm,
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="defaults", help="YAML config path or 'defaults'.")
    parser.add_argument("--out", default=None, help="Override DataConfig.out_path.")
    parser.add_argument("--n-energies", type=int, default=None, help="Smoke-run limit for train energies.")
    parser.add_argument("--n-heldout-energies", type=int, default=None, help="Smoke-run limit for held-out energies.")
    parser.add_argument("--n-geometries", type=int, default=None, help="Smoke-run geometry count per energy.")
    parser.add_argument("--n-histories", type=int, default=512, help="SDE histories per sample.")
    parser.add_argument("--fidelity", choices=["analytic", "sde"], default=None)
    parser.add_argument("--dz-cm", type=float, default=None)
    parser.add_argument("--max-depth-cm", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    if args.out is not None:
        cfg.out_path = args.out
    if args.fidelity is not None:
        cfg.fidelity = args.fidelity
    if args.dz_cm is not None:
        cfg.dz_cm = args.dz_cm
    if args.max_depth_cm is not None:
        cfg.max_depth_cm = args.max_depth_cm

    summary = generate_dataset(
        cfg,
        n_energies=args.n_energies,
        n_heldout_energies=args.n_heldout_energies,
        n_geometries=args.n_geometries,
        n_histories=args.n_histories,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
