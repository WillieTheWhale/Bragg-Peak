#!/usr/bin/env python
"""Generate low/high-statistics SDE dose pairs for future sigma_MC training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from braggtransporter.config import DataConfig
from braggtransporter.data.generate import _geometries, _load_cfg
from braggtransporter.data.physics_engine import simulate_sample
from braggtransporter.schema import Fidelity, pack_perdepth, pack_scalars


def generate_mc_pairs(
    cfg: DataConfig,
    *,
    out_path: str | Path,
    low_histories: int = 64,
    high_histories: int = 2048,
    n_energies: int | None = None,
    n_geometries: int | None = None,
) -> dict[str, Any]:
    """Write a separate HDF5 with matched low/high-statistics SDE dose pairs.

    This is the Phase-4/5 data intervention that makes ``sigma_MC`` identifiable:
    beam, material, geometry, seed, and voxel grid are held fixed while only the
    particle-history count changes.
    """

    energies = list(cfg.energies_mev[: n_energies or len(cfg.energies_mev)])
    ngeom = int(n_geometries or cfg.n_geometries_per_energy)
    geometry_bank = _geometries(ngeom, cfg.max_depth_cm, cfg.seed, high_histories)

    rows: list[dict[str, Any]] = []
    for ei, energy in enumerate(energies):
        for gi, geom in enumerate(geometry_bank):
            seed = int(cfg.seed + 7_000_000 + 100_000 * ei + gi)
            low_geom = dict(geom)
            high_geom = dict(geom)
            low_geom["n_histories"] = int(low_histories)
            high_geom["n_histories"] = int(high_histories)
            low = simulate_sample(energy, low_geom, cfg.dz_cm, Fidelity.SDE, seed)
            high = simulate_sample(energy, high_geom, cfg.dz_cm, Fidelity.SDE, seed)
            rows.append(
                {
                    "x": pack_perdepth(high),
                    "scalars": pack_scalars(high),
                    "z": high.z_cm,
                    "dose_low_stat": low.targets["dose"],
                    "dose_high_stat": high.targets["dose"],
                    "mc_residual": low.targets["dose"] - high.targets["dose"],
                    "r80_high_stat": np.asarray(high.range_r80_cm, dtype=np.float64),
                    "energy_mev": np.asarray(float(energy), dtype=np.float64),
                    "seed": np.asarray(seed, dtype=np.int64),
                    "geometry_json": json.dumps(high.meta.get("geometry", {}), sort_keys=True),
                }
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["schema"] = "braggtransporter_phase4_mc_pairs_v1"
        h5.attrs["data_config_json"] = json.dumps(asdict(cfg), sort_keys=True)
        h5.attrs["low_histories"] = int(low_histories)
        h5.attrs["high_histories"] = int(high_histories)
        h5.attrs["units_json"] = json.dumps(
            {"z": "cm", "dose": "MeV/g per voxel", "energy": "MeV", "dz": "cm"},
            sort_keys=True,
        )
        for key in ("x", "scalars", "z", "dose_low_stat", "dose_high_stat", "mc_residual"):
            h5.create_dataset(key, data=np.stack([r[key] for r in rows]).astype(np.float64))
        for key in ("r80_high_stat", "energy_mev"):
            h5.create_dataset(key, data=np.asarray([r[key] for r in rows], dtype=np.float64))
        h5.create_dataset("seed", data=np.asarray([r["seed"] for r in rows], dtype=np.int64))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("geometry_json", data=[r["geometry_json"] for r in rows], dtype=string_dtype)

    return {
        "out_path": str(out_path),
        "n_pairs": len(rows),
        "low_histories": int(low_histories),
        "high_histories": int(high_histories),
        "dz_cm": float(cfg.dz_cm),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="defaults", help="YAML config path or 'defaults'.")
    parser.add_argument("--out", default="data/generated/phase4_mc_pairs.h5")
    parser.add_argument("--low-histories", type=int, default=64)
    parser.add_argument("--high-histories", type=int, default=2048)
    parser.add_argument("--n-energies", type=int, default=None)
    parser.add_argument("--n-geometries", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    summary = generate_mc_pairs(
        cfg,
        out_path=args.out,
        low_histories=int(args.low_histories),
        high_histories=int(args.high_histories),
        n_energies=args.n_energies,
        n_geometries=args.n_geometries,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
