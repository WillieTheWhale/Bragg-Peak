from __future__ import annotations

import h5py
import numpy as np

from braggtransporter.config import DataConfig
from braggtransporter.data.generate import generate_dataset
from braggtransporter.data.sharp import SharpDataConfig, generate_sharp_dataset
from braggtransporter.metrics import distal_80_20_mm
from scripts import sharp_comparison


def test_sharp_generator_schema_and_falloff(tmp_path):
    sharp_path = tmp_path / "sharp.h5"
    sharp_cfg = SharpDataConfig(
        energies_mev=[90.0],
        heldout_energies_mev=[95.0],
        dz_cm=0.02,
        max_depth_cm=12.0,
        n_geometries_per_energy=4,
        seed=11,
        out_path=str(sharp_path),
        energy_spread_pct=0.06,
        n_paths=3,
    )
    generate_sharp_dataset(sharp_cfg)

    clean_path = tmp_path / "clean.h5"
    clean_cfg = DataConfig(
        energies_mev=[90.0],
        heldout_energies_mev=[95.0],
        dz_cm=0.05,
        max_depth_cm=12.0,
        n_geometries_per_energy=4,
        fidelity="analytic",
        seed=11,
        out_path=str(clean_path),
    )
    generate_dataset(clean_cfg)

    with h5py.File(sharp_path, "r") as h5:
        assert set(["train", "val", "heldout_energy", "norm"]).issubset(h5.keys())
        assert h5["train/x"].shape[-1] == 9
        assert h5["train/scalars"].shape[-1] == 4
        assert h5["train/dose"].shape == h5["train/z"].shape
        assert np.all(h5["train/dose"][...] >= 0.0)
        assert h5["norm/x_mean"].shape == (9,)
        sharp_falloff = _finite_falloffs(h5["train/z"][...], h5["train/dose"][...])

    with h5py.File(clean_path, "r") as h5:
        clean_falloff = _finite_falloffs(h5["train/z"][...], h5["train/dose"][...])

    assert sharp_falloff.size > 0
    assert clean_falloff.size > 0
    assert float(np.median(sharp_falloff)) < float(np.median(clean_falloff))


def test_sharp_comparison_tiny_runner(tmp_path):
    data_path = tmp_path / "sharp_tiny.h5"
    cfg = SharpDataConfig(
        energies_mev=[80.0],
        heldout_energies_mev=[85.0],
        dz_cm=0.04,
        max_depth_cm=10.0,
        n_geometries_per_energy=3,
        seed=3,
        out_path=str(data_path),
        energy_spread_pct=0.08,
        n_paths=2,
    )
    generate_sharp_dataset(cfg)

    out_csv = tmp_path / "sharp_comparison.csv"
    rows = sharp_comparison.main(
        [
            "--data",
            str(data_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--tiny",
            "--max-train-samples",
            "2",
            "--max-val-samples",
            "1",
            "--max-heldout-samples",
            "2",
            "--out-csv",
            str(out_csv),
        ]
    )

    assert out_csv.exists()
    assert out_csv.with_suffix(".npz").exists()
    assert out_csv.with_suffix(".json").exists()
    assert {row["model"] for row in rows} == {"braggtransporter_v0", "fno1d", "conv1d"}
    for row in rows:
        assert row["n_heldout"] == 2.0
        assert "gamma_1pct_1mm_pass_pct_mean" in row
        assert "v0_vs_fno_mean_edge_reduction_mm" in row


def _finite_falloffs(z_batch: np.ndarray, dose_batch: np.ndarray) -> np.ndarray:
    vals = []
    for z, dose in zip(z_batch, dose_batch):
        value = distal_80_20_mm(np.asarray(z, dtype=np.float64), np.asarray(dose, dtype=np.float64))
        if np.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)
