import h5py
import numpy as np
import torch

from braggtransporter.config import DataConfig
from braggtransporter.data.dataset import BraggDataset, make_loaders
from braggtransporter.data.generate import generate_dataset
from braggtransporter.data.physics_engine import simulate_sample
from braggtransporter.physics_prior import compute_prior
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR, Fidelity, PRIOR_FIELDS


def test_compute_prior_shapes_and_normalized_bortfeld():
    z = (np.arange(60, dtype=np.float64) + 0.5) * 0.05
    material = {
        "density_g_cm3": np.ones_like(z),
        "rsp": np.ones_like(z),
        "i_value_ev": np.full_like(z, 78.0),
        "material_class": np.zeros_like(z),
    }
    beam = {"energy_mev": 100.0, "energy_spread_pct": 0.8, "spot_sigma_cm": 0.1}
    prior = compute_prior(z, material, beam)
    assert tuple(prior) == PRIOR_FIELDS
    for value in prior.values():
        assert value.shape == z.shape
        assert value.dtype == np.float64
    assert np.isclose(prior["bortfeld_dose"].max(), 1.0)
    assert np.all(np.diff(prior["wepl_cm"]) > 0.0)
    assert np.all(prior["resid_energy_mev"] >= 0.0)


def test_simulate_sample_analytic_water_contract():
    sample = simulate_sample(
        90.0,
        {"type": "water", "max_depth_cm": 12.0},
        0.1,
        Fidelity.ANALYTIC,
        7,
    )
    sample.validate()
    assert sample.z_cm.dtype == np.float64
    assert sample.targets["dose"].dtype == np.float64
    assert np.all(sample.targets["dose"] >= 0.0)
    assert np.any(sample.targets["letd"] > 0.0)
    assert np.isfinite(sample.range_r80_cm)


def test_generate_hdf5_and_dataset_roundtrip(tmp_path):
    out = tmp_path / "bt_small.h5"
    cfg = DataConfig(
        energies_mev=[80.0, 100.0],
        heldout_energies_mev=[90.0],
        dz_cm=0.1,
        max_depth_cm=12.0,
        n_geometries_per_energy=4,
        fidelity="analytic",
        seed=123,
        out_path=str(out),
    )
    summary = generate_dataset(cfg, n_histories=16)
    assert summary["counts"] == {"train": 6, "val": 2, "heldout_energy": 4}
    assert out.exists()
    assert out.with_suffix(".h5.norm.json").exists()

    with h5py.File(out, "r") as h5:
        assert set(["train", "val", "heldout_energy", "norm"]).issubset(h5.keys())
        assert h5["train/x"].shape[-1] == C_IN_PERDEPTH
        assert h5["train/scalars"].shape[-1] == C_SCALAR
        assert h5["train/dose"][...].min() >= 0.0

    ds = BraggDataset(out, "train")
    item = ds[0]
    assert set(item) == {"x", "scalars", "z", "dose", "letd", "r80", "fidelity"}
    assert item["x"].dtype == torch.float32
    assert item["scalars"].shape == (C_SCALAR,)
    assert item["x"].shape[-1] == C_IN_PERDEPTH
    assert torch.all(item["dose"] >= 0.0)

    train, val, heldout = make_loaders(cfg)
    batch = next(iter(train))
    assert batch["x"].shape[-1] == C_IN_PERDEPTH
    assert len(val.dataset) == 2
    assert len(heldout.dataset) == 4
