from __future__ import annotations

import numpy as np
import torch

from braggtransporter.calibration import coverage
from braggtransporter.config import DataConfig, ModelConfig
from braggtransporter.metrics import gamma_index_1d
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.train import SyntheticBraggDataset


def test_v0_dose_is_nonnegative_on_cpu() -> None:
    torch.manual_seed(0)
    ds = SyntheticBraggDataset(n_samples=2, nz=32, data_cfg=DataConfig(max_depth_cm=12.0), seed=11)
    batch = {k: torch.stack([ds[i][k] for i in range(2)]) for k in ("x", "scalars")}
    model = BraggTransporterV0(
        ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, dropout=0.0, extra={"max_positions": 128})
    ).cpu()

    with torch.no_grad():
        dose = model(batch["x"], batch["scalars"])["dose"]

    assert torch.isfinite(dose).all()
    assert torch.all(dose >= 0.0)


def test_synthetic_range_is_monotonic_with_energy_small_ladder() -> None:
    cfg = DataConfig(energies_mev=[70.0, 90.0, 110.0, 130.0], max_depth_cm=20.0, dz_cm=0.2)
    ds = SyntheticBraggDataset(n_samples=4, nz=80, data_cfg=cfg, seed=19)

    energies = np.asarray([float(ds[i]["scalars"][0]) for i in range(len(ds))], dtype=np.float64)
    ranges = np.asarray([float(ds[i]["r80"]) for i in range(len(ds))], dtype=np.float64)
    order = np.argsort(energies)

    assert np.all(np.diff(ranges[order]) > 0.0)


def test_gamma_identical_curve_is_100_percent() -> None:
    z = np.linspace(0.0, 12.0, 97, dtype=np.float64)
    curve = (0.2 + np.exp(-0.5 * ((z - 8.0) / 0.45) ** 2)) / 1.2

    assert gamma_index_1d(curve, curve, z) == 100.0


def test_calibration_coverage_increases_monotonically_with_temperature() -> None:
    pred = np.zeros(101, dtype=np.float64)
    ref = np.linspace(-2.0, 2.0, pred.size, dtype=np.float64)
    base_sigma = np.full_like(pred, 0.25)
    temperatures = [0.5, 1.0, 2.0, 4.0]

    coverages = [coverage(pred, ref, base_sigma * tau, 0.6826894921370859) for tau in temperatures]

    assert coverages == sorted(coverages)
    assert coverages[-1] > coverages[0]


def test_v0_backcompat_param_count_unchanged() -> None:
    model = BraggTransporterV0(ModelConfig())

    assert model.param_count() == 943_765
