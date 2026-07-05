from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from braggtransporter.data.doserad import (
    DOSERAD_INPUT_CHANNELS,
    DoseRADBeamletDataset,
    extract_bev_pair,
    gamma_index_3d,
    make_doserad_loaders,
)
from braggtransporter.models.bragg3d import Bragg3D


def test_bev_extraction_and_dataset_shapes(tmp_path: Path) -> None:
    root = _write_fake_doserad(tmp_path)
    ds = DoseRADBeamletDataset(
        root,
        ["FAKE001"],
        max_beamlets=2,
        depth_size=12,
        lateral_size=8,
        lateral_extent_mm=28.0,
    )
    assert len(ds) == 2
    item = ds[0]
    assert item["x"].shape == (DOSERAD_INPUT_CHANNELS, 12, 8, 8)
    assert item["dose"].shape == (12, 8, 8)
    assert item["scalars"].shape == (4,)
    assert item["spacing_mm"].shape == (3,)
    assert torch.isfinite(item["x"]).all()
    assert torch.all(item["dose"] >= 0.0)

    train, val = make_doserad_loaders(
        root,
        "FAKE001",
        max_beamlets=2,
        depth_size=12,
        lateral_size=8,
        lateral_extent_mm=28.0,
        batch_size=1,
        seed=4,
    )
    batch = next(iter(train))
    assert batch["x"].shape[1:] == (DOSERAD_INPUT_CHANNELS, 12, 8, 8)
    assert len(val.dataset) == 1


def test_direct_bev_extraction_axis0_depth(tmp_path: Path) -> None:
    root = _write_fake_doserad(tmp_path)
    patient = root / "FAKE001"
    ct = sitk.ReadImage(str(patient / "ct.mha"), sitk.sitkFloat32)
    dose = sitk.ReadImage(str(patient / "dose" / "Dose_B0_R0_L0.mha"), sitk.sitkFloat32)
    bev = extract_bev_pair(
        ct,
        dose,
        ray_source_mm=(15.0, -20.0, 15.0),
        ray_target_mm=(15.0, 30.0, 15.0),
        depth_size=10,
        lateral_size=6,
        lateral_extent_mm=20.0,
    )
    assert bev["x"].shape == (DOSERAD_INPUT_CHANNELS, 10, 6, 6)
    assert bev["dose"].shape == (10, 6, 6)
    profile = np.asarray(bev["dose"]).sum(axis=(1, 2))
    assert int(np.argmax(profile)) > 0


def test_bragg3d_forward_backward_cpu_finite() -> None:
    torch.manual_seed(3)
    model = Bragg3D(d_model=24, n_layers=1, n_heads=4, d_ff=48, max_depth=16)
    x = torch.randn(2, DOSERAD_INPUT_CHANNELS, 10, 6, 6)
    scalars = torch.tensor([[120.0, 0.0, 0.0, 1.0], [150.0, 1.0, 0.5, 0.866]], dtype=torch.float32)
    target = torch.rand(2, 10, 6, 6) * 1e-3
    out = model(x, scalars)
    assert out["dose"].shape == target.shape
    assert torch.all(out["dose"] >= 0.0)
    loss = F.mse_loss(out["dose"], target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert model.param_count() < 5_000_000


def test_gamma3d_identical_is_100_percent() -> None:
    z, y, x = np.mgrid[0:12, 0:7, 0:7]
    dose = np.exp(-0.5 * ((z - 7.0) ** 2 + (y - 3.0) ** 2 + (x - 3.0) ** 2) / 2.0).astype(np.float64)
    gamma = gamma_index_3d(dose, dose, spacing_mm=(2.0, 2.0, 2.0), dose_pct=3.0, dta_mm=3.0)
    assert gamma == 100.0


def _write_fake_doserad(tmp_path: Path) -> Path:
    root = tmp_path / "doserad2026"
    patient = root / "FAKE001"
    dose_dir = patient / "dose"
    dose_dir.mkdir(parents=True)

    shape_zyx = (16, 18, 20)
    z, y, x = np.mgrid[0 : shape_zyx[0], 0 : shape_zyx[1], 0 : shape_zyx[2]]
    ct_arr = (-850.0 + 35.0 * x + 20.0 * z).astype(np.float32)
    ct = sitk.GetImageFromArray(ct_arr)
    ct.SetSpacing((2.0, 2.0, 2.0))
    ct.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(ct, str(patient / "ct.mha"))

    for layer, center_y in enumerate((6.0, 11.0)):
        dose_arr = np.exp(-0.5 * ((y - center_y) ** 2 / 5.0 + (x - 8.0) ** 2 / 6.0 + (z - 7.0) ** 2 / 6.0))
        dose_arr = (dose_arr * (layer + 1) * 1e-3).astype(np.float32)
        dose = sitk.GetImageFromArray(dose_arr)
        dose.SetSpacing((2.0, 2.0, 2.0))
        dose.SetOrigin((0.0, 0.0, 0.0))
        sitk.WriteImage(dose, str(dose_dir / f"Dose_B0_R0_L{layer}.mha"))

    (dose_dir / "Dose_B0_R0_L2.mha").write_bytes(b"corrupt beamlet")
    plan = {
        "iso_center": [15.0, 15.0, 15.0],
        "beams": [
            {
                "beam_idx": 0,
                "gantry_angle": 0.0,
                "rays": [
                    {
                        "ray_idx": 0,
                        "ray_source": [15.0, -20.0, 15.0],
                        "ray_target": [15.0, 30.0, 15.0],
                        "beamlets": [
                            {"beamlet_idx": 0, "energy": 100.0},
                            {"beamlet_idx": 1, "energy": 130.0},
                            {"beamlet_idx": 2, "energy": 160.0},
                        ],
                    }
                ],
            }
        ],
    }
    (patient / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return root
