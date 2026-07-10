from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from braggtransporter.data.doserad import DoseRADBeamletDataset
from braggtransporter.models.bragg3d import Bragg3D
from scripts.train_doserad_gpu import (
    SyntheticDoseRADDataset,
    doserad_relative_loss,
    initialize_lazy_modules,
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint,
    train_step,
)


def test_per_beamlet_normalization_yields_unit_max_targets(tmp_path: Path) -> None:
    root = tmp_path / "doserad2026"
    _write_patient(root, "FAKE001", scales=(1e-3, 3e-4))
    ds = DoseRADBeamletDataset(root, ["FAKE001"], depth_size=12, lateral_size=8, lateral_extent_mm=28.0)

    peaks = [float(ds[i]["dose"].max()) for i in range(len(ds))]
    scales = [float(ds[i]["dose_scale"]) for i in range(len(ds))]

    assert len(ds) == 2
    assert np.allclose(peaks, [1.0, 1.0], atol=1e-6)
    assert all(0.0 < scale < 1e-2 for scale in scales)
    assert not np.isclose(scales[0], scales[1])


def test_multi_patient_dataset_concatenation(tmp_path: Path) -> None:
    root = tmp_path / "doserad2026"
    _write_patient(root, "FAKE001", scales=(1e-3, 8e-4))
    _write_patient(root, "FAKE002", scales=(2e-3, 7e-4))

    ds = DoseRADBeamletDataset(
        root,
        ["FAKE001", "FAKE002"],
        max_beamlets_per_patient=1,
        depth_size=10,
        lateral_size=6,
        lateral_extent_mm=24.0,
    )

    assert len(ds) == 2
    assert {str(ds[i]["meta"]["patient"]) for i in range(len(ds))} == {"FAKE001", "FAKE002"}
    assert all(float(ds[i]["dose"].max()) == 1.0 for i in range(len(ds)))


def test_training_steps_reduce_normalized_loss() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    ds = SyntheticDoseRADDataset(n=4, depth=8, lateral=5, seed=7)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    model = Bragg3D(d_model=8, n_layers=1, n_heads=2, d_ff=16, max_depth=16).to(device)
    initialize_lazy_modules(model, loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=0.0)

    with torch.no_grad():
        before = float(
            doserad_relative_loss(
                model(batch["x"].float(), batch["scalars"].float())["dose"],
                batch["dose"].float(),
                distal_weight=2.0,
            )
        )
    for _ in range(8):
        train_step(model, batch, optimizer, device, amp_enabled=False, distal_weight=2.0)
    with torch.no_grad():
        after = float(
            doserad_relative_loss(
                model(batch["x"].float(), batch["scalars"].float())["dose"],
                batch["dose"].float(),
                distal_weight=2.0,
            )
        )

    assert after < before


def test_resume_from_checkpoint_restores_state(tmp_path: Path) -> None:
    torch.manual_seed(11)
    device = torch.device("cpu")
    loader = DataLoader(SyntheticDoseRADDataset(n=3, depth=8, lateral=5, seed=11), batch_size=2)
    model = Bragg3D(d_model=8, n_layers=1, n_heads=2, d_ff=16, max_depth=16).to(device)
    initialize_lazy_modules(model, loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    train_step(model, next(iter(loader)), optimizer, device, scheduler=scheduler)
    saved_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    saved_scheduler = scheduler.state_dict()
    ckpt_path = tmp_path / "ckpt.pt"

    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=3,
        global_step=17,
        args=argparse.Namespace(seed=11, patients=["SYNTH"]),
        metrics={"val_loss": 0.25},
    )
    for param in model.parameters():
        param.data.add_(torch.randn_like(param))
    scheduler.step()

    ckpt = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, map_location=device)

    assert ckpt["epoch"] == 3
    assert ckpt["global_step"] == 17
    assert scheduler.state_dict() == saved_scheduler
    for name, tensor in model.state_dict().items():
        assert torch.allclose(tensor, saved_model[name])


def test_resolve_resume_latest_downloads_latest_and_best_from_gcs(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool, text: bool, capture_output: bool):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"checkpoint")
        return argparse.Namespace(returncode=0, stderr="")

    monkeypatch.setattr("scripts.train_doserad_gpu.subprocess.run", fake_run)

    path = resolve_resume_path("latest", tmp_path, "gs://bucket/runs/run18")

    assert path == tmp_path / "latest.pt"
    assert (tmp_path / "latest.pt").read_bytes() == b"checkpoint"
    assert (tmp_path / "best.pt").read_bytes() == b"checkpoint"
    assert [cmd[2] for cmd in calls] == ["gs://bucket/runs/run18/latest.pt", "gs://bucket/runs/run18/best.pt"]


def _write_patient(root: Path, patient_id: str, *, scales: tuple[float, ...]) -> None:
    patient = root / patient_id
    dose_dir = patient / "dose"
    dose_dir.mkdir(parents=True)

    shape_zyx = (16, 18, 20)
    z, y, x = np.mgrid[0 : shape_zyx[0], 0 : shape_zyx[1], 0 : shape_zyx[2]]
    ct_arr = (-600.0 + 20.0 * x + 10.0 * z).astype(np.float32)
    ct = sitk.GetImageFromArray(ct_arr)
    ct.SetSpacing((2.0, 2.0, 2.0))
    ct.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(ct, str(patient / "ct.mha"))

    beamlets = []
    for layer, scale in enumerate(scales):
        center_y = 6.0 + 4.0 * layer
        dose_arr = np.exp(-0.5 * ((y - center_y) ** 2 / 5.0 + (x - 8.0) ** 2 / 6.0 + (z - 7.0) ** 2 / 6.0))
        dose_arr = (dose_arr * float(scale)).astype(np.float32)
        dose = sitk.GetImageFromArray(dose_arr)
        dose.SetSpacing((2.0, 2.0, 2.0))
        dose.SetOrigin((0.0, 0.0, 0.0))
        sitk.WriteImage(dose, str(dose_dir / f"Dose_B0_R0_L{layer}.mha"))
        beamlets.append({"beamlet_idx": layer, "energy": 100.0 + 20.0 * layer})

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
                        "beamlets": beamlets,
                    }
                ],
            }
        ],
    }
    (patient / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
