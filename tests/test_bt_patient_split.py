from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from braggtransporter.data.doserad import make_doserad_loaders


def test_patient_split_has_disjoint_train_val_patients_and_prints_ids(tmp_path: Path, capsys) -> None:
    root = _write_multi_patient_doserad(tmp_path, n_patients=3, n_beamlets=4)

    train, val = make_doserad_loaders(
        root,
        ["FAKE001", "FAKE002", "FAKE003"],
        val_frac=0.34,
        batch_size=1,
        seed=12,
        depth_size=8,
        depth_extent_mm=20.0,
        lateral_size=5,
        lateral_extent_mm=8.0,
        split_by="patient",
    )

    train_patients = _loader_patient_ids(train)
    val_patients = _loader_patient_ids(val)
    output = capsys.readouterr().out

    assert train_patients.isdisjoint(val_patients)
    assert set(val.val_patient_ids) == val_patients
    assert "DoseRAD split_by=patient" in output
    assert "val_patients=" in output


def test_beamlet_split_can_overlap_patients(tmp_path: Path) -> None:
    root = _write_multi_patient_doserad(tmp_path, n_patients=3, n_beamlets=4)

    train, val = make_doserad_loaders(
        root,
        ["FAKE001", "FAKE002", "FAKE003"],
        val_frac=0.5,
        batch_size=1,
        seed=0,
        depth_size=8,
        depth_extent_mm=20.0,
        lateral_size=5,
        lateral_extent_mm=8.0,
        split_by="beamlet",
    )

    assert _loader_patient_ids(train) & _loader_patient_ids(val)


def _loader_patient_ids(loader) -> set[str]:
    subset = loader.dataset
    dataset = subset.dataset
    return {dataset.records[int(i)].patient for i in subset.indices}


def _write_multi_patient_doserad(tmp_path: Path, *, n_patients: int, n_beamlets: int) -> Path:
    root = tmp_path / "doserad2026"
    for patient_idx in range(n_patients):
        _write_patient(root, f"FAKE{patient_idx + 1:03d}", n_beamlets=n_beamlets)
    return root


def _write_patient(root: Path, patient_id: str, *, n_beamlets: int) -> None:
    patient = root / patient_id
    dose_dir = patient / "dose"
    dose_dir.mkdir(parents=True)

    shape_zyx = (10, 10, 10)
    z, y, x = np.mgrid[0 : shape_zyx[0], 0 : shape_zyx[1], 0 : shape_zyx[2]]
    ct_arr = np.zeros(shape_zyx, dtype=np.float32)
    ct = sitk.GetImageFromArray(ct_arr)
    ct.SetSpacing((1.0, 1.0, 1.0))
    ct.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(ct, str(patient / "ct.mha"))

    beamlets = []
    for layer in range(n_beamlets):
        dose_arr = np.exp(-0.5 * (((y - 5.0) / 3.0) ** 2 + ((x - 5.0) / 2.0) ** 2 + ((z - 5.0) / 2.0) ** 2))
        dose_arr = (dose_arr * (layer + 1) * 1e-3).astype(np.float32)
        dose = sitk.GetImageFromArray(dose_arr)
        dose.SetSpacing((1.0, 1.0, 1.0))
        dose.SetOrigin((0.0, 0.0, 0.0))
        sitk.WriteImage(dose, str(dose_dir / f"Dose_B0_R0_L{layer}.mha"))
        beamlets.append({"beamlet_idx": layer, "energy": 100.0 + 5.0 * layer})

    plan = {
        "iso_center": [5.0, 5.0, 5.0],
        "beams": [
            {
                "beam_idx": 0,
                "gantry_angle": 0.0,
                "rays": [
                    {
                        "ray_idx": 0,
                        "ray_source": [5.0, -20.0, 5.0],
                        "ray_target": [5.0, 20.0, 5.0],
                        "beamlets": beamlets,
                    }
                ],
            }
        ],
    }
    (patient / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
