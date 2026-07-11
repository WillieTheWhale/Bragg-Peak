from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from package_doserad_run import package_run


def test_package_run_validates_best_and_writes_deterministic_artifacts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    rows = []
    for epoch, gamma in enumerate((70.0, 82.0, 79.0), start=1):
        rows.append(
            {
                "metrics": {
                    "epoch": float(epoch),
                    "global_step": float(epoch * 10),
                    "gamma3d_3pct_3mm": gamma,
                    "val_loss": 1.0 / epoch,
                },
                "patients": ["P1", "P2", "P3"],
                "max_beamlets_per_patient": 500,
                "seed": 0,
                "units": {"spacing": "mm", "dose": "unit-max relative dose"},
            }
        )
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    best_full = tmp_path / "metrics_best_full.json"
    best_full.write_text(json.dumps({"best_gamma3d_3pct_3mm_full": 83.0}), encoding="utf-8")
    test_metrics = tmp_path / "metrics_test.json"
    test_metrics.write_text(json.dumps({"test_gamma3d_1pct_3mm_dota_pymedphys": 97.5}), encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 2,
            "global_step": 20,
            "metrics": {"gamma3d_3pct_3mm": 82.0},
            "args": {"save_best_by": "gamma"},
        },
        checkpoint,
    )
    output_dir = tmp_path / "package"
    args = argparse.Namespace(
        run_name="run-test",
        metrics_jsonl=metrics_path,
        best_full=best_full,
        test_metrics=test_metrics,
        best_checkpoint=checkpoint,
        expected_epochs=3,
        training_commit="train-sha",
        evaluation_commit="eval-sha",
        config=None,
        output_dir=output_dir,
    )

    metadata = package_run(args)
    first_hashes = _artifact_hashes(output_dir)
    package_run(args)
    second_hashes = _artifact_hashes(output_dir)

    assert first_hashes == second_hashes
    assert metadata["checkpoint"]["epoch"] == 2
    assert metadata["checkpoint"]["selection_value"] == 82.0
    assert metadata["epochs"] == {"expected": 3, "observed": 3}
    with np.load(output_dir / "metrics_complete.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["epoch"], np.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(arrays["gamma3d_3pct_3mm"], np.asarray([70.0, 82.0, 79.0]))


def _artifact_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir())
    }
