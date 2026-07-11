#!/usr/bin/env python
"""Validate and package a completed DoseRAD cloud run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
from typing import Any

import numpy as np
import yaml


PAPER_CRITERIA = {
    "gamma3d_1pct_3mm_dota_pymedphys": {
        "dose_difference_percent": 1.0,
        "distance_to_agreement_mm": 3.0,
        "low_dose_cutoff_fraction": 0.001,
        "engine": "PyMedPhys global gamma",
        "interpolation_fraction": 10,
    },
    "gamma3d_1pct_3mm_dota": {
        "dose_difference_percent": 1.0,
        "distance_to_agreement_mm": 3.0,
        "low_dose_cutoff_fraction": 0.001,
        "engine": "voxel-center lower-bound diagnostic",
    },
    "gamma3d_2pct_2mm": {
        "dose_difference_percent": 2.0,
        "distance_to_agreement_mm": 2.0,
        "low_dose_cutoff_fraction": 0.1,
        "engine": "voxel-center diagnostic",
    },
    "gamma3d_3pct_3mm": {
        "dose_difference_percent": 3.0,
        "distance_to_agreement_mm": 3.0,
        "low_dose_cutoff_fraction": 0.1,
        "engine": "voxel-center internal diagnostic",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--best-full", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def package_run(args: argparse.Namespace) -> dict[str, Any]:
    records = read_records(args.metrics_jsonl)
    metrics = [record["metrics"] for record in records]
    validate_progress(metrics, int(args.expected_epochs))
    checkpoint = audit_best_checkpoint(args.best_checkpoint, metrics)
    best_full = read_json_object(args.best_full)
    test_metrics = read_json_object(args.test_metrics)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "metrics_complete.jsonl"
    csv_path = output_dir / "metrics_complete.csv"
    npz_path = output_dir / "metrics_complete.npz"
    metadata_path = output_dir / "run_audit_summary.json"

    jsonl_path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    arrays = numeric_arrays(metrics)
    write_metrics_csv(csv_path, arrays)
    write_deterministic_npz(npz_path, arrays)

    first = records[0]
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "research_only": True,
        "run_name": str(args.run_name),
        "training_commit": str(args.training_commit),
        "evaluation_commit": str(args.evaluation_commit),
        "epochs": {"expected": int(args.expected_epochs), "observed": len(records)},
        "data": {
            "patients": first.get("patients"),
            "max_beamlets_per_patient": first.get("max_beamlets_per_patient"),
            "seed": first.get("seed"),
        },
        "units": first.get("units"),
        "criteria": PAPER_CRITERIA,
        "checkpoint": checkpoint,
        "best_full_metrics": best_full,
        "untouched_test_metrics": test_metrics,
        "config": config,
        "source_sha256": {
            "metrics_jsonl": sha256_file(args.metrics_jsonl),
            "best_full": sha256_file(args.best_full),
            "test_metrics": sha256_file(args.test_metrics),
            "best_checkpoint": sha256_file(args.best_checkpoint),
        },
        "artifact_sha256": {
            "metrics_complete_jsonl": sha256_file(jsonl_path),
            "metrics_complete_csv": sha256_file(csv_path),
            "metrics_complete_npz": sha256_file(npz_path),
        },
    }
    metadata_path.write_text(json.dumps(json_clean(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
            raise ValueError(f"{path}:{line_number} is not a wrapped metrics object")
        records.append(value)
    if not records:
        raise ValueError(f"{path} contains no metrics records")
    return records


def validate_progress(metrics: list[dict[str, Any]], expected_epochs: int) -> None:
    epochs = [int(float(row["epoch"])) for row in metrics]
    expected = list(range(1, int(expected_epochs) + 1))
    if epochs != expected:
        raise ValueError(f"epochs are not contiguous 1..{expected_epochs}: got {epochs[:3]}...{epochs[-3:]}")
    steps = np.asarray([float(row["global_step"]) for row in metrics], dtype=np.float64)
    if np.any(~np.isfinite(steps)) or np.any(np.diff(steps) <= 0.0):
        raise ValueError("global_step must be finite and strictly increasing")
    for row in metrics:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isfinite(float(value)):
                raise ValueError(f"non-finite metric {key!r} at epoch {int(float(row['epoch']))}")


def audit_best_checkpoint(path: Path, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    saved_metrics = checkpoint.get("metrics", {})
    saved_args = checkpoint.get("args", {})
    save_best_by = str(saved_args.get("save_best_by", "gamma"))
    key = "gamma3d_3pct_3mm" if save_best_by == "gamma" else "val_loss"
    selector = max if save_best_by == "gamma" else min
    expected_value = float(selector(float(row[key]) for row in metrics))
    observed_value = float(saved_metrics[key])
    if not np.isclose(observed_value, expected_value, rtol=0.0, atol=1e-12):
        raise ValueError(f"best checkpoint {key}={observed_value} does not match trajectory best {expected_value}")
    trajectory_epoch = int(float(next(row["epoch"] for row in metrics if float(row[key]) == expected_value)))
    if epoch != trajectory_epoch:
        raise ValueError(f"best checkpoint epoch {epoch} does not match trajectory epoch {trajectory_epoch}")
    return {
        "epoch": epoch,
        "global_step": int(checkpoint["global_step"]),
        "selection": save_best_by,
        "selection_metric": key,
        "selection_value": observed_value,
    }


def numeric_arrays(metrics: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = sorted(
        {
            key
            for row in metrics
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        key: np.asarray([float(row.get(key, float("nan"))) for row in metrics], dtype=np.float64)
        for key in keys
    }


def write_metrics_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    keys = list(arrays)
    count = len(next(iter(arrays.values())))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(keys)
        for index in range(count):
            writer.writerow([format(float(arrays[key][index]), ".17g") for key in keys])


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    metadata = package_run(args)
    print(json.dumps({"run_name": metadata["run_name"], "epochs": metadata["epochs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
