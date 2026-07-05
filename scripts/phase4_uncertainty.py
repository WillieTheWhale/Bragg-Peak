#!/usr/bin/env python
"""Train and calibrate a Phase-4 residual uncertainty head."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from braggtransporter.config import DataConfig, ModelConfig, get_device
from braggtransporter.data.dataset import BraggDataset
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.models.uncertainty_head import UncertaintyHeadConfig
from braggtransporter.train import SyntheticBraggDataset
from braggtransporter.uncertainty import (
    UncertaintyTrainConfig,
    calibration_summary,
    decompose_uncertainty,
    load_frozen_v0,
    train_uncertainty_head,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="experiments/bt/phase3/braggtransporter_v0/best.pt")
    parser.add_argument("--data", default="data/generated/phase1_1d.h5")
    parser.add_argument("--head", choices=["heteroscedastic", "flow"], default="heteroscedastic")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true", help="Use a tiny synthetic train/heldout split.")
    parser.add_argument("--flow-samples", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--out", default="docs/results/phase4_calibration.csv")
    args = parser.parse_args(argv)

    torch.manual_seed(int(args.seed))
    device = get_device(args.device)
    train_loader, heldout_loader = _make_loaders(args.data, args.batch_size, args.fast, args.seed)
    v0 = _load_or_make_v0(args.ckpt, device, args.fast)

    train_cfg = UncertaintyTrainConfig(
        head_kind=args.head,
        epochs=int(args.epochs),
        lr=float(args.lr),
        device=str(device),
        seed=int(args.seed),
        flow_samples=int(args.flow_samples),
        flow_steps=int(args.flow_steps),
    )
    head_cfg = UncertaintyHeadConfig(hidden=48 if args.fast else 64, depth=2, use_scalars=True)
    head, history = train_uncertainty_head(v0, train_loader, heldout_loader, train_cfg, head_cfg)

    pred, ref, sigma = _collect_predictions(v0, head, heldout_loader, device, args.flow_samples, args.flow_steps)
    reliability, summary = calibration_summary(pred, ref, sigma)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_calibration_csv(out_path, reliability, summary, args, history)
    _write_sidecar_artifacts(out_path, reliability, summary, args, history)

    payload: dict[str, Any] = {
        "head": args.head,
        "epochs": int(args.epochs),
        "device": str(device),
        "calibration_csv": str(out_path),
        "coverage_68": summary["coverage_68"],
        "coverage_95": summary["coverage_95"],
        "sharpness_mean_sigma": summary["sharpness_mean_sigma"],
        "history_last": history[-1] if history else {},
        "identifiable_channels": ["sigma_aleatoric", "sigma_epistemic_if_ensemble_provided"],
        "data_gated_channels": ["sigma_MC", "sigma_input", "sigma_meas"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _make_loaders(data_path: str, batch_size: int, fast: bool, seed: int) -> tuple[DataLoader, DataLoader]:
    if fast or not Path(data_path).exists():
        ds = SyntheticBraggDataset(
            n_samples=24,
            nz=48,
            data_cfg=DataConfig(max_depth_cm=12.0, dz_cm=0.25),
            seed=int(seed),
        )
        gen = torch.Generator().manual_seed(int(seed))
        train_ds, heldout_ds = random_split(ds, [18, 6], generator=gen)
        return (
            DataLoader(train_ds, batch_size=min(batch_size, 6), shuffle=True, generator=gen),
            DataLoader(heldout_ds, batch_size=min(batch_size, 6), shuffle=False),
        )

    train = BraggDataset(data_path, "train")
    heldout = BraggDataset(data_path, "heldout_energy")
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(heldout, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _load_or_make_v0(ckpt: str, device: torch.device, fast: bool) -> BraggTransporterV0:
    path = Path(ckpt)
    if path.exists():
        return load_frozen_v0(path, device)
    if not fast:
        raise FileNotFoundError(f"frozen v0 checkpoint not found: {path}")
    model = BraggTransporterV0(
        ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, dropout=0.0, extra={"max_positions": 128})
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _collect_predictions(
    v0: BraggTransporterV0,
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    flow_samples: int,
    flow_steps: int,
) -> tuple[Any, Any, Any]:
    preds = []
    refs = []
    sigmas = []
    head.to(device).eval()
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        decomp = decompose_uncertainty(
            v0,
            head,
            x,
            scalars,
            flow_samples=int(flow_samples),
            flow_steps=int(flow_steps),
        )
        preds.append(decomp.mean_dose.detach().cpu())
        refs.append(batch["dose"].to(dtype=torch.float32).cpu())
        sigmas.append(decomp.sigma_total_identifiable.detach().cpu())
    return (
        torch.cat(preds, dim=0).numpy(),
        torch.cat(refs, dim=0).numpy(),
        torch.cat(sigmas, dim=0).numpy(),
    )


def _write_calibration_csv(
    out_path: Path,
    reliability: list[dict[str, float | bool]],
    summary: dict[str, float | bool],
    args: argparse.Namespace,
    history: list[dict[str, float | int]],
) -> None:
    fields = [
        "head",
        "epochs",
        "level",
        "empirical_coverage",
        "abs_error",
        "within_5pct",
        "sharpness_mean_sigma",
        "sharpness_median_sigma",
        "final_train_loss",
        "final_val_loss",
    ]
    last = history[-1] if history else {}
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in reliability:
            writer.writerow(
                {
                    "head": args.head,
                    "epochs": int(args.epochs),
                    "level": row["nominal"],
                    "empirical_coverage": row["empirical"],
                    "abs_error": row["abs_error"],
                    "within_5pct": row["within_5pct"],
                    "sharpness_mean_sigma": summary["sharpness_mean_sigma"],
                    "sharpness_median_sigma": summary["sharpness_median_sigma"],
                    "final_train_loss": last.get("train_loss", ""),
                    "final_val_loss": last.get("val_loss", ""),
                }
            )


def _write_sidecar_artifacts(
    out_path: Path,
    reliability: list[dict[str, float | bool]],
    summary: dict[str, float | bool],
    args: argparse.Namespace,
    history: list[dict[str, float | int]],
) -> None:
    npz_path = out_path.with_suffix(".npz")
    json_path = out_path.with_suffix(".json")
    np.savez(
        npz_path,
        nominal=np.asarray([float(row["nominal"]) for row in reliability], dtype=np.float64),
        empirical=np.asarray([float(row["empirical"]) for row in reliability], dtype=np.float64),
        abs_error=np.asarray([float(row["abs_error"]) for row in reliability], dtype=np.float64),
        sharpness_mean_sigma=np.asarray(float(summary["sharpness_mean_sigma"]), dtype=np.float64),
        sharpness_median_sigma=np.asarray(float(summary["sharpness_median_sigma"]), dtype=np.float64),
    )
    meta = {
        "script": "scripts/phase4_uncertainty.py",
        "checkpoint": args.ckpt,
        "data": args.data,
        "head": args.head,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "units": {"depth": "cm", "energy": "MeV", "dose": "MeV/g per voxel", "sigma": "MeV/g per voxel"},
        "identifiability": {
            "sigma_aleatoric": "residual head trained on target - frozen v0 mean",
            "sigma_epistemic": "requires an ensemble of independently trained v0 checkpoints",
            "sigma_MC": "requires low/high-stat MC pairs with known histories",
            "sigma_input": "requires controlled CT/RSP/SPR perturbation ensembles",
            "sigma_meas": "requires paired MC-vs-measurement residuals and detector metadata",
        },
        "summary": summary,
        "history": history,
    }
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
