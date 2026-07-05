#!/usr/bin/env python
"""Phase 2B backbone ablation for BraggTransporter v3.1.

Trains transformer v0, FNO, and Mamba/SSM backbones on the same 1-D HDF5 data
and ranks them by held-out distal-edge error. Units follow the fixed
BraggTransporter contract: depth cm in data, distal errors reported in mm.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braggtransporter.config import DataConfig, ModelConfig, TrainConfig, get_device
from braggtransporter.data.dataset import BraggDataset
from braggtransporter.metrics import distal_edge_error_mm, gamma_index_1d, peak_depth_cm, rmse_pct
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.models.fno1d import FNO1d
from braggtransporter.models.mamba1d import Mamba1d
from braggtransporter.train import _set_determinism, compute_loss


MODELS = ("braggtransporter_v0", "fno1d", "mamba1d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DataConfig().out_path), help="HDF5 data with train/val/heldout_energy splits.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/results/phase2b.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")

    _set_determinism(args.seed)
    device = get_device(args.device)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=str(device),
        seed=args.seed,
    )
    target_params = BraggTransporterV0(ModelConfig()).param_count()
    constructors = _matched_constructors(target_params)

    train_loader, val_loader, heldout_loader = _loaders(args.data, args.batch_size, args.seed)
    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float | int]]] = {}

    for model_name in MODELS:
        model = constructors[model_name]().to(device)
        count = _param_count(model)
        best_val, history = _train_one(model, model_name, train_loader, val_loader, train_cfg, device)
        histories[model_name] = history
        metrics = _evaluate_heldout(model, heldout_loader, device)
        row = {
            "model": model_name,
            "param_count": count,
            "target_param_count": target_params,
            "param_gap_pct": 100.0 * (count - target_params) / target_params,
            "epochs": args.epochs,
            "best_val_loss": best_val,
            **metrics,
        }
        rows.append(row)

    rows.sort(key=lambda row: float(row["distal_edge_error_mm_mean"]))
    _write_outputs(rows, histories, args, train_cfg, device)
    _print_ranked_table(rows, args.out_csv)


def _matched_constructors(target_params: int) -> dict[str, Any]:
    fno_kwargs = _best_fno_kwargs(target_params)
    cfg = ModelConfig()
    return {
        "braggtransporter_v0": lambda: BraggTransporterV0(cfg),
        "fno1d": lambda: FNO1d(**fno_kwargs),
        "mamba1d": lambda: Mamba1d(cfg),
    }


def _best_fno_kwargs(target_params: int) -> dict[str, int]:
    candidates: list[tuple[float, dict[str, int]]] = []
    for width in (64, 72, 80, 88, 96, 104, 112, 128):
        for projection_width in (128, 160, 192, 224, 256):
            kwargs = {"width": width, "modes": 16, "n_layers": 4, "projection_width": projection_width}
            count = FNO1d(**kwargs).param_count()
            candidates.append((abs(count - target_params) / target_params, kwargs))
    return min(candidates, key=lambda item: item[0])[1]


def _loaders(data_path: Path, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(BraggDataset(data_path, "train"), batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(BraggDataset(data_path, "val"), batch_size=batch_size, shuffle=False),
        DataLoader(BraggDataset(data_path, "heldout_energy"), batch_size=batch_size, shuffle=False),
    )


def _train_one(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, list[dict[str, float | int]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    total_steps = max(1, train_cfg.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, train_cfg, device, optimizer, scheduler)
        val_loss = _run_epoch(model, val_loader, train_cfg, device, None, None)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        print(f"{model_name:>20} epoch {epoch:03d} train_loss={train_loss:.6g} val_loss={val_loss:.6g}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return best_val, history


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(x, scalars)
            loss, _ = compute_loss(outputs, batch, train_cfg)
        if optimizer is not None:
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total += float(loss.detach().cpu())
        n_batches += 1
    return total / max(1, n_batches)


def _evaluate_heldout(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    edge_errors: list[float] = []
    rmses: list[float] = []
    gammas: list[float] = []
    peaks: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            scalars = batch["scalars"].to(device=device, dtype=torch.float32)
            pred = model(x, scalars)["dose"].detach().cpu().numpy()
            ref = batch["dose"].detach().cpu().numpy()
            z = batch["z"].detach().cpu().numpy()
            for i in range(pred.shape[0]):
                pred_i = np.asarray(pred[i], dtype=np.float64)
                ref_i = np.asarray(ref[i], dtype=np.float64)
                z_i = np.asarray(z[i], dtype=np.float64)
                edge_errors.append(distal_edge_error_mm(pred_i, ref_i, z_i))
                rmses.append(rmse_pct(pred_i, ref_i))
                gammas.append(gamma_index_1d(pred_i, ref_i, z_i))
                peaks.append(peak_depth_cm(z_i, pred_i))
    return {
        "n_heldout": float(len(edge_errors)),
        "distal_edge_error_mm_mean": float(np.nanmean(edge_errors)),
        "distal_edge_error_mm_std": float(np.nanstd(edge_errors)),
        "rmse_pct_mean": float(np.nanmean(rmses)),
        "gamma_pass_pct_mean": float(np.nanmean(gammas)),
        "peak_depth_cm_mean": float(np.nanmean(peaks)),
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, float | int]]],
    args: argparse.Namespace,
    train_cfg: TrainConfig,
    device: torch.device,
) -> None:
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.out_csv.with_suffix(".npz")
    np.savez(
        npz_path,
        model=np.asarray([row["model"] for row in rows]),
        param_count=np.asarray([row["param_count"] for row in rows], dtype=np.int64),
        distal_edge_error_mm_mean=np.asarray([row["distal_edge_error_mm_mean"] for row in rows], dtype=np.float64),
        rmse_pct_mean=np.asarray([row["rmse_pct_mean"] for row in rows], dtype=np.float64),
        gamma_pass_pct_mean=np.asarray([row["gamma_pass_pct_mean"] for row in rows], dtype=np.float64),
    )
    meta = {
        "experiment": "phase2b_backbone_ablation",
        "data": str(args.data),
        "csv": str(args.out_csv),
        "npz": str(npz_path),
        "models": list(MODELS),
        "train": asdict(train_cfg),
        "device": str(device),
        "seed": args.seed,
        "units": {"depth": "cm", "distal_edge_error": "mm", "energy": "MeV"},
        "histories": histories,
    }
    args.out_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def _print_ranked_table(rows: list[dict[str, Any]], out_csv: Path) -> None:
    print("\nPhase 2B held-out distal-edge ranking")
    print(f"{'rank':>4} {'model':>20} {'params':>10} {'gap%':>9} {'edge_mm':>10} {'rmse%':>10} {'gamma%':>10}")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4d} {row['model']:>20} {int(row['param_count']):>10d} "
            f"{float(row['param_gap_pct']):>9.2f} {float(row['distal_edge_error_mm_mean']):>10.3f} "
            f"{float(row['rmse_pct_mean']):>10.3f} {float(row['gamma_pass_pct_mean']):>10.2f}"
        )
    print(f"\nSharp-edge owner: {rows[0]['model']} ({float(rows[0]['distal_edge_error_mm_mean']):.3f} mm mean error)")
    print(f"Wrote {out_csv} plus NPZ/JSON metadata.")


def _param_count(model: nn.Module) -> int:
    if hasattr(model, "param_count"):
        return int(model.param_count())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
