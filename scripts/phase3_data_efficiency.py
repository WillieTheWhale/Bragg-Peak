#!/usr/bin/env python
"""Phase 3 Stage-0 data-efficiency study for BraggTransporter v3.1.

Trains supervised v0 from scratch and from Stage-0 masked-pretrained encoder
weights on 25%, 50%, and 100% of the training split, then evaluates held-out
gamma pass rate and distal-edge error. Research software only; no clinical
claims.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braggtransporter.config import DataConfig, ModelConfig, TrainConfig, get_device
from braggtransporter.data.dataset import BraggDataset
from braggtransporter.metrics import distal_edge_error_mm, gamma_index_1d, peak_depth_cm, rmse_pct
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.pretrain import load_pretrained_encoder, pretrain
from braggtransporter.train import RunConfig, SyntheticBraggDataset, compute_loss, load_config


DEFAULT_CONFIG = ROOT / "configs" / "bt_v0_1d.yaml"
DEFAULT_OUT_CSV = ROOT / "docs" / "results" / "phase3_data_efficiency.csv"
DEFAULT_ENCODER = ROOT / "experiments" / "bt" / "pretrain" / "encoder.pt"
FRACTIONS = (0.25, 0.50, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--epochs", type=int, default=None, help="Supervised epochs per condition.")
    parser.add_argument("--pretrain-epochs", type=int, default=5, help="Stage-0 epochs if encoder checkpoint is missing.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="CPU smoke mode with synthetic data and one epoch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data is not None:
        cfg.data.out_path = str(args.data)
    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
        cfg.data.seed = int(args.seed)
    if args.fast:
        cfg.train.epochs = min(cfg.train.epochs, 1)
        cfg.train.batch_size = min(cfg.train.batch_size, 8)
        cfg.model.d_model = min(cfg.model.d_model, 32)
        cfg.model.n_layers = min(cfg.model.n_layers, 1)
        cfg.model.n_heads = min(cfg.model.n_heads, 4)
        cfg.model.d_ff = min(cfg.model.d_ff, 64)
        cfg.model.extra["max_positions"] = min(int(cfg.model.extra.get("max_positions", 128)), 128)
        args.device = "cpu"

    _seed_everything(cfg.train.seed)
    device = get_device(args.device)
    train_cfg = TrainConfig(**asdict(cfg.train))
    train_cfg.device = str(device)
    data_path = Path(cfg.data.out_path)
    train_ds, val_ds, heldout_ds = _datasets(cfg, args.fast or not data_path.exists())

    if not args.encoder.exists() or not _encoder_compatible(args.encoder, cfg.model):
        reason = "not found" if not args.encoder.exists() else "incompatible with this ModelConfig"
        print(f"Stage-0 encoder {reason} at {args.encoder}; running pretraining first.")
        pre_cfg = copy.deepcopy(cfg)
        pre_cfg.train.device = str(device)
        pretrain(pre_cfg, epochs=args.pretrain_epochs, device_override=str(device), fast=args.fast, out_dir=args.encoder.parent)

    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float | int]]] = {}
    for fraction in FRACTIONS:
        subset = _fraction_subset(train_ds, fraction, cfg.train.seed)
        for init_name in ("scratch", "stage0"):
            tag = f"{init_name}_{int(round(100 * fraction))}pct"
            model = BraggTransporterV0(cfg.model).to(device)
            loaded = 0
            skipped = 0
            if init_name == "stage0":
                info = load_pretrained_encoder(model, args.encoder)
                loaded = len(info["loaded"])
                skipped = len(info["skipped"])
                model.to(device)
            train_loader = DataLoader(
                subset,
                batch_size=min(train_cfg.batch_size, max(1, len(subset))),
                shuffle=True,
                generator=torch.Generator().manual_seed(cfg.train.seed + int(1000 * fraction)),
            )
            val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)
            heldout_loader = DataLoader(heldout_ds, batch_size=train_cfg.batch_size, shuffle=False)
            best_val, history = _train_one(model, train_loader, val_loader, train_cfg, device)
            histories[tag] = history
            metrics = _evaluate_heldout(model, heldout_loader, device)
            row = {
                "fraction": fraction,
                "train_samples": len(subset),
                "init": init_name,
                "epochs": train_cfg.epochs,
                "best_val_loss": best_val,
                "loaded_encoder_tensors": loaded,
                "skipped_encoder_tensors": skipped,
                **metrics,
            }
            rows.append(row)
            print(
                f"{tag:>14} val={best_val:.6g} edge_mm={metrics['distal_edge_error_mm_mean']:.3f} "
                f"gamma={metrics['gamma_pass_pct_mean']:.2f}"
            )

    _write_outputs(rows, histories, args, cfg, device)
    _print_help_summary(rows, args.out_csv)


def _datasets(cfg: RunConfig, synthetic: bool) -> tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
    if synthetic:
        n_samples = 24 if cfg.train.epochs <= 2 else 96
        nz = 64 if n_samples <= 24 else max(16, int(round(cfg.data.max_depth_cm / cfg.data.dz_cm)) + 1)
        ds = SyntheticBraggDataset(n_samples=n_samples, nz=nz, data_cfg=cfg.data, seed=cfg.train.seed)
        train_len = max(1, int(0.60 * len(ds)))
        val_len = max(1, int(0.20 * len(ds)))
        heldout_len = len(ds) - train_len - val_len
        return random_split(ds, [train_len, val_len, heldout_len], generator=torch.Generator().manual_seed(cfg.train.seed))
    path = Path(cfg.data.out_path)
    return BraggDataset(path, "train"), BraggDataset(path, "val"), BraggDataset(path, "heldout_energy")


def _fraction_subset(dataset: Dataset[Any], fraction: float, seed: int) -> Subset[Any]:
    n = max(1, int(round(len(dataset) * fraction)))
    generator = torch.Generator().manual_seed(seed + int(round(1000 * fraction)))
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)


def _encoder_compatible(path: Path, model_cfg: ModelConfig) -> bool:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    saved_cfg = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if not isinstance(saved_cfg, dict):
        return False
    current = asdict(model_cfg)
    for key in ("d_model", "n_layers", "n_heads", "d_ff"):
        if int(saved_cfg.get(key, -1)) != int(current.get(key, -2)):
            return False
    saved_extra = saved_cfg.get("extra", {}) or {}
    current_extra = current.get("extra", {}) or {}
    return int(saved_extra.get("max_positions", 2048)) == int(current_extra.get("max_positions", 2048))


def _train_one(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    train_cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, list[dict[str, float | int]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    total_steps = max(1, train_cfg.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    best_val = float("inf")
    best_state: dict[str, TensorLike] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, train_cfg, device, optimizer, scheduler)
        val_loss = _run_epoch(model, val_loader, train_cfg, device, None, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss <= best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return best_val, history


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    train_cfg: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
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


def _evaluate_heldout(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> dict[str, float]:
    model.eval()
    edge_errors: list[float] = []
    gammas: list[float] = []
    rmses: list[float] = []
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
                gammas.append(gamma_index_1d(pred_i, ref_i, z_i))
                rmses.append(rmse_pct(pred_i, ref_i))
                peaks.append(peak_depth_cm(z_i, pred_i))
    return {
        "n_heldout": float(len(edge_errors)),
        "distal_edge_error_mm_mean": _nanmean(edge_errors),
        "distal_edge_error_mm_std": _nanstd(edge_errors),
        "gamma_pass_pct_mean": _nanmean(gammas),
        "rmse_pct_mean": _nanmean(rmses),
        "peak_depth_cm_mean": _nanmean(peaks),
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, float | int]]],
    args: argparse.Namespace,
    cfg: RunConfig,
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
        fraction=np.asarray([row["fraction"] for row in rows], dtype=np.float64),
        init=np.asarray([row["init"] for row in rows]),
        distal_edge_error_mm_mean=np.asarray([row["distal_edge_error_mm_mean"] for row in rows], dtype=np.float64),
        gamma_pass_pct_mean=np.asarray([row["gamma_pass_pct_mean"] for row in rows], dtype=np.float64),
        rmse_pct_mean=np.asarray([row["rmse_pct_mean"] for row in rows], dtype=np.float64),
    )
    meta = {
        "experiment": "phase3_stage0_data_efficiency",
        "config": str(args.config),
        "data": cfg.data.out_path,
        "encoder": str(args.encoder),
        "csv": str(args.out_csv),
        "npz": str(npz_path),
        "device": str(device),
        "seed": cfg.train.seed,
        "train": asdict(cfg.train),
        "model": asdict(cfg.model),
        "units": {"depth": "cm", "distal_edge_error": "mm", "energy": "MeV", "dose": "MeV/g per voxel"},
        "histories": histories,
    }
    args.out_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def _print_help_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    by_fraction: dict[float, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_fraction.setdefault(float(row["fraction"]), {})[str(row["init"])] = row
    helped = 0
    comparable = 0
    print("\nStage-0 data-efficiency summary")
    print(f"{'fraction':>9} {'edge_delta_mm':>14} {'gamma_delta_pct':>16} {'helps':>8}")
    for fraction in sorted(by_fraction):
        pair = by_fraction[fraction]
        if "scratch" not in pair or "stage0" not in pair:
            continue
        comparable += 1
        edge_delta = float(pair["scratch"]["distal_edge_error_mm_mean"]) - float(pair["stage0"]["distal_edge_error_mm_mean"])
        gamma_delta = float(pair["stage0"]["gamma_pass_pct_mean"]) - float(pair["scratch"]["gamma_pass_pct_mean"])
        helps = (np.isfinite(edge_delta) and edge_delta > 0.0) or (np.isfinite(gamma_delta) and gamma_delta > 0.0)
        helped += int(helps)
        print(f"{fraction:>9.2f} {edge_delta:>14.3f} {gamma_delta:>16.2f} {str(helps):>8}")
    verdict = "helps" if comparable and helped >= max(1, comparable // 2 + comparable % 2) else "does not help"
    print(f"\nStage-0 {verdict} by the paired held-out criteria in this run.")
    print(f"Wrote {out_csv} plus NPZ/JSON metadata.")


TensorLike = torch.Tensor


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def _nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.std()) if finite.size else float("nan")


if __name__ == "__main__":
    main()
