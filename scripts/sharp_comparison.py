#!/usr/bin/env python
"""Matched-budget comparison on sharp BraggTransporter 1-D data."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braggtransporter.config import ModelConfig, TrainConfig, get_device
from braggtransporter.data.dataset import BraggDataset
from braggpeak.scoring import compute_bragg_metrics
from braggtransporter.metrics import gamma_index_1d, rmse_pct
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.models.fno1d import FNO1d
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR
from braggtransporter.train import _set_determinism, compute_loss


MODELS = ("braggtransporter_v0", "fno1d", "conv1d")


class ConvBaseline1d(nn.Module):
    """Small local 1-D convolution baseline defined only for this ablation."""

    def __init__(
        self,
        width: int = 96,
        n_layers: int = 5,
        kernel_size: int = 5,
        scalar_width: int = 16,
        projection_width: int = 128,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.scalar_lift = nn.Sequential(
            nn.Linear(C_SCALAR, projection_width),
            nn.GELU(),
            nn.Linear(projection_width, scalar_width),
            nn.GELU(),
        )
        self.input = nn.Conv1d(C_IN_PERDEPTH + scalar_width, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(width, width, kernel_size=kernel_size, padding=pad),
                nn.GroupNorm(1, width),
                nn.GELU(),
                nn.Conv1d(width, width, kernel_size=1),
            )
            for _ in range(n_layers)
        )
        self.projection = nn.Sequential(nn.Linear(width, projection_width), nn.GELU())
        self.dose_head = nn.Linear(projection_width, 1)
        self.letd_head = nn.Linear(projection_width, 1)
        self.r80_head = nn.Sequential(
            nn.Linear(2 * projection_width + scalar_width, projection_width),
            nn.GELU(),
            nn.Linear(projection_width, 1),
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> dict[str, torch.Tensor]:
        scalar_features = self.scalar_lift(scalars)
        scalar_per_depth = scalar_features[:, None, :].expand(-1, x.shape[1], -1)
        y = torch.cat([x, scalar_per_depth], dim=-1).transpose(1, 2)
        y = self.input(y)
        for block in self.blocks:
            y = F.gelu(y + block(y))
        features = y.transpose(1, 2)
        projected = self.projection(features)
        pooled = torch.cat([projected.mean(dim=1), projected.amax(dim=1), scalar_features], dim=-1)
        return {
            "dose": F.softplus(self.dose_head(projected)).squeeze(-1),
            "letd": self.letd_head(projected).squeeze(-1),
            "r80": self.r80_head(pooled).squeeze(-1),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/generated/sharp_1d.h5"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/results/sharp_comparison.csv"))
    parser.add_argument("--tiny", action="store_true", help="Use tiny models for CPU smoke tests.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-heldout-samples", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
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
        distal_edge_weight=7.0,
    )
    constructors, target_params = _matched_constructors(args.tiny)
    train_loader, val_loader, heldout_loader = _loaders(args)

    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float | int]]] = {}
    per_sample: dict[str, dict[str, NDArray[np.float64]]] = {}

    for model_name in MODELS:
        model = constructors[model_name]().to(device)
        count = _param_count(model)
        best_val, history = _train_one(model, model_name, train_loader, val_loader, train_cfg, device)
        histories[model_name] = history
        metrics, arrays = _evaluate_heldout(model, heldout_loader, device)
        per_sample[model_name] = arrays
        rows.append(
            {
                "model": model_name,
                "param_count": count,
                "target_param_count": target_params,
                "param_gap_pct": 100.0 * (count - target_params) / max(1, target_params),
                "epochs": args.epochs,
                "best_val_loss": best_val,
                **metrics,
            }
        )

    paired = _paired_v0_fno(per_sample)
    for row in rows:
        row.update({f"v0_vs_fno_{k}": v for k, v in paired.items() if isinstance(v, (int, float, str, bool))})
    rows.sort(key=lambda row: float(row["distal_edge_error_mm_mean"]))
    _write_outputs(rows, histories, per_sample, paired, args, train_cfg, device)
    _print_ranked_table(rows, paired, args.out_csv)
    return rows


def _matched_constructors(tiny: bool) -> tuple[dict[str, Any], int]:
    if tiny:
        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, dropout=0.0)
        target = BraggTransporterV0(cfg).param_count()
        fno_kwargs = _best_fno_kwargs(target, widths=(16, 24, 32), projections=(32, 48, 64), layers=(1, 2), modes=(4, 8))
        conv_kwargs = _best_conv_kwargs(target, widths=(16, 24, 32), projections=(32, 48, 64), layers=(1, 2, 3))
    else:
        cfg = ModelConfig()
        target = BraggTransporterV0(cfg).param_count()
        fno_kwargs = _best_fno_kwargs(
            target,
            widths=(64, 72, 80, 88, 96, 104, 112, 128),
            projections=(128, 160, 192, 224, 256),
            layers=(4,),
            modes=(16,),
        )
        conv_kwargs = _best_conv_kwargs(
            target,
            widths=(64, 80, 96, 112, 128, 160),
            projections=(128, 160, 192, 224, 256),
            layers=(3, 4, 5, 6, 7),
        )
    return (
        {
            "braggtransporter_v0": lambda: BraggTransporterV0(cfg),
            "fno1d": lambda: FNO1d(**fno_kwargs),
            "conv1d": lambda: ConvBaseline1d(**conv_kwargs),
        },
        target,
    )


def _best_fno_kwargs(
    target_params: int,
    *,
    widths: tuple[int, ...],
    projections: tuple[int, ...],
    layers: tuple[int, ...],
    modes: tuple[int, ...],
) -> dict[str, int]:
    candidates: list[tuple[float, dict[str, int]]] = []
    for width in widths:
        for projection_width in projections:
            for n_layers in layers:
                for mode in modes:
                    kwargs = {"width": width, "modes": mode, "n_layers": n_layers, "projection_width": projection_width}
                    count = FNO1d(**kwargs).param_count()
                    candidates.append((abs(count - target_params) / max(1, target_params), kwargs))
    return min(candidates, key=lambda item: item[0])[1]


def _best_conv_kwargs(
    target_params: int,
    *,
    widths: tuple[int, ...],
    projections: tuple[int, ...],
    layers: tuple[int, ...],
) -> dict[str, int]:
    candidates: list[tuple[float, dict[str, int]]] = []
    for width in widths:
        for projection_width in projections:
            for n_layers in layers:
                kwargs = {"width": width, "n_layers": n_layers, "projection_width": projection_width}
                count = ConvBaseline1d(**kwargs).param_count()
                candidates.append((abs(count - target_params) / max(1, target_params), kwargs))
    return min(candidates, key=lambda item: item[0])[1]


def _loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(args.seed)
    return (
        DataLoader(_limited(BraggDataset(args.data, "train"), args.max_train_samples), batch_size=args.batch_size, shuffle=True, generator=generator),
        DataLoader(_limited(BraggDataset(args.data, "val"), args.max_val_samples), batch_size=args.batch_size, shuffle=False),
        DataLoader(_limited(BraggDataset(args.data, "heldout_energy"), args.max_heldout_samples), batch_size=args.batch_size, shuffle=False),
    )


def _limited(dataset: BraggDataset, limit: int | None) -> BraggDataset | Subset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max(1, int(limit)))))


def _train_one(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, list[dict[str, float | int]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_cfg.epochs * max(1, len(train_loader))),
    )
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, train_cfg, device, optimizer, scheduler)
        val_loss = _run_epoch(model, val_loader, train_cfg, device, None, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        print(f"{model_name:>20} epoch {epoch:03d} train_loss={train_loss:.6g} val_loss={val_loss:.6g}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return float(best_val), history


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
            if train_cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total += float(loss.detach().cpu())
        n_batches += 1
    return total / max(1, n_batches)


def _evaluate_heldout(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, NDArray[np.float64]]]:
    model.eval()
    edge_errors: list[float] = []
    rmses: list[float] = []
    gammas_1_1: list[float] = []
    gammas_2_2: list[float] = []
    r80_ref: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            scalars = batch["scalars"].to(device=device, dtype=torch.float32)
            pred = model(x, scalars)["dose"].detach().cpu().numpy()
            ref = batch["dose"].detach().cpu().numpy()
            z = batch["z"].detach().cpu().numpy()
            r80 = batch["r80"].detach().cpu().numpy()
            for i in range(pred.shape[0]):
                pred_i = np.asarray(pred[i], dtype=np.float64)
                ref_i = np.asarray(ref[i], dtype=np.float64)
                z_i = np.asarray(z[i], dtype=np.float64)
                edge_errors.append(_finite_distal_edge_error_mm(pred_i, ref_i, z_i))
                rmses.append(rmse_pct(pred_i, ref_i))
                gammas_1_1.append(gamma_index_1d(pred_i, ref_i, z_i, dose_pct=1.0, dta_mm=1.0))
                gammas_2_2.append(gamma_index_1d(pred_i, ref_i, z_i, dose_pct=2.0, dta_mm=2.0))
                r80_ref.append(float(r80[i]) * 10.0)
    arrays = {
        "distal_edge_error_mm": np.asarray(edge_errors, dtype=np.float64),
        "rmse_pct": np.asarray(rmses, dtype=np.float64),
        "gamma_1pct_1mm": np.asarray(gammas_1_1, dtype=np.float64),
        "gamma_2pct_2mm": np.asarray(gammas_2_2, dtype=np.float64),
        "ref_r80_mm": np.asarray(r80_ref, dtype=np.float64),
    }
    return (
        {
            "n_heldout": float(len(edge_errors)),
            "distal_edge_error_mm_mean": _nanmean(arrays["distal_edge_error_mm"]),
            "distal_edge_error_mm_std": _nanstd(arrays["distal_edge_error_mm"]),
            "distal_edge_error_mm_median": _nanmedian(arrays["distal_edge_error_mm"]),
            "rmse_pct_mean": _nanmean(arrays["rmse_pct"]),
            "gamma_1pct_1mm_pass_pct_mean": _nanmean(arrays["gamma_1pct_1mm"]),
            "gamma_2pct_2mm_pass_pct_mean": _nanmean(arrays["gamma_2pct_2mm"]),
        },
        arrays,
    )


def _paired_v0_fno(per_sample: dict[str, dict[str, NDArray[np.float64]]]) -> dict[str, Any]:
    v0 = per_sample["braggtransporter_v0"]["distal_edge_error_mm"]
    fno = per_sample["fno1d"]["distal_edge_error_mm"]
    n = min(v0.size, fno.size)
    diff = fno[:n] - v0[:n]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"mean_edge_reduction_mm": float("nan"), "ci95_low_mm": float("nan"), "ci95_high_mm": float("nan"), "p_signflip": float("nan"), "significant": False}
    rng = np.random.default_rng(0)
    boot = np.asarray([np.mean(rng.choice(diff, size=diff.size, replace=True)) for _ in range(5000)], dtype=np.float64)
    p = _signflip_p_value(diff, rng)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_edge_reduction_mm": float(np.mean(diff)),
        "ci95_low_mm": float(low),
        "ci95_high_mm": float(high),
        "p_signflip": float(p),
        "n": int(diff.size),
        "significant": bool(low > 0.0 or high < 0.0),
    }


def _finite_distal_edge_error_mm(pred: NDArray[np.float64], ref: NDArray[np.float64], z: NDArray[np.float64]) -> float:
    pred_r80 = _r80_or_grid_edge_mm(pred, z)
    ref_r80 = _r80_or_grid_edge_mm(ref, z)
    if not np.isfinite(pred_r80) or not np.isfinite(ref_r80):
        return float("nan")
    return float(abs(pred_r80 - ref_r80))


def _r80_or_grid_edge_mm(dose: NDArray[np.float64], z: NDArray[np.float64]) -> float:
    metrics = compute_bragg_metrics(z, np.maximum(dose, 0.0))
    if np.isfinite(metrics.r80_mm):
        return float(metrics.r80_mm)
    peak_idx = int(np.argmax(dose))
    peak = float(np.max(dose))
    if peak <= 0.0:
        return float("nan")
    target = 0.8 * peak
    # If the curve never falls below 80% after the peak, score it as stopping
    # beyond the scored grid rather than dropping the sample from the paired set.
    if bool(np.all(dose[peak_idx:] >= target)):
        return float(z[-1] * 10.0)
    return float(z[peak_idx] * 10.0)


def _signflip_p_value(diff: NDArray[np.float64], rng: np.random.Generator) -> float:
    obs = abs(float(np.mean(diff)))
    n = int(diff.size)
    if n <= 18:
        count = 0
        total = 2**n
        for mask in range(total):
            signs = np.asarray([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)], dtype=np.float64)
            if abs(float(np.mean(signs * diff))) >= obs - 1.0e-15:
                count += 1
        return count / total
    draws = 20000
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, n), replace=True)
    return float((np.sum(np.abs(np.mean(signs * diff[None, :], axis=1)) >= obs) + 1) / (draws + 1))


def _write_outputs(
    rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, float | int]]],
    per_sample: dict[str, dict[str, NDArray[np.float64]]],
    paired: dict[str, Any],
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
        gamma_1pct_1mm_pass_pct_mean=np.asarray([row["gamma_1pct_1mm_pass_pct_mean"] for row in rows], dtype=np.float64),
        gamma_2pct_2mm_pass_pct_mean=np.asarray([row["gamma_2pct_2mm_pass_pct_mean"] for row in rows], dtype=np.float64),
        **{f"{model}_{key}": value for model, arrays in per_sample.items() for key, value in arrays.items()},
    )
    meta = {
        "experiment": "sharp_data_architecture_test",
        "data": str(args.data),
        "csv": str(args.out_csv),
        "npz": str(npz_path),
        "models": list(MODELS),
        "train": asdict(train_cfg),
        "device": str(device),
        "seed": args.seed,
        "tiny": args.tiny,
        "units": {"depth": "cm", "distal_edge_error": "mm", "gamma": "pass percent"},
        "histories": histories,
        "paired_v0_minus_fno": paired,
    }
    args.out_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _print_ranked_table(rows: list[dict[str, Any]], paired: dict[str, Any], out_csv: Path) -> None:
    print("\nSharp held-out distal-edge ranking")
    print(f"{'rank':>4} {'model':>20} {'params':>10} {'gap%':>9} {'edge_mm':>10} {'g1/1%':>10} {'g2/2%':>10}")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4d} {row['model']:>20} {int(row['param_count']):>10d} "
            f"{float(row['param_gap_pct']):>9.2f} {float(row['distal_edge_error_mm_mean']):>10.3f} "
            f"{float(row['gamma_1pct_1mm_pass_pct_mean']):>10.2f} {float(row['gamma_2pct_2mm_pass_pct_mean']):>10.2f}"
        )
    conclusion = _conclusion(paired)
    print(f"\n{conclusion}")
    print(f"Wrote {out_csv} plus NPZ/JSON metadata.")


def _conclusion(paired: dict[str, Any]) -> str:
    delta = float(paired["mean_edge_reduction_mm"])
    low = float(paired["ci95_low_mm"])
    high = float(paired["ci95_high_mm"])
    p = float(paired["p_signflip"])
    if not np.isfinite(delta):
        return "Conclusion: v0 vs FNO significance is undefined because no finite paired held-out edge errors were produced."
    if delta > 0.0 and bool(paired["significant"]):
        return f"Conclusion: transformer beats FNO on the sharp edge by {delta:.3f} mm mean paired error reduction (95% CI {low:.3f}..{high:.3f}, sign-flip p={p:.4g})."
    if delta < 0.0 and bool(paired["significant"]):
        return f"Conclusion: FNO beats transformer on the sharp edge by {-delta:.3f} mm mean paired error reduction (95% CI {low:.3f}..{high:.3f}, sign-flip p={p:.4g})."
    return f"Conclusion: no significant transformer-vs-FNO sharp-edge win; paired v0 reduction is {delta:.3f} mm (95% CI {low:.3f}..{high:.3f}, sign-flip p={p:.4g})."


def _nanmean(values: NDArray[np.float64]) -> float:
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _nanstd(values: NDArray[np.float64]) -> float:
    return float(np.nanstd(values)) if np.any(np.isfinite(values)) else float("nan")


def _nanmedian(values: NDArray[np.float64]) -> float:
    return float(np.nanmedian(values)) if np.any(np.isfinite(values)) else float("nan")


def _param_count(model: nn.Module) -> int:
    if hasattr(model, "param_count"):
        return int(model.param_count())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
