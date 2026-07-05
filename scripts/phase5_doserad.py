#!/usr/bin/env python
"""Phase 5 DoseRAD2026 3-D beamlet proof-of-concept training."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from braggtransporter.config import get_device
from braggtransporter.data.doserad import (
    depth_profile,
    gamma_index_3d,
    make_doserad_loaders,
    r80_mm_from_profile,
    rmse_pct_3d,
)
from braggtransporter.models.bragg3d import Bragg3D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/doserad2026", help="DoseRAD2026 root directory.")
    p.add_argument("--patients", nargs="+", default=["1ABB006"], help="Patient IDs, e.g. 1ABB006.")
    p.add_argument("--max-beamlets", type=int, default=32, help="Maximum valid beamlets to cache/train.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--depth-size", type=int, default=48)
    p.add_argument("--lateral-size", type=int, default=16)
    p.add_argument("--lateral-extent-mm", type=float, default=96.0)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--out", default="docs/results/phase5_doserad.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = get_device(args.device)

    train_loader, val_loader = make_doserad_loaders(
        args.root,
        args.patients,
        max_beamlets=args.max_beamlets,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        depth_size=args.depth_size,
        lateral_size=args.lateral_size,
        lateral_extent_mm=args.lateral_extent_mm,
        rebuild_cache=args.rebuild_cache,
    )

    model = Bragg3D(max_depth=max(128, args.depth_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print(
        f"Phase 5 DoseRAD proof-of-concept: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val beamlets, device={device}, params={model.param_count():,}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device)
        val_loss = _val_loss(model, val_loader, device)
        print(f"epoch {epoch:03d}: train_loss={train_loss:.6g} val_loss={val_loss:.6g}")

    metrics = evaluate(model, val_loader, device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_outputs(out_path, args, metrics)

    print(
        "Held-out beamlets: "
        f"gamma3d_3pct_3mm={metrics['gamma3d_3pct_3mm']:.2f}% "
        f"rmse={metrics['rmse_pct']:.2f}% "
        f"r80_error={metrics['r80_error_mm']:.2f} mm"
    )
    print(
        "Laptop-scale Phase 5 training is a proof-of-concept only; full-scale "
        "DoseRAD gamma remains cloud-gated per the v3.1 plan."
    )


def _train_one_epoch(model: Bragg3D, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        dose = batch["dose"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, scalars)["dose"]
        loss = _relative_mse(pred, dose)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _val_loss(model: Bragg3D, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        dose = batch["dose"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        losses.append(float(_relative_mse(model(x, scalars)["dose"], dose).cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _relative_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Explicit reference-peak-weighted MSE; stored dose tensors are unchanged."""

    peak = target.flatten(1).amax(dim=1).clamp_min(torch.finfo(target.dtype).eps)
    scale = peak.view(-1, 1, 1, 1)
    mask = target >= 0.02 * scale
    sq = ((pred - target) / scale).square()
    return torch.where(mask, sq, 0.1 * sq).mean()


@torch.no_grad()
def evaluate(model: Bragg3D, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    gamma_values: list[float] = []
    rmse_values: list[float] = []
    r80_errors: list[float] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        pred = model(x, scalars)["dose"].detach().cpu().numpy()
        ref = batch["dose"].numpy()
        spacing = batch["spacing_mm"].numpy()
        for i in range(ref.shape[0]):
            gamma_values.append(gamma_index_3d(pred[i], ref[i], spacing[i], dose_pct=3.0, dta_mm=3.0))
            rmse_values.append(rmse_pct_3d(pred[i], ref[i], low_dose_threshold=0.02))
            pred_r80 = r80_mm_from_profile(depth_profile(pred[i]), float(spacing[i][0]))
            ref_r80 = r80_mm_from_profile(depth_profile(ref[i]), float(spacing[i][0]))
            if np.isfinite(pred_r80) and np.isfinite(ref_r80):
                r80_errors.append(abs(pred_r80 - ref_r80))
    return {
        "gamma3d_3pct_3mm": _nanmean(gamma_values),
        "rmse_pct": _nanmean(rmse_values),
        "r80_error_mm": _nanmean(r80_errors),
        "n_val_beamlets": float(len(gamma_values)),
    }


def _write_outputs(out_path: Path, args: argparse.Namespace, metrics: dict[str, float]) -> None:
    row = {
        "patients": " ".join(args.patients),
        "max_beamlets": args.max_beamlets,
        "epochs": args.epochs,
        "depth_size": args.depth_size,
        "lateral_size": args.lateral_size,
        **metrics,
    }
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    np.savez(out_path.with_suffix(".npz"), **{k: np.asarray(v) for k, v in metrics.items()})
    meta = {
        "script": "scripts/phase5_doserad.py",
        "units": {"spacing": "mm", "dose": "DoseRAD MHA voxel values, not implicitly normalised"},
        "rng_seed": args.seed,
        "device": args.device,
        "args": vars(args),
        "metrics": metrics,
        "note": "Proof-of-concept laptop-scale subset; not a clinical claim.",
    }
    out_path.with_suffix(".json").write_text(json.dumps(_json_clean(meta), indent=2), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    return value


if __name__ == "__main__":
    main()
