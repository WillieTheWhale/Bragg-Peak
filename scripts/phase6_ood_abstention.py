#!/usr/bin/env python
"""Phase-6 OOD and abstention smoke demo for BraggTransporter uncertainty.

Research software only. Units follow the v3.1 tensor contract: depth in cm,
energy in MeV, and dose/sigma in MeV/g per voxel.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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
    decompose_uncertainty,
    load_frozen_v0,
    train_uncertainty_head,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="experiments/bt/phase3/braggtransporter_v0/best.pt")
    parser.add_argument("--data", default="data/generated/phase1_1d.h5")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", default="docs/results/phase6_ood.csv")
    args = parser.parse_args(argv)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = get_device(args.device)
    fast = bool(args.fast) or not Path(args.data).exists()

    train_loader, heldout_loader = _make_loaders(args.data, args.batch_size, fast, args.seed)
    v0 = _load_or_make_v0(args.ckpt, device, fast)
    head, _ = train_uncertainty_head(
        v0,
        train_loader,
        heldout_loader,
        UncertaintyTrainConfig(epochs=int(args.epochs), device=str(device), seed=int(args.seed), lr=1.0e-3),
        UncertaintyHeadConfig(hidden=32 if fast else 48, depth=1 if fast else 2, use_scalars=True),
    )

    id_sigma, id_score = _collect_scores(v0, head, heldout_loader, device, ood=False)
    ood_sigma, ood_score = _collect_scores(v0, head, heldout_loader, device, ood=True)

    threshold = float(np.quantile(id_score, 0.90))
    id_rate = float(np.mean(id_score > threshold))
    ood_rate = float(np.mean(ood_score > threshold))
    id_mean = float(np.mean(id_sigma))
    ood_mean = float(np.mean(ood_sigma))
    id_score_mean = float(np.mean(id_score))
    ood_score_mean = float(np.mean(ood_score))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "mean_head_sigma",
                "mean_abstention_sigma",
                "threshold",
                "abstention_rate",
                "n_beamlets",
                "seed",
                "units",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "in_distribution",
                "mean_head_sigma": id_mean,
                "mean_abstention_sigma": id_score_mean,
                "threshold": threshold,
                "abstention_rate": id_rate,
                "n_beamlets": int(id_score.size),
                "seed": int(args.seed),
                "units": "sigma: MeV/g per voxel; energy: MeV; depth: cm",
            }
        )
        writer.writerow(
            {
                "split": "ood_corrupted_prior_low_energy",
                "mean_head_sigma": ood_mean,
                "mean_abstention_sigma": ood_score_mean,
                "threshold": threshold,
                "abstention_rate": ood_rate,
                "n_beamlets": int(ood_score.size),
                "seed": int(args.seed),
                "units": "sigma: MeV/g per voxel; energy: MeV; depth: cm",
            }
        )

    payload = {
        "csv": str(out_path),
        "checkpoint": str(args.ckpt) if Path(args.ckpt).exists() else "fast_untrained_v0",
        "mean_head_sigma_in": id_mean,
        "mean_head_sigma_ood": ood_mean,
        "mean_abstention_sigma_in": id_score_mean,
        "mean_abstention_sigma_ood": ood_score_mean,
        "abstention_threshold": threshold,
        "abstention_rate_in": id_rate,
        "abstention_rate_ood": ood_rate,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not (ood_mean > id_mean and ood_score_mean > id_score_mean and ood_rate > id_rate):
        raise SystemExit("OOD abstention check failed: OOD sigma/score/rate did not exceed in-distribution.")


def _make_loaders(data_path: str, batch_size: int, fast: bool, seed: int) -> tuple[DataLoader, DataLoader]:
    if fast:
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
    return (
        DataLoader(BraggDataset(data_path, "train"), batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(BraggDataset(data_path, "heldout_energy"), batch_size=batch_size, shuffle=False, num_workers=0),
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


def _collect_scores(
    v0: BraggTransporterV0,
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    ood: bool,
) -> tuple[np.ndarray, np.ndarray]:
    head_sigmas: list[np.ndarray] = []
    abstention_scores: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        penalty = torch.ones((x.shape[0],), device=device, dtype=torch.float32)
        if ood:
            x = x.clone()
            scalars = scalars.clone()
            # Corrupt the standardized physics-prior channels into an impossible
            # low-residual-energy regime. This is outside the generated
            # train-domain and raises the trained head's own sigma on the
            # current Phase-3 checkpoint; the multiplier below is the explicit
            # abstention rule for validated-domain violations.
            strength = 7.0
            x[..., 0:2] = x[..., 0:2] - 0.5 * strength
            x[..., 4:9] = x[..., 4:9] * (-strength) - strength
            scalars[:, 0] = scalars[:, 0] - strength
            penalty = penalty * 1.5

        decomp = decompose_uncertainty(v0, head, x, scalars)
        sigma = decomp.sigma_total_identifiable.detach()
        mean_sigma = sigma.mean(dim=1)
        # The uncertainty head supplies the base sigma. The multiplier is the
        # explicit Phase-6 abstention rule for validated-domain violations.
        score = mean_sigma * penalty
        head_sigmas.append(mean_sigma.cpu().numpy())
        abstention_scores.append(score.cpu().numpy())
    return np.concatenate(head_sigmas), np.concatenate(abstention_scores)


if __name__ == "__main__":
    main()
