"""Training loop for BraggTransporter v3.1 models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

from braggtransporter.config import DataConfig, ModelConfig, TrainConfig, get_device
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR, Fidelity


@dataclass
class RunConfig:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return RunConfig(
        data=_section(DataConfig, raw.get("data", raw)),
        train=_section(TrainConfig, raw.get("train", raw)),
        model=_section(ModelConfig, raw.get("model", raw)),
    )


def build_model(name: str, cfg: ModelConfig) -> nn.Module:
    registry: dict[str, Callable[[], type[nn.Module]]] = {
        "mlp": lambda: _lazy_import("braggtransporter.models.mlp", "MLPBaseline"),
        "fno1d": lambda: _lazy_import("braggtransporter.models.fno1d", "FNO1d"),
        "dota": lambda: _lazy_import("braggtransporter.models.dota_transformer", "DoTATransformer"),
        "braggtransporter_v0": lambda: _lazy_import(
            "braggtransporter.models.braggtransporter_v0", "BraggTransporterV0"
        ),
    }
    if name not in registry:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(registry)}")
    cls = registry[name]()
    try:
        return cls(cfg)
    except TypeError:
        return cls()


def compute_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor], train_cfg: TrainConfig) -> tuple[Tensor, dict[str, float]]:
    dose_pred = outputs["dose"]
    dose_target = batch["dose"].to(device=dose_pred.device, dtype=dose_pred.dtype)
    z = batch["z"].to(device=dose_pred.device, dtype=dose_pred.dtype)
    r80_target = batch["r80"].to(device=dose_pred.device, dtype=dose_pred.dtype).view(-1)

    weights = torch.ones_like(dose_target)
    proximal = (z >= (r80_target[:, None] - 2.0)) & (z <= r80_target[:, None])
    weights = torch.where(proximal, torch.full_like(weights, float(train_cfg.distal_edge_weight)), weights)
    dose_loss = (weights * (dose_pred - dose_target).square()).mean()

    letd_loss = dose_loss.new_tensor(0.0)
    if "letd" in outputs and "letd" in batch:
        letd_target = batch["letd"].to(device=dose_pred.device, dtype=dose_pred.dtype)
        letd_loss = F.mse_loss(outputs["letd"], letd_target)

    r80_loss = dose_loss.new_tensor(0.0)
    if "r80" in outputs:
        r80_loss = F.smooth_l1_loss(outputs["r80"].view(-1), r80_target)

    total = dose_loss + letd_loss + r80_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "dose_loss": float(dose_loss.detach().cpu()),
        "letd_loss": float(letd_loss.detach().cpu()),
        "r80_loss": float(r80_loss.detach().cpu()),
    }


class SyntheticBraggDataset(Dataset[dict[str, Tensor]]):
    """Small deterministic data source for fast CPU tests and absent HDF5 files."""

    def __init__(self, n_samples: int, nz: int, data_cfg: DataConfig, seed: int) -> None:
        self.samples = [_synthetic_sample(i, nz, data_cfg, seed) for i in range(n_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {k: v.clone() for k, v in self.samples[idx].items()}


def make_training_loaders(cfg: RunConfig, fast: bool) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    data_path = Path(cfg.data.out_path)
    use_synthetic = fast or not data_path.exists()
    if use_synthetic:
        n_samples = 24 if fast else 96
        nz = 64 if fast else max(16, int(round(cfg.data.max_depth_cm / cfg.data.dz_cm)) + 1)
        ds = SyntheticBraggDataset(n_samples=n_samples, nz=nz, data_cfg=cfg.data, seed=cfg.train.seed)
        train_len = max(1, int(0.75 * len(ds)))
        val_len = len(ds) - train_len
        gen = torch.Generator().manual_seed(cfg.train.seed)
        train_ds, val_ds = random_split(ds, [train_len, val_len], generator=gen)
        batch_size = min(cfg.train.batch_size, 8 if fast else cfg.train.batch_size)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            None,
        )

    from braggtransporter.data.dataset import make_loaders

    return make_loaders(cfg.data)


def train(cfg: RunConfig, *, device_override: str | None = None, fast: bool = False) -> dict[str, Any]:
    if fast:
        cfg.train.epochs = min(cfg.train.epochs, 2)
        cfg.train.batch_size = min(cfg.train.batch_size, 8)
    if device_override:
        cfg.train.device = device_override

    _set_determinism(cfg.train.seed)
    device = get_device(cfg.train.device)
    model = build_model(cfg.train.model, cfg.model).to(device)
    train_loader, val_loader, _ = make_training_loaders(cfg, fast=fast)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total_steps = max(1, cfg.train.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    out_dir = Path(cfg.train.out_dir) / cfg.train.model
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, cfg.train, device, optimizer, scheduler)
        val_metrics = _run_epoch(model, val_loader, cfg.train, device, None, None)
        row: dict[str, float | int] = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"data": asdict(cfg.data), "train": asdict(cfg.train), "model": asdict(cfg.model)},
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "param_count": _param_count(model),
                },
                out_dir / "best.pt",
            )

    metrics = {
        "best_val_loss": best_loss,
        "epochs": cfg.train.epochs,
        "model": cfg.train.model,
        "device": str(device),
        "param_count": _param_count(model),
        "history": history,
        "units": {"depth": "cm", "energy": "MeV", "dose": "MeV/g per voxel", "letd": "keV/um"},
        "seed": cfg.train.seed,
    }
    _write_metrics(out_dir, metrics, history)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    metrics = train(cfg, device_override=args.device, fast=args.fast)
    print(
        json.dumps(
            {
                "model": metrics["model"],
                "best_val_loss": metrics["best_val_loss"],
                "param_count": metrics["param_count"],
                "device": metrics["device"],
            },
            indent=2,
        )
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {"loss": 0.0, "dose_loss": 0.0, "letd_loss": 0.0, "r80_loss": 0.0}
    n_batches = 0
    autocast = torch.autocast(device_type=device.type, enabled=train_cfg.amp) if train_cfg.amp else nullcontext()

    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train), autocast:
            outputs = model(x, scalars)
            loss, parts = compute_loss(outputs, batch, train_cfg)
        if optimizer is not None:
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        for k, v in parts.items():
            totals[k] += v
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in totals.items()}


def _synthetic_sample(i: int, nz: int, data_cfg: DataConfig, seed: int) -> dict[str, Tensor]:
    rng = np.random.default_rng(seed + i)
    z = np.linspace(0.0, data_cfg.max_depth_cm, nz, dtype=np.float32)
    energies = np.asarray(data_cfg.energies_mev, dtype=np.float32)
    energy = float(energies[i % len(energies)] + rng.normal(0.0, 0.15))
    r0 = float(0.0022 * energy**1.77)
    r80 = float(np.clip(0.82 * r0 + rng.normal(0.0, 0.03), 1.0, data_cfg.max_depth_cm - data_cfg.dz_cm))

    peak = np.exp(-0.5 * ((z - r80) / 0.42) ** 2)
    entrance = 0.18 + 0.30 * np.clip(z / max(r80, 1e-6), 0.0, 1.0)
    distal_cut = 1.0 / (1.0 + np.exp(np.clip((z - r80) / 0.16, -60.0, 60.0)))
    dose = (entrance + 1.65 * peak) * distal_cut
    dose = dose / max(float(dose.max()), 1e-6)
    letd = 0.8 + 3.8 * np.clip(z / max(r80, 1e-6), 0.0, 1.25) ** 2
    letd = letd * distal_cut + 0.2 * (1.0 - distal_cut)

    density = np.ones_like(z)
    rsp = np.ones_like(z)
    i_value = np.full_like(z, 75.0)
    material_class = np.zeros_like(z)
    depth_over_r0 = z / max(r0, 1e-6)
    resid_energy = energy * np.clip(1.0 - depth_over_r0, 0.0, 1.0) ** 0.55
    csda_stopping = 1.0 + 6.0 / np.sqrt(np.maximum(resid_energy, 1.0))

    x = np.stack(
        [
            density,
            rsp,
            i_value,
            material_class,
            z,
            csda_stopping.astype(np.float32),
            dose.astype(np.float32),
            depth_over_r0.astype(np.float32),
            resid_energy.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    scalars = np.asarray([energy, 0.5, 0.2, float(Fidelity.SDE)], dtype=np.float32)
    return {
        "x": torch.from_numpy(x),
        "scalars": torch.from_numpy(scalars),
        "z": torch.from_numpy(z),
        "dose": torch.from_numpy(dose.astype(np.float32)),
        "letd": torch.from_numpy(letd.astype(np.float32)),
        "r80": torch.tensor(r80, dtype=torch.float32),
        "fidelity": torch.tensor(int(Fidelity.SDE), dtype=torch.long),
    }


def _write_metrics(out_dir: Path, metrics: dict[str, Any], history: list[dict[str, float | int]]) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if history:
        with (out_dir / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        np.savez(out_dir / "metrics.npz", **{k: np.asarray([row[k] for row in history]) for k in history[0]})


def _section(cls: type[Any], raw: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def _lazy_import(module_name: str, attr: str) -> type[nn.Module]:
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _param_count(model: nn.Module) -> int:
    if hasattr(model, "param_count"):
        return int(model.param_count())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
