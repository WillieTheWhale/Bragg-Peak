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
        "mamba1d": lambda: _lazy_import("braggtransporter.models.mamba1d", "Mamba1d"),
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

    # NaN-safe: a single sample with an undefined R80 (or non-finite target) must
    # not poison the whole batch loss. Mask non-finite entries out of each term.
    r80_finite = torch.isfinite(r80_target)
    r80_safe = torch.nan_to_num(r80_target, nan=0.0, posinf=0.0, neginf=0.0)

    weights = torch.ones_like(dose_target)
    proximal = (z >= (r80_safe[:, None] - 2.0)) & (z <= r80_safe[:, None]) & r80_finite[:, None]
    weights = torch.where(proximal, torch.full_like(weights, float(train_cfg.distal_edge_weight)), weights)
    dose_valid = torch.isfinite(dose_target)
    sq = torch.where(dose_valid, (dose_pred - dose_target).square(), torch.zeros_like(dose_target))
    dose_loss = (weights * sq).sum() / weights[dose_valid].sum().clamp_min(1.0)

    letd_loss = dose_loss.new_tensor(0.0)
    if "letd" in outputs and "letd" in batch:
        letd_target = batch["letd"].to(device=dose_pred.device, dtype=dose_pred.dtype)
        letd_valid = torch.isfinite(letd_target)
        if letd_valid.any():
            letd_loss = F.mse_loss(outputs["letd"][letd_valid], letd_target[letd_valid])

    r80_loss = dose_loss.new_tensor(0.0)
    if "r80" in outputs and r80_finite.any():
        r80_loss = F.smooth_l1_loss(outputs["r80"].view(-1)[r80_finite], r80_target[r80_finite])

    monotonic_range_loss = _monotonic_range_loss(outputs, batch, dose_loss)
    energy_budget_loss = _energy_budget_loss(dose_pred, batch)

    monotonic_term = dose_loss.new_tensor(0.0)
    if train_cfg.monotonic_range_weight > 0.0:
        monotonic_term = float(train_cfg.monotonic_range_weight) * monotonic_range_loss

    energy_budget_term = dose_loss.new_tensor(0.0)
    if train_cfg.constraint_weight > 0.0:
        energy_budget_term = float(train_cfg.constraint_weight) * energy_budget_loss

    total = dose_loss + letd_loss + r80_loss + monotonic_term + energy_budget_term
    return total, {
        "loss": float(total.detach().cpu()),
        "dose_loss": float(dose_loss.detach().cpu()),
        "letd_loss": float(letd_loss.detach().cpu()),
        "r80_loss": float(r80_loss.detach().cpu()),
        "monotonic_range_loss": float(monotonic_term.detach().cpu()),
        "energy_budget_loss": float(energy_budget_term.detach().cpu()),
    }


def _monotonic_range_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor], like: Tensor) -> Tensor:
    """Pairwise hinge loss for the soft physical prior dR80/dE >= 0."""

    if "r80" not in outputs:
        return like.new_tensor(0.0)
    r80 = outputs["r80"].view(-1)
    if "energy_mev" in batch:
        energy = batch["energy_mev"].to(device=r80.device, dtype=r80.dtype).view(-1)
    else:
        energy = batch["scalars"].to(device=r80.device, dtype=r80.dtype)[:, 0].view(-1)
    if r80.numel() < 2 or energy.numel() != r80.numel():
        return like.new_tensor(0.0)

    finite = torch.isfinite(r80) & torch.isfinite(energy)
    if finite.sum() < 2:
        return like.new_tensor(0.0)
    r80 = r80[finite]
    energy = energy[finite]
    higher_energy = energy[:, None] > energy[None, :]
    if not bool(higher_energy.any()):
        return like.new_tensor(0.0)
    violations = F.relu(r80[None, :] - r80[:, None])
    return violations[higher_energy].square().mean()


def _energy_budget_loss(dose_pred: Tensor, batch: dict[str, Tensor]) -> Tensor:
    """One-sided soft energy budget: predicted deposited energy should not exceed incident energy."""

    if "z" not in batch:
        return dose_pred.new_tensor(0.0)
    z = batch["z"].to(device=dose_pred.device, dtype=dose_pred.dtype)
    if z.ndim == 1:
        z = z.unsqueeze(0).expand_as(dose_pred)
    if z.shape != dose_pred.shape:
        return dose_pred.new_tensor(0.0)

    incident = _incident_energy_mev(batch, dose_pred.device, dose_pred.dtype)
    if incident.numel() != dose_pred.shape[0]:
        return dose_pred.new_tensor(0.0)

    dz = _cell_widths_cm(z)
    density = _physical_density_or_ones(batch["x"].to(device=dose_pred.device, dtype=dose_pred.dtype)[..., 0])
    deposit_mev = torch.sum(torch.nan_to_num(dose_pred, nan=0.0, posinf=0.0, neginf=0.0) * density * dz, dim=1)

    valid = torch.isfinite(deposit_mev) & torch.isfinite(incident) & (incident > 0.0)
    if not bool(valid.any()):
        return dose_pred.new_tensor(0.0)
    over_budget = F.relu(deposit_mev[valid] - incident[valid]) / incident[valid].clamp_min(torch.finfo(dose_pred.dtype).eps)
    return over_budget.square().mean()


def _cell_widths_cm(z: Tensor) -> Tensor:
    if z.shape[1] < 2:
        return torch.ones_like(z)
    diffs = torch.diff(z, dim=1).abs()
    first = diffs[:, :1]
    widths = torch.cat([first, diffs], dim=1)
    return torch.where(torch.isfinite(widths) & (widths > 0.0), widths, torch.ones_like(widths))


def _incident_energy_mev(batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype) -> Tensor:
    if "energy_mev" in batch:
        return batch["energy_mev"].to(device=device, dtype=dtype).view(-1)

    scalar_energy = batch["scalars"].to(device=device, dtype=dtype)[:, 0].view(-1)
    if bool((torch.isfinite(scalar_energy) & (scalar_energy > 20.0)).all()):
        return scalar_energy

    if "r80" in batch:
        r80_cm = batch["r80"].to(device=device, dtype=dtype).view(-1)
        finite_positive = torch.isfinite(r80_cm) & (r80_cm > 0.0)
        if bool(finite_positive.any()):
            estimated = torch.pow((r80_cm / (0.82 * 0.0022)).clamp_min(torch.finfo(dtype).eps), 1.0 / 1.77)
            return torch.where(finite_positive, estimated, scalar_energy)

    return scalar_energy


def _physical_density_or_ones(density: Tensor) -> Tensor:
    finite = torch.isfinite(density)
    if bool(finite.all() and (density > 0.0).all() and (density < 25.0).all()):
        return density
    return torch.ones_like(density)


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
    totals: dict[str, float] = {}
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
            totals[k] = totals.get(k, 0.0) + v
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
    # Do NOT force deterministic algorithms on Apple MPS: several backward kernels
    # (e.g. LayerNorm, gather-based interpolation) have no deterministic MPS
    # implementation and, with warn_only=True, silently emit NaN gradients instead
    # of erroring — which poisons training from epoch 1. Seeds above already give
    # reproducible runs; deterministic algorithms are only enabled off-MPS.
    if not torch.backends.mps.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)


def _param_count(model: nn.Module) -> int:
    if hasattr(model, "param_count"):
        return int(model.param_count())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
