"""Stage-0 masked transport pretraining for BraggTransporter v3.1.

Research software only. Units follow the fixed BraggTransporter contract:
depth in cm, energy in MeV, dose in MeV/g per voxel. This module keeps the
supervised v0 model read-only by reusing its encoder path and saving only
encoder-compatible weights for later Stage-1 initialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from braggtransporter.config import ModelConfig, TrainConfig, get_device
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR
from braggtransporter.train import RunConfig, load_config, make_training_loaders


ENCODER_PREFIXES = (
    "input_norm.",
    "input_proj.",
    "pos_embedding",
    "scalar_token.",
    "encoder.",
    "encoder_norm.",
)


@dataclass
class PretrainConfig:
    """Hyperparameters for Stage-0 self-supervision."""

    mask_ratio: float = 0.30
    mask_span_min: int = 2
    mask_span_max: int = 10
    mask_dose: bool = True
    next_slab_offset: int = 4
    x_recon_weight: float = 1.0
    dose_recon_weight: float = 1.0
    next_slab_weight: float = 0.25


class MaskedTransportPretrainer(nn.Module):
    """Masked autoencoding plus next-slab prediction over v0 encoder latents.

    The wrapped :class:`BraggTransporterV0` supplies the frozen Stage-1-compatible
    encoder topology. Stage-0-only modules are the dose-context projection, mask
    tokens, and reconstruction heads; they are intentionally excluded from the
    saved encoder checkpoint.
    """

    def __init__(self, model_cfg: ModelConfig | None = None, pretrain_cfg: PretrainConfig | None = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg if model_cfg is not None else ModelConfig()
        self.pretrain_cfg = pretrain_cfg if pretrain_cfg is not None else PretrainConfig()
        self.backbone = BraggTransporterV0(self.model_cfg)
        d_model = self.backbone.d_model

        self.x_mask_token = nn.Parameter(torch.zeros(C_IN_PERDEPTH))
        self.dose_mask_token = nn.Parameter(torch.zeros(1))
        self.dose_context = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.x_recon_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, C_IN_PERDEPTH))
        self.dose_recon_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.next_slab_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def forward(
        self,
        x: Tensor,
        scalars: Tensor,
        dose: Tensor,
        *,
        generator: torch.Generator | None = None,
        masks: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if x.ndim != 3 or x.shape[-1] != C_IN_PERDEPTH:
            raise ValueError(f"x must have shape (B,Nz,{C_IN_PERDEPTH}), got {tuple(x.shape)}")
        if scalars.ndim != 2 or scalars.shape[-1] != C_SCALAR:
            raise ValueError(f"scalars must have shape (B,{C_SCALAR}), got {tuple(scalars.shape)}")
        if dose.ndim != 2 or dose.shape != x.shape[:2]:
            raise ValueError(f"dose must have shape {tuple(x.shape[:2])}, got {tuple(dose.shape)}")

        mask_payload = masks if masks is not None else self.make_masks(x.shape[0], x.shape[1], x.device, generator)
        x_mask = mask_payload["x_mask"].to(device=x.device, dtype=torch.bool)
        dose_mask = mask_payload["dose_mask"].to(device=x.device, dtype=torch.bool)
        x_masked = torch.where(x_mask.unsqueeze(-1), self.x_mask_token.to(dtype=x.dtype, device=x.device), x)
        dose_masked = torch.where(dose_mask, self.dose_mask_token.to(dtype=dose.dtype, device=dose.device), dose)
        depth_latent = self.encode_masked(x_masked, scalars, dose_masked)
        return {
            "x_recon": self.x_recon_head(depth_latent),
            "dose_recon": self.dose_recon_head(depth_latent).squeeze(-1),
            "next_slab": self.next_slab_head(depth_latent),
            "x_mask": x_mask,
            "dose_mask": dose_mask,
        }

    def encode_masked(self, x_masked: Tensor, scalars: Tensor, dose_masked: Tensor) -> Tensor:
        batch, nz, _ = x_masked.shape
        depth_tokens = self.backbone.input_proj(self.backbone.input_norm(x_masked))
        depth_tokens = depth_tokens + self.backbone._positional_encoding(nz, x_masked.device, x_masked.dtype)
        depth_tokens = depth_tokens + self.dose_context(dose_masked.unsqueeze(-1))

        scale = self.backbone.scalar_scale.to(device=scalars.device, dtype=scalars.dtype)
        scalar_token = self.backbone.scalar_token(scalars / scale)
        tokens = torch.cat([scalar_token.unsqueeze(1), depth_tokens], dim=1)
        encoded = self.backbone.encoder_norm(self.backbone.encoder(tokens))
        return encoded[:, 1:]

    def make_masks(
        self,
        batch: int,
        nz: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        x_mask = mask_depth_spans(
            batch,
            nz,
            self.pretrain_cfg.mask_ratio,
            self.pretrain_cfg.mask_span_min,
            self.pretrain_cfg.mask_span_max,
            device,
            generator,
        )
        if self.pretrain_cfg.mask_dose:
            dose_mask = mask_depth_spans(
                batch,
                nz,
                self.pretrain_cfg.mask_ratio,
                self.pretrain_cfg.mask_span_min,
                self.pretrain_cfg.mask_span_max,
                device,
                generator,
            )
        else:
            dose_mask = torch.zeros(batch, nz, dtype=torch.bool, device=device)
        return {"x_mask": x_mask, "dose_mask": dose_mask}

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        *,
        generator: torch.Generator | None = None,
        masks: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        x = batch["x"].to(dtype=torch.float32, device=next(self.parameters()).device)
        scalars = batch["scalars"].to(dtype=torch.float32, device=x.device)
        dose = batch["dose"].to(dtype=torch.float32, device=x.device)
        out = self(x, scalars, dose, generator=generator, masks=masks)
        cfg = self.pretrain_cfg

        x_valid = out["x_mask"].unsqueeze(-1) & torch.isfinite(x)
        if x_valid.any():
            x_loss = F.mse_loss(out["x_recon"][x_valid], x[x_valid])
        else:
            x_loss = x.new_tensor(0.0)

        dose_valid = out["dose_mask"] & torch.isfinite(dose)
        if dose_valid.any():
            dose_loss = F.mse_loss(out["dose_recon"][dose_valid], dose[dose_valid])
        else:
            dose_loss = x.new_tensor(0.0)

        k = max(1, int(cfg.next_slab_offset))
        if x.shape[1] > k:
            pred = out["next_slab"][:, :-k]
            target = torch.stack([dose[:, k:], x[:, k:, 8]], dim=-1)
            valid = torch.isfinite(target)
            next_loss = F.mse_loss(pred[valid], target[valid]) if valid.any() else x.new_tensor(0.0)
        else:
            next_loss = x.new_tensor(0.0)

        total = cfg.x_recon_weight * x_loss + cfg.dose_recon_weight * dose_loss + cfg.next_slab_weight * next_loss
        parts = {
            "loss": float(total.detach().cpu()),
            "x_recon_loss": float(x_loss.detach().cpu()),
            "dose_recon_loss": float(dose_loss.detach().cpu()),
            "next_slab_loss": float(next_loss.detach().cpu()),
        }
        return total, parts

    def encoder_state_dict(self) -> dict[str, Tensor]:
        state = self.backbone.state_dict()
        return {k: v.detach().cpu() for k, v in state.items() if _is_encoder_key(k)}

    def save_encoder(self, path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state_dict": self.encoder_state_dict(),
                "model_config": asdict(self.model_cfg),
                "pretrain_config": asdict(self.pretrain_cfg),
                "meta": meta or {},
            },
            path,
        )


def mask_depth_spans(
    batch: int,
    nz: int,
    mask_ratio: float,
    span_min: int,
    span_max: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return a boolean ``(batch,nz)`` mask made from random contiguous spans."""

    if batch <= 0 or nz <= 0:
        raise ValueError("batch and nz must be positive.")
    ratio = float(np.clip(mask_ratio, 0.0, 1.0))
    target = max(1, int(round(ratio * nz))) if ratio > 0.0 else 0
    span_min = max(1, int(span_min))
    span_max = max(span_min, int(span_max))
    mask = torch.zeros(batch, nz, dtype=torch.bool)
    for b in range(batch):
        masked = 0
        attempts = 0
        while masked < target and attempts < 4 * nz:
            span = int(torch.randint(span_min, span_max + 1, (1,), generator=generator).item())
            start_hi = max(1, nz - span + 1)
            start = int(torch.randint(0, start_hi, (1,), generator=generator).item())
            before = int(mask[b].sum().item())
            mask[b, start : min(nz, start + span)] = True
            masked += int(mask[b].sum().item()) - before
            attempts += 1
        if target > 0 and not bool(mask[b].any()):
            mask[b, int(torch.randint(0, nz, (1,), generator=generator).item())] = True
    return mask.to(device=device)


def load_pretrained_encoder(model: BraggTransporterV0, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load matching Stage-0 encoder tensors into a supervised v0 model."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("encoder_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain an encoder state dict.")
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    missing = sorted(k for k in state if k not in compatible)
    current.update(compatible)
    model.load_state_dict(current)
    return {"loaded": sorted(compatible), "skipped": missing, "checkpoint": checkpoint}


def pretrain(
    cfg: RunConfig,
    *,
    epochs: int | None = None,
    device_override: str | None = None,
    fast: bool = False,
    out_dir: str | Path = "experiments/bt/pretrain",
    pretrain_cfg: PretrainConfig | None = None,
) -> dict[str, Any]:
    if epochs is not None:
        cfg.train.epochs = int(epochs)
    if fast:
        cfg.train.epochs = min(cfg.train.epochs, 2)
        cfg.train.batch_size = min(cfg.train.batch_size, 8)
        cfg.model.d_model = min(cfg.model.d_model, 32)
        cfg.model.n_layers = min(cfg.model.n_layers, 1)
        cfg.model.n_heads = min(cfg.model.n_heads, 4)
        cfg.model.d_ff = min(cfg.model.d_ff, 64)
        cfg.model.extra["max_positions"] = min(int(cfg.model.extra.get("max_positions", 128)), 128)
        if device_override is None:
            device_override = "cpu"
    if device_override:
        cfg.train.device = device_override

    _seed_everything(cfg.train.seed)
    device = get_device(cfg.train.device)
    model = MaskedTransportPretrainer(cfg.model, pretrain_cfg).to(device)
    train_loader, val_loader, _ = make_training_loaders(cfg, fast=fast)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total_steps = max(1, cfg.train.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.train.seed) + 137)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    autocast = torch.autocast(device_type=device.type, enabled=cfg.train.amp) if cfg.train.amp else nullcontext()

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, scheduler, cfg.train, generator, autocast)
        val_metrics = _run_epoch(model, val_loader, None, None, cfg.train, generator, autocast)
        row: dict[str, float | int] = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if val_metrics["loss"] <= best_val:
            best_val = val_metrics["loss"]
            model.save_encoder(
                out_path / "encoder.pt",
                meta={
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "device": str(device),
                    "seed": cfg.train.seed,
                    "units": {"depth": "cm", "energy": "MeV", "dose": "MeV/g per voxel"},
                    "source": "Stage-0 masked transport pretraining",
                },
            )
        print(f"pretrain epoch {epoch:03d} train_loss={train_metrics['loss']:.6g} val_loss={val_metrics['loss']:.6g}")

    metrics = {
        "best_val_loss": best_val,
        "epochs": cfg.train.epochs,
        "device": str(device),
        "seed": cfg.train.seed,
        "encoder_path": str(out_path / "encoder.pt"),
        "history": history,
        "units": {"depth": "cm", "energy": "MeV", "dose": "MeV/g per voxel"},
    }
    _write_artifacts(out_path, metrics, history)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out-dir", default="experiments/bt/pretrain")
    parser.add_argument("--mask-ratio", type=float, default=PretrainConfig.mask_ratio)
    parser.add_argument("--mask-span-min", type=int, default=PretrainConfig.mask_span_min)
    parser.add_argument("--mask-span-max", type=int, default=PretrainConfig.mask_span_max)
    parser.add_argument("--no-dose-mask", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    p_cfg = PretrainConfig(
        mask_ratio=args.mask_ratio,
        mask_span_min=args.mask_span_min,
        mask_span_max=args.mask_span_max,
        mask_dose=not args.no_dose_mask,
    )
    metrics = pretrain(cfg, epochs=args.epochs, device_override=args.device, fast=args.fast, out_dir=args.out_dir, pretrain_cfg=p_cfg)
    print(json.dumps({"best_val_loss": metrics["best_val_loss"], "encoder_path": metrics["encoder_path"], "device": metrics["device"]}, indent=2))


def _run_epoch(
    model: MaskedTransportPretrainer,
    loader: Any,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_cfg: TrainConfig,
    generator: torch.Generator,
    autocast: Any,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "x_recon_loss": 0.0, "dose_recon_loss": 0.0, "next_slab_loss": 0.0}
    n_batches = 0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train), autocast:
            loss, parts = model.compute_loss(batch, generator=generator)
        if optimizer is not None:
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        for key, value in parts.items():
            totals[key] += value
        n_batches += 1
    return {key: value / max(1, n_batches) for key, value in totals.items()}


def _write_artifacts(out_dir: Path, metrics: dict[str, Any], history: list[dict[str, float | int]]) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if history:
        with (out_dir / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        np.savez(out_dir / "metrics.npz", **{k: np.asarray([row[k] for row in history]) for k in history[0]})


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Do not call torch.use_deterministic_algorithms here: MPS LayerNorm and
    # transformer backward paths are not stable under that setting.


def _is_encoder_key(key: str) -> bool:
    return any(key == prefix or key.startswith(prefix) for prefix in ENCODER_PREFIXES)


if __name__ == "__main__":
    main()
