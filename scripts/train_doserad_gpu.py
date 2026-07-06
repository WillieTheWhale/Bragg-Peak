#!/usr/bin/env python
"""GPU-scale DoseRAD2026 Bragg3D training on normalized beamlet dose."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from braggtransporter.data.doserad import (
    depth_profile,
    download_patients,
    gamma_index_3d,
    make_doserad_loaders,
    r80_mm_from_profile,
    rmse_pct_3d,
)
from braggtransporter.models.bragg3d import Bragg3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/doserad2026", help="DoseRAD2026 root directory.")
    parser.add_argument("--patients", nargs="+", default=["1ABB006"], help="Patient IDs or comma-separated lists.")
    parser.add_argument("--max-beamlets", type=int, default=None, help="Maximum beamlets per patient.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--depth-size", type=int, default=64)
    parser.add_argument("--lateral-size", type=int, default=24)
    parser.add_argument("--lateral-extent-mm", type=float, default=96.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default="experiments/doserad_gpu")
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distal-weight", type=float, default=3.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--gcs", default=None, help="Optional gs:// bucket/prefix for checkpoints and metrics.")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume from latest or a checkpoint path.")
    parser.add_argument("--download", action="store_true", help="Download selected patients before training.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Small CPU-friendly smoke run; uses synthetic data if real data is absent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.epochs = min(args.epochs, 2)
        args.depth_size = min(args.depth_size, 12)
        args.lateral_size = min(args.lateral_size, 8)
        args.batch_size = min(args.batch_size, 2)
        args.max_beamlets = min(args.max_beamlets or 8, 8)
        args.num_workers = 0
    metrics = train(args)
    print(json.dumps({"final": json_clean(metrics)}, sort_keys=True), flush=True)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_seed(int(args.seed))
    device = select_device(str(args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        summary = download_patients(args.patients, args.root, args.max_beamlets)
        print(f"download_summary={json.dumps(summary, sort_keys=True)}", flush=True)

    train_loader, val_loader, source = build_loaders(args)
    model = Bragg3D(
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        d_ff=int(args.d_ff),
        max_depth=max(128, int(args.depth_size)),
    ).to(device)
    initialize_lazy_modules(model, train_loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, int(args.epochs) * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    global_step = 0
    if args.resume:
        resume_path = resolve_resume_path(args.resume, out_dir, args.gcs)
        if resume_path is not None and resume_path.exists():
            ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            restore_rng_state(ckpt.get("rng_state"))
            print(f"resumed checkpoint={resume_path} epoch={start_epoch} step={global_step}", flush=True)

    print(
        "DoseRAD GPU train: "
        f"source={source} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"device={device} amp={amp_enabled} params={model.param_count():,}",
        flush=True,
    )

    latest_metrics: dict[str, float] = {}
    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            loss_value = train_step(
                model,
                batch,
                optimizer,
                device,
                scaler=scaler,
                scheduler=scheduler,
                amp_enabled=amp_enabled,
                grad_clip=float(args.grad_clip),
                distal_weight=float(args.distal_weight),
            )
            global_step += 1
            losses.append(loss_value)
            if int(args.checkpoint_every_steps) > 0 and global_step % int(args.checkpoint_every_steps) == 0:
                latest_metrics = evaluate(model, val_loader, device)
                save_and_upload(out_dir, args, model, optimizer, scheduler, scaler, epoch, global_step, latest_metrics)

        train_loss = float(np.mean(losses)) if losses else float("nan")
        latest_metrics = evaluate(model, val_loader, device)
        latest_metrics["train_loss"] = train_loss
        latest_metrics["epoch"] = float(epoch)
        latest_metrics["global_step"] = float(global_step)
        append_metrics(out_dir / "metrics.jsonl", latest_metrics, args)
        (out_dir / "metrics_latest.json").write_text(json.dumps(json_clean(latest_metrics), indent=2), encoding="utf-8")
        save_and_upload(out_dir, args, model, optimizer, scheduler, scaler, epoch, global_step, latest_metrics)
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6g} "
            f"val_loss={latest_metrics['val_loss']:.6g} "
            f"gamma3d_3pct_3mm={latest_metrics['gamma3d_3pct_3mm']:.2f}% "
            f"rmse={latest_metrics['rmse_pct']:.2f}%",
            flush=True,
        )
    return latest_metrics


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, str]:
    try:
        loaders = make_doserad_loaders(
            args.root,
            args.patients,
            max_beamlets_per_patient=args.max_beamlets,
            val_frac=float(args.val_frac),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            depth_size=int(args.depth_size),
            lateral_size=int(args.lateral_size),
            lateral_extent_mm=float(args.lateral_extent_mm),
            rebuild_cache=bool(args.rebuild_cache),
            num_workers=int(args.num_workers),
        )
        return loaders[0], loaders[1], "doserad"
    except Exception as exc:
        if not args.fast:
            raise
        print(f"real DoseRAD unavailable for --fast smoke ({exc}); using synthetic normalized beamlets", flush=True)
        ds = SyntheticDoseRADDataset(
            n=max(4, int(args.max_beamlets or 8)),
            depth=int(args.depth_size),
            lateral=int(args.lateral_size),
            seed=int(args.seed),
        )
        gen = torch.Generator().manual_seed(int(args.seed))
        val_len = max(1, int(round(len(ds) * float(args.val_frac))))
        train_len = len(ds) - val_len
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_len], generator=gen)
        train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, generator=gen)
        val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)
        return train_loader, val_loader, "synthetic"


def train_step(
    model: Bragg3D,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    amp_enabled: bool = False,
    grad_clip: float = 1.0,
    distal_weight: float = 3.0,
) -> float:
    x = batch["x"].to(device=device, dtype=torch.float32)
    target = batch["dose"].to(device=device, dtype=torch.float32)
    scalars = batch["scalars"].to(device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=amp_enabled):
        pred = model(x, scalars)["dose"]
        loss = doserad_relative_loss(pred, target, distal_weight=distal_weight)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return float(loss.detach().cpu())


def doserad_relative_loss(pred: torch.Tensor, target: torch.Tensor, *, distal_weight: float = 3.0) -> torch.Tensor:
    """Relative loss for unit-maximum normalized beamlet dose."""

    target = target.clamp_min(0.0)
    sq = (pred - target).square()
    voxel_weight = torch.where(target >= 0.02, torch.ones_like(target), torch.full_like(target, 0.1))
    distal = distal_slice_weights(target).to(device=target.device, dtype=target.dtype)
    return (sq * voxel_weight * (1.0 + float(distal_weight) * distal)).mean()


def distal_slice_weights(target: torch.Tensor) -> torch.Tensor:
    profile = target.detach().sum(dim=(2, 3))
    weights = torch.zeros_like(profile)
    depth_idx = torch.arange(profile.shape[1], device=profile.device)
    for i in range(profile.shape[0]):
        peak = torch.max(profile[i])
        if float(peak) <= 0.0:
            continue
        peak_idx = int(torch.argmax(profile[i]).item())
        rel = profile[i] / peak.clamp_min(torch.finfo(profile.dtype).eps)
        distal_band = (depth_idx >= peak_idx) & (rel <= 0.95) & (rel >= 0.02)
        if not bool(torch.any(distal_band)):
            distal_band = depth_idx >= peak_idx
        weights[i, distal_band] = 1.0
    return weights[:, :, None, None]


@torch.no_grad()
def evaluate(model: Bragg3D, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    gamma_values: list[float] = []
    rmse_values: list[float] = []
    r80_errors: list[float] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        target_t = batch["dose"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        pred_t = model(x, scalars)["dose"]
        losses.append(float(doserad_relative_loss(pred_t, target_t).cpu()))
        pred = pred_t.detach().cpu().numpy()
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
        "val_loss": nanmean(losses),
        "gamma3d_3pct_3mm": nanmean(gamma_values),
        "rmse_pct": nanmean(rmse_values),
        "r80_error_mm": nanmean(r80_errors),
        "n_val_beamlets": float(len(gamma_values)),
    }


def save_and_upload(
    out_dir: Path,
    args: argparse.Namespace,
    model: Bragg3D,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> Path:
    ckpt_path = out_dir / f"step_{global_step:08d}.pt"
    save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, global_step, args, metrics)
    latest = out_dir / "latest.pt"
    shutil.copy2(ckpt_path, latest)
    upload_to_gcs([ckpt_path, latest, out_dir / "metrics.jsonl", out_dir / "metrics_latest.json"], args.gcs)
    return ckpt_path


def save_checkpoint(
    path: Path,
    model: Bragg3D,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    args: argparse.Namespace | dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "args": vars(args) if isinstance(args, argparse.Namespace) else dict(args),
            "metrics": metrics or {},
            "rng_state": capture_rng_state(),
            "units": {"spacing": "mm", "dose": "per-beamlet unit-max normalized relative dose"},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: Bragg3D,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def resolve_resume_path(resume: str, out_dir: Path, gcs: str | None) -> Path | None:
    if resume != "latest":
        return Path(resume)
    latest = out_dir / "latest.pt"
    if latest.exists() or not gcs:
        return latest
    latest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil", "cp", f"{gcs.rstrip('/')}/latest.pt", str(latest)]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode == 0 and latest.exists():
            return latest
    except OSError:
        pass
    return latest


def upload_to_gcs(paths: list[Path], gcs: str | None) -> None:
    if not gcs:
        return
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return
    cmd = ["gsutil", "cp", *existing, f"{gcs.rstrip('/')}/"]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            print(f"warning: gsutil upload failed: {proc.stderr[-500:]}", flush=True)
    except OSError as exc:
        print(f"warning: gsutil upload unavailable: {exc}", flush=True)


def append_metrics(path: Path, metrics: dict[str, float], args: argparse.Namespace) -> None:
    row = {
        "metrics": json_clean(metrics),
        "patients": list(args.patients),
        "max_beamlets_per_patient": args.max_beamlets,
        "seed": args.seed,
        "units": {"spacing": "mm", "dose": "per-beamlet unit-max normalized relative dose"},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def initialize_lazy_modules(model: Bragg3D, loader: DataLoader, device: torch.device) -> None:
    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        model(
            batch["x"].to(device=device, dtype=torch.float32),
            batch["scalars"].to(device=device, dtype=torch.float32),
        )


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    return value


class SyntheticDoseRADDataset(Dataset[dict[str, torch.Tensor | dict[str, Any]]]):
    """Small deterministic normalized beamlet set for CPU smoke tests."""

    def __init__(self, *, n: int = 8, depth: int = 12, lateral: int = 8, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(int(seed))
        z = torch.linspace(0.0, 1.0, int(depth)).view(depth, 1, 1)
        y = torch.linspace(-1.0, 1.0, int(lateral)).view(1, lateral, 1)
        x = torch.linspace(-1.0, 1.0, int(lateral)).view(1, 1, lateral)
        self.items: list[dict[str, torch.Tensor | dict[str, Any]]] = []
        for i in range(int(n)):
            center = 0.35 + 0.45 * (i / max(int(n) - 1, 1))
            sigma_z = 0.08 + 0.01 * (i % 3)
            sigma_lat = 0.28 + 0.03 * (i % 2)
            dose = torch.exp(-0.5 * (((z - center) / sigma_z) ** 2 + (y / sigma_lat) ** 2 + (x / sigma_lat) ** 2))
            dose = dose / dose.max().clamp_min(torch.finfo(torch.float32).eps)
            hu = -0.1 + 0.15 * z.expand_as(dose) + 0.01 * torch.randn(dose.shape, generator=gen)
            density = (1.0 + hu).clamp(0.0, 2.5)
            rsp = density.clone()
            self.items.append(
                {
                    "x": torch.stack([hu, density, rsp], dim=0).float(),
                    "dose": dose.float(),
                    "dose_scale": torch.tensor((i + 1) * 1e-3, dtype=torch.float32),
                    "scalars": torch.tensor([100.0 + 2.0 * i, float(i % 8), 0.0, 1.0], dtype=torch.float32),
                    "spacing_mm": torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
                    "meta": {"patient": "SYNTH", "beam_idx": 0, "ray_idx": 0, "layer_idx": i},
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, Any]]:
        return self.items[idx]


if __name__ == "__main__":
    main()
