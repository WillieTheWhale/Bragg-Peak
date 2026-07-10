#!/usr/bin/env python
"""GPU-scale DoseRAD2026 Bragg3D training on normalized beamlet dose."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from braggtransporter.data.doserad import (
    depth_profile,
    download_patients,
    gamma_index_3d,
    gamma_index_3d_fast,
    make_doserad_loaders,
    r80_mm_from_profile,
    rmse_pct_3d,
)

# (metric_key, dose_pct, dta_mm, low_dose_threshold)
SUBSAMPLE_CRITERIA: list[tuple[str, float, float, float]] = [
    ("gamma3d_3pct_3mm", 3.0, 3.0, 0.1),
]
# Full/final evals also report the papers' beamlet criterion (DoTA/ADoTA:
# 1%/3mm with a 0.1% low-dose cutoff) and the plan-level 2%/2mm/10%.
FULL_CRITERIA: list[tuple[str, float, float, float]] = [
    ("gamma3d_3pct_3mm", 3.0, 3.0, 0.1),
    ("gamma3d_1pct_3mm_dota", 1.0, 3.0, 0.001),
    ("gamma3d_2pct_2mm", 2.0, 2.0, 0.1),
]
from braggtransporter.models.bragg3d import Bragg3D
from braggtransporter.models.dota3d import DoTA3D
from braggtransporter.models.dota3d_spatial import DoTA3DSpatial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/doserad2026", help="DoseRAD2026 root directory.")
    parser.add_argument("--patients", nargs="+", default=["1ABB006"], help="Patient IDs or comma-separated lists.")
    parser.add_argument("--max-beamlets", type=int, default=None, help="Maximum beamlets per patient.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=["cosine", "warmrestart", "dota"], default="cosine")
    parser.add_argument("--restart-epochs", type=int, default=28)
    parser.add_argument("--lr-halve-epochs", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--save-best-by", choices=["gamma", "val_loss"], default="gamma")
    parser.add_argument("--eval-subsample", type=int, default=96, help="Fixed held-out beamlets for per-epoch progress gamma.")
    parser.add_argument("--full-eval-every", type=int, default=0, help="Run full held-out gamma every N epochs; 0 means only at end.")
    parser.add_argument("--model", default="bragg3d", choices=["bragg3d", "dota3d", "dota3d_spatial"])
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.0,
        help="Fraction of patients held out as an UNTOUCHED test cohort (patient split only); "
        "evaluated once on the final best checkpoint, never used for selection.",
    )
    parser.add_argument("--depth-size", type=int, default=64)
    parser.add_argument("--depth-extent-mm", type=float, default=400.0)
    parser.add_argument("--lateral-size", type=int, default=24)
    parser.add_argument("--lateral-extent-mm", type=float, default=96.0)
    parser.add_argument("--split-by", choices=["patient", "beamlet"], default="patient")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers; default is min(8, os.cpu_count()) for training and 0 for --fast smoke.",
    )
    parser.add_argument("--out-dir", default="experiments/doserad_gpu")
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distal-weight", type=float, default=3.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=4, help="Patch size for spatial DoTA models.")
    parser.add_argument("--gcs", default=None, help="Optional gs:// bucket/prefix for checkpoints and metrics.")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume from latest or a checkpoint path.")
    parser.add_argument("--download", action="store_true", help="Download selected patients before training.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Small CPU-friendly smoke run; uses synthetic data if real data is absent.")
    parser.add_argument(
        "--allow-coarse-axes",
        nargs="+",
        choices=["depth", "lateral"],
        default=[],
        help="Axes permitted to have grid spacing above the 3mm gamma DTA (hard error otherwise, outside --fast).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        if int(args.epochs) == 20:
            args.epochs = 2
        args.depth_size = min(args.depth_size, 12)
        args.lateral_size = min(args.lateral_size, 8)
        args.batch_size = min(args.batch_size, 2)
        args.max_beamlets = min(args.max_beamlets or 8, 8)
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

    report_grid_resolution(args)
    train_loader, val_loader, test_loader, source = build_loaders(args)
    val_sub_loader = make_eval_subsample_loader(val_loader, args)
    model = build_model(args).to(device)
    initialize_lazy_modules(model, train_loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = build_lr_scheduler(args, optimizer, steps_per_epoch=max(1, len(train_loader)))
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
        f"model={args.model} source={source} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"val_sub={len(val_sub_loader.dataset)} workers={train_loader.num_workers} "
        f"device={device} amp={amp_enabled} params={model.param_count():,}",
        flush=True,
    )

    latest_metrics: dict[str, float] = {}
    best_metrics = load_existing_best_metrics(out_dir, str(args.save_best_by))
    best_gamma_metrics = dict(best_metrics) if str(args.save_best_by) == "gamma" and best_metrics else None
    for epoch in range(start_epoch, int(args.epochs) + 1):
        step_lr_epoch_start(scheduler, str(args.lr_schedule), epoch)
        epoch_lr = current_lr(optimizer)
        model.train()
        losses: list[float] = []
        train_t0 = time.perf_counter()
        for batch in train_loader:
            loss_value = train_step(
                model,
                batch,
                optimizer,
                device,
                scaler=scaler,
                scheduler=scheduler if str(args.lr_schedule) == "cosine" else None,
                amp_enabled=amp_enabled,
                grad_clip=float(args.grad_clip),
                distal_weight=float(args.distal_weight),
            )
            global_step += 1
            losses.append(loss_value)
            if int(args.checkpoint_every_steps) > 0 and global_step % int(args.checkpoint_every_steps) == 0:
                latest_metrics = evaluate(model, val_sub_loader, device)
                save_and_upload(out_dir, args, model, optimizer, scheduler, scaler, epoch, global_step, latest_metrics)

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_seconds = time.perf_counter() - train_t0
        eval_t0 = time.perf_counter()
        latest_metrics = evaluate(model, val_sub_loader, device)
        sub_eval_seconds = time.perf_counter() - eval_t0
        latest_metrics["train_loss"] = train_loss
        latest_metrics["epoch"] = float(epoch)
        latest_metrics["global_step"] = float(global_step)
        latest_metrics["lr"] = epoch_lr
        latest_metrics["eval_subsample"] = float(len(val_sub_loader.dataset))
        latest_metrics["train_seconds"] = float(train_seconds)
        latest_metrics["eval_sub_seconds"] = float(sub_eval_seconds)
        full_eval_seconds = float("nan")
        if should_run_full_eval(epoch, args):
            full_t0 = time.perf_counter()
            full_metrics = evaluate(model, val_loader, device, criteria=FULL_CRITERIA)
            full_eval_seconds = time.perf_counter() - full_t0
            add_full_metric_aliases(latest_metrics, full_metrics)
            latest_metrics["eval_full_seconds"] = float(full_eval_seconds)
        append_metrics(out_dir / "metrics.jsonl", latest_metrics, args)
        (out_dir / "metrics_latest.json").write_text(json.dumps(json_clean(latest_metrics), indent=2), encoding="utf-8")
        save_and_upload(out_dir, args, model, optimizer, scheduler, scaler, epoch, global_step, latest_metrics)
        if is_better_best(latest_metrics, best_gamma_metrics, "gamma"):
            best_gamma_metrics = dict(latest_metrics)
        best_metrics, best_updated = maybe_save_best_checkpoint(
            out_dir,
            args,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            latest_metrics,
            best_metrics,
        )
        best_epoch = int(best_gamma_metrics.get("epoch", epoch)) if best_gamma_metrics else epoch
        best_gamma = (
            float(best_gamma_metrics.get("gamma3d_3pct_3mm", float("nan"))) if best_gamma_metrics else float("nan")
        )
        print(
            f"epoch {epoch:03d}: lr={epoch_lr:.6g} train_loss={train_loss:.6g} "
            f"val_loss={latest_metrics['val_loss']:.6g} "
            f"gamma(sub)={latest_metrics['gamma3d_3pct_3mm']:.2f}% "
            f"{format_full_gamma(latest_metrics)}"
            f"rmse={latest_metrics['rmse_pct']:.2f}% "
            f"best_gamma_so_far={best_gamma:.2f}%@epoch{best_epoch}"
            f" timing=train:{train_seconds:.2f}s eval_sub:{sub_eval_seconds:.2f}s"
            f"{format_full_eval_seconds(full_eval_seconds)}"
            f"{' best_checkpoint_updated' if best_updated else ''}",
            flush=True,
        )
    best_full_metrics = evaluate_best_checkpoint_on_full_val(out_dir, model, val_loader, device)
    if best_full_metrics:
        latest_metrics.update(best_full_metrics)
        (out_dir / "metrics_best_full.json").write_text(json.dumps(json_clean(best_full_metrics), indent=2), encoding="utf-8")
        print(
            f"best checkpoint full eval: gamma(full)={best_full_metrics['best_gamma3d_3pct_3mm_full']:.2f}% "
            f"val_loss(full)={best_full_metrics['best_val_loss_full']:.6g} "
            f"rmse(full)={best_full_metrics['best_rmse_pct_full']:.2f}%",
            flush=True,
        )
        if test_loader is not None:
            test_metrics = evaluate(model, test_loader, device, criteria=FULL_CRITERIA)
            headline = {f"test_{k}": float(v) for k, v in test_metrics.items()}
            latest_metrics.update(headline)
            (out_dir / "metrics_test.json").write_text(json.dumps(json_clean(headline), indent=2), encoding="utf-8")
            upload_to_gcs([out_dir / "metrics_test.json", out_dir / "metrics_best_full.json"], getattr(args, "gcs", None))
            print(
                "HEADLINE (untouched test patients, best checkpoint): "
                f"gamma 1%/3mm/0.1%cut={test_metrics['gamma3d_1pct_3mm_dota']:.2f}% "
                f"gamma 2%/2mm/10%cut={test_metrics['gamma3d_2pct_2mm']:.2f}% "
                f"gamma 3%/3mm/10%cut={test_metrics['gamma3d_3pct_3mm']:.2f}% "
                f"rmse={test_metrics['rmse_pct']:.2f}% r80err={test_metrics['r80_error_mm']:.2f}mm "
                f"n={int(test_metrics['n_val_beamlets'])}",
                flush=True,
            )
    return latest_metrics


class DoTALRScheduler:
    """DoTA epoch schedule: halve every N epochs, hard restart every M epochs."""

    def __init__(self, optimizer: torch.optim.Optimizer, *, lr_halve_epochs: int = 4, restart_epochs: int = 28) -> None:
        if int(lr_halve_epochs) <= 0:
            raise ValueError("lr_halve_epochs must be positive.")
        if int(restart_epochs) <= 0:
            raise ValueError("restart_epochs must be positive.")
        self.optimizer = optimizer
        self.lr_halve_epochs = int(lr_halve_epochs)
        self.restart_epochs = int(restart_epochs)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_epoch = 0

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = int(epoch)
        factor = dota_lr_factor(self.last_epoch, self.lr_halve_epochs, self.restart_epochs)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = float(base_lr) * factor

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lrs": list(self.base_lrs),
            "last_epoch": self.last_epoch,
            "lr_halve_epochs": self.lr_halve_epochs,
            "restart_epochs": self.restart_epochs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_lrs = [float(v) for v in state_dict.get("base_lrs", self.base_lrs)]
        self.last_epoch = int(state_dict.get("last_epoch", self.last_epoch))
        self.lr_halve_epochs = int(state_dict.get("lr_halve_epochs", self.lr_halve_epochs))
        self.restart_epochs = int(state_dict.get("restart_epochs", self.restart_epochs))
        self.step(self.last_epoch)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]


def dota_lr_factor(epoch: int, lr_halve_epochs: int, restart_epochs: int) -> float:
    epoch_in_cycle = (int(epoch) - 1) % int(restart_epochs)
    halvings = epoch_in_cycle // int(lr_halve_epochs)
    return 0.5 ** int(halvings)


def build_lr_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | DoTALRScheduler:
    schedule = str(args.lr_schedule)
    if schedule == "cosine":
        total_steps = max(1, int(args.epochs) * max(1, int(steps_per_epoch)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if schedule == "warmrestart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, int(args.restart_epochs)),
            T_mult=1,
        )
    if schedule == "dota":
        return DoTALRScheduler(
            optimizer,
            lr_halve_epochs=max(1, int(args.lr_halve_epochs)),
            restart_epochs=max(1, int(args.restart_epochs)),
        )
    raise ValueError(f"unknown lr schedule {schedule!r}")


def step_lr_epoch_start(
    scheduler: torch.optim.lr_scheduler.LRScheduler | DoTALRScheduler,
    schedule: str,
    epoch: int,
) -> None:
    if schedule == "warmrestart":
        scheduler.step(int(epoch) - 1)
    elif schedule == "dota":
        scheduler.step(int(epoch))


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def best_metric_name(save_best_by: str) -> str:
    if save_best_by == "gamma":
        return "gamma3d_3pct_3mm"
    if save_best_by == "val_loss":
        return "val_loss"
    raise ValueError(f"unknown save_best_by {save_best_by!r}")


def is_better_best(metrics: dict[str, float], best_metrics: dict[str, float] | None, save_best_by: str) -> bool:
    key = best_metric_name(save_best_by)
    value = float(metrics.get(key, float("nan")))
    if not np.isfinite(value):
        return False
    if not best_metrics:
        return True
    best_value = float(best_metrics.get(key, float("nan")))
    if not np.isfinite(best_value):
        return True
    if save_best_by == "gamma":
        return value > best_value
    return value < best_value


def maybe_save_best_checkpoint(
    out_dir: Path,
    args: argparse.Namespace,
    model: Bragg3D,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | DoTALRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    best_metrics: dict[str, float] | None,
) -> tuple[dict[str, float] | None, bool]:
    save_best_by = str(args.save_best_by)
    if not is_better_best(metrics, best_metrics, save_best_by):
        return best_metrics, False

    updated = dict(metrics)
    updated["best_epoch"] = float(epoch)
    updated["best_global_step"] = float(global_step)
    best_path = out_dir / "best.pt"
    save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, global_step, args, updated)
    upload_to_gcs([best_path], getattr(args, "gcs", None))
    return updated, True


def load_existing_best_metrics(out_dir: Path, save_best_by: str) -> dict[str, float] | None:
    best_path = out_dir / "best.pt"
    if not best_path.exists():
        return None
    try:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"warning: unable to read existing best checkpoint {best_path}: {exc}", flush=True)
        return None
    saved_args = ckpt.get("args")
    if isinstance(saved_args, dict) and saved_args.get("save_best_by") not in (None, save_best_by):
        return None
    metrics = ckpt.get("metrics")
    if not isinstance(metrics, dict):
        return None
    key = best_metric_name(save_best_by)
    if key not in metrics:
        return None
    return {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def report_grid_resolution(args: argparse.Namespace, *, dta_mm: float = 3.0) -> None:
    """Print the resolved BEV grid spacing and warn when it silently degrades gamma.

    Voxel-center gamma search grants distance-to-agreement credit only when a
    neighbouring voxel lies within ``dta_mm``; spacing above that makes the DTA
    term inert and turns "gamma at X%/3mm" into a same-voxel dose test.
    """

    depth_sp = float(args.depth_extent_mm) / float(max(1, int(args.depth_size) - 1))
    lat_sp = float(args.lateral_extent_mm) / float(max(1, int(args.lateral_size) - 1))
    print(
        f"resolved grid: depth {int(args.depth_size)} bins x {depth_sp:.3f}mm, "
        f"lateral {int(args.lateral_size)} bins x {lat_sp:.3f}mm",
        flush=True,
    )
    for axis, spacing in (("depth", depth_sp), ("lateral", lat_sp)):
        if spacing > float(dta_mm):
            message = (
                f"{axis} spacing {spacing:.2f}mm exceeds the {dta_mm:g}mm gamma DTA; "
                f"the DTA term is inert along {axis} and gamma degenerates toward a "
                "same-voxel dose-difference test (harsher than the standard metric)."
            )
            if bool(getattr(args, "fast", False)) or axis in list(getattr(args, "allow_coarse_axes", [])):
                print(f"warning: {message}", flush=True)
            else:
                raise SystemExit(
                    f"error: {message} Pass --allow-coarse-axes {axis} to run anyway, or raise "
                    f"--depth-size/--lateral-size (run17 regression guard)."
                )


def build_model(args: argparse.Namespace) -> Bragg3D | DoTA3D | DoTA3DSpatial:
    model_cls = {"bragg3d": Bragg3D, "dota3d": DoTA3D, "dota3d_spatial": DoTA3DSpatial}[str(args.model)]
    kwargs = {
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "d_ff": int(args.d_ff),
        "max_depth": max(128, int(args.depth_size)),
    }
    if str(args.model) == "dota3d_spatial":
        kwargs["patch_size"] = int(args.patch_size)
    return model_cls(**kwargs)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader | None, str]:
    workers = loader_worker_count(args)
    try:
        loaders = make_doserad_loaders(
            args.root,
            args.patients,
            max_beamlets_per_patient=args.max_beamlets,
            val_frac=float(args.val_frac),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            depth_size=int(args.depth_size),
            depth_extent_mm=float(args.depth_extent_mm),
            lateral_size=int(args.lateral_size),
            lateral_extent_mm=float(args.lateral_extent_mm),
            rebuild_cache=bool(args.rebuild_cache),
            num_workers=0,
            split_by=str(args.split_by),
            test_frac=float(args.test_frac),
        )
        train_loader = clone_loader(
            loaders[0].dataset,
            args,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(args.seed) + 1),
            workers=workers,
        )
        val_loader = clone_loader(loaders[1].dataset, args, shuffle=False, workers=workers)
        test_loader = clone_loader(loaders[2].dataset, args, shuffle=False, workers=workers) if len(loaders) > 2 else None
        return train_loader, val_loader, test_loader, "doserad"
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
        train_ds, val_ds = split_generic_dataset(ds, val_frac=float(args.val_frac), seed=int(args.seed), split_by=str(args.split_by))
        train_loader = clone_loader(
            train_ds,
            args,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(args.seed) + 1),
            workers=workers,
        )
        val_loader = clone_loader(val_ds, args, shuffle=False, workers=workers)
        return train_loader, val_loader, None, "synthetic"


def split_generic_dataset(
    dataset: Dataset,
    *,
    val_frac: float,
    seed: int,
    split_by: str,
) -> tuple[Subset, Subset]:
    mode = str(split_by).lower()
    if mode not in {"patient", "beamlet"}:
        raise ValueError("split_by must be 'patient' or 'beamlet'.")

    patient_by_index = [_dataset_patient_id(dataset, i) for i in range(len(dataset))]
    patient_ids = list(dict.fromkeys(patient_by_index))
    if mode == "patient" and len(patient_ids) > 1:
        shuffled = list(patient_ids)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(shuffled)
        n_val = max(1, int(np.ceil(float(val_frac) * len(shuffled))))
        if n_val >= len(shuffled):
            n_val = len(shuffled) - 1
        val_patient_ids = list(shuffled[:n_val])
        val_patient_set = set(val_patient_ids)
        train_indices = [i for i, patient in enumerate(patient_by_index) if patient not in val_patient_set]
        val_indices = [i for i, patient in enumerate(patient_by_index) if patient in val_patient_set]
        print(f"DoseRAD split_by=patient val_patients={json.dumps(val_patient_ids)}", flush=True)
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    if mode == "patient" and len(patient_ids) <= 1:
        print(
            "warning: split_by=patient requested but only one patient is available; "
            "falling back to beamlet split, validation metrics are optimistic.",
            flush=True,
        )

    val_len = max(1, int(round(len(dataset) * float(val_frac)))) if len(dataset) > 1 else 1
    train_len = max(1, len(dataset) - val_len)
    if train_len + val_len > len(dataset):
        train_len = len(dataset) - val_len
    gen = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len], generator=gen)
    val_patient_ids = list(dict.fromkeys(patient_by_index[int(i)] for i in val_ds.indices))
    print(f"DoseRAD split_by=beamlet val_patients={json.dumps(val_patient_ids)}", flush=True)
    return train_ds, val_ds


def _dataset_patient_id(dataset: Dataset, idx: int) -> str:
    item = dataset[int(idx)]
    if isinstance(item, dict):
        meta = item.get("meta")
        if isinstance(meta, dict) and "patient" in meta:
            return str(meta["patient"])
    return "UNKNOWN"


def loader_worker_count(args: argparse.Namespace) -> int:
    requested = getattr(args, "num_workers", None)
    if requested is not None:
        return max(0, int(requested))
    if bool(getattr(args, "fast", False)):
        return 0
    return min(8, os.cpu_count() or 1)


def clone_loader(
    dataset: Dataset | Subset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    workers: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(workers),
        pin_memory=True,
        persistent_workers=bool(workers > 0),
    )


def make_eval_subsample_loader(val_loader: DataLoader, args: argparse.Namespace) -> DataLoader:
    n_val = len(val_loader.dataset)
    n_sub = min(max(1, int(args.eval_subsample)), n_val) if n_val > 0 else 0
    if n_sub >= n_val:
        dataset = val_loader.dataset
    else:
        rng = np.random.default_rng(int(args.seed))
        indices = np.sort(rng.choice(n_val, size=n_sub, replace=False)).astype(np.int64).tolist()
        dataset = Subset(val_loader.dataset, indices)
    return DataLoader(
        dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=True,
        persistent_workers=bool(val_loader.num_workers > 0),
    )


def should_run_full_eval(epoch: int, args: argparse.Namespace) -> bool:
    full_eval_every = int(args.full_eval_every)
    return full_eval_every > 0 and int(epoch) % full_eval_every == 0


def add_full_metric_aliases(metrics: dict[str, float], full_metrics: dict[str, float]) -> None:
    for key, value in full_metrics.items():
        metrics[f"{key}_full"] = float(value)


def format_full_gamma(metrics: dict[str, float]) -> str:
    value = metrics.get("gamma3d_3pct_3mm_full")
    if value is None or not np.isfinite(float(value)):
        return ""
    return f"gamma(full)={float(value):.2f}% "


def format_full_eval_seconds(seconds: float) -> str:
    if not np.isfinite(float(seconds)):
        return ""
    return f" eval_full:{float(seconds):.2f}s"


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
    x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = batch["dose"].to(device=device, dtype=torch.float32, non_blocking=True)
    scalars = batch["scalars"].to(device=device, dtype=torch.float32, non_blocking=True)
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
def evaluate(
    model: Bragg3D,
    loader: DataLoader,
    device: torch.device,
    *,
    criteria: list[tuple[str, float, float, float]] | None = None,
) -> dict[str, float]:
    model.eval()
    crit = list(criteria) if criteria is not None else list(SUBSAMPLE_CRITERIA)
    losses: list[float] = []
    gamma_values: dict[str, list[float]] = {key: [] for key, *_ in crit}
    rmse_values: list[float] = []
    r80_errors: list[float] = []
    n_seen = 0
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
        target_t = batch["dose"].to(device=device, dtype=torch.float32, non_blocking=True)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32, non_blocking=True)
        pred_t = model(x, scalars)["dose"]
        losses.append(float(doserad_relative_loss(pred_t, target_t).cpu()))
        pred = pred_t.detach().cpu().numpy()
        ref = batch["dose"].numpy()
        spacing = batch["spacing_mm"].numpy()
        for i in range(ref.shape[0]):
            n_seen += 1
            for key, dose_pct, dta_mm, low_cut in crit:
                gamma = safe_gamma_index_3d(
                    pred[i], ref[i], spacing[i], dose_pct=dose_pct, dta_mm=dta_mm, low_dose_threshold=low_cut
                )
                if np.isfinite(gamma):
                    gamma_values[key].append(gamma)
            rmse_values.append(rmse_pct_3d(pred[i], ref[i], low_dose_threshold=0.02))
            pred_r80 = r80_mm_from_profile(depth_profile(pred[i]), float(spacing[i][0]))
            ref_r80 = r80_mm_from_profile(depth_profile(ref[i]), float(spacing[i][0]))
            if np.isfinite(pred_r80) and np.isfinite(ref_r80):
                r80_errors.append(abs(pred_r80 - ref_r80))
    metrics = {
        "val_loss": nanmean(losses),
        "rmse_pct": nanmean(rmse_values),
        "r80_error_mm": nanmean(r80_errors),
        "n_val_beamlets": float(n_seen),
        "n_gamma_beamlets": float(len(gamma_values[crit[0][0]])),
    }
    for key, *_ in crit:
        metrics[key] = nanmean(gamma_values[key])
    return metrics


def safe_gamma_index_3d(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing_mm: np.ndarray,
    *,
    dose_pct: float,
    dta_mm: float,
    low_dose_threshold: float = 0.1,
    peak_epsilon: float = 1e-8,
    max_radius_voxels: int = 8,
) -> float:
    ref_arr = np.asarray(ref, dtype=np.float64)
    peak = float(np.max(ref_arr)) if ref_arr.size else 0.0
    if not np.isfinite(peak) or peak <= float(peak_epsilon):
        return float("nan")
    spacing = np.asarray(tuple(spacing_mm), dtype=np.float64)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0.0):
        return float("nan")
    return gamma_index_3d_fast(
        pred,
        ref_arr,
        spacing,
        dose_pct=dose_pct,
        dta_mm=dta_mm,
        low_dose_threshold=low_dose_threshold,
        max_radius_voxels=max_radius_voxels,
    )


def bounded_gamma_index_3d(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing_mm: np.ndarray,
    *,
    dose_pct: float,
    dta_mm: float,
    low_dose_threshold: float,
    max_radius_voxels: int,
) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    if pred_arr.shape != ref_arr.shape or pred_arr.ndim != 3:
        return float("nan")

    peak = float(np.max(ref_arr))
    dose_norm = (float(dose_pct) / 100.0) * peak
    if not np.isfinite(dose_norm) or dose_norm <= 0.0:
        return float("nan")
    points = np.argwhere(ref_arr >= float(low_dose_threshold) * peak)
    if points.size == 0:
        return float("nan")

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    radii = np.minimum(np.ceil(float(dta_mm) / spacing).astype(int), int(max_radius_voxels))
    passed = 0
    for iz, iy, ix in points:
        z0, z1 = max(0, iz - radii[0]), min(ref_arr.shape[0], iz + radii[0] + 1)
        y0, y1 = max(0, iy - radii[1]), min(ref_arr.shape[1], iy + radii[1] + 1)
        x0, x1 = max(0, ix - radii[2]), min(ref_arr.shape[2], ix + radii[2] + 1)
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        dist_term = (
            ((zz - iz) * spacing[0] / dta_mm) ** 2
            + ((yy - iy) * spacing[1] / dta_mm) ** 2
            + ((xx - ix) * spacing[2] / dta_mm) ** 2
        )
        dose_term = ((pred_arr[z0:z1, y0:y1, x0:x1] - ref_arr[iz, iy, ix]) / dose_norm) ** 2
        if float(np.sqrt(np.min(dist_term + dose_term))) <= 1.0:
            passed += 1
    return float(100.0 * passed / len(points))


def save_and_upload(
    out_dir: Path,
    args: argparse.Namespace,
    model: Bragg3D,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | DoTALRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> Path:
    ckpt_path = out_dir / f"step_{global_step:08d}.pt"
    save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, global_step, args, metrics)
    latest = out_dir / "latest.pt"
    shutil.copy2(ckpt_path, latest)
    upload_to_gcs([ckpt_path, latest, out_dir / "metrics.jsonl", out_dir / "metrics_latest.json"], getattr(args, "gcs", None))
    return ckpt_path


def evaluate_best_checkpoint_on_full_val(
    out_dir: Path,
    model: Bragg3D,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    best_path = out_dir / "best.pt"
    if not best_path.exists():
        return {}
    try:
        load_checkpoint(best_path, model, map_location=device)
    except Exception as exc:
        print(f"warning: unable to re-evaluate best checkpoint {best_path}: {exc}", flush=True)
        return {}
    t0 = time.perf_counter()
    full_metrics = evaluate(model, val_loader, device, criteria=FULL_CRITERIA)
    elapsed = time.perf_counter() - t0
    result = {
        "best_val_loss_full": float(full_metrics["val_loss"]),
        "best_gamma3d_3pct_3mm_full": float(full_metrics["gamma3d_3pct_3mm"]),
        "best_rmse_pct_full": float(full_metrics["rmse_pct"]),
        "best_r80_error_mm_full": float(full_metrics["r80_error_mm"]),
        "best_n_val_beamlets_full": float(full_metrics["n_val_beamlets"]),
        "best_n_gamma_beamlets_full": float(full_metrics["n_gamma_beamlets"]),
        "best_full_eval_seconds": float(elapsed),
    }
    for key, *_ in FULL_CRITERIA:
        result[f"best_{key}_full"] = float(full_metrics[key])
    return result


def save_checkpoint(
    path: Path,
    model: Bragg3D,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | DoTALRScheduler | None,
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | DoTALRScheduler | None = None,
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
    copy_from_gcs(f"{gcs.rstrip('/')}/latest.pt", latest)
    copy_from_gcs(f"{gcs.rstrip('/')}/best.pt", out_dir / "best.pt")
    return latest


def copy_from_gcs(source: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil", "cp", source, str(dest)]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        return proc.returncode == 0 and dest.exists()
    except OSError:
        return False


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
            batch["x"].to(device=device, dtype=torch.float32, non_blocking=True),
            batch["scalars"].to(device=device, dtype=torch.float32, non_blocking=True),
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
    torch.set_rng_state(as_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([as_byte_tensor(s) for s in state["cuda"]])


def as_byte_tensor(value: Any) -> torch.ByteTensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=torch.uint8)
    return torch.tensor(value, dtype=torch.uint8)


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
            wepl = torch.empty_like(rsp)
            wepl[0, :, :] = 0.0
            wepl[1:, :, :] = torch.cumsum(rsp[:-1, :, :], dim=0) * (2.0 / 300.0)
            self.items.append(
                {
                    "x": torch.stack([hu, density, rsp, wepl], dim=0).float(),
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
