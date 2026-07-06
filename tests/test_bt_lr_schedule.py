from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from scripts.train_doserad_gpu import (
    build_lr_scheduler,
    current_lr,
    maybe_save_best_checkpoint,
    step_lr_epoch_start,
)


def test_cosine_lr_sequence_is_valid_and_decays() -> None:
    lrs = _lr_sequence("cosine", epochs=5, restart_epochs=3, lr_halve_epochs=2)

    assert len(lrs) == 5
    assert all(lr >= 0.0 for lr in lrs)
    assert lrs[0] < 0.1
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12)


def test_warmrestart_lr_sequence_jumps_at_restart_boundary() -> None:
    lrs = _lr_sequence("warmrestart", epochs=5, restart_epochs=3, lr_halve_epochs=2)

    assert lrs[:4] == pytest.approx([0.1, 0.075, 0.025, 0.1])
    assert lrs[3] > lrs[2]


def test_dota_lr_sequence_halves_then_hard_restarts() -> None:
    lrs = _lr_sequence("dota", epochs=6, restart_epochs=4, lr_halve_epochs=2)

    assert lrs == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.1, 0.1])
    assert lrs[4] > lrs[3]


def test_best_gamma_checkpoint_keeps_max_not_last(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    args = argparse.Namespace(
        lr_schedule="cosine",
        epochs=3,
        restart_epochs=3,
        lr_halve_epochs=2,
        save_best_by="gamma",
        seed=5,
        patients=["SYNTH"],
        max_beamlets=None,
        gcs=None,
    )
    scheduler = build_lr_scheduler(args, optimizer, steps_per_epoch=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_metrics: dict[str, float] | None = None

    for epoch, gamma in enumerate([10.0, 42.0, 30.0], start=1):
        metrics = {
            "epoch": float(epoch),
            "global_step": float(epoch),
            "val_loss": 1.0 / epoch,
            "gamma3d_3pct_3mm": gamma,
        }
        best_metrics, _ = maybe_save_best_checkpoint(
            tmp_path,
            args,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            epoch,
            metrics,
            best_metrics,
        )

    ckpt = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)

    assert ckpt["epoch"] == 2
    assert ckpt["metrics"]["gamma3d_3pct_3mm"] == 42.0


def _lr_sequence(schedule: str, *, epochs: int, restart_epochs: int, lr_halve_epochs: int) -> list[float]:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    args = argparse.Namespace(
        lr_schedule=schedule,
        epochs=epochs,
        restart_epochs=restart_epochs,
        lr_halve_epochs=lr_halve_epochs,
    )
    scheduler = build_lr_scheduler(args, optimizer, steps_per_epoch=1)
    lrs: list[float] = []
    for epoch in range(1, epochs + 1):
        step_lr_epoch_start(scheduler, schedule, epoch)
        if schedule == "cosine":
            optimizer.step()
            scheduler.step()
        lrs.append(current_lr(optimizer))
    return lrs
