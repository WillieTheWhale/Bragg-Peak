"""Phase-4 residual uncertainty orchestration for BraggTransporter.

The deterministic BraggTransporter-v0 checkpoint is treated as frozen. Phase 4
learns only a separate residual head on ``target_dose - v0_mean_dose``.

Identifiability policy from the v3.1 plan:
- ``sigma_aleatoric`` is identifiable from residuals in the current supervised
  stream and is provided by the heteroscedastic or flow residual head.
- ``sigma_epistemic`` is identifiable only as disagreement across an ensemble of
  independently trained or bootstrapped v0 checkpoints.
- ``sigma_MC`` requires low/high-statistics MC pairs with known histories.
- ``sigma_input`` requires controlled CT/RSP/SPR perturbation ensembles.
- ``sigma_meas`` requires paired MC-vs-measurement residuals plus measurement
  metadata. It is Phase-5+ data-gated and must not be inferred from SDE targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from braggtransporter.config import ModelConfig, get_device
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.models.uncertainty_head import (
    ConditionalFlowMatchingResidualHead,
    HeteroscedasticResidualHead,
    UncertaintyHeadConfig,
)

HeadKind = Literal["heteroscedastic", "flow"]


@dataclass(frozen=True)
class UncertaintyTrainConfig:
    head_kind: HeadKind = "heteroscedastic"
    epochs: int = 4
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    device: str = "auto"
    seed: int = 0
    grad_clip: float = 1.0
    flow_samples: int = 16
    flow_steps: int = 4


@dataclass
class UncertaintyDecomposition:
    """Per-voxel uncertainty channels in dose units (MeV/g per voxel)."""

    mean_dose: Tensor
    sigma_aleatoric: Tensor
    sigma_epistemic: Tensor | None = None
    sigma_mc: Tensor | None = None
    sigma_input: Tensor | None = None
    sigma_meas: Tensor | None = None

    @property
    def sigma_total_identifiable(self) -> Tensor:
        parts = [self.sigma_aleatoric.square()]
        if self.sigma_epistemic is not None:
            parts.append(self.sigma_epistemic.square())
        return torch.sqrt(torch.stack(parts, dim=0).sum(dim=0))


def load_frozen_v0(checkpoint_path: str | Path, device: torch.device | str = "cpu") -> BraggTransporterV0:
    """Load a frozen BraggTransporter-v0 checkpoint without editing v0 code."""

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model_cfg = ModelConfig()
    raw_model_cfg = ckpt.get("config", {}).get("model", {}) if isinstance(ckpt, dict) else {}
    for key, value in raw_model_cfg.items():
        if hasattr(model_cfg, key):
            setattr(model_cfg, key, value)
    model = BraggTransporterV0(model_cfg)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_uncertainty_head(
    kind: HeadKind,
    cfg: UncertaintyHeadConfig | None = None,
) -> HeteroscedasticResidualHead | ConditionalFlowMatchingResidualHead:
    if kind == "heteroscedastic":
        return HeteroscedasticResidualHead(cfg)
    if kind == "flow":
        return ConditionalFlowMatchingResidualHead(cfg)
    raise ValueError("kind must be 'heteroscedastic' or 'flow'")


def train_uncertainty_head(
    frozen_v0: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    cfg: UncertaintyTrainConfig | None = None,
    head_cfg: UncertaintyHeadConfig | None = None,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    """Train a residual uncertainty head on ``target - frozen_v0(x)``."""

    cfg = cfg or UncertaintyTrainConfig()
    torch.manual_seed(int(cfg.seed))
    device = get_device(cfg.device)
    frozen_v0.to(device).eval()
    for param in frozen_v0.parameters():
        param.requires_grad_(False)

    head = build_uncertainty_head(cfg.head_kind, head_cfg).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(cfg.epochs) + 1):
        train_loss = _run_head_epoch(frozen_v0, head, train_loader, device, optimizer, cfg)
        row: dict[str, float | int] = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            row["val_loss"] = _run_head_epoch(frozen_v0, head, val_loader, device, None, cfg)
        history.append(row)
    return head, history


def decompose_uncertainty(
    frozen_v0: nn.Module,
    head: nn.Module,
    x: Tensor,
    scalars: Tensor,
    *,
    ensemble_v0: Iterable[nn.Module] | None = None,
    flow_samples: int = 16,
    flow_steps: int = 4,
) -> UncertaintyDecomposition:
    """Return identifiable uncertainty channels for a batch.

    ``sigma_epistemic`` is returned only when an ensemble of frozen v0 checkpoints
    is supplied. ``sigma_MC``, ``sigma_input``, and ``sigma_meas`` are intentionally
    absent here because their identifiability requires matched data interventions.
    """

    frozen_v0.eval()
    with torch.no_grad():
        mean_dose = frozen_v0(x, scalars)["dose"]
        if isinstance(head, HeteroscedasticResidualHead):
            sigma_aleatoric = head(x, mean_dose, scalars)["sigma"]
        elif isinstance(head, ConditionalFlowMatchingResidualHead):
            sigma_aleatoric = head.predictive_sigma(
                x,
                mean_dose,
                scalars,
                n_samples=flow_samples,
                n_steps=flow_steps,
            )
        else:
            raise TypeError(f"unsupported uncertainty head type {type(head)!r}")

        sigma_epistemic: Tensor | None = None
        if ensemble_v0 is not None:
            preds = []
            for model in ensemble_v0:
                model.eval()
                preds.append(model(x, scalars)["dose"])
            if preds:
                sigma_epistemic = torch.stack(preds, dim=0).std(dim=0, unbiased=False)

    return UncertaintyDecomposition(
        mean_dose=mean_dose,
        sigma_aleatoric=sigma_aleatoric,
        sigma_epistemic=sigma_epistemic,
    )


def calibration_summary(
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    levels: tuple[float, ...] = (0.5, 0.6826894921370859, 0.8, 0.9, 0.95),
) -> tuple[list[dict[str, float | bool]], dict[str, float | bool]]:
    """Reliability rows plus headline 68/95 coverage and sharpness."""

    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if pred_arr.shape != ref_arr.shape or pred_arr.shape != sigma_arr.shape:
        raise ValueError("pred, ref, and sigma must have identical shapes")
    if np.any(sigma_arr < 0.0):
        raise ValueError("sigma must be nonnegative")

    rows = []
    for level in levels:
        empirical = gaussian_coverage(pred_arr, ref_arr, sigma_arr, float(level))
        rows.append(
            {
                "nominal": float(level),
                "empirical": empirical,
                "abs_error": abs(empirical - float(level)),
                "within_5pct": abs(empirical - float(level)) <= 0.05,
            }
        )
    cov68 = gaussian_coverage(pred_arr, ref_arr, sigma_arr, 0.6826894921370859)
    cov95 = gaussian_coverage(pred_arr, ref_arr, sigma_arr, 0.95)
    summary = {
        "coverage_68": cov68,
        "coverage_95": cov95,
        "coverage_68_within_5pct": abs(cov68 - 0.6826894921370859) <= 0.05,
        "coverage_95_within_5pct": abs(cov95 - 0.95) <= 0.05,
        "sharpness_mean_sigma": float(np.mean(sigma_arr)),
        "sharpness_median_sigma": float(np.median(sigma_arr)),
    }
    return rows, summary


def gaussian_coverage(
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    sigma: NDArray[np.float64],
    level: float,
) -> float:
    """Empirical central Gaussian coverage for a predicted dose sigma."""

    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    z = NormalDist().inv_cdf((1.0 + float(level)) / 2.0)
    band = z * np.maximum(np.asarray(sigma, dtype=np.float64), 0.0)
    return float(np.mean(np.abs(np.asarray(ref) - np.asarray(pred)) <= band))


def sigma_mc_from_low_high_stat_pairs(*_: Any, **__: Any) -> Tensor:
    """Hook for ``sigma_MC`` once low/high-stat MC pair data exist.

    Required intervention: the same material/beam/geometry simulated at known
    low and high particle-history counts, so shot noise is the deliberately varied
    source. Use ``scripts/gen_mc_pairs.py`` to create a separate HDF5 stream.
    """

    raise NotImplementedError("sigma_MC is data-gated on low/high-stat MC pairs with known histories")


def sigma_input_from_perturbation_ensemble(*_: Any, **__: Any) -> Tensor:
    """Hook for ``sigma_input`` from controlled CT/RSP/SPR perturbations.

    Required intervention: identical geometry and beam with only input material
    or SPR fields perturbed according to a documented uncertainty model.
    """

    raise NotImplementedError("sigma_input is data-gated on controlled input perturbation ensembles")


def sigma_meas_from_measurement_pairs(*_: Any, **__: Any) -> Tensor:
    """Hook for ``sigma_meas`` once paired measurements exist.

    Required intervention: paired high-fidelity MC and real measured depth-dose
    residuals with detector, beamline, calibration, and acquisition metadata.
    This is Phase-5+ data-gated and must not be inferred from synthetic SDE data.
    """

    raise NotImplementedError("sigma_meas is data-gated on paired MC-vs-measurement residuals")


def _run_head_epoch(
    frozen_v0: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    cfg: UncertaintyTrainConfig,
) -> float:
    is_train = optimizer is not None
    head.train(is_train)
    losses: list[float] = []

    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        scalars = batch["scalars"].to(device=device, dtype=torch.float32)
        dose = batch["dose"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            v0_mean = frozen_v0(x, scalars)["dose"]
        residual = dose - v0_mean

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if isinstance(head, HeteroscedasticResidualHead):
            loss = head.nll_loss(x, v0_mean, residual, scalars)
        elif isinstance(head, ConditionalFlowMatchingResidualHead):
            loss = head.flow_matching_loss(x, v0_mean, residual, scalars)
        else:
            raise TypeError(f"unsupported uncertainty head type {type(head)!r}")
        if optimizer is not None:
            loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg.grad_clip))
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")
