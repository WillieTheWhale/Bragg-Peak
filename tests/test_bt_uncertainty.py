from __future__ import annotations

import numpy as np
import pytest
import torch

from braggtransporter.config import DataConfig
from braggtransporter.models.uncertainty_head import (
    ConditionalFlowMatchingResidualHead,
    HeteroscedasticResidualHead,
    UncertaintyHeadConfig,
)
from braggtransporter.train import SyntheticBraggDataset
from braggtransporter.uncertainty import calibration_summary, gaussian_coverage


def _batch(batch_size: int = 4, nz: int = 23) -> dict[str, torch.Tensor]:
    ds = SyntheticBraggDataset(n_samples=batch_size, nz=nz, data_cfg=DataConfig(max_depth_cm=10.0), seed=77)
    return {k: torch.stack([ds[i][k] for i in range(batch_size)]) for k in ds[0].keys()}


def _v0_mean_and_residual(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    v0_mean = batch["dose"] - 0.08 * torch.tanh(batch["x"][..., 6])
    residual = batch["dose"] - v0_mean
    return v0_mean, residual


def test_heteroscedastic_head_forward_shape_and_nonnegative_sigma() -> None:
    torch.manual_seed(0)
    batch = _batch()
    v0_mean, _ = _v0_mean_and_residual(batch)
    head = HeteroscedasticResidualHead(UncertaintyHeadConfig(hidden=24, depth=1))

    out = head(batch["x"], v0_mean, batch["scalars"])

    assert out["residual_mean"].shape == batch["dose"].shape
    assert out["logvar"].shape == batch["dose"].shape
    assert out["sigma"].shape == batch["dose"].shape
    assert torch.isfinite(out["sigma"]).all()
    assert torch.all(out["sigma"] >= 0.0)


def test_heteroscedastic_nll_decreases_after_two_steps() -> None:
    torch.manual_seed(1)
    batch = _batch()
    v0_mean, residual = _v0_mean_and_residual(batch)
    head = HeteroscedasticResidualHead(UncertaintyHeadConfig(hidden=32, depth=1))
    opt = torch.optim.AdamW(head.parameters(), lr=1.0e-2, weight_decay=0.0)

    initial = head.nll_loss(batch["x"], v0_mean, residual, batch["scalars"])
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = head.nll_loss(batch["x"], v0_mean, residual, batch["scalars"])
        loss.backward()
        opt.step()
    final = head.nll_loss(batch["x"], v0_mean, residual, batch["scalars"])

    assert torch.isfinite(final)
    assert final.item() < initial.item()


def test_flow_head_shape_sigma_and_loss_decreases_after_two_steps() -> None:
    torch.manual_seed(2)
    batch = _batch()
    v0_mean, residual = _v0_mean_and_residual(batch)
    head = ConditionalFlowMatchingResidualHead(UncertaintyHeadConfig(hidden=32, depth=1))
    opt = torch.optim.AdamW(head.parameters(), lr=1.0e-2, weight_decay=0.0)
    noise = torch.linspace(-0.75, 0.75, residual.numel(), dtype=residual.dtype).reshape_as(residual)
    t = torch.full((residual.shape[0], 1), 0.35, dtype=residual.dtype)

    initial = head.flow_matching_loss(batch["x"], v0_mean, residual, batch["scalars"], noise=noise, t=t)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = head.flow_matching_loss(batch["x"], v0_mean, residual, batch["scalars"], noise=noise, t=t)
        loss.backward()
        opt.step()
    final = head.flow_matching_loss(batch["x"], v0_mean, residual, batch["scalars"], noise=noise, t=t)
    sigma = head.predictive_sigma(batch["x"], v0_mean, batch["scalars"], n_samples=3, n_steps=2)

    assert sigma.shape == batch["dose"].shape
    assert torch.isfinite(sigma).all()
    assert torch.all(sigma >= 0.0)
    assert torch.isfinite(final)
    assert final.item() < initial.item()


def test_coverage_monotonic_in_sigma() -> None:
    pred = np.zeros((2, 5), dtype=np.float64)
    ref = np.linspace(-1.0, 1.0, 10, dtype=np.float64).reshape(2, 5)
    small = np.full_like(pred, 0.1)
    large = np.full_like(pred, 10.0)

    assert gaussian_coverage(pred, ref, large, 0.95) >= gaussian_coverage(pred, ref, small, 0.95)
    rows, summary = calibration_summary(pred, ref, large)
    assert rows[-1]["nominal"] == pytest.approx(0.95)
    assert summary["sharpness_mean_sigma"] == pytest.approx(10.0)


def test_real_mps_backward_check_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this machine")
    torch.manual_seed(3)
    device = torch.device("mps")
    batch = {k: v.to(device) for k, v in _batch(batch_size=2, nz=17).items()}
    v0_mean, residual = _v0_mean_and_residual(batch)
    head = HeteroscedasticResidualHead(UncertaintyHeadConfig(hidden=16, depth=1)).to(device)

    loss = head.nll_loss(batch["x"], v0_mean, residual, batch["scalars"])
    loss.backward()

    assert torch.isfinite(loss.detach().cpu())
    grads = [p.grad.detach().cpu() for p in head.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
