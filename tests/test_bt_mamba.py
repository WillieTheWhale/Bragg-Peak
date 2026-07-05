from __future__ import annotations

import pytest
import torch

from braggtransporter.config import ModelConfig
from braggtransporter.models.mamba1d import Mamba1d
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR


def _inputs(batch: int = 2, nz: int = 64, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, nz, C_IN_PERDEPTH, dtype=torch.float32, device=device)
    scalars = torch.tensor(
        [
            [150.0, 1.0, 0.25, 1.0],
            [180.0, 0.5, 0.30, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return x, scalars


def _small_model() -> Mamba1d:
    return Mamba1d(
        ModelConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
            extra={"d_state": 8, "expand": 2, "dt_rank": 8},
        )
    )


def test_mamba1d_forward_shapes_nonnegative_dose_and_param_budget() -> None:
    torch.manual_seed(31)
    model = Mamba1d(ModelConfig())
    x, scalars = _inputs(batch=2, nz=64)

    out = model(x, scalars)

    assert set(out) == {"dose", "letd", "r80"}
    assert out["dose"].shape == (2, 64)
    assert out["letd"].shape == (2, 64)
    assert out["r80"].shape == (2,)
    assert torch.all(out["dose"] >= 0)
    assert torch.isfinite(out["dose"]).all()
    assert torch.isfinite(out["letd"]).all()
    assert torch.isfinite(out["r80"]).all()
    assert 850_000 <= model.param_count() <= 1_050_000
    assert model.param_count() == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_mamba1d_cpu_backward_finite_grads() -> None:
    torch.manual_seed(32)
    model = _small_model()
    x, scalars = _inputs(batch=2, nz=23)

    out = model(x, scalars)
    loss = out["dose"].mean() + out["letd"].square().mean() + out["r80"].mean()
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad and param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all().item() for grad in grads)


def test_mamba1d_mps_backward_finite_grads() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is not available on this host.")

    torch.manual_seed(33)
    device = torch.device("mps")
    model = _small_model().to(device)
    x, scalars = _inputs(batch=2, nz=21, device=device)

    try:
        out = model(x, scalars)
        loss = out["dose"].mean() + out["letd"].square().mean() + out["r80"].mean()
        loss.backward()
        torch.mps.synchronize()
    except NotImplementedError as exc:
        pytest.fail(f"Mamba1d MPS backward raised NotImplementedError: {exc}")

    grads = [param.grad for param in model.parameters() if param.requires_grad and param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all().detach().cpu().item() for grad in grads)
