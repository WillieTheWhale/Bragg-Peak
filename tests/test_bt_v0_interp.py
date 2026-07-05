from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from braggtransporter.config import ModelConfig
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0


def _small_model() -> BraggTransporterV0:
    return BraggTransporterV0(
        ModelConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
            extra={"max_positions": 128},
        )
    )


def _inputs(batch: int = 2, nz: int = 31) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, nz, 9)
    scalars = torch.randn(batch, 4)
    return x, scalars


def test_bt_v0_forward_backward_cpu_finite_grads() -> None:
    torch.manual_seed(10)
    model = _small_model()
    x, scalars = _inputs()

    out = model(x, scalars)
    loss = out["dose"].mean() + out["letd"].square().mean() + out["r80"].mean()
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad and param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_bt_v0_off_grid_coordinate_query_shape_and_finite_values() -> None:
    torch.manual_seed(11)
    model = _small_model()
    x, scalars = _inputs(batch=3, nz=17)
    z_query = torch.tensor([0.0, 0.031, 0.25, 0.501, 0.875, 1.0], dtype=torch.float32)

    out = model(x, scalars, z_query=z_query)

    assert out["dose"].shape == (3, 6)
    assert out["letd"].shape == (3, 6)
    assert out["r80"].shape == (3,)
    assert torch.isfinite(out["dose"]).all()
    assert torch.isfinite(out["letd"]).all()


def test_bt_v0_latent_interpolation_matches_numpy_reference() -> None:
    latent = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 4.0], [3.0, 9.0], [8.0, 16.0], [13.0, 25.0]],
            [[1.0, -2.0], [1.5, -1.0], [4.0, 0.0], [6.0, 2.0], [9.0, 5.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    coords = torch.tensor([[0.0, 0.125, 0.5, 0.9, 1.0], [0.05, 0.33, 0.67, 0.99, 1.0]])

    actual = BraggTransporterV0._interpolate_latent(latent, coords)

    grid = np.linspace(0.0, 1.0, latent.shape[1])
    expected = np.empty(actual.shape, dtype=np.float32)
    latent_np = latent.detach().numpy()
    coords_np = coords.numpy()
    for b in range(latent.shape[0]):
        for c in range(latent.shape[2]):
            expected[b, :, c] = np.interp(coords_np[b], grid, latent_np[b, :, c])

    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-4, atol=1e-4)

    actual.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_bt_v0_two_step_optimization_reduces_simple_dose_mse() -> None:
    torch.manual_seed(12)
    model = _small_model()
    x, scalars = _inputs(batch=4, nz=23)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    with torch.no_grad():
        target = model(x, scalars)["dose"] * 0.25

    initial = F.mse_loss(model(x, scalars)["dose"], target)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x, scalars)["dose"], target)
        loss.backward()
        optimizer.step()
    final = F.mse_loss(model(x, scalars)["dose"], target)

    assert torch.isfinite(final)
    assert final.item() < initial.item()
