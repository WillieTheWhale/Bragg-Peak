from __future__ import annotations

import torch
import torch.nn.functional as F

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS
from braggtransporter.models.dota3d import DoTA3D


def test_dota3d_forward_shape_nonnegative_and_backward_finite() -> None:
    torch.manual_seed(21)
    model = DoTA3D(d_model=32, n_layers=1, n_heads=4, d_ff=64, max_depth=32)
    x, scalars, target = _synthetic_batch()

    out = model(x, scalars)

    assert set(out) == {"dose"}
    assert out["dose"].shape == (2, 32, 16, 16)
    assert torch.all(out["dose"] >= 0.0)
    assert torch.isfinite(out["dose"]).all()
    assert 10_000 < model.param_count() < 8_000_000
    assert model.param_count() == sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss = F.mse_loss(out["dose"], target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all().item() for g in grads)


def test_dota3d_two_step_loss_decrease_cpu() -> None:
    torch.manual_seed(22)
    model = DoTA3D(d_model=32, n_layers=1, n_heads=4, d_ff=64, max_depth=32)
    x, scalars, target = _synthetic_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x, scalars)["dose"], target)
        losses.append(float(loss.detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    assert losses[-1] < losses[0]


def test_dota3d_real_mps_backward_check_if_available() -> None:
    if not torch.backends.mps.is_available():
        return

    torch.manual_seed(23)
    device = torch.device("mps")
    model = DoTA3D(d_model=24, n_layers=1, n_heads=4, d_ff=48, max_depth=12).to(device)
    x, scalars, target = _synthetic_batch(batch=1, depth=12, lateral=8)
    loss = F.mse_loss(
        model(x.to(device), scalars.to(device))["dose"],
        target.to(device),
    )
    loss.backward()
    torch.mps.synchronize()


def _synthetic_batch(
    *,
    batch: int = 2,
    depth: int = 32,
    lateral: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.linspace(0.0, 1.0, depth).view(1, depth, 1, 1)
    y = torch.linspace(-1.0, 1.0, lateral).view(1, 1, lateral, 1)
    x = torch.linspace(-1.0, 1.0, lateral).view(1, 1, 1, lateral)

    doses = []
    inputs = []
    scalars = []
    for i in range(batch):
        center = 0.35 + 0.3 * i / max(batch - 1, 1)
        sigma_z = 0.08 + 0.02 * i
        sigma_lat = 0.30
        dose = torch.exp(-0.5 * (((z - center) / sigma_z) ** 2 + (y / sigma_lat) ** 2 + (x / sigma_lat) ** 2))
        dose = dose.squeeze(0)
        dose = dose / dose.max().clamp_min(torch.finfo(torch.float32).eps)
        hu = -0.05 + 0.15 * z.squeeze(0).expand_as(dose)
        density = (1.0 + hu).clamp(0.0, 2.5)
        rsp = density.clone()
        inputs.append(torch.stack([hu, density, rsp], dim=0))
        doses.append(dose)
        scalars.append(torch.tensor([120.0 + 20.0 * i, float(i), 0.0, 1.0], dtype=torch.float32))

    return (
        torch.stack(inputs).float(),
        torch.stack(scalars).float(),
        torch.stack(doses).float(),
    )


def test_synthetic_batch_contract() -> None:
    x, scalars, dose = _synthetic_batch()
    assert x.shape == (2, DOSERAD_INPUT_CHANNELS, 32, 16, 16)
    assert scalars.shape == (2, 4)
    assert dose.shape == (2, 32, 16, 16)
