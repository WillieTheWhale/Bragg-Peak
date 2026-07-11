from __future__ import annotations

import torch
import torch.nn.functional as F

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS
from braggtransporter.models.dota3d import DoTA3D


def test_dota3d_forward_shape_nonnegative_and_backward_finite() -> None:
    torch.manual_seed(21)
    model = DoTA3D(d_model=32, n_layers=1, n_heads=4, d_ff=64, max_depth=32, lateral_size=16, dropout=0.0)
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
    model = DoTA3D(d_model=32, n_layers=1, n_heads=4, d_ff=64, max_depth=32, lateral_size=16, dropout=0.0)
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
    model = DoTA3D(
        d_model=24,
        n_layers=1,
        n_heads=4,
        d_ff=48,
        max_depth=12,
        lateral_size=8,
        dropout=0.0,
    ).to(device)
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
        wepl = torch.empty_like(rsp)
        wepl[0, :, :] = 0.0
        wepl[1:, :, :] = torch.cumsum(rsp[:-1, :, :], dim=0) * (2.0 / 300.0)
        inputs.append(torch.stack([hu, density, rsp, wepl], dim=0))
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


def test_dota3d_causal_depth_attention_blocks_future_tissue() -> None:
    torch.manual_seed(24)
    model = DoTA3D(
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        max_depth=12,
        lateral_size=8,
        dropout=0.0,
    ).eval()
    x, scalars, _ = _synthetic_batch(batch=1, depth=12, lateral=8)
    changed = x.clone()
    changed[:, :, 7:] = torch.randn_like(changed[:, :, 7:])

    with torch.no_grad():
        before = model(x, scalars)["dose"]
        after = model(changed, scalars)["dose"]

    torch.testing.assert_close(before[:, :7], after[:, :7], rtol=0.0, atol=2e-6)


def test_dota3d_preserves_lateral_position_in_slice_tokens() -> None:
    torch.manual_seed(25)
    model = DoTA3D(
        d_model=48,
        n_layers=1,
        n_heads=4,
        d_ff=48,
        max_depth=4,
        lateral_size=8,
        dropout=0.0,
    ).eval()
    a = torch.zeros(1, DOSERAD_INPUT_CHANNELS, 4, 8, 8)
    b = a.clone()
    a[:, 0, 1, 1, 1] = 1.0
    b[:, 0, 1, 6, 6] = 1.0
    scalars = torch.tensor([[150.0, 0.0, 0.0, 1.0]])

    with torch.no_grad():
        out_a = model(a, scalars)["dose"]
        out_b = model(b, scalars)["dose"]

    assert not torch.allclose(out_a[:, 1], out_b[:, 1], rtol=1e-5, atol=1e-6)


def test_dota3d_wide_two_mm_grid_uses_bounded_published_token_width() -> None:
    model = DoTA3D(
        d_model=576,
        n_layers=1,
        n_heads=16,
        d_ff=576,
        max_depth=4,
        lateral_size=49,
        encoder_channels=4,
        dropout=0.0,
    ).eval()
    x = torch.zeros(1, DOSERAD_INPUT_CHANNELS, 4, 49, 49)
    scalars = torch.tensor([[150.0, 0.0, 0.0, 1.0]])

    with torch.no_grad():
        dose = model(x, scalars)["dose"]

    assert model.encoded_lateral_size**2 * model.encoder_channels == 576
    assert dose.shape == (1, 4, 49, 49)
    assert model.param_count() < 3_000_000
