from __future__ import annotations

import torch
import torch.nn.functional as F

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS
from braggtransporter.models.dota3d_spatial import DoTA3DSpatial


def test_dota3d_spatial_forward_shape_nonnegative_and_backward_finite() -> None:
    torch.manual_seed(31)
    model = DoTA3DSpatial(d_model=32, n_layers=1, n_heads=4, d_ff=64, max_depth=16, patch_size=4)
    x, scalars, target = _synthetic_batch()

    out = model(x, scalars)

    assert set(out) == {"dose"}
    assert out["dose"].shape == (2, 16, 24, 24)
    assert torch.all(out["dose"] >= 0.0)
    assert torch.isfinite(out["dose"]).all()
    assert 10_000 < model.param_count() < 8_000_000
    assert model.param_count() == sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss = F.mse_loss(out["dose"], target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all().item() for g in grads)


def test_dota3d_spatial_lateral_sensitivity() -> None:
    torch.manual_seed(32)
    model = DoTA3DSpatial(d_model=32, n_layers=1, n_heads=4, d_ff=64, max_depth=16, patch_size=4)
    model.eval()
    scalars = torch.tensor([[150.0, 2.0, 0.0, 1.0]], dtype=torch.float32)
    depth_idx = 8
    pos_a = (5, 5)
    pos_b = (18, 18)

    x_a = _constant_bev(batch=1, depth=16, lateral=24)
    x_b = x_a.clone()
    x_a[0, 0, depth_idx, pos_a[0], pos_a[1]] += 1.0
    x_b[0, 0, depth_idx, pos_b[0], pos_b[1]] += 1.0

    with torch.no_grad():
        out_a = model(x_a, scalars)["dose"]
        out_b = model(x_b, scalars)["dose"]

    diff_at_a = torch.abs(out_a[0, depth_idx, pos_a[0], pos_a[1]] - out_b[0, depth_idx, pos_a[0], pos_a[1]])
    diff_at_b = torch.abs(out_a[0, depth_idx, pos_b[0], pos_b[1]] - out_b[0, depth_idx, pos_b[0], pos_b[1]])
    mean_site_diff = 0.5 * (diff_at_a + diff_at_b)
    print(
        "dota3d_spatial_lateral_sensitivity "
        f"diff_at_a={float(diff_at_a):.6g} diff_at_b={float(diff_at_b):.6g} "
        f"mean={float(mean_site_diff):.6g}"
    )

    assert float(diff_at_a) > 1e-3
    assert float(diff_at_b) > 1e-3
    assert float(mean_site_diff) > 1e-3


def test_dota3d_spatial_two_step_loss_decrease_cpu() -> None:
    torch.manual_seed(33)
    model = DoTA3DSpatial(d_model=32, n_layers=1, n_heads=4, d_ff=64, max_depth=16, patch_size=4)
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


def test_dota3d_spatial_real_mps_backward_check_if_available() -> None:
    if not torch.backends.mps.is_available():
        return

    torch.manual_seed(34)
    device = torch.device("mps")
    model = DoTA3DSpatial(d_model=24, n_layers=1, n_heads=4, d_ff=48, max_depth=8, patch_size=4).to(device)
    x, scalars, target = _synthetic_batch(batch=1, depth=8, lateral=8)
    loss = F.mse_loss(
        model(x.to(device), scalars.to(device))["dose"],
        target.to(device),
    )
    loss.backward()
    torch.mps.synchronize()


def _synthetic_batch(
    *,
    batch: int = 2,
    depth: int = 16,
    lateral: int = 24,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.linspace(0.0, 1.0, depth).view(1, depth, 1, 1)
    y = torch.linspace(-1.0, 1.0, lateral).view(1, 1, lateral, 1)
    x = torch.linspace(-1.0, 1.0, lateral).view(1, 1, 1, lateral)

    doses = []
    inputs = []
    scalars = []
    for i in range(batch):
        z_center = 0.35 + 0.3 * i / max(batch - 1, 1)
        y_center = -0.25 + 0.5 * i / max(batch - 1, 1)
        x_center = 0.20 - 0.4 * i / max(batch - 1, 1)
        dose = torch.exp(
            -0.5
            * (
                ((z - z_center) / 0.10) ** 2
                + ((y - y_center) / 0.25) ** 2
                + ((x - x_center) / 0.25) ** 2
            )
        ).squeeze(0)
        dose = dose / dose.max().clamp_min(torch.finfo(torch.float32).eps)
        hu = (-0.05 + 0.15 * z.squeeze(0).expand_as(dose) + 0.25 * dose).clamp(-1.0, 2.5)
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


def _constant_bev(*, batch: int, depth: int, lateral: int) -> torch.Tensor:
    x = torch.zeros(batch, DOSERAD_INPUT_CHANNELS, depth, lateral, lateral)
    x[:, 1] = 1.0
    x[:, 2] = 1.0
    return x


def test_synthetic_batch_contract() -> None:
    x, scalars, dose = _synthetic_batch()
    assert x.shape == (2, DOSERAD_INPUT_CHANNELS, 16, 24, 24)
    assert scalars.shape == (2, 4)
    assert dose.shape == (2, 16, 24, 24)
