from __future__ import annotations

import torch
import torch.nn.functional as F

from braggtransporter.models.dota3d_spatial import DoTA3DSpatial


def test_dota3d_spatial_preserves_absolute_material_level() -> None:
    torch.manual_seed(41)
    model = DoTA3DSpatial(
        c_in=4,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        max_depth=16,
        patch_size=4,
    )
    model.eval()
    scalars = torch.tensor([[150.0, 2.0, 0.0, 1.0]], dtype=torch.float32)
    water = _homogeneous_slab(hu=0.0, density=1.0, rsp=1.0, wepl_stop=1.0)
    dense = _homogeneous_slab(hu=1.0, density=1.9, rsp=1.6, wepl_stop=1.6)

    with torch.no_grad():
        water_dose = model(water, scalars)["dose"]
        dense_dose = model(dense, scalars)["dose"]

    mean_abs_diff = torch.mean(torch.abs(water_dose - dense_dose))
    print(f"dota3d_spatial_water_vs_dense_mean_abs_diff={float(mean_abs_diff):.6g}")

    assert float(mean_abs_diff) > 1e-3


def test_dota3d_spatial_four_channel_shape_backward_and_depth_contract() -> None:
    torch.manual_seed(42)
    model = DoTA3DSpatial(
        c_in=4,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        max_depth=16,
        patch_size=4,
    )
    x = torch.cat(
        [
            _homogeneous_slab(hu=0.0, density=1.0, rsp=1.0, wepl_stop=1.0),
            _homogeneous_slab(hu=1.0, density=1.9, rsp=1.6, wepl_stop=1.6),
        ],
        dim=0,
    )
    scalars = torch.tensor(
        [[150.0, 2.0, 0.0, 1.0], [150.0, 2.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    target = torch.zeros(2, 16, 24, 24, dtype=torch.float32)

    out = model(x, scalars)

    assert set(out) == {"dose"}
    assert out["dose"].shape == (2, 16, 24, 24)
    assert torch.all(out["dose"] >= 0.0)
    assert torch.isfinite(out["dose"]).all()
    assert model.param_count() < 8_000_000

    loss = F.mse_loss(out["dose"], target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all().item() for g in grads)

    variable_depth = _homogeneous_slab(hu=0.0, density=1.0, rsp=1.0, wepl_stop=1.0, depth=11)
    with torch.no_grad():
        variable_out = model(variable_depth, scalars[:1])["dose"]
    assert variable_out.shape == (1, 11, 24, 24)


def _homogeneous_slab(
    *,
    hu: float,
    density: float,
    rsp: float,
    wepl_stop: float,
    depth: int = 16,
    lateral: int = 24,
) -> torch.Tensor:
    z = torch.linspace(0.0, float(wepl_stop), depth, dtype=torch.float32).view(depth, 1, 1)
    slab = torch.empty(4, depth, lateral, lateral, dtype=torch.float32)
    slab[0].fill_(float(hu))
    slab[1].fill_(float(density))
    slab[2].fill_(float(rsp))
    slab[3] = z.expand(depth, lateral, lateral)
    return slab.unsqueeze(0)
