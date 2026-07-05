from __future__ import annotations

import pytest
import torch

from braggtransporter.config import ModelConfig
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR


def _model(decoder_mode: str, use_physics_prior: bool = True) -> BraggTransporterV0:
    return BraggTransporterV0(
        ModelConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            extra={
                "max_positions": 64,
                "decoder_mode": decoder_mode,
                "use_physics_prior": use_physics_prior,
            },
        )
    )


def _inputs(batch: int = 2, nz: int = 23) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(123)
    x = torch.randn(batch, nz, C_IN_PERDEPTH, dtype=torch.float32)
    scalars = torch.randn(batch, C_SCALAR, dtype=torch.float32)
    return x, scalars


@pytest.mark.parametrize("decoder_mode", ["coord_query", "fixed_grid"])
@pytest.mark.parametrize("use_physics_prior", [True, False])
def test_ablation2a_forward_modes_return_depth_grid(decoder_mode: str, use_physics_prior: bool) -> None:
    torch.manual_seed(7)
    model = _model(decoder_mode, use_physics_prior)
    x, scalars = _inputs()

    out = model(x, scalars)

    assert out["dose"].shape == (2, 23)
    assert out["letd"].shape == (2, 23)
    assert out["r80"].shape == (2,)
    assert torch.all(out["dose"] >= 0)
    assert torch.isfinite(out["dose"]).all()
    assert torch.isfinite(out["letd"]).all()
    assert torch.isfinite(out["r80"]).all()


@pytest.mark.parametrize("decoder_mode", ["coord_query", "fixed_grid"])
def test_ablation2a_physics_prior_off_masks_prior_channels(decoder_mode: str) -> None:
    torch.manual_seed(8)
    model = _model(decoder_mode, use_physics_prior=False).eval()
    x_a, scalars = _inputs()
    x_b = x_a.clone()
    x_b[..., 4:9] = torch.randn_like(x_b[..., 4:9]) * 100.0

    with torch.no_grad():
        out_a = model(x_a, scalars)
        out_b = model(x_b, scalars)

    assert torch.allclose(out_a["dose"], out_b["dose"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_a["letd"], out_b["letd"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_a["r80"], out_b["r80"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("decoder_mode", ["coord_query", "fixed_grid"])
def test_ablation2a_off_grid_query_shape(decoder_mode: str) -> None:
    torch.manual_seed(9)
    model = _model(decoder_mode)
    x, scalars = _inputs(batch=3, nz=19)
    z_query = torch.tensor([0.0, 0.25, 0.55, 1.0], dtype=torch.float32)

    out = model(x, scalars, z_query=z_query)

    assert out["dose"].shape == (3, 4)
    assert out["letd"].shape == (3, 4)
    assert out["r80"].shape == (3,)


def test_ablation2a_rejects_unknown_decoder_mode() -> None:
    with pytest.raises(ValueError, match="decoder_mode"):
        _model("unknown")
