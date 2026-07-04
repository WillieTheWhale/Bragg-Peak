from __future__ import annotations

import pytest
import torch

from braggtransporter.models.dota_transformer import DoTATransformer
from braggtransporter.models.fno1d import FNO1d
from braggtransporter.models.mlp import MLPBaseline
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR


@pytest.mark.parametrize(
    "model_cls",
    [
        MLPBaseline,
        FNO1d,
        DoTATransformer,
    ],
)
def test_baseline_model_forward_shapes_nonnegative_dose_and_gradients(model_cls: type[torch.nn.Module]) -> None:
    torch.manual_seed(17)
    batch = 2
    nz = 64
    x = torch.randn(batch, nz, C_IN_PERDEPTH, dtype=torch.float32)
    scalars = torch.tensor(
        [
            [150.0, 1.0, 0.25, 1.0],
            [180.0, 0.5, 0.30, 0.0],
        ],
        dtype=torch.float32,
    )

    model = model_cls()
    out = model(x, scalars)

    assert set(out) == {"dose", "letd", "r80"}
    assert out["dose"].shape == (batch, nz)
    assert out["letd"].shape == (batch, nz)
    assert out["r80"].shape == (batch,)
    assert torch.all(out["dose"] >= 0)
    assert model.param_count() > 0
    assert model.param_count() == sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss = out["dose"].mean() + out["letd"].mean() + out["r80"].mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all().item() for g in grads)
