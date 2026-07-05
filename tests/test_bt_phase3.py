from __future__ import annotations

import torch

from braggtransporter.config import DataConfig, ModelConfig, TrainConfig
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.train import SyntheticBraggDataset, compute_loss


def _batch(batch_size: int = 2, nz: int = 17) -> dict[str, torch.Tensor]:
    ds = SyntheticBraggDataset(n_samples=batch_size, nz=nz, data_cfg=DataConfig(max_depth_cm=10.0), seed=321)
    return {k: torch.stack([ds[i][k] for i in range(batch_size)]) for k in ds[0].keys()}


def _model(quantities: list[str]) -> BraggTransporterV0:
    return BraggTransporterV0(
        ModelConfig(
            d_model=24,
            n_layers=1,
            n_heads=4,
            d_ff=48,
            dropout=0.0,
            quantities=quantities,
            extra={"max_positions": 64},
        )
    )


def test_phase3_heads_present_and_nonnegative_when_enabled() -> None:
    torch.manual_seed(0)
    model = _model(["dose", "letd", "lett", "fluence"])
    batch = _batch(batch_size=2, nz=17)

    out = model(batch["x"], batch["scalars"])

    assert hasattr(model, "lett_head")
    assert hasattr(model, "fluence_head")
    assert out["dose"].shape == (2, 17)
    assert out["letd"].shape == (2, 17)
    assert out["lett"].shape == (2, 17)
    assert out["fluence"].shape == (2, 17)
    assert out["r80"].shape == (2,)
    assert torch.all(out["lett"] >= 0)
    assert torch.all(out["fluence"] >= 0)


def test_phase3_heads_absent_when_quantities_exclude_them() -> None:
    torch.manual_seed(0)
    model = _model(["dose", "letd"])
    batch = _batch(batch_size=2, nz=17)

    out = model(batch["x"], batch["scalars"])

    assert not hasattr(model, "lett_head")
    assert not hasattr(model, "fluence_head")
    assert set(out) == {"dose", "letd", "r80"}


def test_phase3_constraints_default_zero_and_positive_when_enabled() -> None:
    z = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=torch.float32)
    x = torch.ones(2, 3, 9, dtype=torch.float32)
    batch = {
        "x": x,
        "scalars": torch.tensor([[100.0, 0.0, 0.0, 1.0], [150.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        "z": z,
        "dose": torch.zeros(2, 3, dtype=torch.float32),
        "r80": torch.tensor([10.0, 12.0], dtype=torch.float32),
    }
    outputs = {
        "dose": torch.full((2, 3), 100.0, dtype=torch.float32),
        "r80": torch.tensor([10.0, 5.0], dtype=torch.float32),
    }

    _, default_parts = compute_loss(outputs, batch, TrainConfig(device="cpu"))
    assert default_parts["monotonic_range_loss"] == 0.0
    assert default_parts["energy_budget_loss"] == 0.0

    _, constrained_parts = compute_loss(
        outputs,
        batch,
        TrainConfig(device="cpu", monotonic_range_weight=1.0, constraint_weight=1.0),
    )
    assert constrained_parts["monotonic_range_loss"] > 0.0
    assert constrained_parts["energy_budget_loss"] > 0.0
