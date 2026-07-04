from __future__ import annotations

import torch

from braggtransporter.config import DataConfig, ModelConfig, TrainConfig
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.train import SyntheticBraggDataset, compute_loss


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


def _batch(batch_size: int = 2, nz: int = 31) -> dict[str, torch.Tensor]:
    ds = SyntheticBraggDataset(n_samples=batch_size, nz=nz, data_cfg=DataConfig(max_depth_cm=12.0), seed=123)
    keys = ds[0].keys()
    return {k: torch.stack([ds[i][k] for i in range(batch_size)]) for k in keys}


def test_bt_v0_forward_shapes_nonnegative_dose_and_off_grid_query() -> None:
    torch.manual_seed(0)
    model = _small_model()
    batch = _batch(batch_size=2, nz=31)

    out = model(batch["x"], batch["scalars"])
    assert out["dose"].shape == (2, 31)
    assert out["letd"].shape == (2, 31)
    assert out["r80"].shape == (2,)
    assert torch.all(out["dose"] >= 0)

    z_query = torch.tensor([0.0, 0.137, 0.503, 0.997], dtype=torch.float32)
    queried = model(batch["x"], batch["scalars"], z_query=z_query)
    assert queried["dose"].shape == (2, 4)
    assert queried["letd"].shape == (2, 4)
    assert torch.isfinite(queried["dose"]).all()
    assert torch.isfinite(queried["letd"]).all()


def test_bt_v0_two_step_training_smoke_reduces_loss() -> None:
    torch.manual_seed(1)
    model = _small_model()
    batch = _batch(batch_size=4, nz=31)
    cfg = TrainConfig(device="cpu", distal_edge_weight=2.0, grad_clip=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    initial, _ = compute_loss(model(batch["x"], batch["scalars"]), batch, cfg)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_loss(model(batch["x"], batch["scalars"]), batch, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    final, _ = compute_loss(model(batch["x"], batch["scalars"]), batch, cfg)

    assert torch.isfinite(final)
    assert final.item() < initial.item()
