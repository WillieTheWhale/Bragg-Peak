from __future__ import annotations

from pathlib import Path

import torch

from braggtransporter.config import DataConfig, ModelConfig
from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.pretrain import MaskedTransportPretrainer, PretrainConfig, load_pretrained_encoder, mask_depth_spans
from braggtransporter.train import SyntheticBraggDataset


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        d_model=24,
        n_layers=1,
        n_heads=4,
        d_ff=48,
        dropout=0.0,
        extra={"max_positions": 96},
    )


def _batch(batch_size: int = 4, nz: int = 33) -> dict[str, torch.Tensor]:
    ds = SyntheticBraggDataset(
        n_samples=batch_size,
        nz=nz,
        data_cfg=DataConfig(max_depth_cm=12.0, energies_mev=[90.0, 120.0], seed=11),
        seed=11,
    )
    keys = ds[0].keys()
    return {k: torch.stack([ds[i][k] for i in range(batch_size)]) for k in keys}


def test_mask_depth_spans_and_pretrainer_output_shapes() -> None:
    generator = torch.Generator(device="cpu").manual_seed(0)
    mask = mask_depth_spans(3, 31, 0.25, 2, 5, torch.device("cpu"), generator)
    assert mask.shape == (3, 31)
    assert mask.dtype == torch.bool
    assert torch.all(mask.sum(dim=1) > 0)

    model = MaskedTransportPretrainer(_model_cfg(), PretrainConfig(mask_ratio=0.25, mask_span_min=2, mask_span_max=5))
    batch = _batch(batch_size=3, nz=31)
    out = model(batch["x"], batch["scalars"], batch["dose"], generator=torch.Generator(device="cpu").manual_seed(1))
    assert out["x_recon"].shape == batch["x"].shape
    assert out["dose_recon"].shape == batch["dose"].shape
    assert out["next_slab"].shape == (3, 31, 2)
    assert out["x_mask"].shape == (3, 31)
    assert out["dose_mask"].shape == (3, 31)


def test_pretrain_loss_decreases_over_two_steps() -> None:
    torch.manual_seed(2)
    model = MaskedTransportPretrainer(
        _model_cfg(),
        PretrainConfig(mask_ratio=0.35, mask_span_min=2, mask_span_max=6, next_slab_weight=0.1),
    )
    batch = _batch(batch_size=4, nz=33)
    masks = model.make_masks(4, 33, torch.device("cpu"), torch.Generator(device="cpu").manual_seed(99))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    initial, _ = model.compute_loss(batch, masks=masks)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(batch, masks=masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final, _ = model.compute_loss(batch, masks=masks)

    assert torch.isfinite(final)
    assert final.item() < initial.item()


def test_pretrained_encoder_save_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    pretrainer = MaskedTransportPretrainer(_model_cfg(), PretrainConfig(mask_ratio=0.2))
    ckpt = tmp_path / "encoder.pt"
    pretrainer.save_encoder(ckpt, meta={"seed": 3})
    assert ckpt.exists()

    model = BraggTransporterV0(_model_cfg())
    info = load_pretrained_encoder(model, ckpt)
    assert info["loaded"]

    saved = pretrainer.encoder_state_dict()
    loaded = model.state_dict()
    for key, value in saved.items():
        assert key in loaded
        assert torch.allclose(loaded[key].cpu(), value)
