"""Configuration + device selection for BraggTransporter (v3.1).

Hand-written contract module. Device policy: prefer Apple MPS on this M4 Max
(48 GB unified memory — note: the plan text says 128 GB; the real machine is 48 GB,
so batch sizes are sized conservatively here), fall back to CUDA then CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DataConfig:
    energies_mev: list[float] = field(
        default_factory=lambda: [70, 80, 90, 100, 110, 120, 130, 140, 150,
                                 160, 170, 180, 190, 200, 210, 220, 230]
    )
    heldout_energies_mev: list[float] = field(default_factory=lambda: [95, 135, 175, 205])
    dz_cm: float = 0.05
    max_depth_cm: float = 34.0
    n_geometries_per_energy: int = 64   # material profiles (water + slabs) per energy
    fidelity: str = "sde"               # tier-1 default for phase 1
    seed: int = 0
    out_path: str = "data/generated/phase1_1d.h5"


@dataclass
class TrainConfig:
    model: str = "braggtransporter_v0"
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 60
    device: str = "auto"
    distal_edge_weight: float = 5.0     # up-weight the sharp falloff region in the loss
    seed: int = 0
    out_dir: str = "experiments/bt"
    grad_clip: float = 1.0
    amp: bool = False                   # MPS autocast is fragile; default off


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 256
    dropout: float = 0.0
    quantities: list[str] = field(default_factory=lambda: ["dose", "letd"])
    extra: dict[str, Any] = field(default_factory=dict)
