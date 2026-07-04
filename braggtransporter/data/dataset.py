"""Torch Dataset/DataLoader access for Module-A HDF5 data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from braggtransporter.config import DataConfig


class BraggDataset(Dataset):
    """Read one split from a BraggTransporter HDF5 file.

    Returned inputs are standardized with train-split statistics stored in the
    HDF5 ``norm`` group. Targets remain in physical units.
    """

    def __init__(self, path: str | Path, split: str = "train", standardize: bool = True):
        self.path = str(path)
        self.split = split
        self.standardize = bool(standardize)
        with h5py.File(self.path, "r") as h5:
            if split not in h5:
                raise KeyError(f"Split {split!r} not found in {self.path}.")
            self._length = int(h5[split]["x"].shape[0])
            self._x_mean = h5["norm/x_mean"][...].astype(np.float32)
            self._x_std = h5["norm/x_std"][...].astype(np.float32)
            self._scalar_mean = h5["norm/scalar_mean"][...].astype(np.float32)
            self._scalar_std = h5["norm/scalar_std"][...].astype(np.float32)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with h5py.File(self.path, "r") as h5:
            g = h5[self.split]
            x = g["x"][idx].astype(np.float32)
            scalars = g["scalars"][idx].astype(np.float32)
            if self.standardize:
                x = (x - self._x_mean) / self._x_std
                scalars = (scalars - self._scalar_mean) / self._scalar_std
            return {
                "x": torch.from_numpy(x.astype(np.float32, copy=False)),
                "scalars": torch.from_numpy(scalars.astype(np.float32, copy=False)),
                "z": torch.from_numpy(g["z"][idx].astype(np.float32)),
                "dose": torch.from_numpy(g["dose"][idx].astype(np.float32)),
                "letd": torch.from_numpy(g["letd"][idx].astype(np.float32)),
                "r80": torch.tensor(g["r80"][idx], dtype=torch.float32),
                "fidelity": torch.tensor(g["fidelity"][idx], dtype=torch.float32),
            }


def make_loaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train, val, heldout_energy)`` DataLoaders for ``cfg.out_path``."""

    train = BraggDataset(cfg.out_path, "train")
    val = BraggDataset(cfg.out_path, "val")
    heldout = BraggDataset(cfg.out_path, "heldout_energy")
    return (
        DataLoader(train, batch_size=64, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=64, shuffle=False, num_workers=0),
        DataLoader(heldout, batch_size=64, shuffle=False, num_workers=0),
    )
