"""Small per-depth MLP baseline for BraggTransporter."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR


class MLPBaseline(nn.Module):
    """Shared per-depth MLP conditioned on beam/fidelity scalars."""

    def __init__(
        self,
        in_channels: int = C_IN_PERDEPTH,
        scalar_channels: int = C_SCALAR,
        hidden: int = 128,
        scalar_hidden: int = 32,
        n_hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")

        self.scalar_embed = nn.Sequential(
            nn.Linear(scalar_channels, scalar_hidden),
            nn.GELU(),
            nn.Linear(scalar_hidden, scalar_hidden),
            nn.GELU(),
        )

        layers: list[nn.Module] = []
        width_in = in_channels + scalar_hidden
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(width_in if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
        self.trunk = nn.Sequential(*layers)

        self.dose_head = nn.Linear(hidden, 1)
        self.letd_head = nn.Linear(hidden, 1)
        self.r80_head = nn.Sequential(
            nn.Linear(2 * hidden + scalar_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return dose, LETd, and R80 for ``x`` shaped ``(B, Nz, 9)``."""
        if x.ndim != 3:
            raise ValueError("x must have shape (B, Nz, C)")
        if scalars.ndim != 2:
            raise ValueError("scalars must have shape (B, C_scalar)")
        if x.shape[0] != scalars.shape[0]:
            raise ValueError("x and scalars batch dimensions must match")

        scalar_features = self.scalar_embed(scalars)
        scalar_per_depth = scalar_features[:, None, :].expand(-1, x.shape[1], -1)
        features = self.trunk(torch.cat([x, scalar_per_depth], dim=-1))

        mean_pool = features.mean(dim=1)
        max_pool = features.amax(dim=1)
        pooled = torch.cat([mean_pool, max_pool, scalar_features], dim=-1)

        return {
            "dose": F.softplus(self.dose_head(features)).squeeze(-1),
            "letd": self.letd_head(features).squeeze(-1),
            "r80": self.r80_head(pooled).squeeze(-1),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
