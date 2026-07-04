"""DoTA-style depth-token Transformer baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR


class DoTATransformer(nn.Module):
    """Transformer encoder over depth tokens plus one scalar energy token."""

    def __init__(
        self,
        in_channels: int = C_IN_PERDEPTH,
        scalar_channels: int = C_SCALAR,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.0,
        max_depth: int = 2048,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        self.max_depth = max_depth
        self.input_proj = nn.Linear(in_channels, d_model)
        self.scalar_token = nn.Sequential(
            nn.Linear(scalar_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.positional = nn.Parameter(torch.zeros(1, max_depth + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dose_head = nn.Linear(d_model, 1)
        self.letd_head = nn.Linear(d_model, 1)
        self.r80_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return dose, LETd, and R80 for ``x`` shaped ``(B, Nz, 9)``."""
        if x.ndim != 3:
            raise ValueError("x must have shape (B, Nz, C)")
        if scalars.ndim != 2:
            raise ValueError("scalars must have shape (B, C_scalar)")
        if x.shape[0] != scalars.shape[0]:
            raise ValueError("x and scalars batch dimensions must match")
        if x.shape[1] > self.max_depth:
            raise ValueError(f"Nz={x.shape[1]} exceeds max_depth={self.max_depth}")

        depth_tokens = self.input_proj(x)
        energy_token = self.scalar_token(scalars)[:, None, :]
        tokens = torch.cat([energy_token, depth_tokens], dim=1)
        tokens = tokens + self.positional[:, : tokens.shape[1], :]

        encoded = self.norm(self.encoder(tokens))
        energy_encoded = encoded[:, 0, :]
        depth_encoded = encoded[:, 1:, :]
        pooled_depth = depth_encoded.mean(dim=1)
        pooled = torch.cat([energy_encoded, pooled_depth], dim=-1)

        return {
            "dose": F.softplus(self.dose_head(depth_encoded)).squeeze(-1),
            "letd": self.letd_head(depth_encoded).squeeze(-1),
            "r80": self.r80_head(pooled).squeeze(-1),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
