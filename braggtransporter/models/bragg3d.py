"""Small 3-D BraggTransporter for DoseRAD2026 BEV beamlets."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS


class Bragg3D(nn.Module):
    """Depth-token transformer with a 3-D coordinate-query dose decoder.

    Inputs use BEV tensor layout ``x:(B,C,D,H,W)`` with depth along axis 2.
    Scalars are ``[energy_mev, layer_idx, sin(gantry), cos(gantry)]``. Dose is
    nonnegative by a softplus head and is returned on the input grid unless
    explicit normalized query coordinates are provided.
    """

    def __init__(
        self,
        *,
        c_in: int = DOSERAD_INPUT_CHANNELS,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.0,
        max_depth: int = 128,
    ) -> None:
        super().__init__()
        self.c_in = int(c_in)
        self.d_model = int(d_model)
        self.max_depth = int(max_depth)

        self.slice_norm = nn.LayerNorm(self.c_in)
        self.slice_embed = nn.Sequential(
            nn.Conv2d(self.c_in, self.d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.d_model // 2, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.lateral_flat_embed = nn.LazyLinear(self.d_model)
        self.depth_fuse = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
        )

        self.register_buffer(
            "scalar_scale",
            torch.tensor([250.0, 64.0, 1.0, 1.0], dtype=torch.float32),
            persistent=False,
        )
        self.scalar_token = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_depth, self.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers), enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(self.d_model)

        self.query_decoder = nn.Sequential(
            nn.Linear(self.d_model + 3, int(d_ff)),
            nn.GELU(),
            nn.Linear(int(d_ff), int(d_ff) // 2),
            nn.GELU(),
        )
        self.dose_head = nn.Linear(int(d_ff) // 2, 1)
        nn.init.constant_(self.dose_head.bias, -8.0)

    def forward(self, x: Tensor, scalars: Tensor, query_coords: Tensor | None = None) -> dict[str, Tensor]:
        if x.ndim != 5 or x.shape[1] != self.c_in:
            raise ValueError(f"x must have shape (B,{self.c_in},D,H,W), got {tuple(x.shape)}")
        if scalars.ndim != 2 or scalars.shape[-1] != 4:
            raise ValueError(f"scalars must have shape (B,4), got {tuple(scalars.shape)}")

        batch, channels, depth, height, width = x.shape
        slices = x.permute(0, 2, 3, 4, 1).reshape(batch * depth, height, width, channels)
        slices = self.slice_norm(slices).permute(0, 3, 1, 2).contiguous()
        conv_embed = self.slice_embed(slices).reshape(batch, depth, self.d_model)
        flat_embed = self.lateral_flat_embed(slices.reshape(batch, depth, -1))
        depth_tokens = self.depth_fuse(torch.cat([conv_embed, flat_embed], dim=-1))
        depth_tokens = depth_tokens + self._positional_encoding(depth, x.device, x.dtype)

        scalar_token = self.scalar_token(scalars / self.scalar_scale.to(device=scalars.device, dtype=scalars.dtype))
        tokens = torch.cat([scalar_token.unsqueeze(1), depth_tokens], dim=1)
        encoded = self.encoder_norm(self.encoder(tokens))[:, 1:]

        coords = self._query_coords(query_coords, batch, depth, height, width, x.device, x.dtype)
        latent = self._interpolate_depth(encoded, coords[..., 0])
        decoded = self.query_decoder(torch.cat([latent, coords], dim=-1))
        dose = F.softplus(self.dose_head(decoded).squeeze(-1))
        if query_coords is None:
            dose = dose.reshape(batch, depth, height, width)
        return {"dose": dose}

    def param_count(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter)
        )

    def _positional_encoding(self, depth: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos = self.pos_embedding
        if depth != pos.shape[1]:
            pos = F.interpolate(pos.transpose(1, 2), size=depth, mode="linear", align_corners=True).transpose(1, 2)
        return pos.to(device=device, dtype=dtype)

    @staticmethod
    def _query_coords(
        query_coords: Tensor | None,
        batch: int,
        depth: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if query_coords is not None:
            coords = query_coords.to(device=device, dtype=dtype)
            if coords.ndim == 2:
                coords = coords.unsqueeze(0).expand(batch, -1, -1)
            if coords.ndim != 3 or coords.shape[0] != batch or coords.shape[-1] != 3:
                raise ValueError(f"query_coords must have shape (N,3) or ({batch},N,3), got {tuple(coords.shape)}")
            return coords.clamp(0.0, 1.0)

        z = torch.linspace(0.0, 1.0, depth, device=device, dtype=dtype)
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([zz, yy, xx], dim=-1).reshape(1, depth * height * width, 3)
        return coords.expand(batch, -1, -1)

    @staticmethod
    def _interpolate_depth(depth_latent: Tensor, z_coords: Tensor) -> Tensor:
        batch, depth, channels = depth_latent.shape
        pos = z_coords.clamp(0.0, 1.0) * max(depth - 1, 0)
        lower_idx = pos.floor().to(torch.long).clamp(0, max(depth - 1, 0))
        upper_idx = (lower_idx + 1).clamp(0, max(depth - 1, 0))
        upper_weight = (pos - lower_idx.to(pos.dtype)).unsqueeze(-1)
        lower_weight = 1.0 - upper_weight
        lower = torch.gather(depth_latent, 1, lower_idx.unsqueeze(-1).expand(-1, -1, channels))
        upper = torch.gather(depth_latent, 1, upper_idx.unsqueeze(-1).expand(-1, -1, channels))
        return lower * lower_weight + upper * upper_weight
