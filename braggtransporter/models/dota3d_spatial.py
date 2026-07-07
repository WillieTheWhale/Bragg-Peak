"""Spatially preserving DoTA-style 3-D BEV dose Transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS


class FactorizedDepthSpatialBlock(nn.Module):
    """Attention along beam depth, followed by attention across lateral patches."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.depth_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, tokens: Tensor, scalar_token: Tensor) -> Tensor:
        batch, depth, patches, channels = tokens.shape

        depth_tokens = tokens.permute(0, 2, 1, 3).reshape(batch * patches, depth, channels)
        depth_scalar = (
            scalar_token[:, None, :]
            .expand(batch, patches, channels)
            .reshape(batch * patches, 1, channels)
        )
        depth_tokens = self.depth_layer(torch.cat([depth_scalar, depth_tokens], dim=1))[:, 1:]
        tokens = depth_tokens.reshape(batch, patches, depth, channels).permute(0, 2, 1, 3).contiguous()

        spatial_tokens = tokens.reshape(batch * depth, patches, channels)
        spatial_scalar = (
            scalar_token[:, None, :]
            .expand(batch, depth, channels)
            .reshape(batch * depth, 1, channels)
        )
        spatial_tokens = self.spatial_layer(torch.cat([spatial_scalar, spatial_tokens], dim=1))[:, 1:]
        return spatial_tokens.reshape(batch, depth, patches, channels)


class DoTA3DSpatial(nn.Module):
    """Patch-grid DoTA variant that preserves transverse dose position.

    Inputs use DoseRAD BEV layout ``x:(B,C,D,H,W)``. Each depth slice is embedded
    into a lateral patch grid, then factorized Transformer blocks mix dose
    context along both depth and lateral patch axes. Scalars are
    ``[energy_mev, layer_idx, sin(gantry), cos(gantry)]`` and are prepended as
    conditioning tokens inside each depth and spatial attention pass.
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
        patch_size: int = 4,
        max_lateral_patches: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1.")
        if max_lateral_patches < 1:
            raise ValueError("max_lateral_patches must be >= 1.")

        self.c_in = int(c_in)
        self.d_model = int(d_model)
        self.max_depth = int(max_depth)
        self.patch_size = int(patch_size)
        self.max_lateral_patches = int(max_lateral_patches)

        dec_width = max(16, self.d_model // 2)
        self.slice_norm = nn.GroupNorm(1, self.c_in)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                self.c_in,
                self.d_model,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),
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
        self.depth_positional = nn.Parameter(torch.zeros(1, self.max_depth, self.d_model))
        self.lateral_positional = nn.Parameter(
            torch.zeros(1, self.max_lateral_patches, self.max_lateral_patches, self.d_model)
        )
        nn.init.trunc_normal_(self.depth_positional, std=0.02)
        nn.init.trunc_normal_(self.lateral_positional, std=0.02)

        self.blocks = nn.ModuleList(
            [
                FactorizedDepthSpatialBlock(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    d_ff=int(d_ff),
                    dropout=float(dropout),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.d_model)

        self.slice_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.d_model,
                dec_width,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            nn.GELU(),
            nn.Conv2d(dec_width, dec_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dec_width, max(8, dec_width // 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(8, dec_width // 2), 1, kernel_size=3, padding=1),
        )
        self.local_skip = nn.Conv2d(self.c_in, 1, kernel_size=1)
        self._init_dose_head()

    def forward(self, x: Tensor, scalars: Tensor, query_coords: Tensor | None = None) -> dict[str, Tensor]:
        if query_coords is not None:
            raise ValueError("DoTA3DSpatial decodes full BEV dose grids and does not support query_coords.")
        if x.ndim != 5 or x.shape[1] != self.c_in:
            raise ValueError(f"x must have shape (B,{self.c_in},D,H,W), got {tuple(x.shape)}")
        if scalars.ndim != 2 or scalars.shape[-1] != 4:
            raise ValueError(f"scalars must have shape (B,4), got {tuple(scalars.shape)}")
        if x.shape[0] != scalars.shape[0]:
            raise ValueError("x and scalars batch dimensions must match.")

        batch, _, depth, height, width = x.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"height and width must be >= patch_size={self.patch_size}, got H={height}, W={width}"
            )

        slices = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, self.c_in, height, width)
        embedded = self.patch_embed(self.slice_norm(slices))
        _, _, patch_h, patch_w = embedded.shape

        tokens = embedded.reshape(batch, depth, self.d_model, patch_h, patch_w)
        tokens = tokens.permute(0, 1, 3, 4, 2).reshape(batch, depth, patch_h * patch_w, self.d_model)
        tokens = tokens + self._depth_encoding(depth, x.device, x.dtype)
        tokens = tokens + self._lateral_encoding(patch_h, patch_w, x.device, x.dtype)

        scalar_token = self.scalar_token(scalars / self.scalar_scale.to(device=scalars.device, dtype=scalars.dtype))
        for block in self.blocks:
            tokens = block(tokens, scalar_token)
        tokens = self.output_norm(tokens)

        features = tokens.reshape(batch, depth, patch_h, patch_w, self.d_model)
        features = features.permute(0, 1, 4, 2, 3).reshape(batch * depth, self.d_model, patch_h, patch_w)
        logits = self.slice_decoder(features).squeeze(1)
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits[:, None], size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
        skip = self.local_skip(slices).squeeze(1)
        dose = F.softplus(logits + skip)
        return {"dose": dose.reshape(batch, depth, height, width)}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _depth_encoding(self, depth: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos = self.depth_positional
        if depth != pos.shape[1]:
            pos = F.interpolate(pos.transpose(1, 2), size=depth, mode="linear", align_corners=True).transpose(1, 2)
        return pos[:, :, None, :].to(device=device, dtype=dtype)

    def _lateral_encoding(self, patch_h: int, patch_w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos = self.lateral_positional.permute(0, 3, 1, 2)
        if (patch_h, patch_w) != pos.shape[-2:]:
            pos = F.interpolate(pos, size=(patch_h, patch_w), mode="bilinear", align_corners=True)
        pos = pos.permute(0, 2, 3, 1).reshape(1, 1, patch_h * patch_w, self.d_model)
        return pos.to(device=device, dtype=dtype)

    def _init_dose_head(self) -> None:
        head = self.slice_decoder[-1]
        if isinstance(head, nn.Conv2d):
            nn.init.constant_(head.bias, -2.0)
        nn.init.zeros_(self.local_skip.bias)
        with torch.no_grad():
            self.local_skip.weight.zero_()
            self.local_skip.weight[:, 0, :, :] = 0.1
