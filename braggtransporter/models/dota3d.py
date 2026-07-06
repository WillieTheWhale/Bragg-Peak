"""DoTA-faithful 3-D BEV depth Transformer for DoseRAD2026 beamlets."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS


class DoTA3D(nn.Module):
    """CNN slice encoder, depth Transformer, and CNN slice decoder.

    Inputs use DoseRAD BEV layout ``x:(B,C,D,H,W)``. The depth axis is treated
    as a sequence of transverse 2-D slices. Scalars are
    ``[energy_mev, layer_idx, sin(gantry), cos(gantry)]`` and form one
    conditioning token prepended to the slice-token sequence.
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
        decoder_seed_size: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if decoder_seed_size < 2:
            raise ValueError("decoder_seed_size must be >= 2.")

        self.c_in = int(c_in)
        self.d_model = int(d_model)
        self.max_depth = int(max_depth)
        self.decoder_seed_size = int(decoder_seed_size)
        enc_width = max(16, self.d_model // 2)
        dec_width = max(16, self.d_model // 2)

        self.slice_norm = nn.GroupNorm(1, self.c_in)
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(self.c_in, enc_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(enc_width, enc_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(enc_width, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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
        nn.init.trunc_normal_(self.depth_positional, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.depth_transformer = nn.TransformerEncoder(layer, num_layers=int(n_layers), enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(self.d_model)

        self.decoder_seed = nn.Sequential(
            nn.Linear(self.d_model, dec_width * self.decoder_seed_size * self.decoder_seed_size),
            nn.GELU(),
        )
        self.slice_decoder = nn.Sequential(
            nn.Conv2d(dec_width, dec_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dec_width, max(8, dec_width // 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(8, dec_width // 2), 1, kernel_size=3, padding=1),
        )
        self._init_dose_head()

    def forward(self, x: Tensor, scalars: Tensor, query_coords: Tensor | None = None) -> dict[str, Tensor]:
        if query_coords is not None:
            raise ValueError("DoTA3D decodes full BEV dose grids and does not support query_coords.")
        if x.ndim != 5 or x.shape[1] != self.c_in:
            raise ValueError(f"x must have shape (B,{self.c_in},D,H,W), got {tuple(x.shape)}")
        if scalars.ndim != 2 or scalars.shape[-1] != 4:
            raise ValueError(f"scalars must have shape (B,4), got {tuple(scalars.shape)}")
        if x.shape[0] != scalars.shape[0]:
            raise ValueError("x and scalars batch dimensions must match.")

        batch, _, depth, height, width = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, self.c_in, height, width)
        slice_tokens = self.slice_encoder(self.slice_norm(slices)).reshape(batch, depth, self.d_model)
        slice_tokens = slice_tokens + self._depth_encoding(depth, x.device, x.dtype)

        scalar_token = self.scalar_token(scalars / self.scalar_scale.to(device=scalars.device, dtype=scalars.dtype))
        tokens = torch.cat([scalar_token.unsqueeze(1), slice_tokens], dim=1)
        encoded = self.output_norm(self.depth_transformer(tokens))[:, 1:]

        decoded = self.decoder_seed(encoded.reshape(batch * depth, self.d_model))
        seed = decoded.reshape(
            batch * depth,
            -1,
            self.decoder_seed_size,
            self.decoder_seed_size,
        )
        features = F.interpolate(seed, size=(height, width), mode="bilinear", align_corners=False)
        dose = F.softplus(self.slice_decoder(features).squeeze(1))
        return {"dose": dose.reshape(batch, depth, height, width)}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _depth_encoding(self, depth: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos = self.depth_positional
        if depth != pos.shape[1]:
            pos = F.interpolate(pos.transpose(1, 2), size=depth, mode="linear", align_corners=True).transpose(1, 2)
        return pos.to(device=device, dtype=dtype)

    def _init_dose_head(self) -> None:
        head = self.slice_decoder[-1]
        if isinstance(head, nn.Conv2d):
            nn.init.constant_(head.bias, -5.0)
