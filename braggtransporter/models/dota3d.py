"""Published DoTA-style 3-D BEV depth Transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS


class DoTA3D(nn.Module):
    """Shared 2-D CNN encoder, causal depth Transformer, and CNN decoder.

    The architecture follows Pastor-Serrano and Perko (2022): every transverse
    slice is encoded to one spatially flattened token, an energy token is
    prepended, causal self-attention operates along beam depth, and each output
    token is decoded independently to a dose slice. Inputs use DoseRAD BEV axis
    order ``x:(B,C,D,H,W)`` and scalars are
    ``[energy_MeV, layer_idx, sin(gantry), cos(gantry)]``.
    """

    def __init__(
        self,
        *,
        c_in: int = DOSERAD_INPUT_CHANNELS,
        d_model: int = 432,
        n_layers: int = 1,
        n_heads: int = 16,
        d_ff: int = 432,
        dropout: float = 0.2,
        max_depth: int = 201,
        lateral_size: int = 24,
        encoder_channels: int = 12,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if lateral_size < 4:
            raise ValueError("lateral_size must be >= 4.")
        if encoder_channels < 4 or encoder_channels % 4 != 0:
            raise ValueError("encoder_channels must be >= 4 and divisible by 4.")

        self.c_in = int(c_in)
        self.d_model = int(d_model)
        self.max_depth = int(max_depth)
        self.lateral_size = int(lateral_size)
        self.encoder_channels = int(encoder_channels)
        self.encoded_lateral_size = self.lateral_size // 4
        flattened_dim = self.encoder_channels * self.encoded_lateral_size**2

        mean = torch.zeros(self.c_in, dtype=torch.float32)
        std = torch.ones(self.c_in, dtype=torch.float32)
        fixed_mean = torch.tensor([0.0, 1.0, 1.0, 0.5], dtype=torch.float32)
        fixed_std = torch.tensor([0.6, 0.4, 0.4, 0.3], dtype=torch.float32)
        n_fixed = min(self.c_in, fixed_mean.numel())
        mean[:n_fixed] = fixed_mean[:n_fixed]
        std[:n_fixed] = fixed_std[:n_fixed]
        self.register_buffer("input_mean", mean.view(1, self.c_in, 1, 1), persistent=False)
        self.register_buffer("input_std", std.view(1, self.c_in, 1, 1), persistent=False)

        # The paper uses 64, 64, and 12 filters with two 2x downsamplings.
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(self.c_in, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(16, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(16, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, self.encoder_channels, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(4, self.encoder_channels),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.token_projection = (
            nn.Identity() if flattened_dim == self.d_model else nn.Linear(flattened_dim, self.d_model)
        )
        self.decoder_projection = (
            nn.Identity() if flattened_dim == self.d_model else nn.Linear(self.d_model, flattened_dim)
        )

        self.register_buffer("energy_min_mev", torch.tensor(31.73), persistent=False)
        self.register_buffer("energy_span_mev", torch.tensor(200.80 - 31.73), persistent=False)
        self.energy_token = nn.Linear(1, self.d_model)
        self.positional = nn.Parameter(torch.zeros(1, self.max_depth + 1, self.d_model))
        nn.init.trunc_normal_(self.positional, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.depth_transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(n_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.d_model)

        self.slice_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.encoder_channels,
                64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
        )
        nn.init.constant_(self.slice_decoder[-1].bias, -5.0)

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
        if depth > self.max_depth:
            raise ValueError(f"depth={depth} exceeds max_depth={self.max_depth}.")
        if height != self.lateral_size or width != self.lateral_size:
            raise ValueError(
                f"DoTA3D was configured for {self.lateral_size}x{self.lateral_size} lateral grids, "
                f"got {height}x{width}."
            )

        slices = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, self.c_in, height, width)
        standardized = (slices - self.input_mean.to(slices)) / self.input_std.to(slices)
        encoded_slices = self.slice_encoder(standardized)
        slice_tokens = self.token_projection(encoded_slices).reshape(batch, depth, self.d_model)

        energy = scalars[:, :1]
        energy = (energy - self.energy_min_mev.to(energy)) / self.energy_span_mev.to(energy)
        tokens = torch.cat([self.energy_token(energy).unsqueeze(1), slice_tokens], dim=1)
        tokens = tokens + self.positional[:, : depth + 1].to(device=x.device, dtype=x.dtype)
        causal_mask = torch.triu(
            torch.ones(depth + 1, depth + 1, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        transformed = self.output_norm(self.depth_transformer(tokens, mask=causal_mask))[:, 1:]

        decoded = self.decoder_projection(transformed.reshape(batch * depth, self.d_model))
        decoded = decoded.reshape(
            batch * depth,
            self.encoder_channels,
            self.encoded_lateral_size,
            self.encoded_lateral_size,
        )
        logits = self.slice_decoder(decoded)
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        dose = F.softplus(logits.squeeze(1))
        return {"dose": dose.reshape(batch, depth, height, width)}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
