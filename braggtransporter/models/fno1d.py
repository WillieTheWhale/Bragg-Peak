"""One-dimensional Fourier Neural Operator baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR


class SpectralConv1d(nn.Module):
    """1-D spectral convolution with truncated real FFT modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if modes < 1:
            raise ValueError("modes must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, n_depth = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_freq = x_ft.shape[-1]
        n_modes = min(self.modes, n_freq)

        out_ft = x_ft.new_zeros(batch, self.out_channels, n_freq)
        weight = torch.complex(
            self.weight_real[:, :, :n_modes],
            self.weight_imag[:, :, :n_modes],
        )
        out_ft[:, :, :n_modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :n_modes], weight)
        return torch.fft.irfft(out_ft, n=n_depth, dim=-1)


class FNOBlock1d(nn.Module):
    """FNO layer: spectral convolution plus learned pointwise channel mixing."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class FNO1d(nn.Module):
    """Compact 1-D Fourier Neural Operator conditioned on beam scalars."""

    def __init__(
        self,
        in_channels: int = C_IN_PERDEPTH,
        scalar_channels: int = C_SCALAR,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
        scalar_width: int = 8,
        projection_width: int = 128,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.scalar_lift = nn.Sequential(
            nn.Linear(scalar_channels, projection_width),
            nn.GELU(),
            nn.Linear(projection_width, scalar_width),
        )
        self.lift = nn.Sequential(
            nn.Linear(in_channels + scalar_width, projection_width),
            nn.GELU(),
            nn.Linear(projection_width, width),
        )
        self.blocks = nn.ModuleList(FNOBlock1d(width, modes) for _ in range(n_layers))
        self.projection = nn.Sequential(
            nn.Linear(width, projection_width),
            nn.GELU(),
        )
        self.dose_head = nn.Linear(projection_width, 1)
        self.letd_head = nn.Linear(projection_width, 1)
        self.r80_head = nn.Sequential(
            nn.Linear(2 * projection_width + scalar_width, projection_width),
            nn.GELU(),
            nn.Linear(projection_width, 1),
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return dose, LETd, and R80 for ``x`` shaped ``(B, Nz, 9)``."""
        if x.ndim != 3:
            raise ValueError("x must have shape (B, Nz, C)")
        if scalars.ndim != 2:
            raise ValueError("scalars must have shape (B, C_scalar)")
        if x.shape[0] != scalars.shape[0]:
            raise ValueError("x and scalars batch dimensions must match")

        scalar_features = self.scalar_lift(scalars)
        scalar_per_depth = scalar_features[:, None, :].expand(-1, x.shape[1], -1)
        lifted = self.lift(torch.cat([x, scalar_per_depth], dim=-1))

        features = lifted.transpose(1, 2)
        for block in self.blocks:
            features = block(features)
        features = features.transpose(1, 2)
        projected = self.projection(features)

        mean_pool = projected.mean(dim=1)
        max_pool = projected.amax(dim=1)
        pooled = torch.cat([mean_pool, max_pool, scalar_features], dim=-1)

        return {
            "dose": F.softplus(self.dose_head(projected)).squeeze(-1),
            "letd": self.letd_head(projected).squeeze(-1),
            "r80": self.r80_head(pooled).squeeze(-1),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
