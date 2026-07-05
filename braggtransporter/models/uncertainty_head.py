"""Residual uncertainty heads for frozen BraggTransporter-v0 predictions.

Research software only. Inputs keep the v3.1 units: depth in cm, energy in MeV,
and dose/residuals in MeV/g per voxel. These modules do not modify or wrap the
deterministic v0 model; they learn a residual distribution conditioned on the
per-depth input tensor and the frozen v0 mean dose prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR


@dataclass(frozen=True)
class UncertaintyHeadConfig:
    """Small MPS-safe head configuration."""

    hidden: int = 64
    depth: int = 2
    min_sigma: float = 1.0e-5
    max_logvar: float = 8.0
    use_scalars: bool = True


class _ResidualConditioner(nn.Module):
    def __init__(self, cfg: UncertaintyHeadConfig, extra_channels: int) -> None:
        super().__init__()
        in_channels = C_IN_PERDEPTH + 1 + extra_channels
        if cfg.use_scalars:
            in_channels += C_SCALAR
        layers: list[nn.Module] = []
        width = int(cfg.hidden)
        for i in range(max(1, int(cfg.depth))):
            layers.append(nn.Linear(in_channels if i == 0 else width, width))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.use_scalars = bool(cfg.use_scalars)

    def forward(
        self,
        x: Tensor,
        v0_mean: Tensor,
        scalars: Tensor | None = None,
        extra: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != C_IN_PERDEPTH:
            raise ValueError(f"x must have shape (B,Nz,{C_IN_PERDEPTH}), got {tuple(x.shape)}")
        if v0_mean.shape != x.shape[:2]:
            raise ValueError(f"v0_mean must have shape {tuple(x.shape[:2])}, got {tuple(v0_mean.shape)}")

        cols = [x, v0_mean.unsqueeze(-1)]
        if extra is not None:
            if extra.shape[:2] != x.shape[:2]:
                raise ValueError("extra conditioning must match x batch/depth dimensions")
            cols.append(extra)
        if self.use_scalars:
            if scalars is None:
                scalars = x.new_zeros((x.shape[0], C_SCALAR))
            if scalars.ndim != 2 or scalars.shape[-1] != C_SCALAR:
                raise ValueError(f"scalars must have shape (B,{C_SCALAR}), got {tuple(scalars.shape)}")
            cols.append(scalars.to(device=x.device, dtype=x.dtype).unsqueeze(1).expand(-1, x.shape[1], -1))
        return self.net(torch.cat(cols, dim=-1))


class HeteroscedasticResidualHead(nn.Module):
    """Gaussian residual head predicting per-voxel mean and log-variance."""

    def __init__(self, cfg: UncertaintyHeadConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UncertaintyHeadConfig()
        self.conditioner = _ResidualConditioner(self.cfg, extra_channels=0)
        self.out = nn.Linear(int(self.cfg.hidden), 2)

    def forward(self, x: Tensor, v0_mean: Tensor, scalars: Tensor | None = None) -> dict[str, Tensor]:
        h = self.conditioner(x, v0_mean, scalars)
        mean, raw_logvar = self.out(h).unbind(dim=-1)
        logvar = raw_logvar.clamp(max=float(self.cfg.max_logvar))
        sigma = torch.exp(0.5 * logvar).clamp_min(float(self.cfg.min_sigma))
        return {"residual_mean": mean, "logvar": logvar, "sigma": sigma}

    def nll_loss(self, x: Tensor, v0_mean: Tensor, residual: Tensor, scalars: Tensor | None = None) -> Tensor:
        """Gaussian negative log likelihood for ``residual = target - v0_mean``."""

        out = self.forward(x, v0_mean, scalars)
        inv_var = torch.exp(-out["logvar"])
        nll = 0.5 * ((residual - out["residual_mean"]).square() * inv_var + out["logvar"])
        return torch.mean(torch.nan_to_num(nll, nan=0.0, posinf=1.0e6, neginf=1.0e6))


class ConditionalFlowMatchingResidualHead(nn.Module):
    """Few-step conditional flow-matching head for residual distributions.

    The learned vector field transports standard normal residual noise at
    ``t=0`` to observed residual samples at ``t=1``. Sampling several residual
    paths and taking their empirical standard deviation gives the aleatoric dose
    sigma channel.
    """

    def __init__(self, cfg: UncertaintyHeadConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UncertaintyHeadConfig()
        self.conditioner = _ResidualConditioner(self.cfg, extra_channels=2)
        self.velocity = nn.Linear(int(self.cfg.hidden), 1)

    def vector_field(
        self,
        x: Tensor,
        v0_mean: Tensor,
        residual_state: Tensor,
        t: Tensor,
        scalars: Tensor | None = None,
    ) -> Tensor:
        if residual_state.shape != v0_mean.shape:
            raise ValueError("residual_state must match v0_mean shape")
        if t.ndim == 0:
            t_feat = t.to(device=x.device, dtype=x.dtype).expand_as(residual_state)
        elif t.ndim == 1:
            t_feat = t.to(device=x.device, dtype=x.dtype).view(-1, 1).expand_as(residual_state)
        elif t.ndim == 2 and t.shape[1] == 1:
            t_feat = t.to(device=x.device, dtype=x.dtype).expand_as(residual_state)
        else:
            t_feat = t.to(device=x.device, dtype=x.dtype)
        extra = torch.stack([residual_state, t_feat], dim=-1)
        h = self.conditioner(x, v0_mean, scalars, extra=extra)
        return self.velocity(h).squeeze(-1)

    def flow_matching_loss(
        self,
        x: Tensor,
        v0_mean: Tensor,
        residual: Tensor,
        scalars: Tensor | None = None,
        *,
        noise: Tensor | None = None,
        t: Tensor | None = None,
    ) -> Tensor:
        """Conditional flow-matching MSE toward the residual transport velocity."""

        if residual.shape != v0_mean.shape:
            raise ValueError("residual must match v0_mean shape")
        z0 = torch.randn_like(residual) if noise is None else noise.to(device=residual.device, dtype=residual.dtype)
        if t is None:
            t = torch.rand((residual.shape[0], 1), device=residual.device, dtype=residual.dtype)
        else:
            t = t.to(device=residual.device, dtype=residual.dtype)
            if t.ndim == 1:
                t = t.view(-1, 1)
        xt = (1.0 - t) * z0 + t * residual
        target_velocity = residual - z0
        pred_velocity = self.vector_field(x, v0_mean, xt, t, scalars)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample_residuals(
        self,
        x: Tensor,
        v0_mean: Tensor,
        scalars: Tensor | None = None,
        *,
        n_samples: int = 16,
        n_steps: int = 4,
    ) -> Tensor:
        """Draw residual samples with explicit Euler integration.

        Returns a tensor with shape ``(n_samples, B, Nz)``.
        """

        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        samples: list[Tensor] = []
        dt = 1.0 / float(n_steps)
        for _ in range(int(n_samples)):
            state = torch.randn_like(v0_mean)
            for step in range(int(n_steps)):
                t = x.new_full((x.shape[0], 1), (float(step) + 0.5) * dt)
                state = state + dt * self.vector_field(x, v0_mean, state, t, scalars)
            samples.append(state)
        return torch.stack(samples, dim=0)

    @torch.no_grad()
    def predictive_sigma(
        self,
        x: Tensor,
        v0_mean: Tensor,
        scalars: Tensor | None = None,
        *,
        n_samples: int = 16,
        n_steps: int = 4,
    ) -> Tensor:
        residuals = self.sample_residuals(x, v0_mean, scalars, n_samples=n_samples, n_steps=n_steps)
        if residuals.shape[0] == 1:
            return torch.zeros_like(v0_mean).clamp_min(float(self.cfg.min_sigma))
        return residuals.std(dim=0, unbiased=False).clamp_min(float(self.cfg.min_sigma))
