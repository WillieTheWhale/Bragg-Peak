"""Selective state-space 1-D backbone for BraggTransporter.

Research software only. This module implements a compact Mamba/S6-style depth
scan with input-dependent A/Delta/B/C terms and stable negative dynamics. It
follows the fixed BraggTransporter v3.1 forward contract:
``forward(x:(B,Nz,9), scalars:(B,4)) -> {"dose","letd","r80"}``.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from braggtransporter.config import ModelConfig
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR


_SCAN_CHUNK_SIZE = 64


class SelectiveSSMBlock1d(nn.Module):
    """One Mamba-style block with a selective SSM scan over depth."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_state < 1:
            raise ValueError("d_state must be >= 1")
        if expand < 1:
            raise ValueError("expand must be >= 1")
        if dt_rank < 1:
            raise ValueError("dt_rank must be >= 1")

        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.inner_dim = int(expand * d_model)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.inner_dim)
        self.param_proj = nn.Linear(self.inner_dim, dt_rank + 3 * d_state)
        self.dt_proj = nn.Linear(dt_rank, self.inner_dim)

        self.a_log = nn.Parameter(torch.empty(self.inner_dim, d_state))
        self.d_skip = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.a_log, -3.0, -1.0)
        nn.init.constant_(self.dt_proj.bias, -2.0)

    def forward(self, x: Tensor, initial_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return updated tokens and final recurrent state.

        ``initial_state`` is shaped ``(B, inner_dim, d_state)``. The recurrent
        update is:
        h_t = exp(Delta_t A) h_{t-1} + Delta_t B_t u_t
        y_t = C_t h_t + D u_t
        with A constrained negative for stable propagation.
        """

        batch, nz, _ = x.shape
        residual = x
        mixed = self.in_proj(self.norm(x))
        u, gate = mixed.chunk(2, dim=-1)
        u = F.silu(u)

        params = self.param_proj(u)
        dt_raw, a_raw, b_raw, c_raw = torch.split(
            params,
            [self.dt_proj.in_features, self.d_state, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_raw)).clamp(max=20.0)
        a_select = F.softplus(a_raw).clamp(max=5.0)
        b_select = torch.tanh(b_raw)
        c_select = torch.tanh(c_raw)

        a = -torch.exp(self.a_log).to(device=x.device, dtype=x.dtype)
        d_skip = self.d_skip.to(device=x.device, dtype=x.dtype)
        if initial_state is None:
            state = x.new_zeros(batch, self.inner_dim, self.d_state)
        else:
            state = initial_state.to(device=x.device, dtype=x.dtype)
            expected = (batch, self.inner_dim, self.d_state)
            if tuple(state.shape) != expected:
                raise ValueError(f"initial_state must have shape {expected}, got {tuple(state.shape)}")

        y, state = self._scan(delta, u, a_select, b_select, c_select, a, d_skip, state)
        y = self.out_proj(y * torch.sigmoid(gate))
        x = residual + self.dropout(y)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x, state

    def _scan(
        self,
        delta: Tensor,
        u: Tensor,
        a_select: Tensor,
        b_select: Tensor,
        c_select: Tensor,
        a: Tensor,
        d_skip: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Chunked associative prefix scan for diagonal affine SSM updates."""

        nz = delta.shape[1]
        state = initial_state
        outputs: list[Tensor] = []
        for start in range(0, nz, _SCAN_CHUNK_SIZE):
            stop = min(start + _SCAN_CHUNK_SIZE, nz)
            delta_chunk = delta[:, start:stop, :]
            u_chunk = u[:, start:stop, :]
            a_select_chunk = a_select[:, start:stop, :]
            b_select_chunk = b_select[:, start:stop, :]
            c_select_chunk = c_select[:, start:stop, :]

            decay = torch.exp(
                delta_chunk.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0) * a_select_chunk.unsqueeze(2)
            )
            drive = delta_chunk.unsqueeze(-1) * b_select_chunk.unsqueeze(2) * u_chunk.unsqueeze(-1)
            prefix_decay, prefix_drive = self._affine_prefix_scan(decay, drive)

            states = prefix_decay * state.unsqueeze(1) + prefix_drive
            y_chunk = (states * c_select_chunk.unsqueeze(2)).sum(dim=-1)
            y_chunk = y_chunk + d_skip.unsqueeze(0).unsqueeze(0) * u_chunk
            outputs.append(y_chunk)
            state = states[:, -1, :, :]

        return torch.cat(outputs, dim=1), state

    def _affine_prefix_scan(self, decay: Tensor, drive: Tensor) -> tuple[Tensor, Tensor]:
        """Inclusive prefix scan of affine transforms h -> decay * h + drive."""

        prefix_decay = decay
        prefix_drive = drive
        step = 1
        while step < decay.shape[1]:
            right_decay = prefix_decay[:, step:, :, :]
            right_drive = prefix_drive[:, step:, :, :]
            left_decay = prefix_decay[:, :-step, :, :]
            left_drive = prefix_drive[:, :-step, :, :]

            scanned_decay = right_decay * left_decay
            scanned_drive = right_drive + right_decay * left_drive
            prefix_decay = torch.cat((prefix_decay[:, :step, :, :], scanned_decay), dim=1)
            prefix_drive = torch.cat((prefix_drive[:, :step, :, :], scanned_drive), dim=1)
            step *= 2

        return prefix_decay, prefix_drive

    def _scan_reference(
        self,
        delta: Tensor,
        u: Tensor,
        a_select: Tensor,
        b_select: Tensor,
        c_select: Tensor,
        a: Tensor,
        d_skip: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Original sequential scan, retained for numerical equivalence tests."""

        state = initial_state
        outputs: list[Tensor] = []
        for idx in range(delta.shape[1]):
            delta_t = delta[:, idx, :]
            u_t = u[:, idx, :]
            a_t = a.unsqueeze(0) * a_select[:, idx, :].unsqueeze(1)
            b_t = b_select[:, idx, :]
            c_t = c_select[:, idx, :]

            decay = torch.exp(delta_t.unsqueeze(-1) * a_t)
            drive = delta_t.unsqueeze(-1) * b_t.unsqueeze(1) * u_t.unsqueeze(-1)
            state = decay * state + drive
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + d_skip.unsqueeze(0) * u_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1), state


class Mamba1d(nn.Module):
    """Selective SSM backbone for 1-D Bragg depth-dose prediction."""

    def __init__(
        self,
        cfg: ModelConfig | dict[str, Any] | None = None,
        *,
        d_model: int | None = None,
        n_layers: int | None = None,
        d_ff: int | None = None,
        dropout: float | None = None,
        d_state: int | None = None,
        expand: int | None = None,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        params = _model_params(cfg)
        extra = params.get("extra", {}) or {}
        self.d_model = int(d_model if d_model is not None else params.get("d_model", 128))
        self.n_layers = int(n_layers if n_layers is not None else params.get("n_layers", 4))
        self.d_ff = int(d_ff if d_ff is not None else params.get("d_ff", 256))
        self.dropout = float(dropout if dropout is not None else params.get("dropout", 0.0))
        self.d_state = int(d_state if d_state is not None else extra.get("d_state", 16))
        self.expand = int(expand if expand is not None else extra.get("expand", 2))
        default_dt_rank = max(8, self.d_model // 8)
        self.dt_rank = int(dt_rank if dt_rank is not None else extra.get("dt_rank", default_dt_rank))
        self.inner_dim = self.d_model * self.expand

        self.input_norm = nn.LayerNorm(C_IN_PERDEPTH)
        self.input_proj = nn.Linear(C_IN_PERDEPTH, self.d_model)
        self.register_buffer(
            "scalar_scale",
            torch.tensor([250.0, 10.0, 5.0, 4.0], dtype=torch.float32),
            persistent=False,
        )
        self.scalar_bias = nn.Sequential(
            nn.Linear(C_SCALAR, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.scalar_state = nn.ModuleList(
            nn.Linear(C_SCALAR, self.inner_dim * self.d_state) for _ in range(self.n_layers)
        )

        self.blocks = nn.ModuleList(
            SelectiveSSMBlock1d(
                self.d_model,
                self.d_ff,
                d_state=self.d_state,
                expand=self.expand,
                dt_rank=self.dt_rank,
                dropout=self.dropout,
            )
            for _ in range(self.n_layers)
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.point_decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.dose_head = nn.Linear(self.d_ff, 1)
        self.letd_head = nn.Linear(self.d_ff, 1)
        self.r80_head = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, 1),
        )

    def forward(self, x: Tensor, scalars: Tensor) -> dict[str, Tensor]:
        """Predict dose, LETd, and R80 from per-depth features and beam scalars."""

        if x.ndim != 3 or x.shape[-1] != C_IN_PERDEPTH:
            raise ValueError(f"x must have shape (B,Nz,{C_IN_PERDEPTH}), got {tuple(x.shape)}")
        if scalars.ndim != 2 or scalars.shape[-1] != C_SCALAR:
            raise ValueError(f"scalars must have shape (B,{C_SCALAR}), got {tuple(scalars.shape)}")
        if x.shape[0] != scalars.shape[0]:
            raise ValueError("x and scalars batch dimensions must match")

        scalar_features = scalars / self.scalar_scale.to(device=scalars.device, dtype=scalars.dtype)
        tokens = self.input_proj(self.input_norm(x))
        tokens = tokens + self.scalar_bias(scalar_features).unsqueeze(1)

        for layer_idx, block in enumerate(self.blocks):
            init_state = torch.tanh(self.scalar_state[layer_idx](scalar_features))
            init_state = init_state.view(x.shape[0], self.inner_dim, self.d_state)
            tokens, _ = block(tokens, init_state)

        latent = self.output_norm(tokens)
        decoded = self.point_decoder(latent)
        dose = F.softplus(self.dose_head(decoded)).squeeze(-1)
        letd = self.letd_head(decoded).squeeze(-1)

        pooled_mean = latent.mean(dim=1)
        pooled_edge = latent[:, -max(1, latent.shape[1] // 8) :, :].mean(dim=1)
        r80 = F.softplus(self.r80_head(torch.cat([pooled_mean, pooled_edge], dim=-1)).squeeze(-1)) * 40.0
        return {"dose": dose, "letd": letd, "r80": r80}

    def param_count(self) -> int:
        """Number of trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _model_params(cfg: ModelConfig | dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return asdict(ModelConfig())
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)
