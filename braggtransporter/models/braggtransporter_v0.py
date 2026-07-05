"""BraggTransporter v0 model.

Research software only. This model follows the v3.1 tensor contract:
``forward(x:(B,Nz,9), scalars:(B,4))`` returns dose, LETd, and R80 predictions.
When requested by ``ModelConfig.quantities``, the coordinate decoder also returns
nonnegative LETt and fluence profiles. Defaults intentionally keep the Phase 1/2
output and parameter surface unchanged.
Depth is in cm in the data contract; optional coordinate queries are normalized
to [0, 1] unless physical coordinates outside that range are supplied, in which
case they are normalized by their per-sample maximum.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from braggtransporter.config import ModelConfig
from braggtransporter.schema import C_IN_PERDEPTH, C_SCALAR, QUANTITIES


class CalibrationWrapper(nn.Module):
    """Identity hook for later calibration layers."""

    def forward(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return outputs


class BraggTransporterV0(nn.Module):
    """Small transformer encoder with a coordinate-query decoder."""

    def __init__(
        self,
        cfg: ModelConfig | None = None,
        *,
        d_model: int | None = None,
        n_layers: int | None = None,
        n_heads: int | None = None,
        d_ff: int | None = None,
        dropout: float | None = None,
        max_positions: int | None = None,
    ) -> None:
        super().__init__()
        params = _model_params(cfg)
        self.d_model = int(d_model if d_model is not None else params.get("d_model", 128))
        self.n_layers = int(n_layers if n_layers is not None else params.get("n_layers", 4))
        self.n_heads = int(n_heads if n_heads is not None else params.get("n_heads", 8))
        self.d_ff = int(d_ff if d_ff is not None else params.get("d_ff", 256))
        self.dropout = float(dropout if dropout is not None else params.get("dropout", 0.0))
        extra = params.get("extra", {}) or {}
        quantities = params.get("quantities", ["dose", "letd"]) or []
        unknown = sorted(set(quantities) - set(QUANTITIES))
        if unknown:
            raise ValueError(f"unknown ModelConfig.quantities entries: {unknown}")
        self.quantities = tuple(str(q) for q in quantities)
        self.predict_lett = "lett" in self.quantities
        self.predict_fluence = "fluence" in self.quantities
        self.decoder_mode = str(extra.get("decoder_mode", "coord_query"))
        if self.decoder_mode not in {"coord_query", "fixed_grid"}:
            raise ValueError("ModelConfig.extra['decoder_mode'] must be 'coord_query' or 'fixed_grid'")
        self.use_physics_prior = bool(extra.get("use_physics_prior", True))
        self.max_positions = int(max_positions if max_positions is not None else extra.get("max_positions", 2048))

        self.input_norm = nn.LayerNorm(C_IN_PERDEPTH)
        self.input_proj = nn.Linear(C_IN_PERDEPTH, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_positions, self.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.register_buffer(
            "scalar_scale",
            torch.tensor([250.0, 10.0, 5.0, 4.0], dtype=torch.float32),
            persistent=False,
        )
        self.scalar_token = nn.Sequential(
            nn.Linear(C_SCALAR, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(self.d_model)

        if self.decoder_mode == "coord_query":
            self.query_decoder = nn.Sequential(
                nn.Linear(self.d_model + 1, self.d_ff),
                nn.GELU(),
                nn.Linear(self.d_ff, self.d_ff // 2),
                nn.GELU(),
            )
        else:
            self.fixed_grid_decoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Linear(self.d_ff, self.d_ff // 2),
                nn.GELU(),
            )
        hidden = self.d_ff // 2
        self.dose_head = nn.Linear(hidden, 1)
        self.letd_head = nn.Linear(hidden, 1)
        if self.predict_lett:
            self.lett_head = nn.Linear(hidden, 1)
        if self.predict_fluence:
            self.fluence_head = nn.Linear(hidden, 1)
        self.r80_head = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, 1),
        )
        self.calibration = CalibrationWrapper()

    def forward(
        self,
        x: Tensor,
        scalars: Tensor,
        z_query: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict dose/LETd/R80, with optional LETt/fluence heads.

        ``z_query`` is optional and may be ``(Nq,)`` or ``(B,Nq)``. Values in
        [0, 1] are treated as normalized depth coordinates. Values outside that
        range are interpreted as physical coordinates and normalized by the
        per-sample maximum query coordinate before interpolation.
        """

        if x.ndim != 3 or x.shape[-1] != C_IN_PERDEPTH:
            raise ValueError(f"x must have shape (B,Nz,{C_IN_PERDEPTH}), got {tuple(x.shape)}")
        if scalars.ndim != 2 or scalars.shape[-1] != C_SCALAR:
            raise ValueError(f"scalars must have shape (B,{C_SCALAR}), got {tuple(scalars.shape)}")

        if not self.use_physics_prior:
            x = x.clone()
            x[..., 4:9] = 0.0

        batch, nz, _ = x.shape
        depth_tokens = self.input_proj(self.input_norm(x))
        depth_tokens = depth_tokens + self._positional_encoding(nz, x.device, x.dtype)

        scalar_token = self.scalar_token(scalars / self.scalar_scale.to(device=scalars.device, dtype=scalars.dtype))
        tokens = torch.cat([scalar_token.unsqueeze(1), depth_tokens], dim=1)
        encoded = self.encoder_norm(self.encoder(tokens))

        global_token = encoded[:, 0]
        depth_latent = encoded[:, 1:]
        if self.decoder_mode == "coord_query":
            coords = self._query_coords(z_query, batch, nz, x.device, x.dtype)
            query_latent = self._interpolate_latent(depth_latent, coords)
            query_features = torch.cat([query_latent, coords.unsqueeze(-1)], dim=-1)
            decoded = self.query_decoder(query_features)
            dose = F.softplus(self.dose_head(decoded).squeeze(-1))
            letd = self.letd_head(decoded).squeeze(-1)
            outputs = self._profile_outputs(decoded, dose, letd)
        else:
            decoded_grid = self.fixed_grid_decoder(depth_latent)
            dose_grid = F.softplus(self.dose_head(decoded_grid).squeeze(-1))
            letd_grid = self.letd_head(decoded_grid).squeeze(-1)
            outputs_grid = self._profile_outputs(decoded_grid, dose_grid, letd_grid)
            if z_query is None:
                outputs = outputs_grid
            else:
                coords = self._query_coords(z_query, batch, nz, x.device, x.dtype)
                outputs = {
                    name: self._interpolate_values(values, coords)
                    for name, values in outputs_grid.items()
                }
        pooled = depth_latent.mean(dim=1)
        r80 = F.softplus(self.r80_head(torch.cat([global_token, pooled], dim=-1)).squeeze(-1)) * 40.0

        outputs["r80"] = r80
        return self.calibration(outputs)

    def param_count(self) -> int:
        """Number of trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _positional_encoding(self, nz: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos = self.pos_embedding
        if nz != pos.shape[1]:
            pos = F.interpolate(pos.transpose(1, 2), size=nz, mode="linear", align_corners=True).transpose(1, 2)
        return pos.to(device=device, dtype=dtype)

    @staticmethod
    def _query_coords(
        z_query: Tensor | None,
        batch: int,
        nz: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if z_query is None:
            coords = torch.linspace(0.0, 1.0, nz, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)
        else:
            coords = z_query.to(device=device, dtype=dtype)
            if coords.ndim == 1:
                coords = coords.unsqueeze(0).expand(batch, -1)
            elif coords.ndim != 2 or coords.shape[0] != batch:
                raise ValueError(f"z_query must have shape (Nq,) or ({batch},Nq), got {tuple(coords.shape)}")
            if bool((coords < 0).any() or (coords > 1).any()):
                denom = coords.amax(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
                coords = coords / denom
            coords = coords.clamp(0.0, 1.0)
        return coords

    @staticmethod
    def _interpolate_latent(depth_latent: Tensor, coords: Tensor) -> Tensor:
        batch, nz, channels = depth_latent.shape
        if coords.shape[0] != batch:
            raise ValueError(f"coords must have batch dimension {batch}, got {tuple(coords.shape)}")

        position = coords.clamp(0.0, 1.0) * max(nz - 1, 0)
        lower_idx = position.floor().to(dtype=torch.long).clamp(0, max(nz - 1, 0))
        upper_idx = (lower_idx + 1).clamp(0, max(nz - 1, 0))
        upper_weight = (position - lower_idx.to(dtype=position.dtype)).unsqueeze(-1)
        lower_weight = 1.0 - upper_weight

        gather_lower = lower_idx.unsqueeze(-1).expand(-1, -1, channels)
        gather_upper = upper_idx.unsqueeze(-1).expand(-1, -1, channels)
        lower = torch.gather(depth_latent, dim=1, index=gather_lower)
        upper = torch.gather(depth_latent, dim=1, index=gather_upper)
        return lower.mul(lower_weight) + upper.mul(upper_weight)

    @staticmethod
    def _interpolate_values(values: Tensor, coords: Tensor) -> Tensor:
        return BraggTransporterV0._interpolate_latent(values.unsqueeze(-1), coords).squeeze(-1)

    def _profile_outputs(self, decoded: Tensor, dose: Tensor, letd: Tensor) -> dict[str, Tensor]:
        outputs = {"dose": dose, "letd": letd}
        if self.predict_lett:
            outputs["lett"] = F.softplus(self.lett_head(decoded).squeeze(-1))
        if self.predict_fluence:
            outputs["fluence"] = F.softplus(self.fluence_head(decoded).squeeze(-1))
        return outputs


def _model_params(cfg: ModelConfig | dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return asdict(ModelConfig())
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)
