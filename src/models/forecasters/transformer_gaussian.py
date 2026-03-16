from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from src.models.backbones.transformer import (
    TransformerBackbone,
    TransformerBackboneConfig,
)
from src.models.heads.gaussian import GaussianOutput,GaussianHead, GaussianHeadConfig

@dataclass(frozen=True)
class TransformerGaussianForecasterConfig:
    """
    Configuration for the end-to-end probabilistic transformer forecaster.
    """

    input_dim: int
    context_length: int
    horizon: int

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1

    attention_type: str = "causal"
    positional_encoding_type: str = "learned"
    pooling: str = "last"

    head_hidden_dim: int = 128
    head_dropout: float = 0.1
    min_sigma: float = 1e-3
    sigma_activation: str = "softplus"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {self.context_length}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be > 0, got {self.d_ff}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.head_hidden_dim <= 0:
            raise ValueError(
                f"head_hidden_dim must be > 0, got {self.head_hidden_dim}"
            )
        if not (0.0 <= self.head_dropout < 1.0):
            raise ValueError(
                f"head_dropout must be in [0, 1), got {self.head_dropout}"
            )
        if self.min_sigma <= 0.0:
            raise ValueError(f"min_sigma must be > 0, got {self.min_sigma}")



class TransformerGaussianForecaster(nn.Module):
    """
    End-to-end probabilistic transformer forecaster.

    Input:
        X: [B, C, F]

    Output:
        GaussianOutput:
            mu:    [B, H]
            sigma: [B, H]
    """

    def __init__(self, cfg: TransformerGaussianForecasterConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.sigma_activation not in {"softplus"}:
            raise ValueError(f"Unsupported sigma_activation={cfg.sigma_activation!r}")

        backbone_cfg = TransformerBackboneConfig(
            num_features=cfg.input_dim,
            context_length=cfg.context_length,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            attention_type=cfg.attention_type,
            positional_encoding_type=cfg.positional_encoding_type,
            pooling=cfg.pooling,
        )
        self.backbone = TransformerBackbone(backbone_cfg)

        head_cfg = GaussianHeadConfig(
            in_dim=self.backbone.out_dim,
            horizon=cfg.horizon,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.head_dropout,
            min_sigma=cfg.min_sigma,
            sigma_activation=cfg.sigma_activation,
        )
        self.head = GaussianHead(head_cfg)

    def forward(self, x: torch.Tensor) -> GaussianOutput:
        context = self.backbone(x)     # [B, D]
        out = self.head(context)       # GaussianOutput(mu, sigma)
        return out