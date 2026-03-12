from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from src.models.backbones.transformer import (
    TransformerBackbone,
    TransformerBackboneConfig,
)


@dataclass(frozen=True)
class GaussianOutput:
    mu: torch.Tensor
    sigma: torch.Tensor


class GaussianHead(nn.Module):
    """
    Maps a pooled context vector [B, D] to Gaussian parameters
    for a multi-horizon forecast.

    Output:
        mu:    [B, H]
        sigma: [B, H]
    """

    def __init__(
        self,
        *,
        in_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        min_sigma: float = 1e-3,
        sigma_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.min_sigma = float(min_sigma)
        self.sigma_activation = str(sigma_activation).lower()

        if self.sigma_activation not in {"softplus"}:
            raise ValueError(f"Unsupported sigma_activation={sigma_activation!r}")

        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2 * self.horizon),
        )

        self.softplus = nn.Softplus()

    def forward(self, context: torch.Tensor) -> GaussianOutput:
        if context.ndim != 2:
            raise ValueError(f"Expected context [B,D], got {tuple(context.shape)}")

        params = self.net(context)  # [B, 2H]
        mu = params[:, : self.horizon]
        raw_sigma = params[:, self.horizon :]

        sigma = self.softplus(raw_sigma) + self.min_sigma
        return GaussianOutput(mu=mu, sigma=sigma)


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

    def __init__(
        self,
        *,
        num_features: int,
        context_length: int,
        horizon: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        attention_type: str = "causal",
        positional_encoding_type: str = "learned",
        pooling: str = "last",
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
        min_sigma: float = 1e-3,
        sigma_activation: str = "softplus",
    ) -> None:
        super().__init__()

        backbone_cfg = TransformerBackboneConfig(
            num_features=int(num_features),
            context_length=int(context_length),
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            d_ff=int(d_ff),
            dropout=float(dropout),
            attention_type=str(attention_type),
            positional_encoding_type=str(positional_encoding_type),
            pooling=str(pooling),
        )
        self.backbone = TransformerBackbone(backbone_cfg)

        self.head = GaussianHead(
            in_dim=self.backbone.out_dim,
            horizon=int(horizon),
            hidden_dim=int(head_hidden_dim),
            dropout=float(head_dropout),
            min_sigma=float(min_sigma),
            sigma_activation=str(sigma_activation),
        )

    def forward(self, x: torch.Tensor) -> GaussianOutput:
        context = self.backbone(x)     # [B, D]
        out = self.head(context)       # GaussianOutput(mu, sigma)
        return out