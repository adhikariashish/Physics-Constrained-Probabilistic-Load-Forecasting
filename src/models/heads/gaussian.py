from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GaussianHeadConfig:
    """
    Configuration for a Gaussian forecasting head.

    Attributes:
        in_dim: Input feature dimension of the pooled/context vector.
        horizon: Forecast horizon H.
        hidden_dim: Hidden dimension of the MLP head.
        dropout: Dropout probability used between MLP layers.
        min_sigma: Minimum standard deviation added after activation
            for numerical stability.
    """

    in_dim: int
    horizon: int
    hidden_dim: int = 128
    dropout: float = 0.1
    min_sigma: float = 1e-3
    sigma_activation: str = "softplus"

    def __post_init__(self) -> None:
        if self.in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got {self.in_dim}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.min_sigma <= 0.0:
            raise ValueError(f"min_sigma must be > 0, got {self.min_sigma}")


@dataclass(frozen=True)
class GaussianOutput:
    """
    Output container for Gaussian forecast parameters.

    Shapes:
        mu:    [B, H]
        sigma: [B, H]
    """

    mu: torch.Tensor
    sigma: torch.Tensor

    @property
    def variance(self) -> torch.Tensor:
        return self.sigma.pow(2)


class GaussianHead(nn.Module):
    """
    Maps a context vector of shape [B, D] to Gaussian parameters
    for a multi-horizon forecast.

    Input:
        context: [B, in_dim]

    Output:
        GaussianOutput(
            mu=[B, H],
            sigma=[B, H]
        )
    """

    def __init__(self, cfg: GaussianHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.sigma_activation not in {"softplus"}:
            raise ValueError(f"Unsupported sigma_activation={cfg.sigma_activation!r}")

        self.mlp = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.hidden_dim, 2 * cfg.horizon),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize linear layers with Xavier uniform and zero bias.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor) -> GaussianOutput:
        """
        Args:
            context: Tensor of shape [B, in_dim]

        Returns:
            GaussianOutput:
                mu:    [B, H]
                sigma: [B, H], strictly positive
        """
        if context.ndim != 2:
            raise ValueError(
                f"Expected context with shape [B, {self.cfg.in_dim}], "
                f"got {tuple(context.shape)}"
            )

        if context.size(-1) != self.cfg.in_dim:
            raise ValueError(
                f"Expected context.size(-1) == {self.cfg.in_dim}, "
                f"got {context.size(-1)}"
            )

        params = self.mlp(context)  # [B, 2H]
        mu, raw_sigma = torch.split(params, self.cfg.horizon, dim=-1)

        # Softplus ensures positivity; min_sigma prevents collapse to zero.
        sigma = F.softplus(raw_sigma) + self.cfg.min_sigma

        return GaussianOutput(mu=mu, sigma=sigma)