from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GaussianHeadConfig:
    in_dim: int
    horizon: int
    hidden_dim: int = 128
    dropout: float = 0.1
    sigma_min: float = 1e-6


class GaussianHead(nn.Module):
    """
    Maps a context vector [B, in_dim] -> (mu, sigma) each [B, H]
    """

    def __init__(self, cfg: GaussianHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(cfg.hidden_dim, 2 * cfg.horizon),
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu:    [B, H]
            sigma: [B, H] (positive)
        """
        if context.ndim != 2:
            raise ValueError(f"Expected context [B,in_dim], got {tuple(context.shape)}")

        out = self.net(context)  # [B, 2H]
        h = int(self.cfg.horizon)

        mu = out[:, :h]
        raw_sigma = out[:, h:]

        # Ensure positivity + numerical stability
        sigma = F.softplus(raw_sigma) + float(self.cfg.sigma_min)

        return mu, sigma