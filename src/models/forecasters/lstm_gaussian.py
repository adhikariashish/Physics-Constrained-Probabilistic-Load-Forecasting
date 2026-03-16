from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from src.models.backbones.lstm import LSTMEncoder, LSTMEncoderConfig
from src.models.heads.gaussian import GaussianHead, GaussianHeadConfig


@dataclass(frozen=True)
class LSTMGaussianForecasterConfig:
    input_dim: int
    horizon: int = 24
    lstm_hidden_dim: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    bidirectional: bool = False

    head_hidden_dim: int = 128
    head_dropout: float = 0.1
    sigma_min: float = 1e-6
    sigma_activation: str = "softplus"


class LSTMGaussianForecaster(nn.Module):
    """
    End-to-end probabilistic forecaster:
      X: [B,T,D] -> (mu, sigma) each [B,H]
    """

    def __init__(self, cfg: LSTMGaussianForecasterConfig) -> None:
        super().__init__()
        self.cfg = cfg

        enc_cfg = LSTMEncoderConfig(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
            bidirectional=cfg.bidirectional,
        )
        self.encoder = LSTMEncoder(enc_cfg)

        head_cfg = GaussianHeadConfig(
            in_dim=self.encoder.out_dim,
            horizon=cfg.horizon,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.head_dropout,
            min_sigma=cfg.sigma_min,
            sigma_activation=cfg.sigma_activation,
        )
        self.head = GaussianHead(head_cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.encoder(x)
        out = self.head(context)
        return out.mu, out.sigma