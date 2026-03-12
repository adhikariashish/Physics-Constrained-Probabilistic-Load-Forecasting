from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

@dataclass(frozen=True)
class LSTMEncoderConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


class LSTMEncoder(nn.Module):
    """
    Encodes a sequence X: [B, T, D] into a context vector: [B, H]
    using the final hidden state from the top LSTM layer.
    """

    def __init__(self, cfg: LSTMEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # PyTorch LSTM: dropout is applied between LSTM layers when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=float(cfg.dropout) if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )

        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            context: [B, H_out]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B,T,D], got {tuple(x.shape)}")

        # output: [B, T, H_out], (h_n, c_n)
        _, (h_n, _) = self.lstm(x)

        # h_n: [num_layers * num_directions, B, hidden_dim]
        # take the top layer's hidden state(s)
        if self.cfg.bidirectional:
            # last layer has two directions stacked at the end
            # indices: [-2] forward, [-1] backward
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2*hidden_dim]
        else:
            h_last = h_n[-1]  # [B, hidden_dim]

        return h_last