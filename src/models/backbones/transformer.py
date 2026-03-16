from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding for sequences of fixed max length.
    Input/Output shape:
        x: [B, C, D]
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.embedding = nn.Embedding(self.max_len, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,C,D], got {tuple(x.shape)}")

        b, c, d = x.shape
        if c > self.max_len:
            raise ValueError(
                f"Sequence length {c} exceeds max_len={self.max_len} for learned positional encoding."
            )

        pos_idx = torch.arange(c, device=x.device)
        pos = self.embedding(pos_idx)  # [C, D]
        pos = pos.unsqueeze(0)  # [1, C, D]
        return x + pos


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Input/Output shape:
        x: [B, C, D]
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.d_model = int(d_model)

        pe = torch.zeros(self.max_len, self.d_model)  # [C, D]
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)  # [C,1]

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as buffer so it moves with model but is not trainable
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,C,D], got {tuple(x.shape)}")

        _, c, _ = x.shape
        if c > self.max_len:
            raise ValueError(
                f"Sequence length {c} exceeds max_len={self.max_len} for sinusoidal positional encoding."
            )

        return x + self.pe[:, :c, :]


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns an upper-triangular boolean mask of shape [C, C].
    True values are masked (not attendable).
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

@dataclass(frozen=True)
class TransformerBackboneConfig:
    num_features: int
    context_length: int

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1

    attention_type: Literal["causal", "full"] = "causal"
    positional_encoding_type: Literal["learned", "sinusoidal"] = "learned"
    pooling: Literal["last", "mean"] = "last"


# ---------------------------------------------------------
# Backbone
# ---------------------------------------------------------

class TransformerBackbone(nn.Module):
    """
    Pure transformer encoder backbone for time series.

    Input:
        X: [B, C, F]

    Output:
        context: [B, D]   (pooled sequence representation)

    Notes:
    - This backbone does NOT produce forecast outputs.
    - It only encodes the context window into a latent representation.
    - A forecaster head (e.g. Gaussian head) will sit on top.
    """

    def __init__(self, cfg: TransformerBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}"
            )

        # project raw features F -> d_model
        self.input_proj = nn.Linear(int(cfg.num_features), int(cfg.d_model))

        # positional encoding
        if cfg.positional_encoding_type == "learned":
            self.positional_encoding = LearnedPositionalEncoding(
                max_len=int(cfg.context_length),
                d_model=int(cfg.d_model),
            )
        elif cfg.positional_encoding_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                max_len=int(cfg.context_length),
                d_model=int(cfg.d_model),
            )
        else:
            raise ValueError(
                f"Unsupported positional_encoding_type={cfg.positional_encoding_type!r}"
            )

        # transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.d_model),
            nhead=int(cfg.n_heads),
            dim_feedforward=int(cfg.d_ff),
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,  # stable modern default
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(cfg.n_layers),
        )

        self.out_dim = int(cfg.d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode full sequence.

        Args:
            x: [B, C, F]

        Returns:
            h: [B, C, D]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,C,F], got {tuple(x.shape)}")

        b, c, f = x.shape
        if f != self.cfg.num_features:
            raise ValueError(
                f"Expected num_features={self.cfg.num_features}, got input last dim={f}"
            )
        if c > self.cfg.context_length:
            raise ValueError(
                f"Input sequence length {c} exceeds configured context_length={self.cfg.context_length}"
            )

        # [B, C, F] -> [B, C, D]
        h = self.input_proj(x)

        # add position information
        h = self.positional_encoding(h)

        # optional causal mask
        attn_mask: Optional[torch.Tensor] = None
        if self.cfg.attention_type == "causal":
            attn_mask = build_causal_mask(seq_len=c, device=x.device)
        elif self.cfg.attention_type != "full":
            raise ValueError(f"Unsupported attention_type={self.cfg.attention_type!r}")

        # [B, C, D]
        h = self.encoder(h, mask=attn_mask)
        return h

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        """
        Pool encoded sequence into a single context vector.

        Args:
            h: [B, C, D]

        Returns:
            context: [B, D]
        """
        if h.ndim != 3:
            raise ValueError(f"Expected h shape [B,C,D], got {tuple(h.shape)}")

        if self.cfg.pooling == "last":
            return h[:, -1, :]
        elif self.cfg.pooling == "mean":
            return h.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling={self.cfg.pooling!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F]

        Returns:
            context: [B, D]
        """
        h = self.encode(x)
        context = self.pool(h)
        return context