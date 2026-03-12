from __future__ import annotations

import math
from typing import Literal

import torch


def gaussian_nll(
    *,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood for heteroscedastic sigma.
    Shapes:
      y, mu, sigma: [B, H] (or any broadcast-compatible)
    """

    if y.shape != mu.shape or y.shape != sigma.shape:
        raise ValueError(f"Expected y,mu,sigma same shape. Got y={y.shape}, mu={mu.shape}, sigma={sigma.shape}")

    if torch.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive everywhere (check softplus + sigma_min).")

    # NLL = 0.5*log(2π) + log(sigma) + 0.5*((y-mu)/sigma)^2
    nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()

    raise ValueError(f"Unknown reduction={reduction!r}")