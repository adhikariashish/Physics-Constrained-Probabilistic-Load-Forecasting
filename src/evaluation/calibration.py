from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class ReliabilityResult:
    bins: int
    bin_edges: np.ndarray            # [bins+1]
    bin_centers: np.ndarray          # [bins]
    pit_hist: np.ndarray             # [bins] counts normalized to sum=1
    pit_mean: float
    pit_std: float


@torch.no_grad()
def gaussian_pit(
    *,
    y_scaled: torch.Tensor,        # [N,H] scaled
    mu_scaled: torch.Tensor,       # [N,H] scaled
    sigma_scaled: torch.Tensor,    # [N,H] scaled
    bins: int = 10,
    eps: float = 1e-8,
) -> ReliabilityResult:
    """
    PIT = F(y) for predictive CDF F. For a calibrated model, PIT ~ Uniform(0,1).

    We flatten across horizons for one global PIT diagnostic (simple + robust).
    """
    if y_scaled.shape != mu_scaled.shape or sigma_scaled.shape != mu_scaled.shape:
        raise ValueError("y_scaled, mu_scaled, sigma_scaled must have same shape [N,H]")

    sigma = torch.clamp(sigma_scaled, min=eps)
    dist = torch.distributions.Normal(loc=mu_scaled, scale=sigma)
    pit = dist.cdf(y_scaled).clamp(0.0, 1.0)  # [N,H]
    pit_flat = pit.reshape(-1).detach().cpu().numpy()

    bin_edges = np.linspace(0.0, 1.0, int(bins) + 1)
    hist, _ = np.histogram(pit_flat, bins=bin_edges, density=False)
    hist = hist.astype(np.float64)
    hist = hist / max(1.0, hist.sum())

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return ReliabilityResult(
        bins=int(bins),
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        pit_hist=hist,
        pit_mean=float(np.mean(pit_flat)),
        pit_std=float(np.std(pit_flat)),
    )