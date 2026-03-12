from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ProbMetricsResult:
    # scalar summaries
    nll_mean: float
    mae_mean: float
    rmse_mean: float

    # by-horizon arrays (len = H)
    mae_by_h: np.ndarray
    rmse_by_h: np.ndarray
    nll_by_h: np.ndarray

    # calibration: dict[level] -> arrays len H
    coverage_by_h: Dict[float, np.ndarray]
    width_by_h: Dict[float, np.ndarray]


def _z_for_level(level: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns z such that P(|Z| <= z) = level for Z ~ N(0,1).
    i.e. symmetric central interval.
    """
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0,1), got {level}")

    # central interval: alpha = (1 - level)/2, upper quantile = 1 - alpha
    alpha = (1.0 - level) / 2.0
    q = 1.0 - alpha
    dist = torch.distributions.Normal(
        loc=torch.tensor(0.0, device=device, dtype=dtype),
        scale=torch.tensor(1.0, device=device, dtype=dtype),
    )
    return dist.icdf(torch.tensor(q, device=device, dtype=dtype))


@torch.no_grad()
def compute_probabilistic_metrics(
    *,
    mu_scaled: torch.Tensor,      # [N,H] in scaled space
    sigma_scaled: torch.Tensor,   # [N,H] in scaled space (std)
    y_scaled: torch.Tensor,       # [N,H] in scaled space
    y_mw: torch.Tensor,           # [N,H] in MW space
    y_mean_mw: torch.Tensor,      # [N,H] mean prediction in MW (inverse-transformed)
    y_std_mw: Optional[float],    # scaler std for y (MW per 1 scaled unit). None if unscaled.
    levels: List[float],
    eps: float = 1e-8,
) -> ProbMetricsResult:
    """
    Computes:
      - NLL (scaled space)
      - MAE/RMSE (MW)
      - per-horizon versions
      - coverage + interval width per horizon in MW
    """
    device = mu_scaled.device
    dtype = mu_scaled.dtype

    if mu_scaled.ndim != 2:
        raise ValueError(f"mu_scaled expected [N,H], got {tuple(mu_scaled.shape)}")
    if sigma_scaled.shape != mu_scaled.shape or y_scaled.shape != mu_scaled.shape:
        raise ValueError("mu/sigma/y must share same shape [N,H]")
    if y_mw.shape != mu_scaled.shape or y_mean_mw.shape != mu_scaled.shape:
        raise ValueError("y_mw and y_mean_mw must be [N,H]")

    N, H = mu_scaled.shape

    # --- NLL in scaled space ---
    sigma = torch.clamp(sigma_scaled, min=eps)
    # NLL per element: 0.5*log(2π) + log(sigma) + 0.5*((y-mu)/sigma)^2
    nll_elem = 0.5 * np.log(2.0 * np.pi) + torch.log(sigma) + 0.5 * ((y_scaled - mu_scaled) / sigma) ** 2
    nll_by_h = nll_elem.mean(dim=0)                 # [H]
    nll_mean = float(nll_elem.mean().item())

    # --- point metrics in MW space ---
    err = y_mean_mw - y_mw                          # [N,H]
    mae_by_h = err.abs().mean(dim=0)                # [H]
    rmse_by_h = torch.sqrt((err ** 2).mean(dim=0))  # [H]
    mae_mean = float(mae_by_h.mean().item())
    rmse_mean = float(rmse_by_h.mean().item())

    # --- calibration in MW space ---
    coverage_by_h: Dict[float, np.ndarray] = {}
    width_by_h: Dict[float, np.ndarray] = {}

    # sigma in MW: sigma_scaled * std_y (linear scaling)
    if y_std_mw is None:
        sigma_mw = sigma_scaled
    else:
        sigma_mw = sigma_scaled * float(y_std_mw)

    for level in levels:
        z = _z_for_level(level, device=device, dtype=dtype)  # scalar tensor
        lo = y_mean_mw - z * sigma_mw
        hi = y_mean_mw + z * sigma_mw

        covered = ((y_mw >= lo) & (y_mw <= hi)).float().mean(dim=0)  # [H]
        width = (hi - lo).mean(dim=0)                                # [H]

        coverage_by_h[float(level)] = covered.detach().cpu().numpy()
        width_by_h[float(level)] = width.detach().cpu().numpy()

    return ProbMetricsResult(
        nll_mean=nll_mean,
        mae_mean=mae_mean,
        rmse_mean=rmse_mean,
        mae_by_h=mae_by_h.detach().cpu().numpy(),
        rmse_by_h=rmse_by_h.detach().cpu().numpy(),
        nll_by_h=nll_by_h.detach().cpu().numpy(),
        coverage_by_h=coverage_by_h,
        width_by_h=width_by_h,
    )