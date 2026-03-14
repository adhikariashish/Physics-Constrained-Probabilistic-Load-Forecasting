from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class PhysicsStats:
    train_max_mw: float
    ramp_limit_mw: float


def compute_physics_stats(
    train_loader,
    *,
    y_scaler=None,
    ramp_quantile: float = 0.99,
    max_batches: Optional[int] = None,
) -> PhysicsStats:
    """
    Compute physics reference values from TRAINING DATA ONLY.

    Args:
        train_loader: dataloader returning (X, y) or (X, y, meta)
        y_scaler: optional target scaler. If present, y is inverse-transformed to MW.
        ramp_quantile: quantile for absolute hourly ramp limit
        max_batches: optional cap for quick experiments

    Returns:
        PhysicsStats(train_max_mw, ramp_limit_mw)
    """
    ys = []

    for b, batch in enumerate(train_loader):
        if max_batches is not None and b >= max_batches:
            break

        y = batch[1]  # [B, H]
        if y_scaler is not None:
            y = y_scaler.inverse(y)

        ys.append(y.detach().cpu().numpy())

    if not ys:
        raise RuntimeError("No target batches collected for physics stats.")

    y_all = np.concatenate(ys, axis=0)   # [N, H]

    train_max_mw = float(np.max(y_all))

    # absolute ramps along forecast horizon
    ramps = np.abs(np.diff(y_all, axis=1)).reshape(-1)
    if ramps.size == 0:
        ramp_limit_mw = 0.0
    else:
        ramp_limit_mw = float(np.quantile(ramps, ramp_quantile))

    return PhysicsStats(
        train_max_mw=train_max_mw,
        ramp_limit_mw=ramp_limit_mw,
    )