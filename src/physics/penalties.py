from __future__ import annotations

from dataclasses import dataclass

import torch

from src.physics.stats import PhysicsStats


@dataclass
class PhysicsPenaltyOutput:
    total: torch.Tensor
    bounds: torch.Tensor
    ramp: torch.Tensor
    smoothness: torch.Tensor


def bounds_penalty(
    mu_mw: torch.Tensor,
    *,
    upper_bound_mw: float,
) -> torch.Tensor:
    """
    Penalize predictions above the physical upper bound.

    mu_mw: [B, H]
    """
    excess = torch.relu(mu_mw - float(upper_bound_mw))
    return ((excess / float(upper_bound_mw)) ** 2).mean()


def ramp_penalty(
    mu_mw: torch.Tensor,
    *,
    ramp_limit_mw: float,
) -> torch.Tensor:
    """
    Penalize hour-to-hour predicted ramps above allowed ramp limit.

    mu_mw: [B, H]
    """
    if mu_mw.shape[1] < 2:
        return torch.zeros((), device=mu_mw.device, dtype=mu_mw.dtype)

    ramps = torch.abs(mu_mw[:, 1:] - mu_mw[:, :-1])   # [B, H-1]
    excess = torch.relu(ramps - float(ramp_limit_mw))
    scale = max(float(ramp_limit_mw), 1e-6)
    return ((excess/scale) ** 2).mean()


def smoothness_penalty(mu_mw: torch.Tensor) -> torch.Tensor:
    """
    Penalize second-order differences (curvature) for smoother trajectories.

    mu_mw: [B, H]
    """
    if mu_mw.shape[1] < 3:
        return torch.zeros((), device=mu_mw.device, dtype=mu_mw.dtype)

    second_diff = mu_mw[:, 2:] - 2.0 * mu_mw[:, 1:-1] + mu_mw[:, :-2]
    scale = torch.clamp(mu_mw.abs().mean(), min=1.0)
    return ((second_diff/scale) ** 2).mean()


def compute_physics_penalty(
    *,
    mu_mw: torch.Tensor,
    cfg,
    stats: PhysicsStats,
) -> PhysicsPenaltyOutput:
    """
    Compute total weighted physics penalty from config.

    Expected cfg = cfg.train.physics
    """
    device = mu_mw.device
    dtype = mu_mw.dtype

    zero = torch.zeros((), device=device, dtype=dtype)

    bounds_term = zero
    ramp_term = zero
    smooth_term = zero

    # ---- bounds ----
    if bool(cfg.bounds.enabled):
        ub_cfg = cfg.bounds.upper_bound
        strategy = str(ub_cfg.get("strategy", "train_max_margin")).lower()
        if strategy != "train_max_margin":
            raise ValueError(f"Unsupported upper_bound strategy: {strategy!r}")

        margin = float(ub_cfg.get("margin", 0.05))
        upper_bound_mw = float(stats.train_max_mw) * (1.0 + margin)

        bounds_term = bounds_penalty(mu_mw, upper_bound_mw=upper_bound_mw)
        bounds_term = float(cfg.bounds.lambda_) * bounds_term

    # ---- ramp ----
    if bool(cfg.ramp.enabled):
        strategy = str(cfg.ramp.strategy).lower()
        if strategy != "quantile":
            raise ValueError(f"Unsupported ramp strategy: {strategy!r}")

        ramp_limit_mw = float(stats.ramp_limit_mw)
        ramp_term = ramp_penalty(mu_mw, ramp_limit_mw=ramp_limit_mw)
        ramp_term = float(cfg.ramp.lambda_) * ramp_term

    # ---- smoothness ----
    if bool(cfg.smoothness.enabled):
        smooth_term = smoothness_penalty(mu_mw)
        smooth_term = float(cfg.smoothness.lambda_) * smooth_term

    total = bounds_term + ramp_term + smooth_term

    return PhysicsPenaltyOutput(
        total=total,
        bounds=bounds_term,
        ramp=ramp_term,
        smoothness=smooth_term,
    )