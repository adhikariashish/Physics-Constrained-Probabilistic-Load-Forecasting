from __future__ import annotations

import math

import torch

from src.config import load_config
from src.data.datasets import make_dataloaders
from src.data.scaling import fit_standard_scalers


def test_fit_standard_scalers_train_only_smoke() -> None:
    cfg = load_config()

    # physics must be off for milestone 2B
    cfg.train.physics.enabled = False  # safe override for tests

    train_loader, val_loader, test_loader = make_dataloaders(
        cfg=cfg,
        source="system",
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        shuffle_train=False,
        return_meta=False,
        dtype=torch.float32,
    )

    x_scaler, y_scaler = fit_standard_scalers(train_loader, max_batches=2)

    assert math.isfinite(x_scaler.state_dict()["eps"])
    assert math.isfinite(y_scaler.state_dict()["mean"])
    assert math.isfinite(y_scaler.state_dict()["std"])
    assert y_scaler.state_dict()["std"] > 0.0