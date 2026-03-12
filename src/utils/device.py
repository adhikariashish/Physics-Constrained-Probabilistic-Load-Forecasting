from __future__ import annotations

import torch
from src.config.schema import AppConfig


def resolve_device(cfg: AppConfig) -> torch.device:
    name = str(cfg.train.device.name).lower().strip()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] device=cuda requested but CUDA not available; falling back to cpu")
        return torch.device("cpu")
    return torch.device(name)


def resolve_dtype(cfg: AppConfig) -> torch.dtype:
    dt = str(cfg.train.device.precision.dtype).lower().strip()
    if dt == "float64":
        return torch.float64
    return torch.float32