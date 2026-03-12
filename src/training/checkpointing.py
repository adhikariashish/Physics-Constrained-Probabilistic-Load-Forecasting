from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    best_ckpt: Path
    last_ckpt: Path
    scalers_dir: Path


def make_run_dirs(run_root: str, run_name: str) -> RunPaths:
    """
    Create standard run directory structure.
    """
    run_dir = Path(run_root) / str(run_name)
    checkpoints_dir = run_dir / "checkpoints"
    scalers_dir = run_dir / "scalers"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = checkpoints_dir / "best.pt"
    last_ckpt = checkpoints_dir / "last.pt"

    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        best_ckpt=best_ckpt,
        last_ckpt=last_ckpt,
        scalers_dir=scalers_dir,
    )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    metrics: Dict[str, float],
) -> None:
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "step": int(step),
        "metrics": dict(metrics),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        try:
            payload["scheduler_state"] = scheduler.state_dict()
        except Exception:
            # some custom schedulers may not implement state_dict
            payload["scheduler_state"] = None

    torch.save(payload, path)


def load_checkpoint(path: Path, *, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)