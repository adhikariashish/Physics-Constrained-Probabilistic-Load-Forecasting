from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """
    Set global random seeds for reproducibility.

    Args:
        seed: integer seed
        deterministic: if True, tries to make CUDA ops deterministic (may reduce speed)
        cudnn_benchmark: if True, enables cuDNN benchmarking (faster, but less deterministic)
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN flags
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)

    # Optional strict determinism (can raise on some ops)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass