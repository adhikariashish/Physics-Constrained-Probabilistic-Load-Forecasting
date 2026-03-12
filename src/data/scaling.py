from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class StandardScaler1D:
    """StandardScaler for scalar target y."""
    mean_: float = 0.0
    std_: float = 1.0
    eps: float = 1e-8

    def fit(self, y: np.ndarray) -> "StandardScaler1D":
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y))
        if self.std_ < self.eps:
            self.std_ = 1.0
        return self

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean_) / (self.std_ + self.eps)

    def inverse(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * (self.std_ + self.eps) + self.mean_

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean_, "std": self.std_, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "StandardScaler1D":
        return cls(mean_=float(d["mean"]), std_=float(d["std"]), eps=float(d.get("eps", 1e-8)))


@dataclass
class StandardScalerVec:
    """Per-feature StandardScaler for X: [B,T,F]."""
    mean_: Optional[np.ndarray] = None  # [F]
    std_: Optional[np.ndarray] = None   # [F]
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "StandardScalerVec":
        X = np.asarray(X, dtype=np.float64)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < self.eps] = 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScalerVec not fitted.")
        mean = torch.as_tensor(self.mean_, device=x.device, dtype=x.dtype).view(1, 1, -1)
        std = torch.as_tensor(self.std_, device=x.device, dtype=x.dtype).view(1, 1, -1)
        return (x - mean) / (std + self.eps)

    def inverse(self, x_scaled: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScalerVec not fitted.")
        mean = torch.as_tensor(self.mean_, device=x_scaled.device, dtype=x_scaled.dtype).view(1, 1, -1)
        std = torch.as_tensor(self.std_, device=x_scaled.device, dtype=x_scaled.dtype).view(1, 1, -1)
        return x_scaled * (std + self.eps) + mean

    def state_dict(self) -> Dict[str, Any]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScalerVec not fitted.")
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "StandardScalerVec":
        obj = cls(eps=float(d.get("eps", 1e-8)))
        obj.mean_ = np.asarray(d["mean"], dtype=np.float64)
        obj.std_ = np.asarray(d["std"], dtype=np.float64)
        return obj


def _collect_for_fit(loader, *, max_batches: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        X, y = batch[:2]  # supports (X,y) or (X,y,meta)

        Xn = X.detach().cpu().numpy()  # [B,T,F]
        yn = y.detach().cpu().numpy()  # [B,H]

        xs.append(Xn.reshape(-1, Xn.shape[-1]))  # [B*T, F]
        ys.append(yn.reshape(-1))                # [B*H]

    if not xs or not ys:
        raise RuntimeError("No batches collected for scaling fit.")

    X_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    return X_all, y_all


def fit_standard_scalers(train_loader, *, max_batches: Optional[int] = None) -> Tuple[StandardScalerVec, StandardScaler1D]:
    """
    Fit feature and target scalers using ONLY training data.
    """
    X_all, y_all = _collect_for_fit(train_loader, max_batches=max_batches)
    x_scaler = StandardScalerVec().fit(X_all)
    y_scaler = StandardScaler1D().fit(y_all)
    return x_scaler, y_scaler