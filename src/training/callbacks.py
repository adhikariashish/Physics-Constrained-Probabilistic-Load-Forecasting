from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class EarlyStopping:
    """
    Simple early stopping on a scalar metric.

    mode:
      - "min": lower is better
      - "max": higher is better
    """
    patience: int
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"

    best: Optional[float] = None
    bad_epochs: int = 0
    stopped: bool = False

    def step(self, value: float) -> bool:
        """
        Update early stopping state.
        Returns True if improved, else False.
        """
        v = float(value)

        if self.best is None:
            self.best = v
            self.bad_epochs = 0
            return True

        if self.mode == "min":
            improved = (self.best - v) > float(self.min_delta)
        else:
            improved = (v - self.best) > float(self.min_delta)

        if improved:
            self.best = v
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= int(self.patience):
                self.stopped = True

        return improved