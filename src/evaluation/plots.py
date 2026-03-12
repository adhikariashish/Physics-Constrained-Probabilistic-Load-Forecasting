from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.evaluation.calibration import ReliabilityResult


def plot_coverage_by_horizon(
    *,
    out_path: Path,
    coverage_by_h: Dict[float, np.ndarray],  # level -> [H]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = len(next(iter(coverage_by_h.values())))
    x = np.arange(1, H + 1)

    plt.figure()
    for level, cov in sorted(coverage_by_h.items(), key=lambda kv: kv[0]):
        plt.plot(x, cov, label=f"empirical@{level}")
        plt.hlines(level, xmin=1, xmax=H, linestyles="dashed")

    plt.xlabel("Horizon (hours)")
    plt.ylabel("Coverage")
    plt.title("Prediction Interval Coverage by Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_interval_width_by_horizon(
    *,
    out_path: Path,
    width_by_h: Dict[float, np.ndarray],  # level -> [H]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = len(next(iter(width_by_h.values())))
    x = np.arange(1, H + 1)

    plt.figure()
    for level, wid in sorted(width_by_h.items(), key=lambda kv: kv[0]):
        plt.plot(x, wid, label=f"width@{level}")

    plt.xlabel("Horizon (hours)")
    plt.ylabel("Avg Interval Width (MW)")
    plt.title("Prediction Interval Width by Horizon (Sharpness)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_reliability_pit_hist(
    *,
    out_path: Path,
    rel: ReliabilityResult,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.bar(rel.bin_centers, rel.pit_hist, width=(rel.bin_edges[1] - rel.bin_edges[0]))
    plt.hlines(1.0 / rel.bins, xmin=0.0, xmax=1.0, linestyles="dashed")
    plt.xlabel("PIT value")
    plt.ylabel("Probability")
    plt.title(f"PIT Histogram (mean={rel.pit_mean:.3f}, std={rel.pit_std:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()