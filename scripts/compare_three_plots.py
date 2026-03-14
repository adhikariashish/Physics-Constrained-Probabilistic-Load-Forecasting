from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

# resolve project root
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

def load_horizon_metrics(run: str, split: str = "test") -> pd.DataFrame:
    path = Path("reports/metrics") / run / f"{split}_by_horizon.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_three_runs(
    runs: list[str],
    metric: str,
    split: str = "test",
    title: str | None = None,
) -> None:
    plt.figure(figsize=(8, 5))

    for run in runs:
        df = load_horizon_metrics(run, split=split)
        plt.plot(df["h"], df[metric], marker="o", label=run)

    plt.xlabel("Forecast Horizon (hours)")
    plt.ylabel(metric.upper())
    plt.title(title if title else f"{metric.upper()} by Forecast Horizon ({split})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_dir = Path("reports/figures/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{'_vs_'.join(runs)}_{metric}_{split}.png"
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_path.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True, type=str, help="exp_lstm_gaussian_v0")
    ap.add_argument("--run_b", required=True, type=str, help="exp_transformer_gaussian_v0")
    ap.add_argument("--run_c", required=True, type=str, help="exp_transformer_physics_v0")
    args = ap.parse_args()
    run_a = args.run_a
    run_b = args.run_b
    run_c = args.run_c

    runs = [
        run_a,
        run_b,
        run_c,
    ]

    plot_three_runs(
        runs,
        metric="mae_mw",
        split="test",
        title="Test MAE by Forecast Horizon: LSTM vs Transformer vs Transformer+Physics",
    )

    plot_three_runs(
        runs,
        metric="rmse_mw",
        split="test",
        title="Test RMSE by Forecast Horizon: LSTM vs Transformer vs Transformer+Physics",
    )

    plot_three_runs(
        runs,
        metric="nll",
        split="test",
        title="Test NLL by Forecast Horizon: LSTM vs Transformer vs Transformer+Physics",
    )

if __name__ == "__main__":
    main()