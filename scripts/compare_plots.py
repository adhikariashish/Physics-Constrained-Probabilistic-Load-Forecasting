from __future__ import annotations

# resolve project root
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_horizon_metrics(run:str, split: str = "test") -> pd.DataFrame:
    path = Path("reports/metrics")/run/f"{split}_by_horizon.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    df = pd.read_csv(path)
    return df

def plot_metrics(run_a: str, run_b: str, metric: str, split:str = "test"):
    df_a = load_horizon_metrics(run_a, split)
    df_b = load_horizon_metrics(run_b, split)

    plt.figure(figsize = (8,5))

    plt.plot(df_a["h"], df_a[metric], label=run_a, marker="o")
    plt.plot(df_b["h"], df_b[metric], label=run_b, marker="o")

    plt.xlabel("Forecast Horizon (hours)")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} by Forecast Horizon ({split})")


    plt.grid(True)
    plt.legend()

    out_dir = Path("reports/figures/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir/f"{run_a}_{run_b}_{metric}_{split}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"Saved figure to {out_path.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True, type=str, help="exp_lstm_gaussian_v0")
    ap.add_argument("--run_b", required=True, type=str, help="exp_transformer_gaussian_v0")
    args = ap.parse_args()
    run_a = args.run_a
    run_b = args.run_b
    split = "test"

    plot_metrics(run_a, run_b, "mae_mw", split)
    plot_metrics(run_a, run_b, "rmse_mw", split)
    plot_metrics(run_a, run_b, "nll", split)


if __name__ == "__main__":
    main()
