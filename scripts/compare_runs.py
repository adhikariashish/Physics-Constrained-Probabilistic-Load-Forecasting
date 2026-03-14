from __future__ import annotations

# resolve project root
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import csv

from typing import Dict, List


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def compare_overall(run_a: str, run_b: str, split: str = "test") -> None:
    base = Path("reports/metrics")

    a_json = base / run_a / f"{split}_metrics.json"
    b_json = base / run_b / f"{split}_metrics.json"

    A = load_json(a_json)
    B = load_json(b_json)

    print(f"\n=== OVERALL COMPARISON ({split}) ===")
    print(f"{'metric':<15} {run_a:<30} {run_b:<30}")
    print("-" * 80)
    for key in ["mae_mean_mw", "rmse_mean_mw", "nll_mean"]:
        print(f"{key:<15} {A[key]:<30.4f} {B[key]:<30.4f}")


def compare_by_horizon(run_a: str, run_b: str, split: str = "test", out_csv: str | None = None) -> None:
    base = Path("reports/metrics")

    a_csv = base / run_a / f"{split}_by_horizon.csv"
    b_csv = base / run_b / f"{split}_by_horizon.csv"

    A = load_csv_rows(a_csv)
    B = load_csv_rows(b_csv)

    if len(A) != len(B):
        raise ValueError("Horizon lengths do not match.")

    rows = []
    print(f"\n=== BY-HORIZON COMPARISON ({split}) ===")
    print(f"{'h':<4} {'A_mae':<12} {'B_mae':<12} {'Δmae(B-A)':<14} {'A_rmse':<12} {'B_rmse':<12} {'Δrmse(B-A)':<14}")

    for ra, rb in zip(A, B):
        h = int(ra["h"])

        a_mae = float(ra["mae_mw"])
        b_mae = float(rb["mae_mw"])
        a_rmse = float(ra["rmse_mw"])
        b_rmse = float(rb["rmse_mw"])

        d_mae = b_mae - a_mae
        d_rmse = b_rmse - a_rmse

        print(f"{h:<4} {a_mae:<12.2f} {b_mae:<12.2f} {d_mae:<14.2f} {a_rmse:<12.2f} {b_rmse:<12.2f} {d_rmse:<14.2f}")

        row = {
            "h": h,
            f"{run_a}_mae": a_mae,
            f"{run_b}_mae": b_mae,
            "delta_mae_b_minus_a": d_mae,
            f"{run_a}_rmse": a_rmse,
            f"{run_b}_rmse": b_rmse,
            "delta_rmse_b_minus_a": d_rmse,
        }
        rows.append(row)

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[saved] {out_path.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True, type=str, help="exp_lstm_gaussian_v0")
    ap.add_argument("--run_b", required=True, type=str, help="exp_transformer_gaussian_v0")
    args = ap.parse_args()
    run_a = args.run_a
    run_b = args.run_b
    split = "test"

    compare_overall(run_a, run_b, split=split)
    compare_by_horizon(
        run_a,
        run_b,
        split=split,
        out_csv=f"reports/metrics/comparisons/{run_a}_vs_{run_b}_{split}.csv",
    )


if __name__ == "__main__":
    main()