from __future__ import annotations

import argparse
import json

# resolve project root
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional

import pandas as pd
import torch

from src.config import load_config
from src.data.datasets import load_source_df
from src.data.scaling import StandardScalerVec, StandardScaler1D
from src.models.build import build_model


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_scalers(run_dir: Path) -> tuple[Optional[StandardScalerVec], Optional[StandardScaler1D]]:
    scalers_dir = run_dir / "scalers"
    x_path = scalers_dir / "x_scaler.json"
    y_path = scalers_dir / "y_scaler.json"

    if not x_path.exists() or not y_path.exists():
        return None, None

    x_scaler = StandardScalerVec.from_state_dict(load_json(x_path))
    y_scaler = StandardScaler1D.from_state_dict(load_json(y_path))
    return x_scaler, y_scaler


def load_best_checkpoint(run_dir: Path, device: torch.device) -> dict:
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def z_value_for_level(level: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    alpha = (1.0 - float(level)) / 2.0
    q = 1.0 - alpha
    dist = torch.distributions.Normal(
        loc=torch.tensor(0.0, device=device, dtype=dtype),
        scale=torch.tensor(1.0, device=device, dtype=dtype),
    )
    return dist.icdf(torch.tensor(q, device=device, dtype=dtype))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=str, help="Run directory, e.g. runs/exp_transformer_physics_v0")
    ap.add_argument("--source", default="system", choices=["system", "zones"])
    ap.add_argument("--out", default="reports/latest_forecast/next_24h_forecast.json", type=str)
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load processed dataframe + feature columns
    df, feature_cols, target_col = load_source_df(cfg, args.source)

    ts_col = cfg.data.data_schema.timestamp_col
    context_length = int(cfg.data.windows.context_length)
    horizon = int(cfg.data.windows.horizon)

    if len(df) < context_length:
        raise ValueError(
            f"Not enough rows for context window. Need at least {context_length}, got {len(df)}"
        )

    # Most recent context slice
    ctx_df = df.iloc[-context_length:].copy()

    # Build model
    model = build_model(cfg, input_dim=len(feature_cols)).to(device)
    payload = load_best_checkpoint(run_dir, device=device)
    state_key = "model_state" if "model_state" in payload else "model"
    model.load_state_dict(payload[state_key])
    model.eval()

    # Load scalers if present
    x_scaler, y_scaler = load_scalers(run_dir)

    # Prepare X
    X = torch.tensor(
        ctx_df[feature_cols].to_numpy(),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)  # [1, C, F]

    if x_scaler is not None:
        Xs = x_scaler.transform(X)
    else:
        Xs = X

    # Predict
    out = model(Xs)
    if isinstance(out, tuple):
        mu, sigma = out
    else:
        mu, sigma = out.mu, out.sigma

    # [1, H] -> [H]
    mu = mu.squeeze(0)
    sigma = sigma.squeeze(0)

    # Inverse transform target if scaled
    if y_scaler is not None:
        mu_mw = y_scaler.inverse(mu)
        sigma_mw = sigma * float(y_scaler.std_)
    else:
        mu_mw = mu
        sigma_mw = sigma

    # Build forecast timestamps
    last_ts = pd.to_datetime(ctx_df[ts_col].iloc[-1])
    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )

    # Intervals
    z80 = z_value_for_level(0.80, device=mu_mw.device, dtype=mu_mw.dtype)
    z95 = z_value_for_level(0.95, device=mu_mw.device, dtype=mu_mw.dtype)

    lower_80 = mu_mw - z80 * sigma_mw
    upper_80 = mu_mw + z80 * sigma_mw
    lower_95 = mu_mw - z95 * sigma_mw
    upper_95 = mu_mw + z95 * sigma_mw

    rows = []
    for i in range(horizon):
        rows.append(
            {
                "timestamp": future_index[i].isoformat(),
                "mean_mw": float(mu_mw[i].item()),
                "sigma_mw": float(sigma_mw[i].item()),
                "lower_80_mw": float(lower_80[i].item()),
                "upper_80_mw": float(upper_80[i].item()),
                "lower_95_mw": float(lower_95[i].item()),
                "upper_95_mw": float(upper_95[i].item()),
            }
        )

    # -------------------------------------------------
    # Peak / risk detection (NEW)
    # -------------------------------------------------
    mean_values = [r["mean_mw"] for r in rows]
    peak_threshold_mw = 0.95 * max(mean_values)

    risk_rows = []
    for r in rows:
        is_peak_mean = r["mean_mw"] >= peak_threshold_mw
        is_peak_upper = r["upper_95_mw"] >= peak_threshold_mw

        risk_level = "normal"
        if is_peak_mean and is_peak_upper:
            risk_level = "high"
        elif is_peak_mean or is_peak_upper:
            risk_level = "moderate"

        risk_rows.append(
            {
                **r,
                "risk_level": risk_level,
                "is_peak_mean": bool(is_peak_mean),
                "is_peak_upper95": bool(is_peak_upper),
            }
        )

    peak_hour = max(risk_rows, key=lambda x: x["mean_mw"])

    result = {
        "run_name": run_dir.name,
        "source": args.source,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "context_length": context_length,
        "horizon": horizon,
        "last_context_timestamp": last_ts.isoformat(),
        "peak_summary": {
            "peak_threshold_mw": float(peak_threshold_mw),
            "peak_timestamp": peak_hour["timestamp"],
            "peak_mean_mw": float(peak_hour["mean_mw"]),
            "peak_upper_95_mw": float(peak_hour["upper_95_mw"]),
            "peak_risk_level": peak_hour["risk_level"],
        },
        "forecast": risk_rows,
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"[saved] {out_path.resolve()}")


if __name__ == "__main__":
    main()