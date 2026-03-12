from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from src.config import load_config
from src.config.schema import AppConfig
from src.data.datasets import make_dataloaders
from src.data.scaling import StandardScalerVec, StandardScaler1D
from src.models.build import build_model

from src.evaluation.metrics import compute_probabilistic_metrics
from src.evaluation.calibration import gaussian_pit
from src.evaluation.plots import (
    plot_coverage_by_horizon,
    plot_interval_width_by_horizon,
    plot_reliability_pit_hist,
)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _load_scalers(run_dir: Path) -> Tuple[Optional[StandardScalerVec], Optional[StandardScaler1D]]:
    scalers_dir = run_dir / "scalers"
    x_path = scalers_dir / "x_scaler.json"
    y_path = scalers_dir / "y_scaler.json"
    if not x_path.exists() or not y_path.exists():
        return None, None
    x_scaler = StandardScalerVec.from_state_dict(_load_json(x_path))
    y_scaler = StandardScaler1D.from_state_dict(_load_json(y_path))
    return x_scaler, y_scaler


def _load_best_checkpoint(run_dir: Path, device: torch.device) -> Dict:
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


@torch.no_grad()
def _run_split(
    *,
    cfg: AppConfig,
    split: str,
    run_dir: Path,
    source: str,
    device: torch.device,
) -> Dict:
    # dataloaders (use train.yaml batch_size/num_workers as default)
    train_loader, val_loader, test_loader = make_dataloaders(
        cfg=cfg,
        source=source,
        batch_size=int(cfg.train.loop.batch_size),
        num_workers=int(cfg.train.loop.num_workers),
        pin_memory=torch.cuda.is_available(),
        shuffle_train=False,
        return_meta=False,
        dtype=torch.float32,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]

    # infer features
    X0, _ = next(iter(loader))
    num_features = int(X0.shape[-1])

    # build + load weights
    model = build_model(cfg, input_dim=num_features).to(device)
    payload = _load_best_checkpoint(run_dir, device=device)

    state_key = "model_state" if "model_state" in payload else "model"
    model.load_state_dict(payload[state_key])
    model.eval()

    # load scalers
    x_scaler, y_scaler = _load_scalers(run_dir)
    y_std_mw = float(y_scaler.std_) if y_scaler is not None else None

    # gather tensors
    mu_s_list: List[torch.Tensor] = []
    sig_s_list: List[torch.Tensor] = []
    y_s_list: List[torch.Tensor] = []

    y_mw_list: List[torch.Tensor] = []
    mu_mw_list: List[torch.Tensor] = []

    for batch in loader:
        X, y = batch[:2]
        X = X.to(device)
        y = y.to(device)

        if x_scaler is not None and y_scaler is not None:
            Xs = x_scaler.transform(X)
            ys = y_scaler.transform(y)
        else:
            Xs, ys = X, y

        out = model(Xs)
        if isinstance(out, tuple):
            mu_s, sig_s = out
        else:
            mu_s, sig_s = out.mu, out.sigma

        # enforce [B,H]
        if mu_s.ndim == 3 and mu_s.shape[-1] == 1:
            mu_s = mu_s.squeeze(-1)
        if sig_s.ndim == 3 and sig_s.shape[-1] == 1:
            sig_s = sig_s.squeeze(-1)

        mu_s_list.append(mu_s.detach().cpu())
        sig_s_list.append(sig_s.detach().cpu())
        y_s_list.append(ys.detach().cpu())

        if y_scaler is not None:
            mu_mw = y_scaler.inverse(mu_s)
            y_mw = y
        else:
            mu_mw = mu_s
            y_mw = y

        mu_mw_list.append(mu_mw.detach().cpu())
        y_mw_list.append(y_mw.detach().cpu())

    mu_scaled = torch.cat(mu_s_list, dim=0)
    sigma_scaled = torch.cat(sig_s_list, dim=0)
    y_scaled = torch.cat(y_s_list, dim=0)

    mu_mw = torch.cat(mu_mw_list, dim=0)
    y_mw = torch.cat(y_mw_list, dim=0)

    # eval.yaml driven toggles
    levels = list(getattr(cfg.eval.calibration.intervals, "levels", [0.5, 0.8, 0.9, 0.95]))
    bins = int(getattr(cfg.eval.calibration.reliability, "bins", 10))

    m = compute_probabilistic_metrics(
        mu_scaled=mu_scaled,
        sigma_scaled=sigma_scaled,
        y_scaled=y_scaled,
        y_mw=y_mw,
        y_mean_mw=mu_mw,
        y_std_mw=y_std_mw,
        levels=levels,
    )

    rel = None
    if bool(cfg.eval.calibration.enabled):
        rel = gaussian_pit(y_scaled=y_scaled, mu_scaled=mu_scaled, sigma_scaled=sigma_scaled, bins=bins)

    return {
        "split": split,
        "run_dir": str(run_dir),
        "levels": [float(x) for x in levels],
        "bins": bins,
        "horizon": int(mu_scaled.shape[1]),
        "n_samples": int(mu_scaled.shape[0]),
        "nll_mean": m.nll_mean,
        "mae_mean_mw": m.mae_mean,
        "rmse_mean_mw": m.rmse_mean,
        "mae_by_h": m.mae_by_h.tolist(),
        "rmse_by_h": m.rmse_by_h.tolist(),
        "nll_by_h": m.nll_by_h.tolist(),
        "coverage_by_h": {str(k): v.tolist() for k, v in m.coverage_by_h.items()},
        "width_by_h": {str(k): v.tolist() for k, v in m.width_by_h.items()},
        "reliability": None if rel is None else {
            "pit_mean": rel.pit_mean,
            "pit_std": rel.pit_std,
            "pit_hist": rel.pit_hist.tolist(),
            "bin_edges": rel.bin_edges.tolist(),
            "bin_centers": rel.bin_centers.tolist(),
        },
    }


def _save_metrics_and_csv(payload: Dict, *, cfg: AppConfig, run_name: str, split: str) -> None:
    out_cfg = cfg.eval.reporting.output_dir
    metrics_root = Path(out_cfg.metrics_dir) / run_name
    metrics_root.mkdir(parents=True, exist_ok=True)

    out_json = metrics_root / f"{split}_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {out_json.resolve()}")

    # by-horizon CSV (if requested)
    if bool(cfg.eval.metrics.point.by_horizon) or bool(cfg.eval.metrics.probabilistic.by_horizon) or bool(cfg.eval.calibration.intervals.by_horizon):
        out_csv = metrics_root / f"{split}_by_horizon.csv"
        H = int(payload["horizon"])
        levels = payload["levels"]
        cov = payload["coverage_by_h"]
        wid = payload["width_by_h"]

        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            header = ["h"]
            if bool(cfg.eval.metrics.point.enabled):
                header += ["mae_mw", "rmse_mw"]
            if bool(cfg.eval.metrics.probabilistic.enabled):
                header += ["nll"]
            if bool(cfg.eval.calibration.enabled) and bool(cfg.eval.calibration.sharpness["enabled"]):
                for L in levels:
                    header += [f"coverage_{L}", f"width_{L}"]
            writer.writerow(header)

            for h in range(H):
                row = [h + 1]
                if bool(cfg.eval.metrics.point.enabled):
                    row += [payload["mae_by_h"][h], payload["rmse_by_h"][h]]
                if bool(cfg.eval.metrics.probabilistic.enabled):
                    row += [payload["nll_by_h"][h]]
                if bool(cfg.eval.calibration.enabled) and bool(cfg.eval.calibration.sharpness["enabled"]):
                    for L in levels:
                        row += [cov[str(L)][h], wid[str(L)][h]]
                writer.writerow(row)

        print(f"[saved] {out_csv.resolve()}")


def _save_plots(payload: Dict, *, cfg: AppConfig, run_name: str, split: str) -> None:
    if not bool(cfg.eval.plots.enabled):
        return

    out_cfg = cfg.eval.reporting.output_dir
    figures_root = Path(out_cfg.figures_dir) / run_name
    figures_root.mkdir(parents=True, exist_ok=True)

    # coverage by horizon
    if bool(cfg.eval.plots.coverage_by_horizon["enabled"]):
        coverage = {float(k): torch.tensor(v).numpy() for k, v in payload["coverage_by_h"].items()}
        plot_coverage_by_horizon(out_path=figures_root / f"{split}_coverage_by_horizon.png", coverage_by_h=coverage)

    # interval width by horizon
    if bool(cfg.eval.plots.interval_width_by_horizon["enabled"]):
        width = {float(k): torch.tensor(v).numpy() for k, v in payload["width_by_h"].items()}
        plot_interval_width_by_horizon(out_path=figures_root / f"{split}_interval_width_by_horizon.png", width_by_h=width)

    # reliability diagram (PIT hist)
    if bool(cfg.eval.plots.reliability_diagram["enabled"]) and payload.get("reliability") is not None:
        rel = payload["reliability"]
        from src.evaluation.calibration import ReliabilityResult
        import numpy as np

        rr = ReliabilityResult(
            bins=int(payload["bins"]),
            bin_edges=np.array(rel["bin_edges"], dtype=np.float64),
            bin_centers=np.array(rel["bin_centers"], dtype=np.float64),
            pit_hist=np.array(rel["pit_hist"], dtype=np.float64),
            pit_mean=float(rel["pit_mean"]),
            pit_std=float(rel["pit_std"]),
        )
        plot_reliability_pit_hist(out_path=figures_root / f"{split}_pit_hist.png", rel=rr)

    print(f"[plots] saved under: {figures_root.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=str, help="runs/<run_name>")
    ap.add_argument("--source", default="system", choices=["system", "zones"])
    ap.add_argument("--split", default=None, choices=["train", "val", "test"], help="Override cfg.eval.splits")
    args = ap.parse_args()

    cfg = load_config()
    run_dir = Path(args.run)
    run_name = run_dir.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = [str(args.split)] if args.split is not None else list(cfg.eval.splits)

    for split in splits:
        payload = _run_split(cfg=cfg, split=split, run_dir=run_dir, source=str(args.source), device=device)

        print(
            f"[{split}] nll={payload['nll_mean']:.4f} "
            f"mae={payload['mae_mean_mw']:.2f}MW rmse={payload['rmse_mean_mw']:.2f}MW "
            f"(N={payload['n_samples']}, H={payload['horizon']})"
        )

        _save_metrics_and_csv(payload, cfg=cfg, run_name=run_name, split=split)
        _save_plots(payload, cfg=cfg, run_name=run_name, split=split)


if __name__ == "__main__":
    main()