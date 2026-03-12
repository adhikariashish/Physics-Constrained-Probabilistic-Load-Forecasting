from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from src.config import load_config
from src.config.schema import AppConfig
from src.data.datasets import make_dataloaders
from src.data.scaling import fit_standard_scalers, StandardScalerVec, StandardScaler1D
from src.losses.gaussian_nll import gaussian_nll
from src.models.build import build_model
from src.training.callbacks import EarlyStopping
from src.training.checkpointing import make_run_dirs, save_checkpoint
from src.utils.seed import set_global_seed


def resolve_device(cfg: AppConfig) -> torch.device:
    name = str(cfg.train.device.name).lower().strip()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] device=cuda requested but CUDA not available; falling back to cpu")
        return torch.device("cpu")
    return torch.device(name)


def resolve_dtype(cfg: AppConfig) -> torch.dtype:
    dt = str(cfg.train.device.precision.dtype).lower().strip()
    return torch.float64 if dt == "float64" else torch.float32


@torch.no_grad()
def eval_epoch(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    x_scaler: Optional[StandardScalerVec],
    y_scaler: Optional[StandardScaler1D],
) -> Dict[str, float]:
    model.eval()

    nll_sum = 0.0
    mae_sum = 0.0
    rmse_sum = 0.0
    n_batches = 0

    for batch in loader:
        X, y = batch[:2]
        X = X.to(device)
        y = y.to(device)

        if x_scaler is not None and y_scaler is not None:
            Xs = x_scaler.transform(X)
            ys = y_scaler.transform(y)
        else:
            Xs, ys = X, y

        out = model(Xs)  # out.mu/out.sigma in scaled-space if scaling enabled
        if isinstance(out, tuple):
            mu, sigma = out
        else:
            mu, sigma = out.mu, out.sigma
        nll = gaussian_nll(y=ys, mu=mu, sigma=sigma)

        # MW metrics (inverse mean)
        if y_scaler is not None:
            mu_mw = y_scaler.inverse(mu)
            y_mw = y
        else:
            mu_mw = mu
            y_mw = y

        err = mu_mw - y_mw
        mae = err.abs().mean()
        rmse = torch.sqrt((err ** 2).mean())

        nll_sum += float(nll.item())
        mae_sum += float(mae.item())
        rmse_sum += float(rmse.item())
        n_batches += 1

    return {
        "nll": nll_sum / max(1, n_batches),
        "mae": mae_sum / max(1, n_batches),
        "rmse": rmse_sum / max(1, n_batches),
    }


def train(cfg: AppConfig, *, source: str = "system") -> Path:
    # physics not in 2B
    if bool(cfg.train.physics.enabled):
        raise ValueError("Milestone 2B: set train.physics.enabled=false (physics loss comes in Milestone 3).")

    # reproducibility
    set_global_seed(
        cfg.train.reproducibility.seed,
        deterministic=cfg.train.reproducibility.deterministic,
        cudnn_benchmark=cfg.train.reproducibility.cudnn_benchmark,
    )

    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg)
    torch.set_default_dtype(dtype)

    # dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        cfg=cfg,
        source=source,
        batch_size=int(cfg.train.loop.batch_size),
        num_workers=int(cfg.train.loop.num_workers),
        pin_memory=torch.cuda.is_available(),
        shuffle_train=True,
        return_meta=False,
        dtype=torch.float32,  # keep batch tensors float32; model default dtype handles math
    )

    # infer feature dim
    X0, y0 = next(iter(train_loader))
    num_features = int(X0.shape[-1])

    # model
    model = build_model(cfg, input_dim=num_features).to(device)

    # optimizer
    opt = cfg.train.optimization.optimizer
    opt_name = str(opt.name).lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(opt.lr),
            betas=tuple(float(x) for x in opt.betas),
            eps=float(opt.eps),
            weight_decay=float(opt.weight_decay),
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(opt.lr),
            betas=tuple(float(x) for x in opt.betas),
            eps=float(opt.eps),
            weight_decay=float(opt.weight_decay),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name!r}")

    # scheduler (cosine now; warmup later if you want)
    scheduler = None
    sch = cfg.train.optimization.scheduler
    if bool(sch.enabled):
        sch_name = str(sch.name).lower()
        if sch_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(cfg.train.loop.epochs), eta_min=float(sch.min_lr)
            )
        else:
            raise ValueError(f"Milestone 2B supports scheduler.name='cosine' only (got {sch.name!r}).")

    # run dirs
    run = cfg.train.run
    paths = make_run_dirs(run.dir, run.name)

    # scaling (train-only)
    x_scaler: Optional[StandardScalerVec] = None
    y_scaler: Optional[StandardScaler1D] = None
    if bool(cfg.data.scaling.enabled):
        if cfg.data.scaling.fit_on != "train":
            raise ValueError("Scaling fit_on must be 'train'.")
        if cfg.data.scaling.features.method != "standard" or cfg.data.scaling.target.method != "standard":
            raise ValueError("Milestone 2B supports only method='standard' for scaling.")

        x_scaler, y_scaler = fit_standard_scalers(train_loader, max_batches=None)

        (paths.scalers_dir / "x_scaler.json").write_text(json.dumps(x_scaler.state_dict(), indent=2))
        (paths.scalers_dir / "y_scaler.json").write_text(json.dumps(y_scaler.state_dict(), indent=2))
        print(f"[scaling] saved scalers -> {paths.scalers_dir.resolve()}")
    else:
        print("[scaling] disabled (cfg.data.scaling.enabled=false)")

    # early stopping
    es: Optional[EarlyStopping] = None
    if bool(cfg.train.early_stopping.enabled):
        mon = cfg.train.early_stopping.monitor
        if str(mon.metric) != "val/nll":
            raise ValueError("Milestone 2B early stopping supports monitor.metric='val/nll' only.")
        es = EarlyStopping(
            patience=int(cfg.train.early_stopping.patience),
            min_delta=float(cfg.train.early_stopping.min_delta),
            mode=str(mon.mode),
        )

    # checkpoint expectations
    ck = cfg.train.checkpointing
    if str(ck.metric) != "val/nll" or str(ck.mode).lower() != "min":
        raise ValueError("Milestone 2B checkpointing expects metric='val/nll' and mode='min'.")

    epochs = int(cfg.train.loop.epochs)
    log_every = int(cfg.train.loop.log_every_steps)
    eval_every = int(cfg.train.loop.eval_every_epochs)

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()

        for batch in train_loader:
            X, y = batch[:2]
            X = X.to(device)
            y = y.to(device)

            if x_scaler is not None and y_scaler is not None:
                Xs = x_scaler.transform(X)
                ys = y_scaler.transform(y)
            else:
                Xs, ys = X, y

            optimizer.zero_grad(set_to_none=True)
            out = model(Xs)
            if isinstance(out, tuple):
                mu, sigma = out
            else:
                mu, sigma = out.mu, out.sigma

            loss = gaussian_nll(y=ys, mu=mu, sigma=sigma)
            loss.backward()

            clip = float(cfg.model.regularization.gradient_clip_norm)
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

            optimizer.step()
            global_step += 1

            if log_every > 0 and (global_step % log_every == 0):
                with torch.no_grad():
                    if y_scaler is not None:
                        mu_mw = y_scaler.inverse(mu)
                        mae = (mu_mw - y).abs().mean().item()
                    else:
                        mae = (mu - y).abs().mean().item()
                print(f"[train][e{epoch:03d} step{global_step:07d}] nll={loss.item():.4f} mae(MW)={mae:.2f}")

        if scheduler is not None:
            scheduler.step()

        if eval_every > 0 and (epoch % eval_every == 0):
            val_m = eval_epoch(model=model, loader=val_loader, device=device, x_scaler=x_scaler, y_scaler=y_scaler)
            val_nll = float(val_m["nll"])

            # save last
            if bool(ck.enabled) and bool(ck.save_last):
                save_checkpoint(
                    paths.last_ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    metrics={"val/nll": val_nll, "val/mae": float(val_m["mae"]), "val/rmse": float(val_m["rmse"])},
                )

            # save best
            if bool(ck.enabled) and bool(ck.save_best) and val_nll < best_val:
                best_val = val_nll
                save_checkpoint(
                    paths.best_ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    metrics={"val/nll": val_nll, "val/mae": float(val_m["mae"]), "val/rmse": float(val_m["rmse"])},
                )
                print(f"[ckpt] best updated: val/nll={best_val:.6g}")

            print(f"[val][e{epoch:03d}] nll={val_m['nll']:.4f} mae={val_m['mae']:.2f} rmse={val_m['rmse']:.2f}")

            if es is not None:
                _ = es.step(val_nll)
                if es.stopped:
                    print(f"[early-stop] triggered at epoch {epoch}")
                    break

    # load best for test
    if paths.best_ckpt.exists():
        payload = torch.load(paths.best_ckpt, map_location=device)
        model.load_state_dict(payload["model_state"])
        print("[load] loaded best checkpoint for test")

    test_m = eval_epoch(model=model, loader=test_loader, device=device, x_scaler=x_scaler, y_scaler=y_scaler)
    print(f"[test] nll={test_m['nll']:.4f} mae={test_m['mae']:.2f} rmse={test_m['rmse']:.2f}")

    print(f"✅ run saved at: {paths.run_dir.resolve()}")
    return paths.run_dir


def main() -> None:
    cfg = load_config()
    train(cfg, source="system")


if __name__ == "__main__":
    main()