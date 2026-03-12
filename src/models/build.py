from __future__ import annotations

from typing import Optional

from src.config.schema import AppConfig
from src.models.forecasters.lstm_gaussian import (
    LSTMGaussianForecaster,
    LSTMGaussianForecasterConfig,
)


def build_model(cfg: AppConfig, *, input_dim: int):
    """
    Build the forecasting model from YAML config.

    input_dim should come from the dataloader: X.shape[-1]
    """
    m = cfg.model

    # Decide backbone
    use_lstm = bool(getattr(m.backbone.lstm, "enabled", False))
    use_transformer = bool(getattr(m.backbone.transformer, "enabled", False))

    if use_lstm and use_transformer:
        raise ValueError("Config invalid: both LSTM and Transformer are enabled. Choose one.")
    if not use_lstm and not use_transformer:
        raise ValueError("Config invalid: neither LSTM nor Transformer is enabled. Choose one.")

    # Head parameterization
    head_opt = m.head.option
    if head_opt.distribution.lower() != "gaussian":
        raise ValueError(
            f"Milestone 2A supports only Gaussian head for now. Got distribution={head_opt.distribution!r}"
        )

    sigma_act = head_opt.parameterization.sigma_activation.lower()
    if sigma_act != "softplus":
        raise ValueError(
            f"Milestone 2A expects sigma_activation='softplus'. Got {sigma_act!r}"
        )

    min_sigma = float(head_opt.parameterization.min_sigma)

    # Head MLP (optional block we add)
    head_mlp = getattr(m.head, "mlp", None)
    head_hidden_dim = int(getattr(head_mlp, "hidden_dim", 128)) if head_mlp else 128
    head_dropout = float(getattr(head_mlp, "dropout", 0.1)) if head_mlp else 0.1

    horizon = int(m.io.horizon)

    if use_lstm:
        lstm_cfg = m.backbone.lstm

        forecaster_cfg = LSTMGaussianForecasterConfig(
            input_dim=input_dim,
            horizon=horizon,
            lstm_hidden_dim=int(lstm_cfg.hidden_size),
            lstm_layers=int(lstm_cfg.n_layers),
            lstm_dropout=float(lstm_cfg.dropout),
            bidirectional=bool(lstm_cfg.bidirectional),
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            sigma_min=min_sigma,
        )
        return LSTMGaussianForecaster(forecaster_cfg)

    # ---- Transformer path (future Milestone 3) ----
    raise NotImplementedError("Transformer forecaster build will be added in the Transformer milestone.")