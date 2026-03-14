from __future__ import annotations

from src.config.schema import AppConfig
from src.models.forecasters import (
    LSTMGaussianForecaster,
    LSTMGaussianForecasterConfig,
)
from src.models.forecasters.transformer_gaussian import TransformerGaussianForecaster


def build_model(cfg: AppConfig, *, input_dim: int):
    """
    Build the forecasting model from validated YAML config.

    Args:
        cfg: AppConfig
        input_dim: inferred from dataloader, e.g. X.shape[-1]

    Returns:
        Instantiated forecasting model
    """
    m = cfg.model

    # ---------------------------------------------------------
    # Backbone selection checks
    # ---------------------------------------------------------
    use_lstm = bool(getattr(m.backbone.lstm, "enabled", False))
    use_transformer = bool(getattr(m.backbone.transformer, "enabled", False))

    if use_lstm and use_transformer:
        raise ValueError("Config invalid: both LSTM and Transformer are enabled. Choose one.")
    if not use_lstm and not use_transformer:
        raise ValueError("Config invalid: neither LSTM nor Transformer is enabled. Choose one.")

    # Optional consistency check against model.name
    model_name = str(getattr(m, "name", "")).lower().strip()
    if use_lstm and model_name not in {"", "lstm_forecaster"}:
        raise ValueError(
            f"Config mismatch: backbone.lstm.enabled=True but model.name={m.name!r}. "
            f"Expected 'lstm_forecaster'."
        )
    if use_transformer and model_name not in {"", "transformer_forecaster"}:
        raise ValueError(
            f"Config mismatch: backbone.transformer.enabled=True but model.name={m.name!r}. "
            f"Expected 'transformer_forecaster'."
        )

    # ---------------------------------------------------------
    # Shared head checks
    # ---------------------------------------------------------
    head_opt = m.head.option

    if str(head_opt.distribution).lower() != "gaussian":
        raise ValueError(
            f"Only Gaussian head is supported right now. Got distribution={head_opt.distribution!r}"
        )

    sigma_act = str(head_opt.parameterization.sigma_activation).lower()
    if sigma_act != "softplus":
        raise ValueError(
            f"Expected sigma_activation='softplus'. Got {sigma_act!r}"
        )

    min_sigma = float(head_opt.parameterization.min_sigma)

    # Head MLP config
    head_mlp = getattr(m.head, "mlp", None)
    head_hidden_dim = int(getattr(head_mlp, "hidden_dim", 128)) if head_mlp is not None else 128
    head_dropout = float(getattr(head_mlp, "dropout", 0.1)) if head_mlp is not None else 0.1

    horizon = int(m.io.horizon)
    context_length = int(m.io.context_length)

    # ---------------------------------------------------------
    # LSTM path
    # ---------------------------------------------------------
    if use_lstm:
        lstm_cfg = m.backbone.lstm

        forecaster_cfg = LSTMGaussianForecasterConfig(
            input_dim=int(input_dim),
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

    # ---------------------------------------------------------
    # Transformer path
    # ---------------------------------------------------------
    transformer_cfg = m.backbone.transformer

    attention_type = str(transformer_cfg.attention.get("type", "causal")).lower()
    positional_encoding_type = str(
        transformer_cfg.positional_encoding.get("type", "learned")
    ).lower()

    return TransformerGaussianForecaster(
        num_features=int(input_dim),
        context_length=context_length,
        horizon=horizon,
        d_model=int(transformer_cfg.d_model),
        n_heads=int(transformer_cfg.n_heads),
        n_layers=int(transformer_cfg.n_layers),
        d_ff=int(transformer_cfg.d_ff),
        dropout=float(transformer_cfg.dropout),
        attention_type=attention_type,
        positional_encoding_type=positional_encoding_type,
        pooling="last",
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        min_sigma=min_sigma,
        sigma_activation=sigma_act,
    )