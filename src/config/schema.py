from __future__ import annotations

from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# -----------------------------
# data.yaml schema
# -----------------------------
class DataSourceCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "ercot_hourly"
    version: str = "v0"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    years: list[int] = Field(default_factory=lambda : [2019,2021,2022,2023,2024,2025])
    sheet_name: str = "Sheet1"


class DataSchemaCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_col: str = "timestamp"
    target_col: str = "load_mw"
    timezone: str = "America/Chicago"
    freq: str = "H"
    hour_convention: str = "ending"

class DataTargetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "system_load"
    column: str = "ERCOT"
    unit: str = "MW"


class WindowCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_length: int = 168
    horizon: int = 24
    stride: int = 1
    allow_partial_windows: bool = False


class TimeSplitRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str = "YYYY-MM-DD"
    end: str = "YYYY-MM-DD"


class TimeSplitCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: TimeSplitRange = Field(default_factory=TimeSplitRange)
    val: TimeSplitRange = Field(default_factory=TimeSplitRange)
    test: TimeSplitRange = Field(default_factory=TimeSplitRange)


class RollingSplitCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    folds: int = 5
    step_size: int = 168


class SplitCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["time", "rolling"] = "time"
    time: TimeSplitCfg = Field(default_factory=TimeSplitCfg)
    rolling: RollingSplitCfg = Field(default_factory=RollingSplitCfg)


class ImputeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "ffill"
    limit: int = 3


class MissingnessCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["drop", "impute"] = "drop"
    max_missing_pct: float = 0.01
    impute: ImputeCfg = Field(default_factory=ImputeCfg)


class ScalingTargetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["standard", "robust", "none"] = "standard"


class ScalingFeaturesCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["standard", "robust", "none"] = "standard"


class ScalingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    target: ScalingTargetCfg = Field(default_factory=ScalingTargetCfg)
    features: ScalingFeaturesCfg = Field(default_factory=ScalingFeaturesCfg)
    fit_on: Literal["train"] = "train"


class CalendarHourCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    encoding: Literal["sin_cos"] = "sin_cos"


class CalendarDowCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    encoding: Literal["sin_cos"] = "sin_cos"


class CalendarCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    hour_of_day: CalendarHourCfg = Field(default_factory=CalendarHourCfg)
    day_of_week: CalendarDowCfg = Field(default_factory=CalendarDowCfg)
    is_weekend: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    is_holiday: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "calendar_name": "US"})


class LagsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    values: List[int] = Field(default_factory=lambda: [1, 2, 3, 24, 48, 72, 168])


class RollingFeaturesCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    windows: List[int] = Field(default_factory=lambda: [24, 168])
    stats: List[str] = Field(default_factory=lambda: ["mean", "std"])


class ExogenousCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    weather: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "source": "NOAA"})
    renewables: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    price: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})


class FeaturesCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calendar: CalendarCfg = Field(default_factory=CalendarCfg)
    lags: LagsCfg = Field(default_factory=LagsCfg)
    rolling: RollingFeaturesCfg = Field(default_factory=RollingFeaturesCfg)
    exogenous: ExogenousCfg = Field(default_factory=ExogenousCfg)


class DataOutputCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    processed_filename: str = "ercot_hourly_v0.parquet"
    zones_filename: str = "ercot_native_load_zones_v0.parquet"
    save_zones: bool = False


class DataCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: DataSourceCfg = Field(default_factory=DataSourceCfg)
    data_schema: DataSchemaCfg = Field(default_factory=DataSchemaCfg)
    target: DataTargetCfg = Field(default_factory=DataTargetCfg)
    windows: WindowCfg = Field(default_factory=WindowCfg)
    split: SplitCfg = Field(default_factory=SplitCfg)
    missingness: MissingnessCfg = Field(default_factory=MissingnessCfg)
    scaling: ScalingCfg = Field(default_factory=ScalingCfg)
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    output: DataOutputCfg = Field(default_factory=DataOutputCfg)


# -----------------------------
# model.yaml schema
# -----------------------------
class ModelIOCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_length: int = 168
    horizon: int = 24
    num_features: int = -1   # placeholder; filled after feature building
    target_dim: int = 1


class TransformerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    attention: Dict[str, Any] = Field(default_factory=lambda: {"type": "causal"})
    positional_encoding: Dict[str, Any] = Field(default_factory=lambda: {"type": "learned"})


class LSTMCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    hidden_size: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


class BackboneCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transformer: TransformerCfg = Field(default_factory=TransformerCfg)
    lstm: LSTMCfg = Field(default_factory=LSTMCfg)


class HeadParamCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sigma_activation: Literal["softplus"] = "softplus"
    min_sigma: float = 1.0e-3


class HeadOptionCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    distribution: Literal["gaussian"] = "gaussian"
    parameterization: HeadParamCfg = Field(default_factory=HeadParamCfg)
    predict: Dict[str, Any] = Field(default_factory=lambda: {"mean": True, "scale": True})

class HeadMlpCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hidden_dim: int = 128
    dropout: float = 0.1

class HeadCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "probabilistic_head"
    mlp: HeadMlpCfg = Field(default_factory=HeadMlpCfg)
    option: HeadOptionCfg = Field(default_factory=HeadOptionCfg)

class RegularizationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "transformer_forecaster"
    version: str = "v0"
    io: ModelIOCfg = Field(default_factory=ModelIOCfg)
    backbone: BackboneCfg = Field(default_factory=BackboneCfg)
    head: HeadCfg = Field(default_factory=HeadCfg)
    regularization: RegularizationCfg = Field(default_factory=RegularizationCfg)


# -----------------------------
# train.yaml schema
# -----------------------------
class RunCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: str = "runs"
    name: str = "exp_transformer_gaussian_v0"
    tags: List[str] = Field(default_factory=lambda: ["variantA", "hourly", "probabilistic", "physics"])
    notes: str = "placeholder run notes"


class PrecisionCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dtype: Literal["float32", "float64"] = "float32"
    amp: bool = False


class DeviceCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["auto", "cpu", "cuda"] = "auto"
    precision: PrecisionCfg = Field(default_factory=PrecisionCfg)


class ReproCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False


class OptimizerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["adamw", "adam"] = "adamw"
    lr: float = 3.0e-4
    betas: List[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    weight_decay: float = 1.0e-4


class SchedulerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    name: Literal["cosine", "reduce_on_plateau"] = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1.0e-6


class OptimizationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    scheduler: SchedulerCfg = Field(default_factory=SchedulerCfg)


class LoopCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 4
    log_every_steps: int = 50
    eval_every_epochs: int = 1


class EarlyStopMonitorCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str = "val/nll"
    mode: Literal["min", "max"] = "min"


class EarlyStoppingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    monitor: EarlyStopMonitorCfg = Field(default_factory=EarlyStopMonitorCfg)
    patience: int = 10
    min_delta: float = 0.0


class CheckpointingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    save_best: bool = True
    save_last: bool = True
    metric: str = "val/nll"
    mode: Literal["min", "max"] = "min"


class PhysicsBoundsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    lambda_: float = Field(0.1, alias="lambda")
    upper_bound: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "strategy": "train_max_margin",
        "margin": 0.05
    })


class PhysicsRampCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    lambda_: float = Field(0.1, alias="lambda")
    strategy: Literal["quantile"] = "quantile"
    quantile: float = 0.99


class PhysicsSmoothnessCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    lambda_: float = Field(0.01, alias="lambda")


class PhysicsOptionCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    apply_to: Literal["mean"] = "mean"


class PhysicsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    enabled: bool = True
    option: PhysicsOptionCfg = Field(default_factory=PhysicsOptionCfg)
    bounds: PhysicsBoundsCfg = Field(default_factory=PhysicsBoundsCfg)
    ramp: PhysicsRampCfg = Field(default_factory=PhysicsRampCfg)
    smoothness: PhysicsSmoothnessCfg = Field(default_factory=PhysicsSmoothnessCfg)


class ArtifactsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics_format: List[str] = Field(default_factory=lambda: ["json", "csv"])
    predictions_format: str = "parquet"


class LoggingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    save_resolved_config: bool = True
    save_predictions: bool = True
    save_plots: bool = True
    save_metrics: bool = True
    artifacts: ArtifactsCfg = Field(default_factory=ArtifactsCfg)


class TrainCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run: RunCfg = Field(default_factory=RunCfg)
    device: DeviceCfg = Field(default_factory=DeviceCfg)
    reproducibility: ReproCfg = Field(default_factory=ReproCfg)
    optimization: OptimizationCfg = Field(default_factory=OptimizationCfg)
    loop: LoopCfg = Field(default_factory=LoopCfg)
    early_stopping: EarlyStoppingCfg = Field(default_factory=EarlyStoppingCfg)
    checkpointing: CheckpointingCfg = Field(default_factory=CheckpointingCfg)
    physics: PhysicsCfg = Field(default_factory=PhysicsCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)


# -----------------------------
# eval.yaml schema
# -----------------------------
class MetricsPointCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    list: List[str] = Field(default_factory=lambda: ["mae", "rmse"])
    by_horizon: bool = True


class MetricsProbCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    list: List[str] = Field(default_factory=lambda: ["nll"])
    by_horizon: bool = True


class MetricsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    point: MetricsPointCfg = Field(default_factory=MetricsPointCfg)
    probabilistic: MetricsProbCfg = Field(default_factory=MetricsProbCfg)


class CalibrationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    intervals: Dict[str, Any] = Field(default_factory=lambda: {"levels": [0.5, 0.8, 0.9, 0.95], "by_horizon": True})
    reliability: Dict[str, Any] = Field(default_factory=lambda: {"bins": 10})
    sharpness: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "report_interval_width": True})


class PlotsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    forecast_examples: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "num_examples": 6})
    coverage_by_horizon: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    interval_width_by_horizon: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    reliability_diagram: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    error_heatmap: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "group_by": ["hour_of_day", "day_of_week"]})

class OutputDirCfg(BaseModel):
    reports_root: str = "reports"
    figures_dir: str = "reports/figures"
    metrics_dir: str = "reports/metrics"

class ReportingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: OutputDirCfg = Field(default_factory=OutputDirCfg)
    run_artifacts_dir: str = "runs"


class EvalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    splits: List[str] = Field(default_factory=lambda: ["val", "test"])
    metrics: MetricsCfg = Field(default_factory=MetricsCfg)
    calibration: CalibrationCfg = Field(default_factory=CalibrationCfg)
    plots: PlotsCfg = Field(default_factory=PlotsCfg)
    reporting: ReportingCfg = Field(default_factory=ReportingCfg)


# -----------------------------
# Root config object (what you import everywhere)
# -----------------------------
class AppConfig(BaseModel):
    """
    Unified, validated configuration object.
    Usage: cfg.data..., cfg.model..., cfg.train..., cfg.eval...
    """
    model_config = ConfigDict(extra="forbid")
    data: DataCfg = Field(default_factory=DataCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)